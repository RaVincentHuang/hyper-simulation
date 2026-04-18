from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


DEFAULT_MODEL = "qwen3.5-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


ENTITY_TAGS = """
PERSON: Human being, individual, or specific character.
COUNTRY: A nation with its own government.
LOC: Geographical location, natural region, body of water.
ORG: Organization, institution, company, government body.
FAC: Physical building, facility, structure.
GPE: Geopolitical entity, such as cities, states, provinces (but not countries).
NORP: Nationalities, religious or political groups.
PRODUCT: Physical object, vehicle, device, manufactured good.
WORK_OF_ART: Piece of art, publication, show.
LAW: Legal document, binding agreement.
LANGUAGE: Spoken or written human language.
OCCUPATION: Job, profession, trade.
EVENT: Phenomenon, historical event, sports match.
TEMPORAL: Time period, specific date, unit of time.
NUMBER: Mathematical number, quantity.
CONCEPT: Abstract idea, theoretical concept.
ORGANISM: Living being, such as animal, plant, or microorganism.
FOOD: Edible substance, dish, or cuisine.
MEDICAL: Medical condition, disease, symptom, or treatment.
ANATOMY: Body part, organ, or anatomical structure.
SUBSTANCE: Chemical element, compound, or material.
ASTRO: Astronomical object, such as a star, planet, or galaxy.
AWARD: Prize, honor, or recognition given to a person or organization.
VEHICLE: Means of transportation, such as a car, airplane, or bicycle.
THEORY: Scientific or philosophical theory, principle, or framework.
GROUP: Collection of individuals likes a family, team, class, or social group.
FEATURE: Distinctive attribute, property, or characteristic of an entity or concept.
ECONOMIC: Economic entity, such as a market, industry, or economic concept.
SOCIOLOGY: Concepts related to society, culture, sociology, or social interactions.
PHENOMENON: Natural or social phenomenon, such as climate change, pandemic.
ACTION: Action, behavior, or process, such as a specific activity, event, or process.
NOT_ENT: Use this if the synset does not fit any of the above categories.
""".strip()


PROMPT_TEMPLATE = """
# Role
You are an expert Legal NLP Data Processor. Your task is to transform raw, messy, and template-based legal documents (such as Non-Disclosure Agreements) into clean, coherent, and continuous narrative texts.

# Objectives
1. Text Normalization: Eliminate all non-semantic visual formatting, special bullet points, and fragmented line breaks. Make the text flow as a continuous, readable narrative.
2. Noise Reduction: Completely remove meaningless sections such as signature blocks, witness attestations, and page numbers.
3. Placeholder Replacement: Identify any template blanks (e.g., "_____", "[COMPANY]", "[DATE]") or empty fields, infer the missing entity type from the surrounding context, and replace it exactly with one of the provided <TAG>s.

# Rules & Guidelines
- Remove visual line breaks (`\\n`) that interrupt sentences. Merge them into a continuous sentence with proper spacing.
- Convert multi-level list markers (e.g., "1.", "a.", "(i)") into natural prose or simple continuous text if they break the narrative flow.
- Convert ALL-CAPS paragraphs (often used for legal emphasis) into standard Sentence case to improve coherence, while keeping proper nouns capitalized.
- Remove signature blocks entirely (e.g., anything following "IN WITNESS WHEREOF..." that merely lists "By: ___", "Title: ___", "Date: ___").
- Fix broken punctuation caused by blanks (e.g., "day of , 2013." should become "day of <TEMPORAL>, <TEMPORAL>.").
- For the blanks/placeholders, use ONLY the tags from the <Entity Tags> list below. Format the replacement as `<TAG>`.

# <Entity Tags>
{entity_tags}

# Example

## Input Text:
MUTUAL NON-DISCLOSURE/CONFIDENTIALITY AGREEMENT
THIS AGREEMENT effective the ____ day of ________, 2013.
BETWEEN:
[COMPANY NAME], located at _________________
OF THE FIRST PART
-and-
________________, as represented by the Chief Administrative Officer
OF THE SECOND PART

1. The confidential information includes:
 trade secrets
 financial data

IN WITNESS WHEREOF:
By: ____________________
Name: __________________

## Output Text:
This mutual non-disclosure and confidentiality agreement is effective the <TEMPORAL> day of <TEMPORAL>, <TEMPORAL>. This agreement is between <ORG>, located at <LOC>, of the first part, and <ORG>, as represented by the <OCCUPATION>, of the second part. The confidential information includes trade secrets and financial data.

# Task
Now, process the following input text according to the rules, objectives, and tags defined above. Return ONLY the processed text, without any introductory or concluding conversational filler.

## Input Text:
{user_input_text}
""".strip()


def _clean_angle_brackets(text: str) -> str:
	# 后处理：把 <TAG> 变成 TAG
	return re.sub(r"<\s*([A-Z_]+)\s*>", r"\1", text)


def _normalize_whitespace(text: str) -> str:
	text = re.sub(r"[ \t]+", " ", text)
	text = re.sub(r"\n{3,}", "\n\n", text)
	return text.strip()


class DashScopeChatClient:
	def __init__(self, model: str, base_url: str, api_key: str | None = None) -> None:
		resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY")
		if not resolved_key:
			raise EnvironmentError("Missing DASHSCOPE_API_KEY environment variable.")
		self._client = OpenAI(api_key=resolved_key, base_url=base_url)
		self.model = model

	def invoke(self, prompt: str) -> str:
		response = self._client.chat.completions.create(
			model=self.model,
			messages=[
				{"role": "system", "content": "You are a precise legal text cleaner. Output only the cleaned text."},
				{"role": "user", "content": prompt},
			],
			temperature=0.0,
			top_p=1.0,
			extra_body={"enable_thinking": False},
		)
		content = response.choices[0].message.content or ""
		if isinstance(content, list):
			return "\n".join(
				part if isinstance(part, str) else json.dumps(part, ensure_ascii=False)
				for part in content
			)
		return str(content)


def _build_prompt(raw_text: str) -> str:
	return PROMPT_TEMPLATE.format(entity_tags=ENTITY_TAGS, user_input_text=raw_text)


def _clean_with_llm(client: DashScopeChatClient, raw_text: str) -> str:
	prompt = _build_prompt(raw_text)
	cleaned = client.invoke(prompt)
	cleaned = _clean_angle_brackets(cleaned)
	cleaned = _normalize_whitespace(cleaned)
	return cleaned


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as fin:
		for line_no, line in enumerate(fin, start=1):
			text = line.strip()
			if not text:
				continue
			obj = json.loads(text)
			if not isinstance(obj, dict):
				raise ValueError(f"Line {line_no} is not a JSON object")
			rows.append(obj)
	return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as fout:
		for row in rows:
			fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def _find_target_keys(record: dict[str, Any]) -> list[str]:
	keys: list[str] = []
	# 兼容你提到的 promise / 部分，以及数据中常见的 premise
	for key in ("premise", "promise", "部分"):
		if key in record and isinstance(record.get(key), str) and record[key].strip():
			keys.append(key)
	return keys


def clean_contract_nli(
	input_path: str,
	output_path: str,
	model: str = DEFAULT_MODEL,
	base_url: str = DEFAULT_BASE_URL,
	limit: int | None = None,
	in_place: bool = False,
) -> dict[str, Any]:
	in_file = Path(input_path)
	out_file = Path(output_path)

	rows = _read_jsonl(in_file)
	if limit is not None:
		rows = rows[:limit]

	client = DashScopeChatClient(model=model, base_url=base_url)

	cleaned_count = 0
	touched_records = 0
	failed: list[dict[str, Any]] = []

	for idx, row in enumerate(tqdm(rows, desc="cleaning", unit="row"), start=1):
		target_keys = _find_target_keys(row)
		if not target_keys:
			continue

		touched_records += 1
		for key in target_keys:
			raw_text = row[key]
			try:
				new_text = _clean_with_llm(client=client, raw_text=raw_text)
			except Exception as exc:
				failed.append({"row_index": idx, "id": row.get("id"), "field": key, "reason": f"{type(exc).__name__}: {exc}"})
				continue

			if new_text and new_text != raw_text:
				row[key] = new_text
				cleaned_count += 1

	final_out = in_file if in_place else out_file
	_write_jsonl(final_out, rows)

	return {
		"input": str(in_file.resolve()),
		"output": str(final_out.resolve()),
		"rows": len(rows),
		"records_with_target_fields": touched_records,
		"cleaned_fields": cleaned_count,
		"failed": len(failed),
		"failed_examples": failed[:20],
		"model": model,
		"base_url": base_url,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Clean contract_nli JSONL fields with DashScope-compatible LLM.")
	parser.add_argument("--input", type=str, default="data/nli/contract_nli.jsonl", help="Input JSONL path")
	parser.add_argument(
		"--output",
		type=str,
		default="data/nli/contract_nli.cleaned.jsonl",
		help="Output JSONL path (ignored when --in-place is set)",
	)
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
	parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
	parser.add_argument("--limit", type=int, default=None, help="Only process first N rows")
	parser.add_argument("--in-place", action="store_true", help="Overwrite input file")
	args = parser.parse_args()

	summary = clean_contract_nli(
		input_path=args.input,
		output_path=args.output,
		model=args.model,
		base_url=args.base_url,
		limit=args.limit,
		in_place=args.in_place,
	)
	print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
