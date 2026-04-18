"""
Split over-limit contract_nli premise tokens into two semantic chunks using LLM.
Preserves referential integrity and semantic boundaries.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import spacy
from openai import OpenAI
from tqdm import tqdm


DEFAULT_MODEL = "qwen3.5-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TOKEN_LIMIT = 4096
DEFAULT_NUM_WORKERS = 4
DEFAULT_MAX_RETRIES = 5
DEFAULT_THREE_CHUNK_RATIO = 1.75


SPLIT_PROMPT_TEMPLATE = """
# Role
You are a legal document processing expert. Your task is to split a long legal premise into {num_chunks} coherent, semantically meaningful chunks.

# Objectives
1. Split the premise at natural semantic boundaries (e.g., between paragraphs, after a complete thought)
2. Preserve referential integrity: pronouns, references, and anaphora should remain clear in both chunks
3. Each chunk should be independently understandable
4. Minimize redundancy while ensuring no critical context is lost

# Rules
- Split at section breaks, paragraph boundaries, or after a complete legal clause when possible
- Avoid splitting mid-sentence or mid-reference
- Both chunks should roughly balance the token count if possible, but prioritize semantic coherence
- Each chunk should maintain legal/technical meaning without requiring the other chunk for comprehension
- If pronouns (e.g., "this agreement", "the party") will be clarified in a later chunk, repeat the antecedent in that chunk
- Return response in strict JSON format ONLY (no markdown, no code blocks)

# Output Format
Return ONLY valid JSON with this exact structure:
{output_schema}

# Input Premise to Split
{input_text}

---
Respond with ONLY the JSON object above, no additional text.
""".strip()


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
				{"role": "system", "content": "You are a legal text processing expert. Return valid JSON only."},
				{"role": "user", "content": prompt},
			],
			temperature=0.3,
			top_p=0.95,
			extra_body={"enable_thinking": False},
		)
		content = response.choices[0].message.content or ""
		if isinstance(content, list):
			return "\n".join(
				part if isinstance(part, str) else json.dumps(part, ensure_ascii=False)
				for part in content
			)
		return str(content)


def _setup_spacy_gpu() -> Any:
	"""Load spaCy model with optional GPU enable."""
	try:
		require_gpu_fn = getattr(spacy, "require_gpu", None)
		if callable(require_gpu_fn):
			require_gpu_fn()
		nlp = spacy.load("en_core_web_trf")
		print("[INFO] Loaded en_core_web_trf")
	except OSError:
		print("[WARN] en_core_web_trf not found, falling back to en_core_web_sm")
		nlp = spacy.load("en_core_web_sm")
	return nlp


def _extract_json_object(text: str) -> dict[str, Any]:
	"""Extract JSON object from model response with tolerant parsing."""
	text = text.strip()
	text = re.sub(r"^```(?:json)?\s*", "", text)
	text = re.sub(r"\s*```$", "", text)
	text = text.strip()

	# Fast path: direct JSON object
	try:
		obj = json.loads(text)
		if isinstance(obj, dict):
			return obj
	except Exception:
		pass

	# Fallback: extract first {...} block
	start = text.find("{")
	end = text.rfind("}")
	if start != -1 and end != -1 and end > start:
		candidate = text[start : end + 1]
		obj = json.loads(candidate)
		if isinstance(obj, dict):
			return obj

	raise ValueError("No valid JSON object found in model response")


def _count_tokens(nlp: Any, text: str) -> int:
	"""Count tokens using spaCy without components."""
	doc = nlp(text)
	return len(doc)


def _validate_chunks(nlp: Any, chunks: list[str], token_limit: int) -> bool:
	if not chunks or any(not chunk for chunk in chunks):
		return False
	for chunk in chunks:
		if _count_tokens(nlp, chunk) >= token_limit:
			return False
	return True


def _select_num_chunks(total_tokens: int, token_limit: int, three_chunk_ratio: float) -> int:
	if total_tokens > int(token_limit * three_chunk_ratio):
		return 3
	return 2


def _build_output_schema(num_chunks: int) -> str:
	lines = ["{"]
	for idx in range(1, num_chunks + 1):
		lines.append(f'  "chunk_{idx}": "...",')
	lines.append('  "split_rationale": "Brief explanation of where/why split was made"')
	lines.append("}")
	return "\n".join(lines)


def _fallback_split_locally(
	premise: str,
	nlp: Any,
	token_limit: int,
	num_chunks: int,
) -> tuple[list[str], str] | None:
	"""
	Local semantic fallback splitter: prefer sentence boundaries near midpoint.
	This avoids losing samples when LLM formatting/parsing fails.
	"""
	doc = nlp(premise)
	tokens = list(doc)
	if not tokens:
		return None

	total_tokens = len(tokens)
	if total_tokens <= token_limit:
		return None

	# Candidate sentence boundaries (token offsets)
	boundaries: list[int] = []
	for sent in doc.sents:
		end_idx = sent.end
		if 0 < end_idx < total_tokens:
			boundaries.append(end_idx)

	if not boundaries:
		# Fallback to evenly-spaced token offsets.
		boundaries = [int(total_tokens * k / num_chunks) for k in range(1, num_chunks)]

	# Choose split points close to evenly-spaced targets.
	targets = [int(total_tokens * k / num_chunks) for k in range(1, num_chunks)]
	chosen: list[int] = []
	used = set()
	for target in targets:
		candidate_order = sorted(boundaries, key=lambda x: abs(x - target))
		picked = None
		for c in candidate_order:
			if c not in used and 0 < c < total_tokens:
				picked = c
				break
		if picked is None:
			picked = target
		chosen.append(picked)
		used.add(picked)

	chosen = sorted(set(chosen))
	if len(chosen) != num_chunks - 1:
		return None

	all_points = [0] + chosen + [total_tokens]
	chunks: list[str] = []
	for start, end in zip(all_points[:-1], all_points[1:]):
		part = doc[start:end].text.strip()
		chunks.append(part)

	if _validate_chunks(nlp, chunks, token_limit):
		return chunks, "local_fallback_sentence_boundary"

	return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
	"""Read JSONL file."""
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
	"""Write JSONL file."""
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as fout:
		for row in rows:
			fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_split_prompt(text: str, num_chunks: int) -> str:
	"""Build split prompt with input text."""
	return SPLIT_PROMPT_TEMPLATE.format(
		input_text=text,
		num_chunks=num_chunks,
		output_schema=_build_output_schema(num_chunks),
	)


def _split_premise_with_llm(
	client: DashScopeChatClient,
	premise: str,
	nlp: Any,
	token_limit: int,
	max_retries: int,
	num_chunks: int,
) -> tuple[list[str], str] | None:
	"""
	Use LLM to split premise into two semantic chunks.
	Returns (chunk_1, chunk_2, rationale) or None on failure.
	"""
	base_prompt = _build_split_prompt(premise, num_chunks=num_chunks)
	for attempt in range(1, max_retries + 1):
		prompt = base_prompt
		if attempt > 1:
			prompt = (
				base_prompt
				+ "\n\nIMPORTANT: Return STRICT JSON only. Do not add any explanation outside JSON."
			)
		try:
			response = client.invoke(prompt)
			result = _extract_json_object(response)

			chunks = [str(result.get(f"chunk_{i}", "")).strip() for i in range(1, num_chunks + 1)]
			rationale = str(result.get("split_rationale", "")).strip() or "llm_split"

			if _validate_chunks(nlp, chunks, token_limit):
				return chunks, rationale
		except Exception as exc:
			if attempt == max_retries:
				print(f"[WARN] LLM split failed after retries: {exc}")

	# Final fallback: local sentence-boundary split
	return _fallback_split_locally(
		premise=premise,
		nlp=nlp,
		token_limit=token_limit,
		num_chunks=num_chunks,
	)


def _split_task_worker(
	idx: int,
	row: dict[str, Any],
	nlp: Any,
	model: str,
	base_url: str,
	token_limit: int,
	max_retries: int,
	three_chunk_ratio: float,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
	"""
	Worker function for parallel LLM split execution.
	Returns (index, updated_row_on_success, error_dict_on_failure).
	"""
	client = DashScopeChatClient(model=model, base_url=base_url)
	premise = row.get("premise", "").strip()
	record_id = row.get("id", str(idx))
	
	if not premise:
		return idx, None, None

	total_tokens = _count_tokens(nlp, premise)
	num_chunks = _select_num_chunks(
		total_tokens=total_tokens,
		token_limit=token_limit,
		three_chunk_ratio=three_chunk_ratio,
	)
	
	result = _split_premise_with_llm(
		client=client,
		premise=premise,
		nlp=nlp,
		token_limit=token_limit,
		max_retries=max_retries,
		num_chunks=num_chunks,
	)
	if result is None:
		return idx, None, {
			"index": idx,
			"id": record_id,
			"reason": "split failed after llm retries and local fallback",
		}
	
	chunks, rationale = result
	
	# Modify original record: premise becomes list, add chunked flag
	updated_row = row.copy()
	updated_row["premise"] = chunks
	updated_row["chunked"] = True
	updated_row["num_chunks"] = len(chunks)
	updated_row["_split_rationale"] = rationale
	
	return idx, updated_row, None


def split_contract_nli(
	input_path: str,
	output_path: str,
	model: str = DEFAULT_MODEL,
	base_url: str = DEFAULT_BASE_URL,
	token_limit: int = DEFAULT_TOKEN_LIMIT,
	num_workers: int = DEFAULT_NUM_WORKERS,
	max_retries: int = DEFAULT_MAX_RETRIES,
	three_chunk_ratio: float = DEFAULT_THREE_CHUNK_RATIO,
) -> dict[str, Any]:
	"""
	Split over-limit premises in contract_nli JSONL using LLM.
	
	Returns summary with statistics.
	"""
	in_file = Path(input_path)
	out_file = Path(output_path)
	
	rows = _read_jsonl(in_file)
	nlp = _setup_spacy_gpu()
	
	split_count = 0
	total_over_limit = 0
	failed_splits: list[dict[str, Any]] = []
	output_rows: list[dict[str, Any]] = []
	
	print("[INFO] Scanning for over-limit premises...")
	over_limit_indices: list[int] = []
	for idx, row in enumerate(rows):
		premise = row.get("premise", "").strip()
		if not premise:
			continue
		
		token_count = _count_tokens(nlp, premise)
		if token_count > token_limit:
			over_limit_indices.append(idx)
			total_over_limit += 1
	
	print(f"[INFO] Found {total_over_limit} over-limit premises to split")
	
	# Create a dictionary to store results by index, preserving order
	over_limit_set = set(over_limit_indices)
	result_dict: dict[int, dict[str, Any]] = {}
	
	# Process over-limit premises with thread pool
	print(f"[INFO] Starting parallel processing with {num_workers} workers...")
	
	with ThreadPoolExecutor(max_workers=num_workers) as executor:
		# Submit all over-limit tasks
		futures = {}
		for idx in over_limit_indices:
			row = rows[idx]
			future = executor.submit(
				_split_task_worker,
				idx,
				row,
				nlp,
				model,
				base_url,
				token_limit,
				max_retries,
				three_chunk_ratio,
			)
			futures[future] = idx
		
		# Collect results with progress bar
		for future in tqdm(as_completed(futures), total=len(futures), desc="splitting", unit="premise"):
			idx, updated_row, error = future.result()
			
			if error:
				failed_splits.append(error)
				result_dict[idx] = rows[idx]  # Keep original on failure
			elif updated_row:
				result_dict[idx] = updated_row
				split_count += 1
			else:
				# No update needed (empty premise)
				result_dict[idx] = rows[idx]
	
	# Build output preserving original order
	for idx, row in enumerate(rows):
		if idx in over_limit_set:
			output_rows.append(result_dict[idx])
		else:
			output_rows.append(row)
	
	_write_jsonl(out_file, output_rows)
	
	return {
		"input": str(in_file.resolve()),
		"output": str(out_file.resolve()),
		"total_rows": len(rows),
		"premises_over_limit": total_over_limit,
		"premises_split_success": split_count,
		"split_failed": len(failed_splits),
		"failed_examples": failed_splits[:10],
		"model": model,
		"base_url": base_url,
		"token_limit": token_limit,
		"num_workers": num_workers,
		"max_retries": max_retries,
		"three_chunk_ratio": three_chunk_ratio,
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Split over-limit contract_nli premises into semantic chunks using LLM."
	)
	parser.add_argument(
		"--input",
		type=str,
		default="data/nli/contract_nli.cleaned.jsonl",
		help="Input JSONL path",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="data/nli/contract_nli.split.jsonl",
		help="Output JSONL path",
	)
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
	parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
	parser.add_argument("--token-limit", type=int, default=DEFAULT_TOKEN_LIMIT)
	parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of parallel workers")
	parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Retry times per premise when LLM output is invalid")
	parser.add_argument("--three-chunk-ratio", type=float, default=DEFAULT_THREE_CHUNK_RATIO, help="If tokens > token_limit * ratio, split into three chunks")
	args = parser.parse_args()
	
	summary = split_contract_nli(
		input_path=args.input,
		output_path=args.output,
		model=args.model,
		base_url=args.base_url,
		token_limit=args.token_limit,
		num_workers=args.num_workers,
		max_retries=args.max_retries,
		three_chunk_ratio=args.three_chunk_ratio,
	)
	print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
