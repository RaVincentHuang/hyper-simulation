import ast
import json
import re
from pathlib import Path


INPUT_PATH = Path("data/nli/econ.jsonl")
OUTPUT_PATH = Path("data/nli/econ_qa.jsonl")


QA_PATTERN = re.compile(r"^\s*Question:\s*(.*?)\s*Answer:\s*(.*?)\s*$", re.DOTALL)


def extract_original_premise(premise: str) -> str | None:
	if not isinstance(premise, str):
		return None

	text = premise.strip()
	if not text.startswith("{"):
		return None

	try:
		parsed = ast.literal_eval(text)
	except (ValueError, SyntaxError):
		return None

	if not isinstance(parsed, dict):
		return None

	original_value = None
	for key, value in parsed.items():
		if str(key).strip().lower() == "original":
			original_value = value
			break

	if original_value is None:
		return None

	if isinstance(original_value, list):
		return " ".join(str(item).strip() for item in original_value if str(item).strip())

	original_text = str(original_value).strip()
	return original_text if original_text else None


def extract_question_answer(hypothesis: str) -> tuple[str, str] | None:
	if not isinstance(hypothesis, str):
		return None

	match = QA_PATTERN.match(hypothesis)
	if not match:
		return None

	question = match.group(1).strip()
	answer = match.group(2).strip()
	if not answer:
		return None

	return question, answer


def main() -> None:
	kept = 0

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	with INPUT_PATH.open("r", encoding="utf-8") as src, OUTPUT_PATH.open("w", encoding="utf-8") as dst:
		for line in src:
			row = json.loads(line)

			premise = extract_original_premise(row.get("premise", ""))
			if not premise:
				continue

			qa = extract_question_answer(row.get("hypothesis", ""))
			if not qa:
				continue

			question, answer = qa
			converted = {
				"id": row.get("id"),
				"premise": premise,
				"hypothesis": question,
				"label": answer,
			}
			dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
			kept += 1

	print(f"Saved {kept} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
