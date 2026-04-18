"""
Count NLI document tokens in batch with spaCy (no fastcoref).

This script only does one thing:
1) Count tokens for each premise via spaCy nlp.pipe batch mode.
2) Record rows whose token count is greater than a threshold (default 4096).
3) Save results to a JSON file.

Examples:
	python tests/tasks/count_tokens.py --dataset contract_nli
	python tests/tasks/count_tokens.py --dataset docnli
	python tests/tasks/count_tokens.py --dataset docnli --batch-size 64 --token-limit 4096
"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Literal

import jsonlines
import spacy
from spacy.language import Language
from tqdm import tqdm


logger = logging.getLogger(__name__)


DatasetName = Literal["contract_nli", "docnli"]

DEFAULT_DATASET: DatasetName = "contract_nli"
DEFAULT_MODEL = "en_core_web_trf"


def default_input_path(dataset: DatasetName) -> str:
	if dataset == "docnli":
		return "data/nli/docnli.jsonl"
	return "data/nli/contract_nli.cleaned.jsonl"


def default_output_path(dataset: DatasetName) -> str:
	if dataset == "docnli":
		return "data/nli/docnli_over_4096_tokens.json"
	return "data/nli/contract_nli_over_4096_tokens.json"


def setup_spacy_gpu(model_name: str) -> Language:
	"""Load spaCy model and try to enable GPU. No fastcoref is used here."""
	try:
		require_gpu_fn = getattr(spacy, "require_gpu", None)
		if callable(require_gpu_fn) and require_gpu_fn():
			logger.info("GPU is enabled for spaCy")
		else:
			logger.warning("GPU not available for spaCy, fallback to CPU")
	except Exception as exc:
		logger.warning("GPU check failed (%s), fallback to CPU", exc)

	try:
		return spacy.load(model_name)
	except OSError as exc:
		logger.error("spaCy model %s not found", model_name)
		raise exc


def load_rows(input_path: str) -> list[dict]:
	path = Path(input_path)
	if not path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")
	if not path.is_file() or path.suffix != ".jsonl":
		raise FileNotFoundError(f"Unsupported input path: {input_path}")

	rows: list[dict] = []
	with jsonlines.open(path, "r") as reader:
		for row in reader:
			if isinstance(row, dict):
				rows.append(row)
	return rows


def count_over_limit_tokens(
	rows: list[dict],
	nlp: Language,
	token_limit: int,
	batch_size: int,
) -> list[dict]:
	"""Count premise tokens with nlp.pipe and return rows above token_limit."""
	texts: list[str] = []
	metas: list[dict] = []

	for idx, row in enumerate(rows):
		premise = (row.get("premise") or "").strip()
		texts.append(premise)
		metas.append(
			{
				"row_index": idx,
				"id": str(row.get("id", f"row-{idx}")),
				"dataset": row.get("dataset", ""),
				"subset": row.get("subset", ""),
				"label": row.get("label", ""),
				"hypothesis": row.get("hypothesis", ""),
				"premise": premise,
			}
		)

	over_limit: list[dict] = []

	# Disable all model components during pipe so only tokenization runs.
	# This avoids unrelated model limits while still using spaCy batch processing.
	disable_components = list(nlp.pipe_names)
	for doc, meta in tqdm(
		zip(
			nlp.pipe(texts, batch_size=max(1, batch_size), disable=disable_components),
			metas,
		),
		total=len(texts),
		desc="Counting premise tokens",
		unit="row",
	):
		token_count = len(doc)
		if token_count > token_limit:
			over_limit.append(
				{
					"row_index": meta["row_index"],
					"id": meta["id"],
					"dataset": meta["dataset"],
					"subset": meta["subset"],
					"label": meta["label"],
					"token_count": token_count,
					"hypothesis": meta["hypothesis"],
					"premise": meta["premise"],
				}
			)

	return over_limit


def main() -> None:
	parser = ArgumentParser(description="Count NLI premise tokens and record rows over token limit")
	parser.add_argument(
		"--dataset",
		type=str,
		choices=["contract_nli", "docnli"],
		default=DEFAULT_DATASET,
		help="Dataset preset for default input/output paths",
	)
	parser.add_argument("--input", type=str, default="", help="Input JSONL path. If empty, use dataset preset")
	parser.add_argument("--output", type=str, default="", help="Output JSON path. If empty, use dataset preset")
	parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL, help="spaCy model name")
	parser.add_argument("--batch-size", type=int, default=32, help="spaCy nlp.pipe batch size")
	parser.add_argument("--token-limit", type=int, default=4096, help="Token threshold to record")
	args = parser.parse_args()
	dataset = args.dataset
	resolved_input = args.input.strip() or default_input_path(dataset)
	resolved_output = args.output.strip() or default_output_path(dataset)

	rows = load_rows(resolved_input)
	nlp = setup_spacy_gpu(args.model_name)
	over_limit = count_over_limit_tokens(
		rows=rows,
		nlp=nlp,
		token_limit=max(1, args.token_limit),
		batch_size=max(1, args.batch_size),
	)

	output_path = Path(resolved_output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	payload = {
		"summary": {
			"dataset": dataset,
			"input": str(Path(resolved_input).resolve()),
			"output": str(output_path.resolve()),
			"model_name": args.model_name,
			"batch_size": max(1, args.batch_size),
			"token_limit": max(1, args.token_limit),
			"total_rows": len(rows),
			"over_limit_rows": len(over_limit),
		},
		"over_limit_items": over_limit,
	}

	output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

	print("\n" + "=" * 60)
	print("Contract NLI token counting finished")
	print("=" * 60)
	print(f"Total rows:      {len(rows)}")
	print(f"Over limit rows: {len(over_limit)}")
	print(f"Token limit:     {max(1, args.token_limit)}")
	print(f"Output:          {output_path.resolve()}")
	print("=" * 60 + "\n")


if __name__ == "__main__":
	main()
