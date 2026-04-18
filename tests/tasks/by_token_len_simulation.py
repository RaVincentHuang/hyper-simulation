from __future__ import annotations

import argparse
import hashlib
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import spacy
from spacy.language import Language
from tqdm import tqdm

from hyper_simulation.query_instance import build_query_instance_for_task
from hyper_simulation.question_answer.utils.load_data import load_data

from refine_hypergraph import load_task_dataset


DEFAULT_MODEL = "en_core_web_sm"
DEFAULT_OUTPUT_ROOT = Path("data/debug/split")
DEFAULT_MUSIQUE_DATASETS = [
	"/home/vincent/.dataset/musique/sample1000/musique_answerable.jsonl",
	"/home/vincent/.dataset/musique/rest/musique_answerable.jsonl",
]
DEFAULT_CONTRA_NLI_DATASET = "data/nli/contract_nli.split.jsonl"
TOKEN_BUCKETS = [1000, 2000, 3000, 4000, 5000]
SAMPLE_PER_BUCKET = 15


def generate_instance_id(query: str) -> str:
	normalized = "".join(query.split()).lower()
	return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TokenBucketResult:
	upper_bound: int
	items: list[dict[str, Any]]


def setup_spacy_tokenizer(model_name: str = DEFAULT_MODEL) -> Language:
	"""Load spaCy and fall back to a blank English tokenizer if needed."""
	try:
		return spacy.load(model_name)
	except Exception:
		return spacy.blank("en")


def _count_tokens(nlp: Language, texts: Iterable[str], batch_size: int) -> list[int]:
	cleaned_texts = [text or "" for text in texts]
	disable_components = list(nlp.pipe_names)
	return [
		len(doc)
		for doc in nlp.pipe(
cleaned_texts,
batch_size=max(1, batch_size),
disable=disable_components,
)
	]


def _load_musique_items(dataset_paths: list[str]) -> list[dict[str, Any]]:
	items: list[dict[str, Any]] = []
	for dataset_path in dataset_paths:
		items.extend(load_data(dataset_path, task="musique", use_supporting_only=False))
	return items


def _load_contra_nli_items(dataset_path: str) -> list[dict[str, Any]]:
	return load_task_dataset(task="contract_nli", dataset_path=dataset_path)


def _split_texts_by_bucket(total_token_count: int) -> int | None:
	for threshold in TOKEN_BUCKETS:
		if total_token_count <= threshold:
			return threshold
	return None


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _finalize_records(
items: list[dict[str, Any]],
nlp: Language,
batch_size: int,
task_name: str,
) -> list[dict[str, Any]]:
	query_texts = [item["query_text"] for item in items]
	query_counts = _count_tokens(nlp, query_texts, batch_size=batch_size)

	all_data_texts: list[str] = []
	for item in items:
		item["data_start"] = len(all_data_texts)
		all_data_texts.extend(item["data_texts"])
		item["data_end"] = len(all_data_texts)

	data_counts = _count_tokens(nlp, all_data_texts, batch_size=batch_size) if all_data_texts else []

	records: list[dict[str, Any]] = []
	for idx, item in enumerate(items):
		query_count = query_counts[idx]
		data_count = sum(data_counts[item["data_start"] : item["data_end"]])
		total_count = query_count + data_count
		bucket = _split_texts_by_bucket(total_count)
		records.append(
{
"task": task_name,
"row_index": item["row_index"],
"instance_id": item["instance_id"],
"source_id": item["source_id"],
"query": item["query_text"],
"data": item["data_texts"],
"query_token_count": query_count,
"data_token_count": data_count,
"token_count": total_count,
"token_bucket": bucket,
}
)

	return records


def _build_musique_records(dataset_paths: list[str], nlp: Language, batch_size: int, limit_items: int | None) -> list[dict[str, Any]]:
	raw_items = _load_musique_items(dataset_paths)
	items: list[dict[str, Any]] = []

	for row_index, raw_item in enumerate(tqdm(raw_items, desc="Collect MuSiQue items", unit="item")):
		if limit_items is not None and limit_items > 0 and len(items) >= limit_items:
			break
		qi = build_query_instance_for_task(raw_item, task="musique")
		question = (qi.query or "").strip()
		data_texts = [(text or "").strip() for text in (qi.data or []) if (text or "").strip()]
		if not question or not data_texts:
			continue
		items.append(
{
"row_index": row_index,
"instance_id": generate_instance_id(question),
"source_id": str(raw_item.get("_id", f"row-{row_index}")),
"query_text": question,
"data_texts": data_texts,
}
)

	return _finalize_records(items, nlp, batch_size, task_name="musique")


def _build_contra_nli_records(dataset_path: str, nlp: Language, batch_size: int, limit_items: int | None) -> list[dict[str, Any]]:
	raw_items = _load_contra_nli_items(dataset_path)
	items: list[dict[str, Any]] = []

	for row_index, raw_item in enumerate(tqdm(raw_items, desc="Collect Contract NLI items", unit="item")):
		if limit_items is not None and limit_items > 0 and len(items) >= limit_items:
			break
		question = (raw_item.get("question") or raw_item.get("hypothesis") or "").strip()
		premise_chunks = raw_item.get("premise_chunks") or raw_item.get("context_docs") or []
		if not isinstance(premise_chunks, list):
			premise_chunks = []
		data_texts = [text.strip() for text in premise_chunks if isinstance(text, str) and text.strip()]
		if not question or not data_texts:
			continue
		instance_id = generate_instance_id(f"{raw_item.get('_id', '')}:{question}")
		items.append(
{
"row_index": row_index,
"instance_id": instance_id,
"source_id": str(raw_item.get("_id", f"row-{row_index}")),
"query_text": question,
"data_texts": data_texts,
}
)

	return _finalize_records(items, nlp, batch_size, task_name="contra_nli")


def _write_split_outputs(task_name: str, records: list[dict[str, Any]], output_root: Path) -> None:
	task_dir = output_root / task_name
	task_dir.mkdir(parents=True, exist_ok=True)

	available_bucket_files: dict[int, list[dict[str, Any]]] = {threshold: [] for threshold in TOKEN_BUCKETS}
	sampled_bucket_files: dict[int, list[dict[str, Any]]] = {threshold: [] for threshold in TOKEN_BUCKETS}
	overflow: list[dict[str, Any]] = []
	for record in records:
		bucket = record.get("token_bucket")
		if bucket in available_bucket_files:
			available_bucket_files[bucket].append(record)
		else:
			overflow.append(record)

	rng = random.Random(42)
	for threshold, rows in available_bucket_files.items():
		if len(rows) <= SAMPLE_PER_BUCKET:
			sampled = list(rows)
		else:
			sampled = rng.sample(rows, SAMPLE_PER_BUCKET)
		sampled.sort(key=lambda item: (item.get("token_count", 0), item.get("instance_id", "")))
		sampled_bucket_files[threshold] = sampled
		_dump_jsonl(task_dir / f"{threshold}.jsonl", sampled)
	_dump_jsonl(task_dir / "overflow.jsonl", overflow)

	summary = {
		"task": task_name,
		"output_dir": str(task_dir.resolve()),
		"total_items": len(records),
		"available_bucket_counts": {str(threshold): len(rows) for threshold, rows in available_bucket_files.items()},
		"sampled_bucket_counts": {str(threshold): len(rows) for threshold, rows in sampled_bucket_files.items()},
		"sample_per_bucket": SAMPLE_PER_BUCKET,
		"overflow_count": len(overflow),
	}
	(task_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

	print("\n" + "=" * 72)
	print(f"{task_name} token split finished")
	print("=" * 72)
	print(f"Total items: {len(records)}")
	for threshold in TOKEN_BUCKETS:
		print(f"<= {threshold}: {len(sampled_bucket_files[threshold])} / {len(available_bucket_files[threshold])}")
	print(f"Overflow: {len(overflow)}")
	print(f"Saved to: {task_dir.resolve()}")
	print("=" * 72)


def main() -> None:
	parser = argparse.ArgumentParser(description="Split MuSiQue and Contract NLI by query+data token length")
	parser.add_argument("--musique-datasets", nargs="+", default=DEFAULT_MUSIQUE_DATASETS)
	parser.add_argument("--contra-nli-dataset", type=str, default=DEFAULT_CONTRA_NLI_DATASET)
	parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
	parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--limit-items", type=int, default=0, help="Limit items per dataset for quick testing")
	args = parser.parse_args()

	nlp = setup_spacy_tokenizer(args.model_name)
	output_root = Path(args.output_root)
	limit_items = args.limit_items or None

	musique_records = _build_musique_records(args.musique_datasets, nlp, max(1, args.batch_size), limit_items)
	_write_split_outputs("musique", musique_records, output_root)

	contra_records = _build_contra_nli_records(args.contra_nli_dataset, nlp, max(1, args.batch_size), limit_items)
	_write_split_outputs("contra_nli", contra_records, output_root)


if __name__ == "__main__":
	main()
