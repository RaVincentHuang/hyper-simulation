from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import jsonlines
import spacy
from spacy.language import Language
from tqdm import tqdm

from hyper_simulation.component.build_hypergraph import (
	clean_text_for_spacy,
	doc_to_hypergraph,
	text_to_hypergraph,
)
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph


logger = logging.getLogger(__name__)


DEFAULT_SOURCE_ROOT = "data/debug/split/contra_nli"
DEFAULT_OUTPUT_ROOT = "data/debug/split/contra_nli/hypergraphs"
DEFAULT_MODEL = "en_core_web_trf"
LOCAL_MODEL_PATH = "/home/vincent/.cache/huggingface/hub/models--biu-nlp--lingmess-coref/snapshots/fa5d8a827a09388d03adbe9e800c7d8c509c3935"
BUCKET_NAMES = ["1000", "2000", "3000", "4000", "5000", "overflow"]


def setup_gpu_nlp(model_name: str = DEFAULT_MODEL) -> Language:
	"""Initialize spaCy with optional GPU and fastcoref."""
	try:
		require_gpu_fn = getattr(spacy, "require_gpu", None)
		if callable(require_gpu_fn) and require_gpu_fn():
			logger.info("GPU is enabled for spaCy")
		else:
			logger.warning("GPU not available for spaCy, fallback to CPU")
	except Exception as exc:
		logger.warning("GPU check failed (%s), fallback to CPU", exc)

	try:
		nlp = spacy.load(model_name)
	except OSError as exc:
		logger.error("spaCy model %s not found", model_name)
		raise exc

	if "fastcoref" not in nlp.pipe_names:
		try:
			nlp.add_pipe(
				"fastcoref",
				config={
					"model_architecture": "LingMessCoref",
					"model_path": LOCAL_MODEL_PATH,
					"device": "cuda",
				},
			)
			logger.info("fastcoref added with CUDA")
		except Exception as exc:
			logger.warning("Failed to add fastcoref: %s", exc)

	return nlp


def _load_rows(jsonl_path: Path) -> list[dict[str, Any]]:
	if not jsonl_path.exists():
		raise FileNotFoundError(f"Input file not found: {jsonl_path}")
	rows: list[dict[str, Any]] = []
	with jsonlines.open(jsonl_path, "r") as reader:
		for row in reader:
			if isinstance(row, dict):
				rows.append(row)
	return rows


def _list_bucket_files(source_root: Path) -> list[Path]:
	files: list[Path] = []
	for name in BUCKET_NAMES:
		path = source_root / f"{name}.jsonl"
		if path.exists():
			files.append(path)
	return files


def _build_hypergraphs(
	nlp: Language,
	texts_with_metadata: list[dict[str, Any]],
	batch_size: int = 32,
	is_query: bool = False,
) -> tuple[list[tuple[dict[str, Any], LocalHypergraph | None]], float]:
	texts = [clean_text_for_spacy(item["text"]) for item in texts_with_metadata]
	metadatas = [item["meta"] for item in texts_with_metadata]
	original_texts = [item["text"] for item in texts_with_metadata]
	component_cfg = {"fastcoref": {"resolve_text": True}} if "fastcoref" in nlp.pipe_names else {}
	results: list[tuple[dict[str, Any], LocalHypergraph | None]] = []
	doc_to_hypergraph_time = 0.0

	try:
		docs = list(
			nlp.pipe(
				texts,
				component_cfg=component_cfg,
				batch_size=max(1, batch_size),
			)
		)
		for doc, meta, original in zip(docs, metadatas, original_texts):
			try:
				start_time = perf_counter()
				hg = doc_to_hypergraph(doc, original, is_query=is_query)
				doc_to_hypergraph_time += perf_counter() - start_time
				results.append((meta, hg))
			except Exception as exc:
				meta["error"] = f"{type(exc).__name__}: {exc}"
				results.append((meta, None))
	except Exception as exc:
		logger.error(
			"Batch processing failed (%s: %s). Stop immediately in strict batch mode.",
			type(exc).__name__,
			exc,
		)
		raise RuntimeError(
			f"Batch processing failed in strict mode: {type(exc).__name__}: {exc}"
		) from exc

	return results, doc_to_hypergraph_time


def _normalize_split_row(raw: dict[str, Any], row_idx: int, bucket_name: str) -> dict[str, Any] | None:
	question = (raw.get("query") or raw.get("question") or "").strip()
	data = raw.get("data") or []
	if not question or not isinstance(data, list):
		return None
	data_texts = [(text or "").strip() for text in data if isinstance(text, str) and text.strip()]
	if not data_texts:
		return None

	instance_id = str(raw.get("instance_id") or f"{bucket_name}-{row_idx}")
	return {
		"instance_id": instance_id,
		"source_id": str(raw.get("source_id", "")),
		"question": question,
		"data_texts": data_texts,
		"token_count": raw.get("token_count"),
		"token_bucket": raw.get("token_bucket", bucket_name),
		"row_index": raw.get("row_index", row_idx),
	}


def _build_bucket_hypergraphs(
	jsonl_path: Path,
	nlp: Language,
	output_root: Path,
	force_rebuild: bool = False,
	batch_size: int = 32,
) -> dict[str, Any]:
	bucket_name = jsonl_path.stem
	rows = _load_rows(jsonl_path)
	instances_root = output_root / bucket_name
	instances_root.mkdir(parents=True, exist_ok=True)

	built_count = 0
	skipped_count = 0
	failed: list[dict[str, Any]] = []
	save_time = 0.0
	docs_retrieval_start = perf_counter()

	items: list[dict[str, Any]] = []
	for idx, raw in enumerate(rows):
		normalized = _normalize_split_row(raw, idx, bucket_name)
		if normalized is None:
			skipped_count += 1
			continue
		instance_dir = instances_root / normalized["instance_id"]
		instance_dir.mkdir(parents=True, exist_ok=True)
		query_path = instance_dir / "query_hypergraph.pkl"
		metadata_path = instance_dir / "metadata.json"
		if metadata_path.exists() and not force_rebuild:
			skipped_count += 1
			continue
		normalized["instance_dir"] = str(instance_dir)
		normalized["query_path"] = str(query_path)
		normalized["metadata_path"] = str(metadata_path)
		items.append(normalized)

	docs_retrieval_time = perf_counter() - docs_retrieval_start

	query_batch = [
		{
			"text": item["question"],
			"meta": {
				"instance_id": item["instance_id"],
				"instance_dir": item["instance_dir"],
				"query_path": item["query_path"],
				"metadata_path": item["metadata_path"],
				"source_id": item["source_id"],
				"bucket": bucket_name,
				"row_index": item["row_index"],
				"token_count": item["token_count"],
				"token_bucket": item["token_bucket"],
			},
		}
		for item in items
	]

	context_batch: list[dict[str, Any]] = []
	for item in items:
		for doc_idx, doc_text in enumerate(item["data_texts"]):
			context_batch.append(
				{
					"text": doc_text,
					"meta": {
						"instance_id": item["instance_id"],
						"instance_dir": item["instance_dir"],
						"doc_idx": doc_idx,
					},
				}
			)

	query_results: dict[str, LocalHypergraph] = {}
	query_hypergraph_time = 0.0
	if query_batch:
		query_pairs, query_hypergraph_time = _build_hypergraphs(nlp, query_batch, batch_size=batch_size, is_query=True)
		for meta, hypergraph in tqdm(query_pairs, total=len(query_pairs), desc=f"[{bucket_name}] Build query hypergraphs", unit="query"):
			if hypergraph is not None:
				query_results[meta["instance_id"]] = hypergraph

	context_results: dict[tuple[str, int], LocalHypergraph] = {}
	context_hypergraph_time = 0.0
	if context_batch:
		context_pairs, context_hypergraph_time = _build_hypergraphs(nlp, context_batch, batch_size=batch_size, is_query=False)
		for meta, hypergraph in tqdm(context_pairs, total=len(context_pairs), desc=f"[{bucket_name}] Build data hypergraphs", unit="doc"):
			if hypergraph is not None:
				context_results[(meta["instance_id"], meta["doc_idx"])] = hypergraph

	for item in tqdm(items, desc=f"[{bucket_name}] Save outputs", unit="instance"):
		try:
			instance_dir = Path(item["instance_dir"])
			query_path = Path(item["query_path"])
			metadata_path = Path(item["metadata_path"])
			instance_id = item["instance_id"]
			question = item["question"]
			data_texts = item["data_texts"]

			query_hypergraph = query_results.get(instance_id)
			if query_hypergraph is None:
				failed.append(
					{
						"instance_id": instance_id,
						"error": "Query hypergraph processing failed",
					}
				)
				continue

			save_start = perf_counter()
			query_hypergraph.save(str(query_path))

			data_files: list[str] = []
			for idx in range(len(data_texts)):
				key = (instance_id, idx)
				if key not in context_results:
					continue
				data_file = f"data_hypergraph{idx}.pkl"
				context_results[key].save(str(instance_dir / data_file))
				data_files.append(data_file)

			metadata = {
				"instance_id": instance_id,
				"bucket": bucket_name,
				"source_id": item["source_id"],
				"row_index": item["row_index"],
				"question": question,
				"token_count": item["token_count"],
				"token_bucket": item["token_bucket"],
				"num_data": len(data_texts),
				"saved_data_hypergraphs": len(data_files),
				"files": {
					"query": query_path.name,
					"data": data_files,
				},
			}
			metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
			built_count += 1
			save_time += perf_counter() - save_start
		except Exception as exc:
			failed.append(
				{
					"instance_id": item["instance_id"],
					"error": f"{type(exc).__name__}: {exc}",
				}
			)

	summary = {
		"bucket": bucket_name,
		"dataset_path": str(jsonl_path.resolve()),
		"output_root": str(instances_root.resolve()),
		"total_rows": len(rows),
		"built": built_count,
		"skipped": skipped_count,
		"failed": len(failed),
		"docs_retrieval_time": docs_retrieval_time,
		"query_doc_to_hypergraph_time": query_hypergraph_time,
		"data_doc_to_hypergraph_time": context_hypergraph_time,
		"doc_to_hypergraph_time": query_hypergraph_time + context_hypergraph_time,
		"save_time": save_time,
		"total_build_time": docs_retrieval_time + query_hypergraph_time + context_hypergraph_time + save_time,
	}

	(instances_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
	if failed:
		(instances_root / "failed.json").write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")

	return summary


def build_by_token_length(
	source_root: str = DEFAULT_SOURCE_ROOT,
	output_root: str = DEFAULT_OUTPUT_ROOT,
	force_rebuild: bool = False,
	model_name: str = DEFAULT_MODEL,
	batch_size: int = 32,
) -> dict[str, Any]:
	source = Path(source_root)
	if not source.exists():
		raise FileNotFoundError(f"Source root not found: {source}")

	bucket_files = _list_bucket_files(source)
	if not bucket_files:
		raise FileNotFoundError(f"No split jsonl files found under: {source}")

	nlp = setup_gpu_nlp(model_name)
	output = Path(output_root)
	output.mkdir(parents=True, exist_ok=True)

	results: list[dict[str, Any]] = []
	for jsonl_path in bucket_files:
		results.append(
			_build_bucket_hypergraphs(
				jsonl_path=jsonl_path,
				nlp=nlp,
				output_root=output,
				force_rebuild=force_rebuild,
				batch_size=batch_size,
			)
		)

	global_summary = {
		"source_root": str(source.resolve()),
		"output_root": str(output.resolve()),
		"buckets": results,
	}
	(output / "summary.json").write_text(json.dumps(global_summary, indent=2, ensure_ascii=False), encoding="utf-8")
	return global_summary


def main() -> None:
	parser = argparse.ArgumentParser(description="Build hypergraphs for token-length split Contract NLI buckets")
	parser.add_argument("--source-root", type=str, default=DEFAULT_SOURCE_ROOT)
	parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
	parser.add_argument("--force-rebuild", action="store_true")
	parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
	parser.add_argument("--batch-size", type=int, default=32)
	args = parser.parse_args()

	summary = build_by_token_length(
		source_root=args.source_root,
		output_root=args.output_root,
		force_rebuild=args.force_rebuild,
		model_name=args.model_name,
		batch_size=args.batch_size,
	)
	print("\n" + "=" * 60)
	print("Split Contract NLI hypergraph build finished")
	print("=" * 60)
	print(f"Source: {summary['source_root']}")
	print(f"Output: {summary['output_root']}")
	for bucket in summary["buckets"]:
		print(f"{bucket['bucket']}: built={bucket['built']} skipped={bucket['skipped']} failed={bucket['failed']}")
	print("=" * 60 + "\n")


if __name__ == "__main__":
	main()
