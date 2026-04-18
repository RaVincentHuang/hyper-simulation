"""
Build LegalBench hypergraphs with multi-format schema normalization.

Supported file schemas under a directory like:
  /home/vincent/.dataset/LegalBench/sample975

Common variants handled here:
1) contract_qa style
   - fields: question, text, answer
2) consumer_contracts_qa style
   - fields: question, contract, answer
3) privacy_policy_qa style
   - fields: question, text, answer

Examples:
  python tests/tasks/build_legalbench.py --use-gpu-batch --batch-size 32
  python tests/tasks/build_legalbench.py --force-rebuild
"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Iterator

import jsonlines
import spacy
from spacy.language import Language
from tqdm import tqdm

from hyper_simulation.component.build_hypergraph import (
	clean_text_for_spacy,
	doc_to_hypergraph,
	generate_instance_id,
	text_to_hypergraph,
)
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph


logger = logging.getLogger(__name__)


target_dir = "data/debug/legalbench/sample975/"
dataset_path = "/home/vincent/.dataset/LegalBench/sample975"
local_model_path = "/home/vincent/.cache/huggingface/hub/models--biu-nlp--lingmess-coref/snapshots/fa5d8a827a09388d03adbe9e800c7d8c509c3935"


def setup_gpu_nlp(model_name: str = "en_core_web_trf") -> Language:
	"""Initialize spaCy model with optional GPU and fastcoref."""
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
					"model_path": local_model_path,
					"device": "cuda",
				},
			)
			logger.info("fastcoref added with CUDA")
		except Exception as exc:
			logger.warning("Failed to add fastcoref: %s", exc)

	return nlp


def _normalize_item(raw: dict, source_file: str, row_idx: int) -> dict | None:
	"""Normalize different LegalBench schemas into one build-ready format."""
	question = (raw.get("question") or "").strip()
	answer = (raw.get("answer") or "").strip()
	if not question:
		return None

	# Schema A/C: context in `text` (contract_qa / privacy_policy_qa)
	text = (raw.get("text") or "").strip()

	# Schema B: context in `contract` (consumer_contracts_qa)
	contract_text = (raw.get("contract") or "").strip()

	if contract_text:
		context_text = contract_text
		context_type = "consumer_contract"
	elif text:
		if "privacy" in source_file:
			context_type = "privacy_policy"
		elif "contract" in source_file:
			context_type = "contract"
		else:
			context_type = "legal_document"
		context_text = text
	else:
		# Fallback for edge cases with different field names.
		fallback = (raw.get("context") or raw.get("passage") or "").strip()
		if not fallback:
			return None
		context_text = fallback
		context_type = "legal_document"

	source_id = raw.get("id")
	if source_id is None:
		source_id = raw.get("index", f"{source_file}:{row_idx}")

	return {
		"_id": str(source_id),
		"question": question,
		"answer": answer,
		"context_docs": [context_text],
		"context_type": context_type,
		"source_file": source_file,
	}


def _load_legalbench_items(path_str: str) -> list[dict]:
	"""Load all jsonl files and normalize mixed LegalBench schemas."""
	path = Path(path_str)
	files: list[Path]
	if path.is_dir():
		files = sorted(path.glob("*.jsonl"))
	elif path.is_file() and path.suffix == ".jsonl":
		files = [path]
	else:
		raise FileNotFoundError(f"Unsupported path: {path_str}")

	if not files:
		raise FileNotFoundError(f"No jsonl files found under: {path_str}")

	data: list[dict] = []
	for file in files:
		with jsonlines.open(file, "r") as reader:
			for idx, item in enumerate(reader):
				normalized = _normalize_item(item, file.name, idx)
				if normalized is not None:
					data.append(normalized)
	return data


def batch_text_to_hypergraph(
	nlp: Language,
	texts_with_metadata: list[dict],
	batch_size: int = 32,
	is_query: bool = False,
) -> Iterator[tuple[dict, LocalHypergraph | None]]:
	"""Convert texts to hypergraphs with nlp.pipe and safe fallback."""
	texts = [clean_text_for_spacy(item["text"]) for item in texts_with_metadata]
	metadatas = [item["meta"] for item in texts_with_metadata]
	original_texts = [item["text"] for item in texts_with_metadata]

	component_cfg = {"fastcoref": {"resolve_text": True}} if "fastcoref" in nlp.pipe_names else {}

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
				hg = doc_to_hypergraph(doc, original, is_query=is_query)
				yield meta, hg
			except Exception as exc:
				meta["error"] = f"{type(exc).__name__}: {exc}"
				yield meta, None
	except Exception as exc:
		logger.warning(
			"Batch processing failed (%s: %s). Fallback to per-text mode.",
			type(exc).__name__,
			exc,
		)
		for text, meta, original in zip(texts, metadatas, original_texts):
			try:
				doc = nlp(text)
				hg = doc_to_hypergraph(doc, original, is_query=is_query)
				yield meta, hg
			except Exception as inner_exc:
				meta["error"] = f"{type(inner_exc).__name__}: {inner_exc}"
				yield meta, None


def _build_all_hypergraphs_single(
	dataset_path: str,
	target_dir: str,
	force_rebuild: bool = False,
) -> dict:
	"""Single-text processing mode."""
	data = _load_legalbench_items(dataset_path)
	out_root = Path(target_dir)
	out_root.mkdir(parents=True, exist_ok=True)

	built_count = 0
	skipped_count = 0
	failed: list[dict] = []

	for item in tqdm(data, desc="Building legalbench hypergraphs (single)"):
		try:
			question = (item.get("question") or "").strip()
			docs = item.get("context_docs") or []
			if not question or not docs:
				skipped_count += 1
				continue

			instance_id = generate_instance_id(question)
			instance_dir = out_root / instance_id
			instance_dir.mkdir(parents=True, exist_ok=True)

			query_path = instance_dir / "query_hypergraph.pkl"
			metadata_path = instance_dir / "metadata.json"

			if metadata_path.exists() and not force_rebuild:
				skipped_count += 1
				continue

			query_hypergraph = text_to_hypergraph(question, is_query=True)
			query_hypergraph.save(str(query_path))

			data_files: list[str] = []
			for idx, doc_text in enumerate(docs):
				text = (doc_text or "").strip()
				if not text:
					continue
				data_hypergraph = text_to_hypergraph(text, is_query=False)
				data_file = f"data_hypergraph{idx}.pkl"
				data_hypergraph.save(str(instance_dir / data_file))
				data_files.append(data_file)

			metadata = {
				"instance_id": instance_id,
				"source_id": item.get("_id", ""),
				"source_file": item.get("source_file", ""),
				"question": question,
				"answer": item.get("answer", ""),
				"context_type": item.get("context_type", "legal_document"),
				"num_data": len(docs),
				"saved_data_hypergraphs": len(data_files),
				"files": {
					"query": query_path.name,
					"data": data_files,
				},
			}
			metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

			built_count += 1
		except Exception as exc:
			failed.append(
				{
					"id": item.get("_id", ""),
					"question": item.get("question", ""),
					"error": f"{type(exc).__name__}: {exc}",
				}
			)

	summary = {
		"dataset_path": dataset_path,
		"target_dir": str(out_root.resolve()),
		"total_questions": len(data),
		"built": built_count,
		"skipped": skipped_count,
		"failed": len(failed),
		"mode": "single",
	}

	(out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
	if failed:
		(out_root / "failed.json").write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")

	return summary


def _build_all_hypergraphs_gpu_batch(
	dataset_path: str,
	target_dir: str,
	force_rebuild: bool = False,
	batch_size: int = 32,
) -> dict:
	"""GPU + batch processing mode."""
	logger.info("Starting GPU batch mode")
	nlp = setup_gpu_nlp()

	data = _load_legalbench_items(dataset_path)
	out_root = Path(target_dir)
	out_root.mkdir(parents=True, exist_ok=True)

	built_count = 0
	skipped_count = 0
	failed: list[dict] = []

	queries_to_process = []
	contexts_to_process = []

	for item_idx, item in enumerate(tqdm(data, desc="[1/4] Collect items", unit="items")):
		try:
			question = (item.get("question") or "").strip()
			docs = item.get("context_docs") or []
			if not question or not docs:
				skipped_count += 1
				continue

			instance_id = generate_instance_id(question)
			instance_dir = out_root / instance_id
			instance_dir.mkdir(parents=True, exist_ok=True)

			query_path = instance_dir / "query_hypergraph.pkl"
			metadata_path = instance_dir / "metadata.json"

			if metadata_path.exists() and not force_rebuild:
				skipped_count += 1
				continue

			queries_to_process.append(
				{
					"item": item,
					"instance_id": instance_id,
					"instance_dir": instance_dir,
					"query_path": query_path,
					"metadata_path": metadata_path,
					"question": question,
					"docs": docs,
				}
			)

			for doc_idx, doc_text in enumerate(docs):
				text = (doc_text or "").strip()
				if text:
					contexts_to_process.append(
						{
							"instance_id": instance_id,
							"doc_idx": doc_idx,
							"doc_text": text,
							"item_idx": item_idx,
						}
					)
		except Exception as exc:
			failed.append(
				{
					"id": item.get("_id", ""),
					"question": item.get("question", ""),
					"error": f"{type(exc).__name__}: {exc}",
				}
			)

	query_results: dict[str, LocalHypergraph] = {}
	if queries_to_process:
		query_batch = [
			{
				"text": q["question"],
				"meta": {
					"instance_id": q["instance_id"],
				},
			}
			for q in queries_to_process
		]

		query_pbar = tqdm(total=len(query_batch), desc="[2/4] Build query hypergraphs", unit="query")
		for meta, hypergraph in batch_text_to_hypergraph(
			nlp,
			query_batch,
			batch_size=batch_size,
			is_query=True,
		):
			if hypergraph is not None:
				query_results[meta["instance_id"]] = hypergraph
			query_pbar.update(1)
		query_pbar.close()

	context_results: dict[tuple[str, int], LocalHypergraph] = {}
	if contexts_to_process:
		context_batch = [
			{
				"text": c["doc_text"],
				"meta": {
					"instance_id": c["instance_id"],
					"doc_idx": c["doc_idx"],
				},
			}
			for c in contexts_to_process
		]

		context_pbar = tqdm(total=len(context_batch), desc="[3/4] Build context hypergraphs", unit="doc")
		for meta, hypergraph in batch_text_to_hypergraph(
			nlp,
			context_batch,
			batch_size=batch_size,
			is_query=False,
		):
			if hypergraph is not None:
				context_results[(meta["instance_id"], meta["doc_idx"])] = hypergraph
			context_pbar.update(1)
		context_pbar.close()

	for q in tqdm(queries_to_process, desc="[4/4] Save outputs", unit="instance"):
		try:
			item = q["item"]
			instance_id = q["instance_id"]
			query_path = q["query_path"]
			metadata_path = q["metadata_path"]
			instance_dir = q["instance_dir"]
			question = q["question"]
			docs = q["docs"]

			query_hypergraph = query_results.get(instance_id)
			if query_hypergraph is None:
				failed.append(
					{
						"id": item.get("_id", ""),
						"question": question,
						"error": "Query hypergraph processing failed",
					}
				)
				continue

			query_hypergraph.save(str(query_path))

			data_files: list[str] = []
			for idx in range(len(docs)):
				key = (instance_id, idx)
				if key not in context_results:
					continue
				data_file = f"data_hypergraph{idx}.pkl"
				context_results[key].save(str(instance_dir / data_file))
				data_files.append(data_file)

			metadata = {
				"instance_id": instance_id,
				"source_id": item.get("_id", ""),
				"source_file": item.get("source_file", ""),
				"question": question,
				"answer": item.get("answer", ""),
				"context_type": item.get("context_type", "legal_document"),
				"num_data": len(docs),
				"saved_data_hypergraphs": len(data_files),
				"files": {
					"query": query_path.name,
					"data": data_files,
				},
			}
			metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
			built_count += 1
		except Exception as exc:
			failed.append(
				{
					"id": q["item"].get("_id", ""),
					"question": q["question"],
					"error": f"{type(exc).__name__}: {exc}",
				}
			)

	summary = {
		"dataset_path": dataset_path,
		"target_dir": str(out_root.resolve()),
		"total_questions": len(data),
		"built": built_count,
		"skipped": skipped_count,
		"failed": len(failed),
		"mode": "gpu_batch",
		"batch_size": batch_size,
	}

	(out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
	if failed:
		(out_root / "failed.json").write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")

	return summary


def build_all_hypergraphs(
	dataset_path: str,
	target_dir: str,
	force_rebuild: bool = False,
	use_gpu_batch: bool = False,
	batch_size: int = 32,
) -> dict:
	if use_gpu_batch:
		return _build_all_hypergraphs_gpu_batch(
			dataset_path=dataset_path,
			target_dir=target_dir,
			force_rebuild=force_rebuild,
			batch_size=batch_size,
		)
	return _build_all_hypergraphs_single(
		dataset_path=dataset_path,
		target_dir=target_dir,
		force_rebuild=force_rebuild,
	)


def _print_summary(summary: dict) -> None:
	print("\n" + "=" * 60)
	print("LegalBench hypergraph build finished")
	print("=" * 60)
	print(f"Total:   {summary.get('total_questions', 0)}")
	print(f"Built:   {summary.get('built', 0)}")
	print(f"Skipped: {summary.get('skipped', 0)}")
	print(f"Failed:  {summary.get('failed', 0)}")
	print(f"Mode:    {summary.get('mode', 'unknown')}")
	if "batch_size" in summary:
		print(f"Batch:   {summary['batch_size']}")
	print(f"Output:  {summary.get('target_dir', 'N/A')}")
	print("=" * 60 + "\n")


def main() -> None:
	parser = ArgumentParser(description="Build and store LegalBench question hypergraphs")
	parser.add_argument("--dataset-path", type=str, default=dataset_path)
	parser.add_argument("--target-dir", type=str, default=target_dir)
	parser.add_argument("--force-rebuild", action="store_true")
	parser.add_argument("--use-gpu-batch", action="store_true")
	parser.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU batch mode")
	args = parser.parse_args()

	summary = build_all_hypergraphs(
		dataset_path=args.dataset_path,
		target_dir=args.target_dir,
		force_rebuild=args.force_rebuild,
		use_gpu_batch=args.use_gpu_batch,
		batch_size=args.batch_size,
	)
	_print_summary(summary)


if __name__ == "__main__":
	main()
