from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any

from langchain_ollama import ChatOllama
from tqdm import tqdm

from hyper_simulation.baselines.CDIT import judge_similarity_batch
from hyper_simulation.baselines.contradoc import judge_contradiction_batch
from hyper_simulation.llm.chat_completion import get_invoke
from hyper_simulation.question_answer.vmdit.metrics import (
	exact_match_score,
	match,
	metric_max_over_ground_truths,
	qa_f1_score,
)
from refine_hypergraph import load_dataset_index, load_task_dataset


DEFAULT_PROMPTS_DIR = "data/legalbench/prompts"
DEFAULT_OUTPUT_DIR = "data/baseline/legalbench"
DEFAULT_METHODS = ["bsim", "her", "sentli", "sparsecl", "cdit", "contradoc", "vanilla"]
DEFAULT_DATASET_PATHS = ["/home/vincent/.dataset/LegalBench/sample975"]
FALLBACK_LEGALBENCH_PROMPTS_ROOT = Path("data/legalbench")
DIRECT_CONTEXT_METHODS = {"bsim", "her", "sentli", "sparsecl"}
SIMILARITY_METHODS = {"cdit"}
CONTRADICT_METHODS = {"contradoc"}
RAW_CONTEXT_METHODS = {"vanilla"}
LEGALBENCH_SOURCE_FILES = {
	"contract_qa.jsonl",
	"consumer_contracts_qa.jsonl",
	"privacy_policy_qa.jsonl",
}


LEGALBENCH_PROMPT_CONTRACT = """### Legal Document:
{context_text}

### Question:
{question}

### Instructions:
Answer the question using only the legal document.
Output exactly ONE line in the following format:
### Final Answer: Yes
or
### Final Answer: No
Do not include reasoning, explanations, or extra text.

### Response:
"""


LEGALBENCH_PROMPT_CONSUMER_CONTRACT = """### Legal Document:
{context_text}

### Question:
{question}

### Instructions:
Answer the question using only the legal document.
Output exactly ONE line in the following format:
### Final Answer: Yes
or
### Final Answer: No
Do not include reasoning, explanations, or extra text.

### Response:
"""


LEGALBENCH_PROMPT_PRIVACY = """### Privacy Policy:
{context_text}

### Question:
{question}

### Instructions:
Decide whether the policy content is relevant to the privacy concern raised by the question.
Output exactly ONE line in the following format:
### Final Answer: Relevant
or
### Final Answer: Irrelevant
Do not include reasoning, explanations, or extra text.

### Response:
"""


CONSISTENT_TAG = "[consistent]"
INCONSISTENT_TAG = "[inconsisitent]"


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
	tmp_path.replace(path)


def _row_key(instance_id: Any, question: str) -> str:
	inst = str(instance_id).strip() if instance_id is not None else ""
	q = str(question or "").strip()
	return inst if inst else q


def _load_jsonl(file_path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	if not file_path.exists():
		return rows
	with file_path.open("r", encoding="utf-8") as fp:
		for line in fp:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue
			if isinstance(obj, dict):
				rows.append(obj)
	return rows


def _source_file_from_context_type(context_type: str) -> str:
	ctx = (context_type or "").strip().lower()
	if ctx in {"contract", "legal_document", "contract_clause"}:
		return "contract_qa.jsonl"
	if ctx in {"tos", "terms_of_service", "consumer_contract"}:
		return "consumer_contracts_qa.jsonl"
	if ctx in {"privacy_policy", "privacy"}:
		return "privacy_policy_qa.jsonl"
	return ""


def _load_direct_prompt_rows(method: str, prompts_root: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], str]:
	candidates: list[Path] = []

	# Legacy single-folder layout: data/legalbench/prompts/{method}_prompts.jsonl
	legacy_file = prompts_root / f"{method}_prompts.jsonl"
	if legacy_file.exists():
		candidates.append(legacy_file)

	# New layout: data/legalbench/<subtask>/prompts/{method}_prompts.jsonl
	for root in [prompts_root, FALLBACK_LEGALBENCH_PROMPTS_ROOT]:
		if root.exists():
			candidates.extend(sorted(root.glob(f"*/prompts/{method}_prompts.jsonl")))

	# 去重并按路径稳定排序
	seen: set[str] = set()
	unique_candidates: list[Path] = []
	for path in sorted(candidates):
		key = str(path.resolve())
		if key in seen:
			continue
		seen.add(key)
		unique_candidates.append(path)

	rows: list[dict[str, Any]] = []
	for file_path in unique_candidates:
		for row in _load_jsonl(file_path):
			if not isinstance(row, dict):
				continue
			item = dict(row)
			source_file = str(item.get("source_file") or item.get("_source_file") or "")
			if not source_file:
				source_file = _source_file_from_context_type(str(item.get("context_type") or ""))
			if source_file:
				item["source_file"] = source_file
			rows.append(item)
			if limit is not None and limit > 0 and len(rows) >= limit:
				return rows, f"merged:{','.join(str(p) for p in unique_candidates)}"

	return rows, f"merged:{','.join(str(p) for p in unique_candidates)}"


def _load_existing_results_map(out_file: Path) -> dict[str, dict[str, Any]]:
	if not out_file.exists():
		return {}
	try:
		payload = json.loads(out_file.read_text(encoding="utf-8"))
	except Exception:
		return {}
	rows = payload.get("results", []) if isinstance(payload, dict) else []
	if not isinstance(rows, list):
		return {}
	res: dict[str, dict[str, Any]] = {}
	for one in rows:
		if not isinstance(one, dict):
			continue
		key = _row_key(one.get("instance_id"), str(one.get("question") or ""))
		if key:
			res[key] = one
	return res


def _load_context_cache_map(cache_file: Path) -> dict[str, dict[str, Any]]:
	if not cache_file.exists():
		return {}
	try:
		payload = json.loads(cache_file.read_text(encoding="utf-8"))
	except Exception:
		return {}
	rows = payload.get("rows", []) if isinstance(payload, dict) else []
	if not isinstance(rows, list):
		return {}
	cache: dict[str, dict[str, Any]] = {}
	for one in rows:
		if not isinstance(one, dict):
			continue
		key = _row_key(one.get("instance_id"), str(one.get("question") or ""))
		if key:
			cache[key] = one
	return cache


def _normalize_answer(text: Any) -> str:
	if not text:
		return ""
	cleaned = str(text).replace("</s>", "").replace("</think>", "").strip()
	patterns = [
		r"###\s*Final\s*Answer:\s*(.+?)(?:\n|$)",
		r"ANSWER:\s*(.+?)(?:\n|$)",
		r"Answer:\s*(.+?)(?:\n|$)",
		r"###\s*Answer:\s*(.+?)(?:\n|$)",
	]
	for pattern in patterns:
		match_obj = re.search(pattern, cleaned, re.IGNORECASE)
		if match_obj:
			cleaned = match_obj.group(1).strip()
			break
	cleaned = cleaned.splitlines()[-1].strip()
	cleaned = cleaned.strip(" .,;:!?\"'()[]")
	if cleaned.lower() in {"", "unanswerable", "unknown", "none", "not mentioned", "cannot be determined"}:
		return "unanswerable"
	return cleaned


def _normalize_legalbench_label(label: str, source_file: str) -> str:
	value = (label or "").strip().lower()
	if source_file == "privacy_policy_qa.jsonl":
		if value in {"relevant", "rel", "yes", "true", "1"}:
			return "Relevant"
		if value in {"irrelevant", "irrel", "no", "false", "0"}:
			return "Irrelevant"
		return "Irrelevant"
	if value in {"yes", "true", "1"}:
		return "Yes"
	if value in {"no", "false", "0"}:
		return "No"
	return "No"


def _evaluate_answer(prediction: str, ground_truth: list[str] | str) -> dict[str, float]:
	if isinstance(ground_truth, list):
		ground_truths = [gt for gt in ground_truth if isinstance(gt, str) and gt.strip()]
	else:
		ground_truths = [ground_truth] if isinstance(ground_truth, str) and ground_truth.strip() else []
	if not ground_truths:
		return {"exact_match": 0.0, "f1": 0.0, "match": 0.0, "hit": 0.0}
	em_score = float(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
	f1_score = max(qa_f1_score(prediction, gt) for gt in ground_truths)
	match_score = float(match(prediction, ground_truths))
	hit = em_score > 0.0 or f1_score > 0.0 or match_score > 0.0
	return {"exact_match": em_score, "f1": f1_score, "match": match_score, "hit": float(hit)}


def _extract_context_text(item: dict[str, Any]) -> str:
	context_docs = item.get("context_docs") or []
	if isinstance(context_docs, list) and context_docs:
		parts = [str(doc).strip() for doc in context_docs if str(doc).strip()]
		if parts:
			return "\n\n".join(parts)
	context = item.get("context") or []
	if isinstance(context, list):
		parts: list[str] = []
		for record in context:
			if not (isinstance(record, (list, tuple)) and len(record) >= 2):
				continue
			sentences = record[1]
			if not isinstance(sentences, list):
				continue
			text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip()).strip()
			if text:
				parts.append(text)
		if parts:
			return "\n\n".join(parts)
	for key in ["text", "contract", "passage"]:
		value = item.get(key)
		if isinstance(value, str) and value.strip():
			return value.strip()
	return ""


def _extract_context_docs(item: dict[str, Any], max_docs: int = 20) -> list[str]:
	context_text = _extract_context_text(item)
	if not context_text:
		return []
	return [context_text][:max_docs]


def _build_prompt(question: str, context_text: str, source_file: str) -> str:
	if source_file == "privacy_policy_qa.jsonl":
		template = LEGALBENCH_PROMPT_PRIVACY
	elif source_file == "consumer_contracts_qa.jsonl":
		template = LEGALBENCH_PROMPT_CONSUMER_CONTRACT
	else:
		template = LEGALBENCH_PROMPT_CONTRACT
	return template.format(context_text=context_text, question=question)


def _load_legalbench_items(dataset_paths: list[str], limit: int | None = None) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	seen_keys: set[str] = set()
	for one_path in dataset_paths:
		path = Path(one_path)
		if not path.exists():
			continue
		for item in load_task_dataset(task="legalbench", dataset_path=str(path)):
			if not isinstance(item, dict):
				continue
			source_file = str(item.get("_source_file") or item.get("source_file") or "")
			if source_file not in LEGALBENCH_SOURCE_FILES:
				continue
			question = (item.get("question") or "").strip()
			if not question:
				continue
			key = str(item.get("_id") or question)
			if key in seen_keys:
				continue
			seen_keys.add(key)
			item = dict(item)
			item["source_file"] = source_file
			rows.append(item)
			if limit is not None and limit > 0 and len(rows) >= limit:
				return rows
	return rows


def _load_dataset_index_multi(dataset_paths: list[str], target_ids: set[str]) -> dict[str, dict[str, Any]]:
	merged: dict[str, dict[str, Any]] = {}
	for path in dataset_paths:
		p = Path(path)
		if not p.exists():
			continue
		one = load_dataset_index(task="legalbench", dataset_path=str(p))
		for key, value in one.items():
			if key not in merged:
				merged[key] = value
	return merged


def _find_dataset_item_by_question(dataset_index: dict[str, dict[str, Any]], question: str) -> dict[str, Any] | None:
	if not question:
		return None
	for item in dataset_index.values():
		if (item.get("question") or "").strip() == question.strip():
			return item
	return None


def _build_marked_context_for_method(method: str, question: str, docs: list[str], model: ChatOllama) -> str:
	if not docs:
		return ""
	if method == "cdit":
		judgments = judge_similarity_batch(query_str=question, doc_list=docs, model=model)
		marked_docs: list[str] = []
		for idx, doc in enumerate(docs):
			is_similar = judgments[idx] if idx < len(judgments) else False
			tag = CONSISTENT_TAG if is_similar else INCONSISTENT_TAG
			marked_docs.append(f"{tag}\n{doc}")
		return "\n\n".join(marked_docs)
	if method == "contradoc":
		judgments = judge_contradiction_batch(doc_a_list=[question] * len(docs), doc_b_list=docs, model=model)
		marked_docs = []
		for idx, doc in enumerate(docs):
			has_contradiction, evidence = judgments[idx] if idx < len(judgments) else (False, "")
			tag = INCONSISTENT_TAG if has_contradiction else CONSISTENT_TAG
			if has_contradiction and evidence:
				marked_docs.append(f"{tag}\n{doc}\nEvidence:\n{evidence}")
			else:
				marked_docs.append(f"{tag}\n{doc}")
		return "\n\n".join(marked_docs)
	return "\n\n".join(docs)


def _prepare_method_rows(
	method: str,
	prompts_root: Path,
	dataset_index: dict[str, dict[str, Any]],
	dataset_paths: list[str],
	model: ChatOllama,
	context_cache_file: Path | None = None,
	existing_result_map: dict[str, dict[str, Any]] | None = None,
	limit: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
	if method in DIRECT_CONTEXT_METHODS:
		direct_rows, source = _load_direct_prompt_rows(method=method, prompts_root=prompts_root, limit=limit)
		if direct_rows:
			return direct_rows, source

	dataset_rows = _load_legalbench_items(dataset_paths=dataset_paths, limit=limit)
	generated_rows: list[dict[str, Any]] = []
	cache_map = _load_context_cache_map(context_cache_file) if context_cache_file else {}
	existing_result_map = existing_result_map or {}

	pbar = tqdm(dataset_rows, desc=f"build_context/{method}", unit="q")
	for item in pbar:
		question = (item.get("question") or "").strip()
		instance_id = item.get("_id") or question
		key = _row_key(instance_id, question)
		pbar.set_postfix(instance_id=str(instance_id), refresh=False)
		if not question:
			continue
		if key in existing_result_map:
			generated_rows.append(
				{
					"instance_id": instance_id,
					"question": question,
					"source_file": item.get("source_file", item.get("_source_file", "")),
					"reference_answer": existing_result_map[key].get("reference_answer", []),
					"context_text": "",
				}
			)
			continue
		if key in cache_map:
			generated_rows.append(cache_map[key])
			continue
		docs = _extract_context_docs(item, max_docs=20)
		if not docs:
			continue
		context_text = _build_marked_context_for_method(method=method, question=question, docs=docs, model=model)
		answer = _normalize_legalbench_label(str(item.get("answer", "")), source_file=str(item.get("source_file") or ""))
		source_file = str(item.get("source_file") or item.get("_source_file") or "")
		new_row = {
			"instance_id": instance_id,
			"question": question,
			"source_file": source_file,
			"reference_answer": [answer] if answer else [],
			"context_count": len(docs),
			"context_text": context_text,
			"dataset_item": item,
		}
		generated_rows.append(new_row)
		cache_map[key] = new_row
		if context_cache_file:
			_safe_write_json(context_cache_file, {"method": method, "rows": list(cache_map.values())})

	return generated_rows, "generated_from_dataset_context"


def run_legalbench_baseline(
	prompts_dir: str = DEFAULT_PROMPTS_DIR,
	output_dir: str = DEFAULT_OUTPUT_DIR,
	methods: list[str] | None = None,
	dataset_paths: list[str] | None = None,
	model_name: str = "qwen3.5:9b",
	temperature: float = 0.1,
	limit: int | None = None,
) -> dict[str, Any]:
	methods = methods or list(DEFAULT_METHODS)
	dataset_paths = dataset_paths or list(DEFAULT_DATASET_PATHS)
	prompts_root = Path(prompts_dir)
	out_root = Path(output_dir)
	out_root.mkdir(parents=True, exist_ok=True)

	model = ChatOllama(model=model_name, temperature=temperature, top_p=1, reasoning=False, num_predict=8192)
	global_summary: dict[str, Any] = {
		"prompts_dir": str(prompts_root.resolve()),
		"output_dir": str(out_root.resolve()),
		"dataset_paths": dataset_paths,
		"methods": methods,
		"model_name": model_name,
		"temperature": temperature,
		"results": {},
	}

	all_ids: set[str] = set()
	for method in methods:
		in_file = prompts_root / f"{method}_prompts.jsonl"
		for row in _load_jsonl(in_file):
			instance_id = row.get("instance_id")
			if isinstance(instance_id, str) and instance_id.strip():
				all_ids.add(instance_id.strip())

	dataset_index = _load_dataset_index_multi(dataset_paths=dataset_paths, target_ids=all_ids)

	for method in methods:
		out_file = out_root / f"legalbench_{method}.json"
		existing_result_map = _load_existing_results_map(out_file)
		context_cache_file = out_root / f"legalbench_{method}_contexts.json"

		rows, input_source = _prepare_method_rows(
			method=method,
			prompts_root=prompts_root,
			dataset_index=dataset_index,
			dataset_paths=dataset_paths,
			model=model,
			context_cache_file=context_cache_file,
			existing_result_map=existing_result_map,
			limit=limit,
		)

		if not rows:
			global_summary["results"][method] = {"status": "skipped", "reason": f"empty_or_missing: {input_source}"}
			_safe_write_json(out_root / "legalbench_baseline_summary.json", global_summary)
			continue

		method_results: list[dict[str, Any]] = []
		method_f1: list[float] = []
		method_em: list[float] = []
		method_match: list[float] = []
		method_hit: list[float] = []

		def _method_summary() -> dict[str, Any]:
			return {
				"method": method,
				"input_file": input_source,
				"total": len(method_results),
				"avg_f1": fmean(method_f1) if method_f1 else 0.0,
				"avg_exact_match": fmean(method_em) if method_em else 0.0,
				"avg_match": fmean(method_match) if method_match else 0.0,
				"avg_hit": fmean(method_hit) if method_hit else 0.0,
			}

		_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})

		pbar_exec = tqdm(rows, desc=f"baseline/{method}", unit="q")
		for row in pbar_exec:
			question = (row.get("question") or "").strip()
			instance_id = row.get("instance_id")
			current_key = _row_key(instance_id, question)
			pbar_exec.set_postfix(instance_id=str(instance_id), refresh=False)

			if not question:
				tqdm.write(f"[ERROR][baseline/{method}] instance_id={instance_id} empty question, skip")
				_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})
				global_summary["results"][method] = {"status": "running", "output_file": str(out_file.resolve()), **_method_summary()}
				_safe_write_json(out_root / "legalbench_baseline_summary.json", global_summary)
				continue

			dataset_item = row.get("dataset_item")
			if not isinstance(dataset_item, dict):
				dataset_item = _find_dataset_item_by_question(dataset_index, question)

			existing = existing_result_map.get(current_key)
			if isinstance(existing, dict) and "prediction" in existing and "metrics" in existing:
				metrics_exist = existing.get("metrics", {}) or {}
				try:
					method_f1.append(float(metrics_exist.get("f1", 0.0)))
					method_em.append(float(metrics_exist.get("exact_match", 0.0)))
					method_match.append(float(metrics_exist.get("match", 0.0)))
					method_hit.append(float(metrics_exist.get("hit", 0.0)))
				except Exception:
					pass
				method_results.append(existing)
				continue

			source_file = str(row.get("source_file") or row.get("_source_file") or "")
			reference_answer = row.get("reference_answer", [])
			if not source_file and isinstance(dataset_item, dict):
				source_file = str(dataset_item.get("source_file") or dataset_item.get("_source_file") or "")
			if not source_file:
				source_file = _source_file_from_context_type(str(row.get("context_type") or ""))

			if method in DIRECT_CONTEXT_METHODS:
				context_text = (row.get("context_text") or "").strip()
				if not context_text:
					context_text = _extract_context_text(dataset_item or {})
				if not source_file and isinstance(dataset_item, dict):
					source_file = str(dataset_item.get("source_file") or dataset_item.get("_source_file") or "")
				prompt = row.get("prompt") or _build_prompt(question=question, context_text=context_text, source_file=source_file)
			elif method in SIMILARITY_METHODS | CONTRADICT_METHODS | RAW_CONTEXT_METHODS:
				context_text = (row.get("context_text") or "").strip()
				if not context_text and isinstance(dataset_item, dict):
					context_text = _extract_context_text(dataset_item)
				prompt = _build_prompt(question=question, context_text=context_text, source_file=source_file)
			else:
				context_text = (row.get("context_text") or "").strip()
				prompt = row.get("prompt") or _build_prompt(question=question, context_text=context_text, source_file=source_file)

			try:
				raw_answer = get_invoke(model, prompt)
				prediction = _normalize_legalbench_label(_normalize_answer(raw_answer), source_file=source_file)
			except Exception as exc:
				tqdm.write(
					f"[ERROR][baseline/{method}] instance_id={instance_id} question={question[:80]} err={type(exc).__name__}: {exc}"
				)
				continue

			if not reference_answer and isinstance(dataset_item, dict):
				reference_answer = [
					_normalize_legalbench_label(str(dataset_item.get("answer", "")), source_file=source_file)
				]
			elif isinstance(reference_answer, str):
				reference_answer = [_normalize_legalbench_label(reference_answer, source_file=source_file)]

			if method in SIMILARITY_METHODS:
				reference_answer = reference_answer or []
			elif method in CONTRADICT_METHODS:
				reference_answer = reference_answer or []

			metrics = _evaluate_answer(prediction, reference_answer)
			method_f1.append(metrics["f1"])
			method_em.append(metrics["exact_match"])
			method_match.append(metrics["match"])
			method_hit.append(metrics["hit"])

			method_results.append(
				{
					"instance_id": instance_id,
					"source_file": source_file,
					"question": question,
					"reference_answer": reference_answer,
					"prediction": prediction,
					"metrics": metrics,
					"context_text": context_text,
				}
			)

			_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})
			global_summary["results"][method] = {"status": "running", "output_file": str(out_file.resolve()), **_method_summary()}
			_safe_write_json(out_root / "legalbench_baseline_summary.json", global_summary)

		method_summary = _method_summary()
		payload = {"summary": method_summary, "results": method_results}
		_safe_write_json(out_file, payload)
		global_summary["results"][method] = {"status": "ok", "output_file": str(out_file.resolve()), **method_summary}
		_safe_write_json(out_root / "legalbench_baseline_summary.json", global_summary)

	overall_file = out_root / "legalbench_baseline_summary.json"
	_safe_write_json(overall_file, global_summary)
	global_summary["summary_file"] = str(overall_file.resolve())
	return global_summary


def main() -> None:
	parser = argparse.ArgumentParser(description="Run LegalBench baseline on stored prompt jsonl files")
	parser.add_argument("--prompts-dir", type=str, default=DEFAULT_PROMPTS_DIR)
	parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--dataset-paths", type=str, default=",".join(DEFAULT_DATASET_PATHS))
	parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
	parser.add_argument("--model-name", type=str, default="qwen3.5:9b")
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--limit", type=int, default=0)
	args = parser.parse_args()

	methods = [one.strip() for one in args.methods.split(",") if one.strip()]
	dataset_paths = [one.strip() for one in args.dataset_paths.split(",") if one.strip()]
	report = run_legalbench_baseline(
		prompts_dir=args.prompts_dir,
		output_dir=args.output_dir,
		methods=methods,
		dataset_paths=dataset_paths,
		model_name=args.model_name,
		temperature=args.temperature,
		limit=(args.limit or None),
	)
	print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
