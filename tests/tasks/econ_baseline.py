from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama
from tqdm import tqdm

from hyper_simulation.baselines.contradoc import judge_contradiction_batch
from refine_hypergraph import load_task_dataset


DEFAULT_OUTPUT_DIR = "data/baseline/econ"
DEFAULT_DATASET_PATHS = ["data/nli/econ_qa.jsonl"]
EVAL_VERSION = "contradoc_binary_v3"


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
	tmp_path.replace(path)


def _row_key(instance_id: Any, hypothesis: str) -> str:
	inst = str(instance_id).strip() if instance_id is not None else ""
	h = str(hypothesis or "").strip()
	return inst if inst else h


def _normalize_econ_label(label: str) -> str:
	value = (label or "").strip().lower()

	if value in {"yes", "true", "1"}:
		return "Yes"
	if value in {"no", "false", "0"}:
		return "No"
	return "No"


def _normalize_binary_from_econ_label(label: str) -> str:
	# In econ QA, answer "No" means the hypothesis is contradicted by the premise.
	normalized = _normalize_econ_label(label)
	return "contradiction" if normalized == "No" else "non-contradiction"


def _evaluate_answer(prediction: str, ground_truth: str) -> dict[str, float | int]:
	if not ground_truth:
		return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

	pred_is_positive = prediction == "contradiction"
	gold_is_positive = ground_truth == "contradiction"

	tp = 1 if pred_is_positive and gold_is_positive else 0
	fp = 1 if pred_is_positive and not gold_is_positive else 0
	fn = 1 if (not pred_is_positive) and gold_is_positive else 0
	tn = 1 if (not pred_is_positive) and (not gold_is_positive) else 0
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
	return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def _is_reusable_existing_result(existing: dict[str, Any]) -> bool:
	if existing.get("eval_version") != EVAL_VERSION:
		return False
	metrics = existing.get("metrics")
	if not isinstance(metrics, dict):
		return False
	return all(key in metrics for key in ("tp", "fp", "fn"))


def _extract_context_text(item: dict[str, Any]) -> str:
	for key in ["premise", "text", "context", "passage"]:
		value = item.get(key)
		if isinstance(value, str) and value.strip():
			return value.strip()
	return ""


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
		key = _row_key(one.get("instance_id"), str(one.get("hypothesis") or ""))
		if key:
			res[key] = one
	return res


def _load_econ_items(dataset_paths: list[str], limit: int | None = None) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	seen_keys: set[str] = set()
	for one_path in dataset_paths:
		path = Path(one_path)
		if not path.exists():
			continue
		for item in load_task_dataset(task="econ", dataset_path=str(path)):
			if not isinstance(item, dict):
				continue
			hypothesis = (item.get("hypothesis") or "").strip()
			if not hypothesis:
				continue
			key = str(item.get("_id") or hypothesis)
			if key in seen_keys:
				continue
			seen_keys.add(key)
			item = dict(item)
			rows.append(item)
			if limit is not None and limit > 0 and len(rows) >= limit:
				return rows
	return rows


def run_econ_baseline(
	output_dir: str = DEFAULT_OUTPUT_DIR,
	dataset_paths: list[str] | None = None,
	model_name: str = "qwen3.5:9b",
	temperature: float = 0.1,
	limit: int | None = None,
) -> dict[str, Any]:
	dataset_paths = dataset_paths or list(DEFAULT_DATASET_PATHS)
	out_root = Path(output_dir)
	out_root.mkdir(parents=True, exist_ok=True)

	model = ChatOllama(model=model_name, temperature=temperature, top_p=1, reasoning=False, num_predict=8192)
	global_summary: dict[str, Any] = {
		"output_dir": str(out_root.resolve()),
		"dataset_paths": dataset_paths,
		"method": "contradoc_only",
		"model_name": model_name,
		"temperature": temperature,
		"results": {},
	}

	method = "contradoc"
	out_file = out_root / f"econ_{method}.json"
	existing_result_map = _load_existing_results_map(out_file)

	dataset_rows = _load_econ_items(dataset_paths=dataset_paths, limit=limit)
	if not dataset_rows:
		global_summary["results"][method] = {"status": "skipped", "reason": "empty_or_missing"}
		_safe_write_json(out_root / "econ_baseline_summary.json", global_summary)
		return global_summary

	method_results: list[dict[str, Any]] = []
	total_tp = 0
	total_fp = 0
	total_fn = 0
	total_tn = 0

	def _method_summary() -> dict[str, Any]:
		precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
		recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
		f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
		return {
			"method": method,
			"total": len(method_results),
			"tp": total_tp,
			"fp": total_fp,
			"fn": total_fn,
			"tn": total_tn,
			"predicted_positive": total_tp + total_fp,
			"gold_positive": total_tp + total_fn,
			"precision": precision,
			"recall": recall,
			"f1": f1,
		}

	_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})

	pbar_exec = tqdm(dataset_rows, desc=f"baseline/{method}", unit="q")
	for item in pbar_exec:
		hypothesis = (item.get("hypothesis") or "").strip()
		instance_id = item.get("_id") or hypothesis
		current_key = _row_key(instance_id, hypothesis)
		pbar_exec.set_postfix(instance_id=str(instance_id), refresh=False)

		if not hypothesis:
			continue

		existing = existing_result_map.get(current_key)
		if isinstance(existing, dict) and "prediction" in existing and _is_reusable_existing_result(existing):
			method_results.append(existing)
			total_tp += int(existing.get("metrics", {}).get("tp", 0))
			total_fp += int(existing.get("metrics", {}).get("fp", 0))
			total_fn += int(existing.get("metrics", {}).get("fn", 0))
			total_tn += int(existing.get("metrics", {}).get("tn", 0))
			_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})
			global_summary["results"][method] = {"status": "running", "output_file": str(out_file.resolve()), **_method_summary()}
			_safe_write_json(out_root / "econ_baseline_summary.json", global_summary)
			continue

		context_text = _extract_context_text(item)
		if not context_text:
			continue

		# Contradoc-only baseline: no second-stage LLM call.
		judgments = judge_contradiction_batch(doc_a_list=[hypothesis], doc_b_list=[context_text], model=model)
		has_contradiction, evidence = judgments[0] if judgments else (False, "")
		prediction = "contradiction" if has_contradiction else "non-contradiction"

		reference_answer = _normalize_binary_from_econ_label(str(item.get("label", "")))
		metrics = _evaluate_answer(prediction, reference_answer)

		total_tp += int(metrics["tp"])
		total_fp += int(metrics["fp"])
		total_fn += int(metrics["fn"])
		total_tn += int(metrics.get("tn", 0))

		method_results.append(
			{
				"instance_id": instance_id,
				"eval_version": EVAL_VERSION,
				"hypothesis": hypothesis,
				"reference_answer": reference_answer,
				"prediction": prediction,
				"metrics": metrics,
				"contradiction_detected": bool(has_contradiction),
				"contradiction_evidence": str(evidence or "")[:500],
			}
		)

		_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})
		global_summary["results"][method] = {"status": "running", "output_file": str(out_file.resolve()), **_method_summary()}
		_safe_write_json(out_root / "econ_baseline_summary.json", global_summary)

	method_summary = _method_summary()
	payload = {"summary": method_summary, "results": method_results}
	_safe_write_json(out_file, payload)
	global_summary["results"][method] = {"status": "ok", "output_file": str(out_file.resolve()), **method_summary}
	_safe_write_json(out_root / "econ_baseline_summary.json", global_summary)

	overall_file = out_root / "econ_baseline_summary.json"
	_safe_write_json(overall_file, global_summary)
	global_summary["summary_file"] = str(overall_file.resolve())

	print("\n" + "=" * 72)
	print("Econ Baseline (contradoc) results")
	print("=" * 72)
	print(f"Total evaluated: {method_summary['total']}")
	print(f"Overall Precision: {method_summary['precision']:.4f}")
	print(f"Overall Recall: {method_summary['recall']:.4f}")
	print(f"Overall F1: {method_summary['f1']:.4f}")
	print(f"Saved to: {out_file}")
	print("=" * 72)

	return global_summary


def main() -> None:
	parser = argparse.ArgumentParser(description="Econ NLI baseline (contradoc method only)")
	parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--dataset-paths", type=str, default=",".join(DEFAULT_DATASET_PATHS))
	parser.add_argument("--model-name", type=str, default="qwen3.5:9b")
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--limit", type=int, default=0)
	args = parser.parse_args()

	dataset_paths = [one.strip() for one in args.dataset_paths.split(",") if one.strip()]
	report = run_econ_baseline(
		output_dir=args.output_dir,
		dataset_paths=dataset_paths,
		model_name=args.model_name,
		temperature=args.temperature,
		limit=(args.limit or None),
	)
	print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
