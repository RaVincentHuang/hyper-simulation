from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from langchain_ollama import ChatOllama
from tqdm import tqdm

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.llm.chat_completion import get_invoke

from refine_hypergraph import load_dataset_index


DEFAULT_INSTANCES_ROOT = "data/debug/docnli/sample50"
DEFAULT_DATASET_PATH = "data/nli/docnli_50.jsonl"
DEFAULT_OUTPUT_PATH = "data/debug/docnli/docnli.json"


DOCNLI_PROMPT = """### Task:
Given a document premise and a hypothesis, determine whether the premise entails the hypothesis.

### Premise:
{context_text}

### Hypothesis:
{question}

### Hyper-simulation Hints:
The following semantic alignments between hypothesis and premise are trusted anchors:
{non_conflict_items}

### Label Space:
- entailment: premise supports hypothesis
- non-entailment: premise does not support hypothesis

### Output format (STRICT):
Output exactly one line:
### Final Answer: entailment
or
### Final Answer: non-entailment

### Response:
"""


def _sorted_index_from_name(path: Path) -> int:
	match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
	if match_obj is None:
		return 10**9
	return int(match_obj.group(1))


def _is_content_vertex(vertex: Vertex) -> bool:
	if vertex.is_verb() or vertex.is_virtual():
		return False
	text = vertex.text().strip()
	return bool(text)


def _extract_docnli_context_text(item: dict[str, Any]) -> str:
	premise = item.get("premise")
	if isinstance(premise, list):
		parts = [str(one).strip() for one in premise if str(one).strip()]
		return "\n\n".join(parts)
	if isinstance(premise, str):
		return premise.strip()
	for key in ["text", "context", "passage"]:
		value = item.get(key)
		if isinstance(value, str) and value.strip():
			return value.strip()
	return ""


def _load_instance_graphs(instance_dir: Path, item: dict[str, Any]) -> tuple[LocalHypergraph | None, list[dict[str, Any]]]:
	query_path = instance_dir / "query_hypergraph.pkl"
	if not query_path.exists():
		return None, []

	try:
		query_hg = LocalHypergraph.load(str(query_path))
	except Exception:
		return None, []

	context_text = _extract_docnli_context_text(item)
	data_paths = sorted(instance_dir.glob("data_hypergraph*.pkl"), key=_sorted_index_from_name)

	evidence_items: list[dict[str, Any]] = []
	for data_path in data_paths:
		match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", data_path.name)
		if match_obj is None:
			continue
		data_idx = int(match_obj.group(1))
		try:
			data_hg = LocalHypergraph.load(str(data_path))
		except Exception:
			data_hg = None

		if data_hg is None:
			continue

		evidence_items.append(
			{
				"index": data_idx,
				"path": str(data_path),
				"hypergraph": data_hg,
				"title": "docnli",
				"text": context_text,
			}
		)

	return query_hg, evidence_items


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
	return cleaned


def _normalize_binary_nli_label(label: str) -> str:
	value = (label or "").strip().lower()
	if value in {"entailment", "entails", "entailed", "yes", "support"}:
		return "entailment"
	return "non-entailment"


def _evaluate_answer(prediction: str, ground_truth: str) -> dict[str, float | int]:
	if not ground_truth:
		return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

	pred_is_positive = prediction == "entailment"
	gold_is_positive = ground_truth == "entailment"

	tp = 1 if pred_is_positive and gold_is_positive else 0
	fp = 1 if pred_is_positive and not gold_is_positive else 0
	fn = 1 if (not pred_is_positive) and gold_is_positive else 0
	tn = 1 if (not pred_is_positive) and (not gold_is_positive) else 0
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
	return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def _build_hyper_simulation_hints(
	query_hg: LocalHypergraph,
	mapping: Mapping[int, set[int] | list[int]],
	q_map: dict[int, Vertex],
	d_map: dict[int, Vertex],
	max_items: int = 24,
) -> tuple[list[str], str]:
	matched_pairs: list[str] = []

	for q_id in sorted(mapping.keys()):
		query_vertex = q_map.get(q_id)
		if query_vertex is None or not _is_content_vertex(query_vertex):
			continue
		d_ids = sorted(int(d_id) for d_id in mapping.get(q_id, []) if int(d_id) in d_map)
		if not d_ids:
			continue
		for d_id in d_ids:
			data_vertex = d_map[d_id]
			matched_pairs.append(f"('{query_vertex.text()}', '{data_vertex.text()}')")
			if len(matched_pairs) >= max_items:
				break
		if len(matched_pairs) >= max_items:
			break

	non_conflict_items = "\n".join(matched_pairs) if matched_pairs else "- (none)"
	return matched_pairs, non_conflict_items


def run_docnli_multihop_evaluation(
	instances_root: str = DEFAULT_INSTANCES_ROOT,
	dataset_path: str = DEFAULT_DATASET_PATH,
	output_path: str = DEFAULT_OUTPUT_PATH,
	hit0_output_path: str = "",
	model_name: str = "qwen3.5:9b",
	temperature: float = 0.2,
	limit_instances: int | None = None,
) -> dict[str, Any]:
	root = Path(instances_root)
	if not root.exists():
		raise FileNotFoundError(f"Instances root not found: {root}")

	instance_dirs = sorted([path for path in root.iterdir() if path.is_dir() and (path / "query_hypergraph.pkl").exists()])
	if limit_instances is not None and limit_instances > 0:
		instance_dirs = instance_dirs[:limit_instances]
	if not instance_dirs:
		raise FileNotFoundError(f"No valid instance directories found under: {root}")

	dataset_index = load_dataset_index(task="docnli", dataset_path=dataset_path)

	model = ChatOllama(
		model=model_name,
		temperature=temperature,
		reasoning=False,
		num_predict=8192,
	)

	results: list[dict[str, Any]] = []
	hit0_cases: list[dict[str, Any]] = []
	total_tp = 0
	total_fp = 0
	total_fn = 0
	total_tn = 0

	for instance_dir in tqdm(instance_dirs, desc="DocNLI", unit="inst"):
		item = dataset_index.get(instance_dir.name)
		if item is None:
			results.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": "dataset_item_not_found",
				}
			)
			continue

		query_hg, evidence_items = _load_instance_graphs(instance_dir, item)
		if query_hg is None or not evidence_items:
			results.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": "missing_graphs",
				}
			)
			continue

		valid_hgs = [entry["hypergraph"] for entry in evidence_items if entry.get("hypergraph") is not None]
		if not valid_hgs:
			results.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": "no_valid_evidence_graphs",
				}
			)
			continue

		fusion = MultiHopFusion()
		merged_hg, _provenance = fusion.merge_hypergraphs(valid_hgs)

		mapping, q_map, d_map = compute_hyper_simulation(query_hg, merged_hg)
		matched_pairs, non_conflict_items = _build_hyper_simulation_hints(
			query_hg=query_hg,
			mapping=mapping,
			q_map=q_map,
			d_map=d_map,
		)

		context_text = "\n\n".join((entry.get("text") or "").strip() for entry in evidence_items if (entry.get("text") or "").strip())
		question = (item.get("question") or item.get("hypothesis") or "").strip()
		prompt = DOCNLI_PROMPT.format(
			context_text=context_text,
			question=question,
			non_conflict_items=non_conflict_items,
		)

		raw_answer = get_invoke(model, prompt)
		prediction = _normalize_binary_nli_label(_normalize_answer(raw_answer))

		final_ground_truth = _normalize_binary_nli_label(str(item.get("label", "")))
		final_metrics = _evaluate_answer(prediction, final_ground_truth)

		total_tp += int(final_metrics["tp"])
		total_fp += int(final_metrics["fp"])
		total_fn += int(final_metrics["fn"])
		total_tn += int(final_metrics.get("tn", 0))

		if float(final_metrics.get("f1", 0.0)) == 0.0:
			hit0_cases.append(
				{
					"instance_id": instance_dir.name,
					"hypothesis": question,
					"premise": context_text,
					"prediction": prediction,
					"non_conflict_items": non_conflict_items,
					"ground_truth": final_ground_truth,
					"metrics": final_metrics,
				}
			)

		tqdm.write(
			"\n"
			+ "-" * 72
			+ f"\nHypothesis: {question}"
			+ f"\nLLM Label: {prediction}"
			+ f"\nGold Label: {final_ground_truth or 'N/A'}"
			+ f"\nPrecision: {float(final_metrics['precision']):.4f}"
			+ f"\nRecall: {float(final_metrics['recall']):.4f}"
			+ f"\nF1: {final_metrics['f1']:.4f}"
			+ "\n"
			+ "-" * 72
		)

		instance_report = {
			"instance_id": instance_dir.name,
			"status": "ok",
			"hypothesis": question,
			"final": {
				"hypothesis": question,
				"prediction": prediction,
				"ground_truth": final_ground_truth,
				"metrics": final_metrics,
				"non_conflict_matches": matched_pairs,
			},
		}
		results.append(instance_report)

	overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
	overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
	overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
	ne_tp = total_tn
	ne_fp = total_fn
	ne_fn = total_fp
	non_entailment_precision = ne_tp / (ne_tp + ne_fp) if (ne_tp + ne_fp) > 0 else 0.0
	non_entailment_recall = ne_tp / (ne_tp + ne_fn) if (ne_tp + ne_fn) > 0 else 0.0
	non_entailment_f1 = (
		(2 * non_entailment_precision * non_entailment_recall) / (non_entailment_precision + non_entailment_recall)
		if (non_entailment_precision + non_entailment_recall) > 0
		else 0.0
	)

	summary = {
		"instances_root": str(root.resolve()),
		"dataset_path": str(Path(dataset_path).resolve()),
		"total_instances": len(results),
		"evaluated_instances": sum(1 for data_item in results if data_item.get("status") == "ok"),
		"tp": total_tp,
		"fp": total_fp,
		"fn": total_fn,
		"tn": total_tn,
		"predicted_positive": total_tp + total_fp,
		"gold_positive": total_tp + total_fn,
		"overall_precision": overall_precision,
		"overall_recall": overall_recall,
		"overall_f1": overall_f1,
		"non_entailment_precision": non_entailment_precision,
		"non_entailment_recall": non_entailment_recall,
		"non_entailment_f1": non_entailment_f1,
		"final_count": sum(1 for data_item in results if data_item.get("status") == "ok"),
	}

	payload = {
		"summary": summary,
		"results": results,
	}

	out_path = Path(output_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

	if hit0_output_path:
		hit0_out_path = Path(hit0_output_path)
	else:
		hit0_out_path = out_path.with_name(f"{out_path.stem}_hit0.json")

	hit0_payload = {
		"summary": {
			"instances_root": str(root.resolve()),
			"dataset_path": str(Path(dataset_path).resolve()),
			"hit0_count": len(hit0_cases),
			"note": "These are final-step cases with hit=0.",
		},
		"cases": hit0_cases,
	}
	hit0_out_path.parent.mkdir(parents=True, exist_ok=True)
	hit0_out_path.write_text(json.dumps(hit0_payload, indent=2, ensure_ascii=False), encoding="utf-8")

	print("\n" + "=" * 72)
	print("DocNLI results")
	print("=" * 72)
	print(f"Total instances: {summary['total_instances']}")
	print(f"Evaluated instances: {summary['evaluated_instances']}")
	print(f"Overall Precision: {summary['overall_precision']:.4f}")
	print(f"Overall Recall: {summary['overall_recall']:.4f}")
	print(f"Overall F1: {summary['overall_f1']:.4f}")
	print(f"Saved to: {out_path}")
	print(f"Hit=0 cases saved to: {hit0_out_path}")
	print("=" * 72)

	return payload


def main() -> None:
	parser = argparse.ArgumentParser(description="DocNLI with hyper-simulation hints")
	parser.add_argument("--instances-root", type=str, default=DEFAULT_INSTANCES_ROOT)
	parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--hit0-output-path", type=str, default="")
	parser.add_argument("--model-name", type=str, default="qwen3.5:9b")
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--limit-instances", type=int, default=0)
	args = parser.parse_args()

	run_docnli_multihop_evaluation(
		instances_root=args.instances_root,
		dataset_path=args.dataset_path,
		output_path=args.output_path,
		hit0_output_path=args.hit0_output_path,
		model_name=args.model_name,
		temperature=args.temperature,
		limit_instances=args.limit_instances or None,
	)


if __name__ == "__main__":
	main()
