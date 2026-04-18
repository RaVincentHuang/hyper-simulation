import argparse
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from langchain_ollama import ChatOllama
from tqdm import tqdm

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.llm.chat_completion import get_invoke
from hyper_simulation.question_answer.vmdit.metrics import (
	exact_match_score,
	match,
	metric_max_over_ground_truths,
	qa_f1_score,
)

from refine_hypergraph import load_dataset_index


DEFAULT_INSTANCES_ROOT = "data/debug/legalbench/sample975"
DEFAULT_DATASET_PATH = "/home/vincent/.dataset/LegalBench/sample975"
DEFAULT_OUTPUT_PATH = "data/debug/legalbench/multihop_qa.json"


LEGALBENCH_PROMPT_CONSUMER_CONTRACT = """### Legal Document:
{context_text}

### Question:
{question}

### Hyper-simulation guidance:
The following query/data alignments are non-conflicting and can be trusted as consistent semantic anchors:
{non_conflict_items}

The following query-side key items were not matched in the document by hyper-simulation:
{unmatched_items}
Treat unmatched items as uncertainty signals and avoid over-claiming.

### Task:
Decide whether the document supports a YES or NO answer to the question.

### Output format (STRICT):
- Output exactly one line: ### Final Answer: Yes
- or: ### Final Answer: No
- No extra text.

### Response:
"""


LEGALBENCH_PROMPT_CONTRACT = """### Contract Text:
{context_text}

### Question:
{question}

### Hyper-simulation guidance:
The following query/data alignments are non-conflicting and can be trusted as consistent semantic anchors:
{non_conflict_items}

The following query-side key items were not matched in the contract by hyper-simulation:
{unmatched_items}
Treat unmatched items as uncertainty signals and avoid over-claiming.

### Task:
Answer whether the contract entails the asked condition.

### Output format (STRICT):
- Output exactly one line: ### Final Answer: Yes
- or: ### Final Answer: No
- No extra text.

### Response:
"""


LEGALBENCH_PROMPT_PRIVACY = """### Privacy Policy Text:
{context_text}

### User Question:
{question}

### Hyper-simulation guidance:
The following query/data alignments are non-conflicting and can be trusted as consistent semantic anchors:
{non_conflict_items}

The following query-side key items were not matched in the policy by hyper-simulation:
{unmatched_items}
Treat unmatched items as uncertainty signals and avoid over-claiming.

### Task:
Classify whether the policy content is relevant to the asked privacy concern.

### Output format (STRICT):
- Output exactly one line: ### Final Answer: Relevant
- or: ### Final Answer: Irrelevant
- No extra text.

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


def _extract_legalbench_context_text(item: dict[str, Any]) -> str:
	for key in ["contract", "text", "context", "passage"]:
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

	context_text = _extract_legalbench_context_text(item)
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
				"title": item.get("_source_file", "legalbench"),
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


def _build_hyper_simulation_hints(
	query_hg: LocalHypergraph,
	mapping: Mapping[int, set[int] | list[int]],
	q_map: dict[int, Vertex],
	d_map: dict[int, Vertex],
	max_items: int = 24,
) -> tuple[list[str], list[str], str, str]:
	matched_pairs: list[str] = []
	matched_qids: set[int] = set()

	for q_id in sorted(mapping.keys()):
		query_vertex = q_map.get(q_id)
		if query_vertex is None or not _is_content_vertex(query_vertex):
			continue
		d_ids = sorted(int(d_id) for d_id in mapping.get(q_id, []) if int(d_id) in d_map)
		if not d_ids:
			continue
		matched_qids.add(q_id)
		for d_id in d_ids:
			data_vertex = d_map[d_id]
			matched_pairs.append(f"- Q[{query_vertex.id}] {query_vertex.text()} <-> D[{data_vertex.id}] {data_vertex.text()}")
			if len(matched_pairs) >= max_items:
				break
		if len(matched_pairs) >= max_items:
			break

	unmatched_queries: list[str] = []
	for vertex in query_hg.vertices:
		if not _is_content_vertex(vertex):
			continue
		if vertex.id in matched_qids:
			continue
		unmatched_queries.append(f"- Q[{vertex.id}] {vertex.text()}")
		if len(unmatched_queries) >= max_items:
			break

	non_conflict_items = "\n".join(matched_pairs) if matched_pairs else "- (none)"
	unmatched_items = "\n".join(unmatched_queries) if unmatched_queries else "- (none)"
	return matched_pairs, unmatched_queries, non_conflict_items, unmatched_items


def _build_task_prompt(
	question: str,
	context_text: str,
	source_file: str,
	non_conflict_items: str,
	unmatched_items: str,
) -> str:
	if source_file == "privacy_policy_qa.jsonl":
		template = LEGALBENCH_PROMPT_PRIVACY
	elif source_file == "consumer_contracts_qa.jsonl":
		template = LEGALBENCH_PROMPT_CONSUMER_CONTRACT
	else:
		template = LEGALBENCH_PROMPT_CONTRACT

	return template.format(
		context_text=context_text,
		question=question,
		non_conflict_items=non_conflict_items,
		unmatched_items=unmatched_items,
	)


def run_legalbench_multihop_evaluation(
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

	instance_dirs = sorted(
		[path for path in root.iterdir() if path.is_dir() and (path / "query_hypergraph.pkl").exists()]
	)
	if limit_instances is not None and limit_instances > 0:
		instance_dirs = instance_dirs[:limit_instances]
	if not instance_dirs:
		raise FileNotFoundError(f"No valid instance directories found under: {root}")

	dataset_index = load_dataset_index(task="legalbench", dataset_path=dataset_path)

	model = ChatOllama(
		model=model_name,
		temperature=temperature,
		reasoning=False,
		num_predict=8192,
	)

	results: list[dict[str, Any]] = []
	hit0_cases: list[dict[str, Any]] = []
	all_f1_scores: list[float] = []
	all_em_scores: list[float] = []
	all_match_scores: list[float] = []
	all_hit_scores: list[float] = []

	for instance_dir in tqdm(instance_dirs, desc="LegalBench QA", unit="inst"):
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
		matched_pairs, unmatched_queries, non_conflict_items, unmatched_items = _build_hyper_simulation_hints(
			query_hg=query_hg,
			mapping=mapping,
			q_map=q_map,
			d_map=d_map,
		)

		context_text = "\n\n".join((entry.get("text") or "").strip() for entry in evidence_items if (entry.get("text") or "").strip())
		source_file = str(item.get("_source_file") or "")
		question = (item.get("question") or "").strip()
		prompt = _build_task_prompt(
			question=question,
			context_text=context_text,
			source_file=source_file,
			non_conflict_items=non_conflict_items,
			unmatched_items=unmatched_items,
		)

		raw_answer = get_invoke(model, prompt)
		prediction = _normalize_legalbench_label(_normalize_answer(raw_answer), source_file=source_file)

		final_ground_truth = [_normalize_legalbench_label(str(item.get("answer", "")), source_file=source_file)]
		final_metrics = _evaluate_answer(prediction, final_ground_truth)

		all_f1_scores.append(final_metrics["f1"])
		all_em_scores.append(final_metrics["exact_match"])
		all_match_scores.append(final_metrics["match"])
		all_hit_scores.append(final_metrics["hit"])

		if final_metrics.get("hit", 0.0) == 0.0:
			hit0_cases.append(
				{
					"instance_id": instance_dir.name,
					"source_file": source_file,
					"question": question,
					"prediction": prediction,
					"ground_truth": final_ground_truth,
					"metrics": final_metrics,
				}
			)

		tqdm.write(
			"\n" + "-" * 72
			+ f"\nTask: {source_file or 'unknown'}"
			+ f"\nQuestion: {question}"
			+ f"\nLLM Answer: {prediction}"
			+ f"\nStandard Answer: {', '.join(gt for gt in final_ground_truth if gt) or 'N/A'}"
			+ f"\nF1: {final_metrics['f1']:.4f}"
			+ f"\nExact Match: {final_metrics['exact_match']:.4f}"
			+ f"\nMatch: {final_metrics['match']:.4f}"
			+ f"\nHit: {final_metrics['hit']:.4f}"
			+ "\n" + "-" * 72
		)

		instance_report = {
			"instance_id": instance_dir.name,
			"status": "ok",
			"source_file": source_file,
			"question": question,
			"final": {
				"question": question,
				"prediction": prediction,
				"ground_truth": final_ground_truth,
				"metrics": final_metrics,
				"non_conflict_matches": matched_pairs,
				"query_unmatched_items": unmatched_queries,
			},
		}
		results.append(instance_report)

	summary = {
		"instances_root": str(root.resolve()),
		"dataset_path": str(Path(dataset_path).resolve()),
		"total_instances": len(results),
		"evaluated_instances": sum(1 for data_item in results if data_item.get("status") == "ok"),
		"overall_f1": fmean(all_f1_scores) if all_f1_scores else 0.0,
		"overall_exact_match": fmean(all_em_scores) if all_em_scores else 0.0,
		"overall_match": fmean(all_match_scores) if all_match_scores else 0.0,
		"overall_hit": fmean(all_hit_scores) if all_hit_scores else 0.0,
		"final_count": len(all_f1_scores),
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
	print("LegalBench QA results")
	print("=" * 72)
	print(f"Total instances: {summary['total_instances']}")
	print(f"Evaluated instances: {summary['evaluated_instances']}")
	print(f"Overall F1: {summary['overall_f1']:.4f}")
	print(f"Overall Exact Match: {summary['overall_exact_match']:.4f}")
	print(f"Overall Match: {summary['overall_match']:.4f}")
	print(f"Overall Hit: {summary['overall_hit']:.4f}")
	print(f"Saved to: {out_path}")
	print(f"Hit=0 cases saved to: {hit0_out_path}")
	print("=" * 72)

	return payload


def main() -> None:
	parser = argparse.ArgumentParser(description="LegalBench QA with hyper-simulation hints")
	parser.add_argument("--instances-root", type=str, default=DEFAULT_INSTANCES_ROOT)
	parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--hit0-output-path", type=str, default="")
	parser.add_argument("--model-name", type=str, default="qwen3.5:9b")
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--limit-instances", type=int, default=0)
	args = parser.parse_args()

	run_legalbench_multihop_evaluation(
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
