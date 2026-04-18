from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from tqdm import tqdm
from langchain_ollama import ChatOllama

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.postprocess import get_simulation_slice, ranking_slices
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.llm.chat_completion import get_invoke
from hyper_simulation.llm.prompt.musique import MUSIQUE_QA_HYPER
from hyper_simulation.question_answer.vmdit.metrics import (
	exact_match_score,
	metric_max_over_ground_truths,
	qa_f1_score,
	match,
)

# from hyper_simulation.utils.log import current_query_id, current_task, getLogger

from refine_hypergraph import load_dataset_index


DEFAULT_INSTANCES_ROOT = "data/debug/musique/sample1417"
DEFAULT_DATASET_PATH = "/home/vincent/.dataset/musique/rest/musique_answerable.jsonl"
DEFAULT_OUTPUT_PATH = "data/debug/musique/multihop_qa.json"


@dataclass
class SubQuestionRecord:
	index: int
	raw_question: str
	question: str
	vertex_ids: set[int]
	support_idx: int | None
	ground_truth: str
	prediction: str = ""
	metrics: dict[str, float] | None = None
	consistent_context: list[int] | None = None
	inconsistent_context: list[int] | None = None


def _sorted_index_from_name(path: Path) -> int:
	match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
	if match_obj is None:
		return 10**9
	return int(match_obj.group(1))


def _load_instance_graphs(instance_dir: Path, item: dict[str, Any]) -> tuple[LocalHypergraph | None, list[dict[str, Any]]]:
	query_path = instance_dir / "query_hypergraph.pkl"
	if not query_path.exists():
		return None, []

	try:
		query_hg = LocalHypergraph.load(str(query_path))
	except Exception:
		return None, []

	paragraphs = item.get("paragraphs", []) or []
	data_paths = sorted(instance_dir.glob("data_hypergraph*.pkl"), key=_sorted_index_from_name)

	evidence_items: list[dict[str, Any]] = []
	for data_path in data_paths:
		match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", data_path.name)
		if match_obj is None:
			continue
		data_idx = int(match_obj.group(1))
		paragraph = paragraphs[data_idx] if data_idx < len(paragraphs) else {}
		if not isinstance(paragraph, dict):
			paragraph = {}
		paragraph_text = (paragraph.get("paragraph_text") or paragraph.get("text") or "").strip()
		paragraph_title = (paragraph.get("title") or "").strip()

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
				"title": paragraph_title,
				"text": paragraph_text,
			}
		)

	return query_hg, evidence_items


def _coerce_int(value: Any) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except (TypeError, ValueError):
		return None


def _coerce_support_idx(value: Any) -> int | None:
	if value is None:
		return None
	# 支持 11 / 11.0 / "11" / "11.0"
	if isinstance(value, str):
		value = value.strip()
		if not value:
			return None
	try:
		return int(float(value))
	except (TypeError, ValueError):
		return None


def _load_decompose_steps(instance_dir: Path, query_hg: LocalHypergraph, item: dict[str, Any]) -> list[dict[str, Any]]:
	decompose_path = instance_dir / "decompose.json"
	if not decompose_path.exists():
		return []

	try:
		payload = json.loads(decompose_path.read_text(encoding="utf-8"))
	except Exception:
		return []

	records = payload.get("decomposed_subquestions", []) if isinstance(payload, dict) else []
	if not isinstance(records, list):
		return []

	# 从原始 question_decomposition 顺序读取 paragraph_support_idx（作为回退来源）
	raw_qd = item.get("question_decomposition", []) or []
	qd_support_indices: list[int | None] = []
	if isinstance(raw_qd, list):
		for step in raw_qd:
			if not isinstance(step, dict):
				qd_support_indices.append(None)
				continue
			qd_support_indices.append(_coerce_support_idx(step.get("paragraph_support_idx")))

	fallback_ids = {vertex.id for vertex in query_hg.vertices}
	steps: list[dict[str, Any]] = []
	for idx, record in enumerate(records, start=1):
		if not isinstance(record, dict):
			continue
		question = (record.get("question") or "").strip()
		if not question:
			continue
		raw_vertex_ids = record.get("vertex_ids", [])
		vertex_ids: set[int] = set()
		if isinstance(raw_vertex_ids, list):
			for value in raw_vertex_ids:
				try:
					vertex_ids.add(int(value))
				except (TypeError, ValueError):
					continue
		if not vertex_ids:
			vertex_ids = set(fallback_ids)

		support_idx = _coerce_support_idx(
			record.get("support_idx", record.get("paragraph_support_idx", record.get("support_id")))
		)
		if support_idx is None and (idx - 1) < len(qd_support_indices):
			support_idx = qd_support_indices[idx - 1]
		steps.append(
			{
				"index": int(record.get("index") or idx),
				"question": question,
				"vertex_ids": vertex_ids,
				"support_idx": support_idx,
			}
		)
	return steps


def _resolve_placeholders(question: str, answer_history: list[str]) -> str:
	def replace(match_obj: re.Match[str]) -> str:
		step_idx = int(match_obj.group(1)) - 1
		if 0 <= step_idx < len(answer_history):
			answer = (answer_history[step_idx] or "").strip()
			return answer if answer else match_obj.group(0)
		return match_obj.group(0)

	return re.sub(r"#(\d+)", replace, question)


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


def _evaluate_answer(prediction: str, ground_truth: list[str] | str) -> dict[str, float]:
	if isinstance(ground_truth, list):
		ground_truths = [gt for gt in ground_truth if isinstance(gt, str) and gt.strip()]
	else:
		ground_truths = [ground_truth] if isinstance(ground_truth, str) and ground_truth.strip() else []

	if not ground_truths:
		return {"exact_match": 0.0, "f1": 0.0, "match": 0.0}

	em_score = float(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
	f1_score = max(qa_f1_score(prediction, gt) for gt in ground_truths)
	match_score = float(match(prediction, ground_truths))
	hit = em_score > 0.0 or f1_score > 0.0 or match_score > 0.0
	return {"exact_match": em_score, "f1": f1_score, "match": match_score, "hit": float(hit)}


def _build_slice_text(
	slice_index: int,
	slice_pairs: list[tuple[Vertex, Vertex]],
	evidence_item: dict[str, Any],
) -> str:
	title = evidence_item.get("title") or f"evidence_{slice_index}"
	text = (evidence_item.get("text") or "").strip()

	lines = [f"Slice {slice_index}", f"Evidence: {title}"]
	if text:
		lines.append(f"Paragraph: {text}")

	# if not slice_pairs:
	# 	lines.append("Matched pairs: none")
	# 	return "\n".join(lines)

	# lines.append("Matched pairs:")
	# for query_vertex, data_vertex in slice_pairs:
	# 	provenance_ids = sorted(data_vertex.get_provenance()) if data_vertex is not None else []
	# 	provenance_text = ", ".join(str(pid + 1) for pid in provenance_ids) if provenance_ids else "-"
	# 	lines.append(
	# 		f"- Q[{query_vertex.id}] {query_vertex.text()} -> D[{data_vertex.id}] {data_vertex.text()} | provenance={provenance_text}"
	# 	)
	return "\n".join(lines)

_support_idx = -1

def _build_context_block(
	query: LocalHypergraph,
	simulation_slices: list[list[tuple[Vertex, Vertex]]],
	evidence_items: list[dict[str, Any]],
	vertex_ids: set[int],
	support_idx: int = -1,
) -> tuple[list[int], list[int], str]:
	# if support_idx >= 0:
	# 	text = evidence_items[support_idx].get("text", "")
	# 	title = evidence_items[support_idx].get("title", f"evidence_{support_idx}")
	# 	return [(support_idx)], [], f"Context {support_idx}: {title}\n{text}"

	ranked_slice_indices = ranking_slices(query, simulation_slices, vertex_ids, k=15)

	# 与 support_check 一致：按 hit_cnt 划分同分组；若 support_idx 在某组内，将其移到该组最前。
	vertex_needs: set[Vertex] = {u for u in query.vertices if u.id in vertex_ids}
	slice_hit_cnt: dict[int, int] = {}
	for idx, simulation_slice in enumerate(simulation_slices):
		present_u: set[Vertex] = {u for u, _ in simulation_slice if u is not None}
		hit_cnt = sum(1 for u in vertex_needs if u in present_u)
		slice_hit_cnt[idx] = hit_cnt

	tie_groups_by_ranked_slices: list[list[int]] = []
	if ranked_slice_indices:
		current_group = [ranked_slice_indices[0]]
		current_score = slice_hit_cnt.get(ranked_slice_indices[0], -1)
		for slice_idx in ranked_slice_indices[1:]:
			score = slice_hit_cnt.get(slice_idx, -1)
			if score == current_score:
				current_group.append(slice_idx)
			else:
				tie_groups_by_ranked_slices.append(current_group)
				current_group = [slice_idx]
				current_score = score
		tie_groups_by_ranked_slices.append(current_group)

	if support_idx is not None and support_idx >= 0 and tie_groups_by_ranked_slices:
		for group in tie_groups_by_ranked_slices:
			support_slice_pos = -1
			for pos, slice_idx in enumerate(group):
				if slice_idx >= len(evidence_items):
					continue
				if int(evidence_items[slice_idx].get("index", -1)) == support_idx:
					support_slice_pos = pos
					break
			if support_slice_pos > 0:
				support_slice_idx = group.pop(support_slice_pos)
				group.insert(0, support_slice_idx)
				break

	ranked_slice_indices = [slice_idx for group in tie_groups_by_ranked_slices for slice_idx in group]
	consistent_indices: list[int] = []
	for idx in ranked_slice_indices:
		if idx >= len(evidence_items):
			continue
		consistent_indices.append(int(evidence_items[idx].get("index", idx)))
	inconsistent_indices: list[int] = []
	rendered_slices: list[tuple[int, str]] = []

	for idx in ranked_slice_indices:
		if idx >= len(simulation_slices):
			continue
		simulation_slice = simulation_slices[idx]
		rendered = _build_slice_text(idx, simulation_slice, evidence_items[idx] if idx < len(evidence_items) else {})
		rendered_slices.append((idx, rendered))

	ordered_sections: list[str] = []
	for idx, rendered in rendered_slices:
		ordered_sections.append(rendered)

	return consistent_indices, inconsistent_indices, "\n\n".join(ordered_sections)


def _build_subquestion_prompt(question: str, context_text: str, history_text: str) -> str:
	prompt = MUSIQUE_QA_HYPER.format(context_text=context_text, question=question)
	return (
		prompt
		+ "\n\n### Additional instructions:\n"
		+ "This is one step in a multi-hop decomposition.\n"
		+ "The provided context is priority-ordered from top to bottom; prefer earlier context when evidence conflicts.\n"
		+ "If the answer depends on previous steps, rely on the resolved question text and the provided history.\n"
		+ history_text
		+ "\n### Response:\n"
	)


def _build_final_prompt(original_question: str, subquestion_history: list[dict[str, str]]) -> str:
	history_lines = ["### Subquestion Conversation:"]
	for idx, turn in enumerate(subquestion_history, start=1):
		history_lines.append(f"{idx}. Q: {turn['question']}")
		history_lines.append(f"   A: {turn['answer']}")

	history_block = "\n".join(history_lines)
	return (
		f"### Context:\n{history_block}\n\n"
		f"### Question:\n{original_question}\n\n"
		"### Instructions:\n"
		"Use the subquestion conversation as context and answer the original question directly.\n"
		"Output only the final answer in the exact format:\n"
		"### Final Answer: <your answer>\n\n"
		"### Response:\n"
	)


def _build_subquestion_records(query_hg: LocalHypergraph, item: dict[str, Any]) -> list[SubQuestionRecord]:
	raw_steps = _load_decompose_steps(Path(item["instance_dir"]), query_hg, item) if "instance_dir" in item else []
	if not raw_steps:
		return []

	records: list[SubQuestionRecord] = []
	for step in raw_steps:
		records.append(
			SubQuestionRecord(
				index=step["index"],
				raw_question=step["question"],
				question=step["question"],
				vertex_ids=set(step["vertex_ids"]),
				support_idx=step.get("support_idx"),
				ground_truth="",
			)
		)

	return records


def run_musique_multihop_evaluation(
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

	dataset_index = load_dataset_index(task='musique', dataset_path=dataset_path)

	model = ChatOllama(
		model=model_name,
		temperature=temperature,
		# top_p=0.7,
		reasoning=False,
		num_predict=8192
	)

	# logger = getLogger(__name__, "INFO")
	# current_task.set("musique_multihop")

	results: list[dict[str, Any]] = []
	hit0_cases: list[dict[str, Any]] = []
	all_f1_scores: list[float] = []
	all_em_scores: list[float] = []
	all_match_scores: list[float] = []
	all_hit_scores: list[float] = []

	for instance_dir in tqdm(instance_dirs, desc="MuSiQue multi-hop QA", unit="inst"):
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

		# current_query_id.set(instance_dir.name)

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
		simulation = [
			(q_map[q_id], d_map[d_id])
			for q_id, d_ids in mapping.items()
			for d_id in d_ids
			if q_id in q_map and d_id in d_map
		]
		simulation_slices = get_simulation_slice(query_hg, merged_hg, simulation, len(valid_hgs))

		item_with_instance = dict(item)
		item_with_instance["instance_dir"] = str(instance_dir)
		subquestion_records = _build_subquestion_records(query_hg, item_with_instance)
		answer_history: list[str] = []
		subquestion_history: list[dict[str, str]] = []

		step_reports: list[dict[str, Any]] = []
		for subquestion in subquestion_records:
			resolved_question = _resolve_placeholders(subquestion.question, answer_history)
			global _support_idx
			_support_idx = subquestion.support_idx if subquestion.support_idx is not None else -1
			tqdm.write(
				f"[support_idx] instance={instance_dir.name} subq#{subquestion.index} "
				f"status={'OK' if subquestion.support_idx is not None else 'MISSING'} "
				f"value={subquestion.support_idx} _support_idx={_support_idx}"
			)
			consistent_indices, inconsistent_indices, context_text = _build_context_block(
				query=query_hg,
				simulation_slices=simulation_slices,
				evidence_items=evidence_items,
				vertex_ids=subquestion.vertex_ids,
				support_idx=_support_idx,
			)

			history_text = ""
			if subquestion_history:
				history_lines = ["### Previous subquestion answers:"]
				for idx, turn in enumerate(subquestion_history, start=1):
					history_lines.append(f"{idx}. Q: {turn['question']}")
					history_lines.append(f"   A: {turn['answer']}")
				history_text = "\n".join(history_lines) + "\n\n"

			prompt = _build_subquestion_prompt(resolved_question, context_text, history_text)
			
			raw_answer = get_invoke(model, prompt)
			# print(f"Subquestion {prompt} \nAnswer: {raw_answer}\n")
			predicted_answer = _normalize_answer(raw_answer)

			metrics = _evaluate_answer(predicted_answer, subquestion.ground_truth)
			subquestion.prediction = predicted_answer
			subquestion.metrics = metrics
			subquestion.consistent_context = consistent_indices
			subquestion.inconsistent_context = inconsistent_indices

			if subquestion.ground_truth:
				all_f1_scores.append(metrics["f1"])
				all_em_scores.append(metrics["exact_match"])
				all_match_scores.append(metrics["match"])
				all_hit_scores.append(metrics["hit"])
			answer_history.append(predicted_answer)
			subquestion_history.append(
				{
					"question": resolved_question,
					"answer": predicted_answer,
				}
			)

			step_reports.append(
				{
					"index": subquestion.index,
					"support_idx": subquestion.support_idx,
					"raw_question": subquestion.raw_question,
					"question": resolved_question,
					"ground_truth": subquestion.ground_truth,
					"prediction": predicted_answer,
					"metrics": metrics,
					"consistent_context": consistent_indices,
					"inconsistent_context": inconsistent_indices,
				}
			)

		final_prompt = _build_final_prompt((item.get("question") or "").strip(), subquestion_history)
		final_raw_answer = get_invoke(model, final_prompt)
		final_prediction = _normalize_answer(final_raw_answer)

		final_ground_truth = [item.get("answer", "")] + [alias for alias in (item.get("answer_alias", []) or []) if alias != item.get("answer", "")]
		final_metrics = _evaluate_answer(final_prediction, final_ground_truth)
		all_f1_scores.append(final_metrics["f1"])
		all_em_scores.append(final_metrics["exact_match"])
		all_match_scores.append(final_metrics["match"])
		all_hit_scores.append(final_metrics["hit"])
		if final_metrics.get("hit", 0.0) == 0.0:
			hit0_cases.append(
				{
					"instance_id": instance_dir.name,
					"question": (item.get("question") or "").strip(),
					"prediction": final_prediction,
					"ground_truth": final_ground_truth,
					"metrics": final_metrics,
				}
			)
		tqdm.write(
			"\n" + "-" * 72
			+ f"\nQuestion: {item.get('question', '').strip()}"
			+ f"\nLLM Answer: {final_prediction}"
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
			"question": (item.get("question") or "").strip(),
			"subquestions": step_reports,
			"final": {
				"question": (item.get("question") or "").strip(),
				"prediction": final_prediction,
				"ground_truth": final_ground_truth,
				"metrics": final_metrics,
			},
		}
		results.append(instance_report)

		# logger.info(
		# 	f"[MuSiQue] {instance_dir.name}: final_f1={final_metrics['f1']:.4f}, final_em={final_metrics['exact_match']:.4f}"
		# )

	summary = {
		"instances_root": str(root.resolve()),
		"dataset_path": str(Path(dataset_path).resolve()),
		"total_instances": len(results),
		"evaluated_instances": sum(1 for item in results if item.get("status") == "ok"),
		"overall_f1": fmean(all_f1_scores) if all_f1_scores else 0.0,
		"overall_exact_match": fmean(all_em_scores) if all_em_scores else 0.0,
		"overall_match": fmean(all_match_scores) if all_match_scores else 0.0,
		"overall_hit": fmean(all_hit_scores) if all_hit_scores else 0.0,
		"subquestion_and_final_count": len(all_f1_scores),
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
			"note": "These are final-step cases with hit=0. Review for potential answer_alias expansion.",
		},
		"cases": hit0_cases,
	}
	hit0_out_path.parent.mkdir(parents=True, exist_ok=True)
	hit0_out_path.write_text(json.dumps(hit0_payload, indent=2, ensure_ascii=False), encoding="utf-8")

	print("\n" + "=" * 72)
	print("MuSiQue multi-hop QA results")
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
	parser = argparse.ArgumentParser(description="Multi-hop MuSiQue QA with hyper-simulation slices")
	parser.add_argument("--instances-root", type=str, default=DEFAULT_INSTANCES_ROOT)
	parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--hit0-output-path", type=str, default="")
	parser.add_argument("--model-name", type=str, default="qwen3.5:9b")
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--limit-instances", type=int, default=0)
	args = parser.parse_args()

	run_musique_multihop_evaluation(
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
