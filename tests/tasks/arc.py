from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any

from langchain_ollama import ChatOllama
from tqdm import tqdm

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.postprocess import get_simulation_slice, ranking_slices
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.llm.chat_completion import get_invoke
from hyper_simulation.llm.prompt.arc import ARC_BASE, ARC_HYPER
from hyper_simulation.question_answer.vmdit.metrics import (
	exact_match_score,
	match,
	metric_max_over_ground_truths,
	qa_f1_score,
)

from refine_hypergraph import load_dataset_index


DEFAULT_INSTANCES_ROOT = "data/debug/arc/sample_challenge"
DEFAULT_DATASET_PATH = "/home/vincent/.dataset/ARC/sample_ARC"
DEFAULT_OUTPUT_PATH = "data/debug/arc/arc.json"


def _sorted_index_from_name(path: Path) -> int:
	match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
	if match_obj is not None:
		return int(match_obj.group(1))
	match_obj = re.fullmatch(r"data_(\d+)\.pkl", path.name)
	if match_obj is not None:
		return int(match_obj.group(1))
	return 10**9


def _find_query_file(instance_dir: Path) -> Path | None:
	for name in ("query_hypergraph.pkl", "query.pkl"):
		candidate = instance_dir / name
		if candidate.exists():
			return candidate
	return None


def _list_data_files(instance_dir: Path) -> list[Path]:
	all_files: dict[str, Path] = {}
	for pattern in ("data_hypergraph*.pkl", "data_*.pkl"):
		for path in instance_dir.glob(pattern):
			all_files[path.name] = path
	return sorted(all_files.values(), key=lambda path: (_sorted_index_from_name(path), path.name))


def _extract_arc_doc_entries(item: dict[str, Any]) -> list[dict[str, str]]:
	entries: list[dict[str, str]] = []
	context = item.get("context")
	if isinstance(context, list):
		for record in context:
			if isinstance(record, (list, tuple)) and len(record) >= 2:
				title = str(record[0] or "").strip()
				sentences = record[1]
				if not isinstance(sentences, list):
					sentences = []
				text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip()).strip()
				entries.append({"title": title, "text": text})
			elif isinstance(record, dict):
				title = str(record.get("title") or "").strip()
				text = str(record.get("text") or record.get("paragraph_text") or "").strip()
				if text:
					entries.append({"title": title, "text": text})

	paragraphs = item.get("paragraphs") or []
	if not entries and isinstance(paragraphs, list):
		for record in paragraphs:
			if not isinstance(record, dict):
				continue
			title = str(record.get("title") or "").strip()
			text = str(record.get("text") or record.get("paragraph_text") or "").strip()
			if text:
				entries.append({"title": title, "text": text})

	ctxs = item.get("ctxs") or []
	if not entries and isinstance(ctxs, list):
		for record in ctxs:
			if not isinstance(record, dict):
				continue
			title = str(record.get("title") or "").strip()
			text = str(record.get("text") or "").strip()
			if text:
				entries.append({"title": title, "text": text})

	context_docs = item.get("context_docs") or []
	if not entries and isinstance(context_docs, list):
		for record in context_docs:
			text = str(record or "").strip()
			if text:
				entries.append({"title": "", "text": text})

	context_text = str(item.get("context_text") or "").strip()
	if not entries and context_text:
		entries.append({"title": "", "text": context_text})

	return entries


def _load_instance_graphs(instance_dir: Path, item: dict[str, Any]) -> tuple[LocalHypergraph | None, list[dict[str, Any]]]:
	query_path = _find_query_file(instance_dir)
	if query_path is None:
		return None, []

	try:
		query_hg = LocalHypergraph.load(str(query_path))
	except Exception:
		return None, []

	doc_entries = _extract_arc_doc_entries(item)
	data_paths = _list_data_files(instance_dir)

	evidence_items: list[dict[str, Any]] = []
	for data_path in data_paths:
		data_idx = _sorted_index_from_name(data_path)
		doc_entry = doc_entries[data_idx] if data_idx < len(doc_entries) else {}
		title = str(doc_entry.get("title") or "").strip() if isinstance(doc_entry, dict) else ""
		text = str(doc_entry.get("text") or "").strip() if isinstance(doc_entry, dict) else ""

		try:
			data_hg = LocalHypergraph.load(str(data_path))
		except Exception:
			data_hg = None

		evidence_items.append(
			{
				"index": data_idx,
				"path": str(data_path),
				"hypergraph": data_hg,
				"title": title,
				"text": text,
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
	if cleaned.lower() in {"", "unanswerable", "unknown", "none", "not mentioned", "cannot be determined"}:
		return "unanswerable"
	return cleaned


def _normalize_arc_label(text: str, choice_labels: list[str], choice_texts: list[str]) -> str:
	value = (text or "").strip()
	if not value:
		return ""

	labels = [str(label).strip().upper() for label in choice_labels if str(label).strip()]
	if not labels:
		labels = ["A", "B", "C", "D"]

	v_upper = value.upper()
	if v_upper in labels:
		return v_upper

	match_obj = re.match(r"^([A-Z])[\)\.:\-\s].*", v_upper)
	if match_obj and match_obj.group(1) in labels:
		return match_obj.group(1)

	digit_match = re.match(r"^(\d+)$", value)
	if digit_match:
		idx = int(digit_match.group(1)) - 1
		if 0 <= idx < len(labels):
			return labels[idx]

	for idx, option_text in enumerate(choice_texts):
		normalized_option = re.sub(r"\s+", " ", str(option_text or "").strip()).lower()
		if normalized_option and normalized_option == re.sub(r"\s+", " ", value).lower():
			return labels[idx] if idx < len(labels) else ""

	inline_match = re.search(r"\b([A-D])\b", v_upper)
	if inline_match:
		cand = inline_match.group(1)
		if cand in labels:
			return cand

	return v_upper


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


def _build_slice_text(slice_index: int, evidence_item: dict[str, Any]) -> str:
	title = evidence_item.get("title") or f"evidence_{slice_index}"
	text = (evidence_item.get("text") or "").strip()

	lines = [f"Slice {slice_index}", f"Evidence: {title}"]
	if text:
		lines.append(f"Paragraph: {text}")
	return "\n".join(lines)


def _build_context_block(
	query: LocalHypergraph,
	simulation_slices: list[list[tuple[Vertex, Vertex]]],
	evidence_items: list[dict[str, Any]],
	vertex_ids: set[int],
	k: int = 10,
) -> tuple[list[int], list[int], str]:
	ranked_slice_indices = ranking_slices(query, simulation_slices, vertex_ids, k=k)

	consistent_indices: list[int] = []
	rendered_slices: list[str] = []
	for idx in ranked_slice_indices:
		if idx >= len(evidence_items):
			continue
		consistent_indices.append(int(evidence_items[idx].get("index", idx)))
		rendered_slices.append(_build_slice_text(idx, evidence_items[idx]))

	return consistent_indices, [], "\n\n".join(rendered_slices)


def _build_fallback_context(evidence_items: list[dict[str, Any]]) -> tuple[list[int], list[int], str]:
	consistent_indices: list[int] = []
	rendered_slices: list[str] = []
	for idx, evidence_item in enumerate(evidence_items):
		consistent_indices.append(int(evidence_item.get("index", idx)))
		rendered_slices.append(_build_slice_text(idx, evidence_item))
	return consistent_indices, [], "\n\n".join(rendered_slices)


def _extract_question_with_options(question: str, choice_labels: list[str], choice_texts: list[str]) -> str:
	base = (question or "").strip()
	if not base:
		return ""
	if "\n\nOptions:" in base:
		return base

	option_lines: list[str] = []
	for idx, text in enumerate(choice_texts):
		label = choice_labels[idx] if idx < len(choice_labels) else str(idx + 1)
		option_lines.append(f"{label}) {str(text).strip()}")

	if not option_lines:
		return base
	return f"{base}\n\nOptions:\n" + "\n".join(option_lines)


def _build_final_prompt(question: str, context_text: str) -> str:
	return ARC_BASE.format(context_text=context_text, question=question)


def _extract_choice_info(item: dict[str, Any]) -> tuple[list[str], list[str]]:
	choices = item.get("choices")
	if isinstance(choices, dict):
		labels = [str(label).strip() for label in (choices.get("label") or []) if str(label).strip()]
		texts = [str(text).strip() for text in (choices.get("text") or []) if str(text).strip()]
		return labels, texts

	option_labels = item.get("option_labels") or []
	options = item.get("options") or []
	labels = [str(label).strip() for label in option_labels if str(label).strip()]
	texts = [str(text).strip() for text in options if str(text).strip()]
	return labels, texts


def run_arc_multihop_evaluation(
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

	instance_dirs = sorted([path for path in root.iterdir() if path.is_dir() and _find_query_file(path) is not None])
	if limit_instances is not None and limit_instances > 0:
		instance_dirs = instance_dirs[:limit_instances]
	if not instance_dirs:
		raise FileNotFoundError(f"No valid instance directories found under: {root}")

	dataset_index = load_dataset_index(task="ARC", dataset_path=dataset_path)

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

	for instance_dir in tqdm(instance_dirs, desc="ARC multi-hop QA", unit="inst"):
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
		if query_hg is None:
			results.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": "missing_query_graph",
				}
			)
			continue

		question_text = (item.get("question") or "").strip()
		choice_labels, choice_texts = _extract_choice_info(item)
		question_with_options = _extract_question_with_options(question_text, choice_labels, choice_texts)

		valid_hgs = [entry["hypergraph"] for entry in evidence_items if entry.get("hypergraph") is not None]
		if valid_hgs:
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

			def allowed_vertex(vertex: Vertex) -> bool:
				return not (vertex.is_verb() or vertex.is_virtual())

			full_query_vertex_ids = {vertex.id for vertex in query_hg.vertices if allowed_vertex(vertex)}
			consistent_indices, inconsistent_indices, context_text = _build_context_block(
				query=query_hg,
				simulation_slices=simulation_slices,
				evidence_items=evidence_items,
				vertex_ids=full_query_vertex_ids,
			)
		else:
			consistent_indices, inconsistent_indices, context_text = _build_fallback_context(evidence_items)

		prompt = _build_final_prompt(question_with_options, context_text)
		raw_answer = get_invoke(model, prompt)
		prediction = _normalize_arc_label(_normalize_answer(raw_answer), choice_labels=choice_labels, choice_texts=choice_texts)

		answer_label = str(item.get("answer_label") or item.get("answerKey") or item.get("answer") or "").strip()
		final_ground_truth = [answer_label] if answer_label else []
		final_metrics = _evaluate_answer(prediction, final_ground_truth)

		all_f1_scores.append(final_metrics["f1"])
		all_em_scores.append(final_metrics["exact_match"])
		all_match_scores.append(final_metrics["match"])
		all_hit_scores.append(final_metrics["hit"])

		if final_metrics.get("hit", 0.0) == 0.0:
			hit0_cases.append(
				{
					"instance_id": instance_dir.name,
					"question": question_with_options,
					"prediction": prediction,
					"ground_truth": final_ground_truth,
					"metrics": final_metrics,
				}
			)

		tqdm.write(
			"\n"
			+ "-" * 72
			+ f"\nQuestion: {question_with_options}"
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
			"question": question_with_options,
			"final": {
				"question": question_with_options,
				"prediction": prediction,
				"ground_truth": final_ground_truth,
				"metrics": final_metrics,
				"consistent_context": consistent_indices,
				"inconsistent_context": inconsistent_indices,
				"context_count": len(evidence_items),
			},
		}
		results.append(instance_report)

	summary = {
		"instances_root": str(root.resolve()),
		"dataset_path": str(Path(dataset_path).resolve()),
		"total_instances": len(results),
		"evaluated_instances": sum(1 for item in results if item.get("status") == "ok"),
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
			"note": "These are final-step cases with hit=0. Review option-label normalization and context ranking.",
		},
		"cases": hit0_cases,
	}
	hit0_out_path.parent.mkdir(parents=True, exist_ok=True)
	hit0_out_path.write_text(json.dumps(hit0_payload, indent=2, ensure_ascii=False), encoding="utf-8")

	print("\n" + "=" * 72)
	print("ARC multi-hop QA results")
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
	parser = argparse.ArgumentParser(description="ARC QA with hyper-simulation slice ranking")
	parser.add_argument("--instances-root", type=str, default=DEFAULT_INSTANCES_ROOT)
	parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--hit0-output-path", type=str, default="")
	parser.add_argument("--model-name", type=str, default="qwen3.5:9b")
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--limit-instances", type=int, default=0)
	args = parser.parse_args()

	run_arc_multihop_evaluation(
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