from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = ROOT / "tests" / "tasks"
SRC_DIR = ROOT / "src"

for candidate in (str(SRC_DIR), str(TASKS_DIR)):
	if candidate not in sys.path:
		sys.path.insert(0, candidate)

from tqdm import tqdm

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.postprocess import get_simulation_slice, ranking_slices
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.utils.log import current_query_id
from refine_hypergraph import load_dataset_index


def format_vertex(vertex: Vertex) -> str:
    nodes = "\n".join(
        f"    - '{node.text}' (pos={node.pos.name}, dep={node.dep.name}, ent={node.ent.name}, ENT={node.entity.name if node.entity else 'None'})"
        for node in vertex.nodes
    )
    return f"[{vertex.id}] '{vertex.text()}'\n{nodes}"

DEFAULT_INSTANCES_ROOT = "data/debug/musique/sample1000"
DEFAULT_DATASET_PATH = "/home/vincent/.dataset/musique/sample1000/musique_answerable.jsonl"


def _sorted_index_from_name(path: Path) -> int:
	match = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
	if match is None:
		return 10**9
	return int(match.group(1))


def _coerce_int(value: Any) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except (TypeError, ValueError):
		return None


def _extract_subquestion_steps(item: dict[str, Any]) -> list[dict[str, Any]]:
	decomposition = item.get("question_decomposition", []) or []
	if not isinstance(decomposition, list):
		return []

	if decomposition and all(isinstance(step, dict) and "id" in step for step in decomposition):
		decomposition = sorted(decomposition, key=lambda step: step.get("id"))

	steps: list[dict[str, Any]] = []
	for idx, step in enumerate(decomposition, start=1):
		if not isinstance(step, dict):
			continue
		question = (step.get("question") or "").strip()
		if not question:
			continue
		steps.append(
			{
				"index": idx,
				"question": question,
				"support_id": _coerce_int(step.get("paragraph_support_idx")),
			}
		)
	return steps


def _load_decomposed_vertex_ids(instance_dir: Path) -> list[set[int]]:
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

	pairs: list[tuple[int, set[int]]] = []
	for i, record in enumerate(records, start=1):
		if not isinstance(record, dict):
			continue
		order = _coerce_int(record.get("index")) or i
		raw_ids = record.get("vertex_ids")
		if not isinstance(raw_ids, list):
			pairs.append((order, set()))
			continue

		ids: set[int] = set()
		for one in raw_ids:
			coerced = _coerce_int(one)
			if coerced is not None:
				ids.add(coerced)
		pairs.append((order, ids))

	pairs.sort(key=lambda item: item[0])
	return [vertex_ids for _, vertex_ids in pairs]


def _build_subquestion_specs(query_hg: LocalHypergraph, instance_dir: Path, item: dict[str, Any]) -> list[dict[str, Any]]:
	steps = _extract_subquestion_steps(item)
	if not steps:
		return []

	all_vertex_ids = {vertex.id for vertex in query_hg.vertices}
	vertex_ids_from_decompose = _load_decomposed_vertex_ids(instance_dir)

	specs: list[dict[str, Any]] = []
	for i, step in enumerate(steps):
		if i < len(vertex_ids_from_decompose) and vertex_ids_from_decompose[i]:
			vertex_ids = set(vertex_ids_from_decompose[i])
		else:
			vertex_ids = set(all_vertex_ids)
		specs.append(
			{
				"index": int(step["index"]),
				"question": str(step["question"]),
				"support_id": step.get("support_id"),
				"vertex_ids": vertex_ids,
			}
		)

	return specs


def _load_instance_graphs(instance_dir: Path, item: dict[str, Any], max_data_graphs: int | None = None) -> tuple[LocalHypergraph | None, list[dict[str, Any]]]:
    query_path = instance_dir / "query_hypergraph.pkl"
    if not query_path.exists():
        return None, []

    query_hg = LocalHypergraph.load(str(query_path))
    paragraphs = item.get("paragraphs", []) or []
    data_paths = sorted(instance_dir.glob("data_hypergraph*.pkl"), key=_sorted_index_from_name)
    if max_data_graphs is not None:
        data_paths = data_paths[:max_data_graphs]

    evidence_items: list[dict[str, Any]] = []
    for data_path in data_paths:
        match = re.fullmatch(r"data_hypergraph(\d+)\.pkl", data_path.name)
        if match is None:
            continue
        data_idx = int(match.group(1))
        if data_idx >= len(paragraphs):
            continue
        paragraph = paragraphs[data_idx]
        if not isinstance(paragraph, dict):
            continue
        paragraph_text = (paragraph.get("paragraph_text") or "").strip()
        if not paragraph_text:
            continue
        try:
            data_hg = LocalHypergraph.load(str(data_path))
        except Exception:
            continue
        evidence_items.append(
            {
                "index": data_idx,
                "path": str(data_path),
                "hypergraph": data_hg,
                "text": paragraph_text,
            }
        )

    return query_hg, evidence_items


def _get_consistent_context_set(
    query_hg: LocalHypergraph,
    slices: list[list[tuple[Vertex, Vertex]]],
    evidence_items: list[dict[str, Any]],
    vertex_ids: set[int],
) -> set[int]:
    ranked_context_indices, consistent_context_ids = _get_ranked_and_consistent_contexts(
        query_hg=query_hg,
        slices=slices,
        evidence_items=evidence_items,
        vertex_ids=vertex_ids,
    )
    _ = ranked_context_indices
    return consistent_context_ids


def _get_ranked_and_consistent_contexts(
    query_hg: LocalHypergraph,
    slices: list[list[tuple[Vertex, Vertex]]],
    evidence_items: list[dict[str, Any]],
    vertex_ids: set[int],
) -> tuple[list[int], set[int], list[list[int]]]:
    ranked_slice_indices = ranking_slices(
        query=query_hg,
        simulation_slices=slices,
        vertex_ids=vertex_ids,
        k=20,
    )

    vertex_needs: set[Vertex] = {u for u in query_hg.vertices if u.id in vertex_ids}
    slice_hit_cnt: dict[int, int] = {}
    for idx, simulation_slice in enumerate(slices):
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

    ranked_context_indices: list[int] = []
    consistent_context_ids: set[int] = set()
    for slice_idx in ranked_slice_indices:
        if slice_idx >= len(evidence_items):
            continue
        context_idx = int(evidence_items[slice_idx]["index"])
        ranked_context_indices.append(context_idx)
        consistent_context_ids.add(context_idx)

    tie_groups: list[list[int]] = []
    for group in tie_groups_by_ranked_slices:
        context_group: list[int] = []
        for slice_idx in group:
            if slice_idx >= len(evidence_items):
                continue
            context_group.append(int(evidence_items[slice_idx]["index"]))
        if context_group:
            tie_groups.append(context_group)

    return ranked_context_indices, consistent_context_ids, tie_groups


def evaluate_support_batch(
	instances_root: str = DEFAULT_INSTANCES_ROOT,
	dataset_path: str = DEFAULT_DATASET_PATH,
	limit_instances: int | None = None,
	max_data_graphs: int | None = None,
	show_live_score: bool = True,
) -> dict[str, Any]:
    root = Path(instances_root)
    if not root.exists():
        raise FileNotFoundError(f"Instances root not found: {root}")

    instance_dirs = sorted([path for path in root.iterdir() if path.is_dir() and (path / "query_hypergraph.pkl").exists()])
    if limit_instances is not None and limit_instances > 0:
        instance_dirs = instance_dirs[:limit_instances]
    if not instance_dirs:
        raise FileNotFoundError(f"No valid instance directories found under: {root}")

    target_ids = {path.name for path in instance_dirs}
    dataset_index = load_dataset_index(dataset_path=dataset_path, target_ids=target_ids)

    results: list[dict[str, Any]] = []
    running_ok = 0
    running_skipped = 0

    subq_total = 0
    subq_with_support = 0
    subq_support_matched = 0
    extra_sizes_with_support: list[float] = []
    support_ranks_with_support: list[float] = []
    support_rank_miss_count = 0
    support_ranks_adjusted_with_support: list[float] = []
    support_rank_adjusted_miss_count = 0
    support_rank_counter: Counter[int] = Counter()
    support_rank_adjusted_counter: Counter[int] = Counter()

    pbar = tqdm(instance_dirs, desc="Support check")
    for instance_dir in pbar:
        item = dataset_index.get(instance_dir.name)
        if item is None:
            running_skipped += 1
            results.append(
                {
                    "instance_id": instance_dir.name,
                    "status": "skipped",
                    "reason": "dataset_item_not_found",
                }
            )
            pbar.set_postfix(ok=running_ok, skip=running_skipped)
            continue

        query_hg, evidence_items = _load_instance_graphs(instance_dir, item, max_data_graphs=max_data_graphs)
        if query_hg is None or not evidence_items:
            running_skipped += 1
            results.append(
                {
                    "instance_id": instance_dir.name,
                    "status": "skipped",
                    "reason": "missing_graphs",
                }
            )
            pbar.set_postfix(ok=running_ok, skip=running_skipped)
            continue

        subquestion_specs = _build_subquestion_specs(query_hg=query_hg, instance_dir=instance_dir, item=item)
        if not subquestion_specs:
            running_skipped += 1
            results.append(
                {
                    "instance_id": instance_dir.name,
                    "status": "skipped",
                    "reason": "missing_subquestions",
                }
            )
            pbar.set_postfix(ok=running_ok, skip=running_skipped)
            continue

        current_query_id.set(instance_dir.name)
        valid_hgs = [entry["hypergraph"] for entry in evidence_items]

        fusion = MultiHopFusion()
        merged_hg, _provenance = fusion.merge_hypergraphs(valid_hgs)
        mapping, q_map, d_map = compute_hyper_simulation(query_hg, merged_hg)
        simulation = [
            (q_map[q_id], d_map[d_id])
            for q_id, d_ids in mapping.items()
            for d_id in d_ids
            if q_id in q_map and d_id in d_map
        ]
        slices = get_simulation_slice(query_hg, merged_hg, simulation, len(valid_hgs))

        per_subquestion: list[dict[str, Any]] = []
        for spec in subquestion_specs:
            support_id = spec.get("support_id")
            vertex_ids = set(spec["vertex_ids"])
            sub_question = spec["question"]
            print(f"Checking subq '{sub_question}'")
            for vertex in query_hg.vertices:
                if vertex.id in vertex_ids:
                    print(f"- [{vertex.id}] {vertex.text()}")
            ranked_context_indices, consistent_context_set, tie_groups = _get_ranked_and_consistent_contexts(
                query_hg=query_hg,
                slices=slices,
                evidence_items=evidence_items,
                vertex_ids=vertex_ids,
            )

            if support_id is None:
                match_support_id: bool | None = None
                extra_context_size: float | None = None
                support_rank: int | None = None
                support_rank_adjusted: int | None = None
            else:
                tqdm.write(f"Checking subq {spec['index']} with support_id={support_id} against {consistent_context_set}")
                match_support_id = int(support_id) in consistent_context_set
                extra_context_size = float(len(consistent_context_set - {int(support_id)}))
                support_id_int = int(support_id)
                try:
                    support_rank = ranked_context_indices.index(support_id_int) + 1
                    support_ranks_with_support.append(float(support_rank))
                    support_rank_counter[support_rank] += 1
                except ValueError:
                    support_rank = None
                    support_rank_miss_count += 1

                if support_rank is None:
                    support_rank_adjusted = None
                    support_rank_adjusted_miss_count += 1
                else:
                    adjusted_rank = support_rank
                    offset = 0
                    for group in tie_groups:
                        if support_id_int in group:
                            adjusted_rank = offset + 1
                            break
                        offset += len(group)
                    support_rank_adjusted = adjusted_rank
                    support_ranks_adjusted_with_support.append(float(support_rank_adjusted))
                    support_rank_adjusted_counter[support_rank_adjusted] += 1

                tqdm.write(
                    f"subq {spec['index']} rank original={support_rank}, adjusted={support_rank_adjusted}, ties={tie_groups}"
                )
                subq_with_support += 1
                if match_support_id:
                    subq_support_matched += 1
                extra_sizes_with_support.append(extra_context_size)

            subq_total += 1

            per_subquestion.append(
                {
                    "index": spec["index"],
                    "question": spec["question"],
                    "support_id": support_id,
                    "vertex_ids": sorted(vertex_ids),
                    "ranked_context_indices": ranked_context_indices,
                    "rank_tie_groups": tie_groups,
                    "consistent_context_set": sorted(consistent_context_set),
                    "match_support_id": match_support_id,
                    "support_rank": support_rank,
                    "support_rank_adjusted": support_rank_adjusted,
                    "extra_context_size": extra_context_size,
                }
            )

        running_ok += 1
        match_rate = (subq_support_matched / subq_with_support) if subq_with_support else 0.0
        avg_extra = fmean(extra_sizes_with_support) if extra_sizes_with_support else 0.0
        pbar.set_postfix(ok=running_ok, skip=running_skipped, match=f"{match_rate:.3f}", extra=f"{avg_extra:.3f}")

        if show_live_score:
            hit = sum(1 for one in per_subquestion if one["match_support_id"] is True)
            valid = sum(1 for one in per_subquestion if one["match_support_id"] is not None)
            extras = [float(one["extra_context_size"]) for one in per_subquestion if one["extra_context_size"] is not None]
            mean_extra = fmean(extras) if extras else 0.0
            tqdm.write(
                (
                    f"[{instance_dir.name}] "
                    f"subq={len(per_subquestion)}, "
                    f"matched={hit}/{valid}, "
                    f"avg_extra={mean_extra:.3f}, "
                    f"running_match={match_rate:.3f}, "
                    f"running_avg_extra={avg_extra:.3f}"
                )
            )

        results.append(
            {
                "instance_id": instance_dir.name,
                "status": "ok",
                "question": (item.get("question") or "").strip(),
                "subquestions": per_subquestion,
            }
        )

    summary = {
        "instances_root": str(root.resolve()),
        "dataset_path": str(Path(dataset_path).resolve()),
        "processed": running_ok,
        "skipped": running_skipped,
        "subquestion_total": subq_total,
        "subquestion_with_support_id": subq_with_support,
        "support_id_match_count": subq_support_matched,
        "support_id_match_rate": (subq_support_matched / subq_with_support) if subq_with_support else 0.0,
        "support_id_rank_count": len(support_ranks_with_support),
        "support_id_avg_rank": fmean(support_ranks_with_support) if support_ranks_with_support else None,
        "support_id_rank_miss_count": support_rank_miss_count,
        "support_id_adjusted_rank_count": len(support_ranks_adjusted_with_support),
        "support_id_adjusted_avg_rank": fmean(support_ranks_adjusted_with_support) if support_ranks_adjusted_with_support else None,
        "support_id_adjusted_rank_miss_count": support_rank_adjusted_miss_count,
        "support_id_rank_distribution": {str(rank): cnt for rank, cnt in sorted(support_rank_counter.items())},
        "support_id_adjusted_rank_distribution": {
            str(rank): cnt for rank, cnt in sorted(support_rank_adjusted_counter.items())
        },
        "avg_context_minus_support_size": fmean(extra_sizes_with_support) if extra_sizes_with_support else 0.0,
    }

    return {
        "summary": summary,
        "results": results,
    }


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print("Batch statistics:")
    print(f"  processed: {summary['processed']}")
    print(f"  skipped: {summary['skipped']}")
    print(f"  subquestion_total: {summary['subquestion_total']}")
    print(f"  subquestion_with_support_id: {summary['subquestion_with_support_id']}")
    print(f"  support_id_match_count: {summary['support_id_match_count']}")
    print(f"  support_id_match_rate: {summary['support_id_match_rate']:.4f}")
    print(f"  support_id_rank_count: {summary['support_id_rank_count']}")
    print(f"  support_id_avg_rank: {summary['support_id_avg_rank']}")
    print(f"  support_id_rank_miss_count: {summary['support_id_rank_miss_count']}")
    print(f"  support_id_adjusted_rank_count: {summary['support_id_adjusted_rank_count']}")
    print(f"  support_id_adjusted_avg_rank: {summary['support_id_adjusted_avg_rank']}")
    print(f"  support_id_adjusted_rank_miss_count: {summary['support_id_adjusted_rank_miss_count']}")
    print(f"  support_id_rank_distribution: {summary['support_id_rank_distribution']}")
    print(f"  support_id_adjusted_rank_distribution: {summary['support_id_adjusted_rank_distribution']}")
    print(f"  avg_context_minus_support_size: {summary['avg_context_minus_support_size']:.4f}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Support coverage check for MuSiQue hypergraph matches.")
	parser.add_argument("--instances-root", type=str, default=DEFAULT_INSTANCES_ROOT)
	parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--limit-instances", type=int, default=0)
	parser.add_argument("--max-data-graphs", type=int, default=0)
	parser.add_argument("--output-path", type=str, default="")
	parser.add_argument("--disable-live-score", action="store_true")
	args = parser.parse_args()

	report = evaluate_support_batch(
		instances_root=args.instances_root,
		dataset_path=args.dataset_path,
		limit_instances=args.limit_instances or None,
		max_data_graphs=args.max_data_graphs or None,
		show_live_score=not args.disable_live_score,
	)

	if args.output_path:
		output_path = Path(args.output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

	_print_report(report)


if __name__ == "__main__":
	main()
