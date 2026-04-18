from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean
from typing import Any

from tqdm import tqdm

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph
from hyper_simulation.hypergraph.union import MultiHopFusion


DEFAULT_SPLIT_ROOT = "data/debug/split/contra_nli"
DEFAULT_HYPERGRAPHS_ROOT = "data/debug/split/contra_nli/hypergraphs"
DEFAULT_OUTPUT_PATH = "contract_nli_split_time.json"
DEFAULT_BUCKETS = ["1000", "2000", "3000", "4000", "5000", "overflow"]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _sorted_index_from_name(path: Path) -> int:
    name = path.name
    if not name.startswith("data_hypergraph") or not name.endswith(".pkl"):
        return 10**9
    middle = name[len("data_hypergraph") : -len(".pkl")]
    return int(middle) if middle.isdigit() else 10**9


def _load_instance_graphs(instance_dir: Path) -> tuple[LocalHypergraph | None, list[LocalHypergraph]]:
    query_path = instance_dir / "query_hypergraph.pkl"
    if not query_path.exists():
        return None, []

    try:
        query_hg = LocalHypergraph.load(str(query_path))
    except Exception:
        return None, []

    data_hgs: list[LocalHypergraph] = []
    data_paths = sorted(instance_dir.glob("data_hypergraph*.pkl"), key=_sorted_index_from_name)
    for data_path in data_paths:
        try:
            data_hgs.append(LocalHypergraph.load(str(data_path)))
        except Exception:
            continue

    return query_hg, data_hgs


def _evaluate_bucket(
    bucket_name: str,
    rows: list[dict[str, Any]],
    hypergraphs_root: Path,
    sigma: float,
    delta: float,
    b: int,
) -> dict[str, Any]:
    fusion = MultiHopFusion()

    timings: list[float] = []
    skipped_missing_dir = 0
    skipped_missing_query = 0
    skipped_no_data = 0

    bucket_hg_dir = hypergraphs_root / bucket_name

    for row in tqdm(rows, desc=f"Bucket {bucket_name}", unit="inst"):
        instance_id = str(row.get("instance_id") or "").strip()
        if not instance_id:
            skipped_missing_dir += 1
            continue

        instance_dir = bucket_hg_dir / instance_id
        if not instance_dir.exists():
            skipped_missing_dir += 1
            continue

        query_hg, data_hgs = _load_instance_graphs(instance_dir)
        if query_hg is None:
            skipped_missing_query += 1
            continue
        if not data_hgs:
            skipped_no_data += 1
            continue

        merged_hg, _ = fusion.merge_hypergraphs(data_hgs)

        start = time.time()
        compute_hyper_simulation(
            query_hg,
            merged_hg,
            sigma_threshold=sigma,
            b_threshold=b,
            delta_threshold=delta,
        )
        timings.append(time.time() - start)

    executed_count = len(timings)
    total_time_cost = float(sum(timings))
    average_time_cost = float(fmean(timings)) if timings else 0.0

    return {
        "bucket": bucket_name,
        "requested_count": len(rows),
        "executed_count": executed_count,
        "total_time_cost": total_time_cost,
        "average_time_cost": average_time_cost,
        "skipped_missing_dir": skipped_missing_dir,
        "skipped_missing_query": skipped_missing_query,
        "skipped_no_data": skipped_no_data,
    }


def run_contract_nli_split_timing(
    split_root: str = DEFAULT_SPLIT_ROOT,
    hypergraphs_root: str = DEFAULT_HYPERGRAPHS_ROOT,
    output_path: str = DEFAULT_OUTPUT_PATH,
    buckets: list[str] | None = None,
    limit_per_bucket: int | None = None,
    sigma: float = 0.75,
    delta: float = 0.7,
    b: int = 5,
) -> dict[str, Any]:
    split_dir = Path(split_root)
    hg_dir = Path(hypergraphs_root)

    if not split_dir.exists():
        raise FileNotFoundError(f"Split root not found: {split_dir}")
    if not hg_dir.exists():
        raise FileNotFoundError(f"Hypergraphs root not found: {hg_dir}")

    selected_buckets = buckets or list(DEFAULT_BUCKETS)
    bucket_summaries: list[dict[str, Any]] = []

    for bucket in selected_buckets:
        jsonl_path = split_dir / f"{bucket}.jsonl"
        rows = _load_jsonl(jsonl_path)
        if limit_per_bucket is not None and limit_per_bucket > 0:
            rows = rows[:limit_per_bucket]

        summary = _evaluate_bucket(
            bucket_name=bucket,
            rows=rows,
            hypergraphs_root=hg_dir,
            sigma=sigma,
            delta=delta,
            b=b,
        )
        summary["jsonl_path"] = str(jsonl_path.resolve()) if jsonl_path.exists() else str(jsonl_path)
        bucket_summaries.append(summary)

    valid_avgs = [item["average_time_cost"] for item in bucket_summaries if item["executed_count"] > 0]
    total_executed = sum(item["executed_count"] for item in bucket_summaries)
    total_time = sum(item["total_time_cost"] for item in bucket_summaries)

    run_summary = {
        "split_root": str(split_dir.resolve()),
        "hypergraphs_root": str(hg_dir.resolve()),
        "sigma": sigma,
        "delta": delta,
        "b": b,
        "limit_per_bucket": limit_per_bucket,
        "bucket_results": bucket_summaries,
        "overall": {
            "bucket_count": len(bucket_summaries),
            "executed_count": total_executed,
            "total_time_cost": total_time,
            "average_time_cost": float(fmean(valid_avgs)) if valid_avgs else 0.0,
            "weighted_average_time_cost": (total_time / total_executed) if total_executed > 0 else 0.0,
        },
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {"runs": []}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                payload["runs"] = existing["runs"]
        except Exception:
            pass

    payload["runs"].append(run_summary)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Contract NLI split timing finished")
    print("=" * 72)
    for item in bucket_summaries:
        print(
            f"{item['bucket']:>8}: executed={item['executed_count']:>3} "
            f"avg={item['average_time_cost']:.4f}s total={item['total_time_cost']:.2f}s"
        )
    print("-" * 72)
    print(
        "Overall: "
        f"executed={run_summary['overall']['executed_count']} "
        f"weighted_avg={run_summary['overall']['weighted_average_time_cost']:.4f}s "
        f"total={run_summary['overall']['total_time_cost']:.2f}s"
    )
    print(f"Saved to: {out_path.resolve()}")
    print("=" * 72 + "\n")

    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch timing for Contract NLI split buckets")
    parser.add_argument("--split-root", type=str, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--hypergraphs-root", type=str, default=DEFAULT_HYPERGRAPHS_ROOT)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--buckets", nargs="+", default=DEFAULT_BUCKETS)
    parser.add_argument("--limit-per-bucket", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.75)
    parser.add_argument("--delta", type=float, default=0.7)
    parser.add_argument("--b", type=int, default=5)
    args = parser.parse_args()

    run_contract_nli_split_timing(
        split_root=args.split_root,
        hypergraphs_root=args.hypergraphs_root,
        output_path=args.output_path,
        buckets=args.buckets,
        limit_per_bucket=args.limit_per_bucket or None,
        sigma=args.sigma,
        delta=args.delta,
        b=args.b,
    )


if __name__ == "__main__":
    main()
