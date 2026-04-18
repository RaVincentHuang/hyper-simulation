from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from statistics import fmean
from typing import Any

from tqdm import tqdm

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph
from hyper_simulation.hypergraph.union import MultiHopFusion

from refine_hypergraph import load_dataset_index


DEFAULT_INSTANCES_ROOT = "data/debug/control/sample80"
DEFAULT_DATASET_PATH = "data/nli/ConTRoL50.jsonl"
DEFAULT_OUTPUT_PATH = "control_time.json"


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
			}
		)

	return query_hg, evidence_items


def run_control_hyper_simulation_timing(
	instances_root: str = DEFAULT_INSTANCES_ROOT,
	dataset_path: str = DEFAULT_DATASET_PATH,
	output_path: str = DEFAULT_OUTPUT_PATH,
	limit_instances: int | None = None,
	sigma: float = 0.75,
	b: int = 5,
	delta: float = 0.7,
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

	dataset_index = load_dataset_index(task="control", dataset_path=dataset_path)

	time_costs: list[float] = []

	for instance_dir in tqdm(instance_dirs, desc="ConTRoL hyper-simulation timing", unit="inst"):
		item = dataset_index.get(instance_dir.name)
		if item is None:
			continue

		query_hg, evidence_items = _load_instance_graphs(instance_dir, item)
		if query_hg is None or not evidence_items:
			continue

		valid_hgs = [entry["hypergraph"] for entry in evidence_items if entry.get("hypergraph") is not None]
		if not valid_hgs:
			continue

		fusion = MultiHopFusion()
		merged_hg, _provenance = fusion.merge_hypergraphs(valid_hgs)

		start_time = time.time()
		compute_hyper_simulation(
			query_hg,
			merged_hg,
			sigma_threshold=sigma,
			b_threshold=b,
			delta_threshold=delta,
		)
		end_time = time.time()

		simulation_time = end_time - start_time
		time_costs.append(simulation_time)
		tqdm.write(f"Instance {instance_dir.name}: simulation time = {simulation_time:.2f} seconds")

	total_time_cost = sum(time_costs)
	average_time_cost = fmean(time_costs) if time_costs else 0
	print(f"\nTotal simulation time for {len(time_costs)} instances: {total_time_cost:.2f} seconds")
	print(f"Average simulation time per instance: {average_time_cost:.2f} seconds")

	summary = {
		"instances_root": str(root.resolve()),
		"dataset_path": str(Path(dataset_path).resolve()),
		"instance_count": len(time_costs),
		"total_time_cost": total_time_cost,
		"average_time_cost": average_time_cost,
		"sigma": sigma,
		"delta": delta,
		"b": b,
	}

	out_path = Path(output_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	runs_payload: dict[str, Any] = {"runs": []}
	if out_path.exists():
		try:
			existing_payload = json.loads(out_path.read_text(encoding="utf-8"))
			if isinstance(existing_payload, dict):
				runs = existing_payload.get("runs")
				if isinstance(runs, list):
					runs_payload["runs"] = runs
		except Exception:
			# If existing output is invalid JSON, start a new run history.
			pass

	runs_payload["runs"].append(summary)
	out_path.write_text(json.dumps(runs_payload, indent=2, ensure_ascii=False), encoding="utf-8")

	print("\n" + "=" * 72)

	return {"summary": summary}


def main() -> None:
	parser = argparse.ArgumentParser(description="ConTRoL hyper-simulation timing only")
	parser.add_argument("--instances-root", type=str, default=DEFAULT_INSTANCES_ROOT)
	parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--limit-instances", type=int, default=0)
	parser.add_argument("--sigma", type=float, default=0.75)
	parser.add_argument("--delta", type=float, default=0.7)
	parser.add_argument("--b", type=int, default=5)
	args = parser.parse_args()

	run_control_hyper_simulation_timing(
		instances_root=args.instances_root,
		dataset_path=args.dataset_path,
		output_path=args.output_path,
		limit_instances=args.limit_instances or None,
		sigma=args.sigma,
		delta=args.delta,
		b=args.b,
	)


if __name__ == "__main__":
	main()
