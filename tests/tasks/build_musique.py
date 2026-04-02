

from argparse import ArgumentParser
import json
from pathlib import Path

from tqdm import tqdm

from hyper_simulation.component.build_hypergraph import generate_instance_id, text_to_hypergraph
from hyper_simulation.question_answer.utils.load_data import load_data
from hyper_simulation.query_instance import build_query_instance_for_task


target_dir = "data/debug/musique/sample1000/"
dataset_path = "/home/vincent/.dataset/musique/sample1000/musique_answerable.jsonl"


def build_all_hypergraphs(
	dataset_path: str,
	target_dir: str,
	force_rebuild: bool = False,
	using_support_only: bool = False,
) -> dict:
	data = load_data(dataset_path, task="musique", use_supporting_only=using_support_only)
	out_root = Path(target_dir)
	out_root.mkdir(parents=True, exist_ok=True)

	built_count = 0
	skipped_count = 0
	failed: list[dict] = []

	for item in tqdm(data, desc="Building musique hypergraphs"):
		try:
			qi = build_query_instance_for_task(item, task="musique")
			question = (qi.query or "").strip()
			if not question:
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

			# 1) Build query hypergraph.
			query_hypergraph = text_to_hypergraph(question, is_query=True)
			query_hypergraph.save(str(query_path))

			# 2) Build all context hypergraphs for this question.
			data_files = []
			for idx, doc_text in enumerate(qi.data):
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
				"question": question,
				"num_data": len(qi.data),
				"saved_data_hypergraphs": len(data_files),
				"files": {
					"query": query_path.name,
					"data": data_files,
				},
			}
			metadata_path.write_text(
				json.dumps(metadata, indent=2, ensure_ascii=False),
				encoding="utf-8",
			)

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
	}

	(out_root / "summary.json").write_text(
		json.dumps(summary, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)
	if failed:
		(out_root / "failed.json").write_text(
			json.dumps(failed, indent=2, ensure_ascii=False),
			encoding="utf-8",
		)

	return summary


def main() -> None:
	parser = ArgumentParser(description="Build and store all MuSiQue question hypergraphs.")
	parser.add_argument("--dataset-path", type=str, default=dataset_path)
	parser.add_argument("--target-dir", type=str, default=target_dir)
	parser.add_argument("--force-rebuild", action="store_true")
	parser.add_argument("--using-support-only", action="store_true")
	args = parser.parse_args()

	summary = build_all_hypergraphs(
		dataset_path=args.dataset_path,
		target_dir=args.target_dir,
		force_rebuild=args.force_rebuild,
		using_support_only=args.using_support_only,
	)
	print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()

