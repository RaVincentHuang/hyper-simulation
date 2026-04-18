from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean
from typing import Any

from langchain_ollama import ChatOllama
from tqdm import tqdm

from hyper_simulation.baselines.contradoc import judge_contradiction_batch
from refine_hypergraph import load_task_dataset


DEFAULT_OUTPUT_PATH = "contradoc_time.json"
DEFAULT_DATASET_PATHS = {
	"musique": [
		"/home/vincent/.dataset/musique/sample1000/musique_answerable.jsonl",
		"/home/vincent/.dataset/musique/rest/musique_answerable.jsonl",
	],
	"hotpotqa": [
		"/home/vincent/.dataset/HotpotQA/sample1000",
	],
}
DEFAULT_TASKS = ["musique", "hotpotqa"]
DEFAULT_MODEL_NAME = "qwen3.5:9b"


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
	tmp_path.replace(path)


def _load_existing_runs(out_file: Path) -> list[dict[str, Any]]:
	if not out_file.exists():
		return []
	try:
		payload = json.loads(out_file.read_text(encoding="utf-8"))
	except Exception:
		return []
	runs = payload.get("runs", []) if isinstance(payload, dict) else []
	return runs if isinstance(runs, list) else []


def _normalize_question(text: str) -> str:
	return " ".join(str(text or "").split()).strip()


def _load_task_rows(task: str, dataset_paths: list[str], limit: int | None = None) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	seen_keys: set[str] = set()

	for one_path in dataset_paths:
		path = Path(one_path)
		if not path.exists():
			continue
		for item in load_task_dataset(task=task, dataset_path=str(path)):
			if not isinstance(item, dict):
				continue
			question = _normalize_question(item.get("question", ""))
			if not question:
				continue
			instance_id = str(item.get("id") or item.get("instance_id") or item.get("_id") or "").strip()
			key = instance_id if instance_id else question
			if key in seen_keys:
				continue
			seen_keys.add(key)
			rows.append(dict(item))
			if limit is not None and limit > 0 and len(rows) >= limit:
				return rows

	return rows


def _extract_musique_docs(item: dict[str, Any], max_docs: int = 20) -> list[str]:
	docs: list[str] = []

	paragraphs = item.get("paragraphs") or []
	if isinstance(paragraphs, list):
		for paragraph in paragraphs:
			if not isinstance(paragraph, dict):
				continue
			title = (paragraph.get("title") or "").strip()
			text = (paragraph.get("paragraph_text") or paragraph.get("text") or "").strip()
			if not text:
				continue
			docs.append(f"{title}\n{text}" if title else text)

	if docs:
		return docs[:max_docs]

	for key in ["context_docs", "ctxs", "context"]:
		value = item.get(key)
		if isinstance(value, list):
			for one in value:
				if isinstance(one, dict):
					title = (one.get("title") or "").strip()
					text = (one.get("text") or one.get("paragraph_text") or "").strip()
					if text:
						docs.append(f"{title}\n{text}" if title else text)
				else:
					text = str(one).strip()
					if text:
						docs.append(text)
		elif isinstance(value, str) and value.strip():
			docs.append(value.strip())

	return docs[:max_docs]


def _extract_hotpotqa_docs(item: dict[str, Any], max_docs: int = 20) -> list[str]:
	docs: list[str] = []
	context = item.get("context")

	if isinstance(context, dict):
		titles = context.get("title", [])
		sent_groups = context.get("sentences", [])
		if isinstance(titles, list):
			for idx, title in enumerate(titles):
				sentences = sent_groups[idx] if isinstance(sent_groups, list) and idx < len(sent_groups) else []
				if not isinstance(sentences, list):
					sentences = []
				text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip()).strip()
				if text:
					docs.append(f"{title}\n{text}" if str(title or "").strip() else text)
			return docs[:max_docs]

	if isinstance(context, list):
		for record in context:
			if not (isinstance(record, (list, tuple)) and len(record) >= 2):
				continue
			title, sentences = record[0], record[1]
			if not isinstance(sentences, list):
				sentences = []
			text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip()).strip()
			if text:
				docs.append(f"{title}\n{text}" if str(title or "").strip() else text)

	if docs:
		return docs[:max_docs]

	for key in ["context_docs", "paragraphs", "ctxs", "passage"]:
		value = item.get(key)
		if isinstance(value, list):
			for one in value:
				if isinstance(one, dict):
					title = (one.get("title") or "").strip()
					text = (one.get("text") or one.get("paragraph_text") or "").strip()
					if text:
						docs.append(f"{title}\n{text}" if title else text)
				else:
					text = str(one).strip()
					if text:
						docs.append(text)
		elif isinstance(value, str) and value.strip():
			docs.append(value.strip())

	return docs[:max_docs]


def _extract_docs(task: str, item: dict[str, Any]) -> list[str]:
	if task == "hotpotqa":
		return _extract_hotpotqa_docs(item)
	return _extract_musique_docs(item)


def _build_summary(
	task: str,
	dataset_paths: list[str],
	rows: list[dict[str, Any]],
	results: list[dict[str, Any]],
	timings: list[float],
	skipped: dict[str, int],
) -> dict[str, Any]:
	return {
		"task": task,
		"dataset_paths": dataset_paths,
		"row_count": len(rows),
		"instance_count": len(timings),
		"result_count": len(results),
		"total_time_cost": float(sum(timings)),
		"average_time_cost": float(fmean(timings)) if timings else 0.0,
		"skipped_missing_question": skipped.get("missing_question", 0),
		"skipped_missing_context": skipped.get("missing_context", 0),
		"skipped_empty": skipped.get("empty", 0),
	}


def _append_run(out_path: Path, payload: dict[str, Any]) -> None:
	runs_payload = {"runs": _load_existing_runs(out_path)}
	runs_payload["runs"].append(payload)
	_safe_write_json(out_path, runs_payload)


def _run_one_task(
	task: str,
	dataset_paths: list[str],
	output_path: Path,
	model_name: str,
	temperature: float,
	limit: int | None = None,
) -> dict[str, Any]:
	rows = _load_task_rows(task=task, dataset_paths=dataset_paths, limit=limit)
	model = ChatOllama(model=model_name, temperature=temperature, top_p=1, reasoning=False, num_predict=1024)

	results: list[dict[str, Any]] = []
	timings: list[float] = []
	skipped = {"missing_question": 0, "missing_context": 0, "empty": 0}

	for item in tqdm(rows, desc=f"contradoc/{task}", unit="q"):
		question = _normalize_question(item.get("question", ""))
		instance_id = str(item.get("id") or item.get("instance_id") or item.get("_id") or question).strip()

		if not question:
			skipped["missing_question"] += 1
			continue

		docs = _extract_docs(task, item)
		if not docs:
			skipped["missing_context"] += 1
			continue

		start_time = time.perf_counter()
		judgments = judge_contradiction_batch(doc_a_list=[question] * len(docs), doc_b_list=docs, model=model)
		elapsed = time.perf_counter() - start_time
		timings.append(elapsed)

		results.append(
			{
				"instance_id": instance_id,
				"question": question,
				"doc_count": len(docs),
				"timing": elapsed,
				"judgment_count": len(judgments),
			}
		)
		tqdm.write(f"Instance {instance_id}: contradoc time = {elapsed:.4f} seconds")

	summary = _build_summary(
		task=task,
		dataset_paths=dataset_paths,
		rows=rows,
		results=results,
		timings=timings,
		skipped=skipped,
	)
	payload = {"summary": summary, "results": results}
	_append_run(output_path, payload)

	print("\n" + "=" * 72)
	print(f"{task} contradoc timing finished")
	print("=" * 72)
	print(f"Instances executed: {summary['instance_count']}")
	print(f"Total time: {summary['total_time_cost']:.4f} seconds")
	print(f"Average time per instance: {summary['average_time_cost']:.4f} seconds")
	print(f"Saved to: {output_path.resolve()}")
	print("=" * 72)

	return payload


def run_contradoc_timing(
	task: str = "all",
	output_path: str = DEFAULT_OUTPUT_PATH,
	musique_dataset_paths: list[str] | None = None,
	hotpotqa_dataset_paths: list[str] | None = None,
	model_name: str = DEFAULT_MODEL_NAME,
	temperature: float = 0.1,
	limit: int | None = None,
) -> dict[str, Any]:
	selected_tasks = DEFAULT_TASKS if task == "all" else [task]
	out_path = Path(output_path)
	runs: list[dict[str, Any]] = []

	for one_task in selected_tasks:
		if one_task == "musique":
			dataset_paths = musique_dataset_paths or list(DEFAULT_DATASET_PATHS["musique"])
		elif one_task == "hotpotqa":
			dataset_paths = hotpotqa_dataset_paths or list(DEFAULT_DATASET_PATHS["hotpotqa"])
		else:
			raise ValueError(f"Unsupported task: {one_task}")

		runs.append(
			_run_one_task(
				task=one_task,
				dataset_paths=dataset_paths,
				output_path=out_path,
				model_name=model_name,
				temperature=temperature,
				limit=limit,
			)
		)

	if len(runs) == 1:
		return runs[0]

	instance_count = sum(run["summary"]["instance_count"] for run in runs)
	total_time_cost = sum(run["summary"]["total_time_cost"] for run in runs)
	return {
		"summary": {
			"task": "all",
			"instance_count": instance_count,
			"total_time_cost": total_time_cost,
			"average_time_cost": (total_time_cost / instance_count) if instance_count > 0 else 0.0,
		},
		"runs": runs,
	}


def _split_paths(value: str) -> list[str]:
	return [one.strip() for one in value.split(",") if one.strip()]


def main() -> None:
	parser = argparse.ArgumentParser(description="Contradoc timing for MuSiQue and HotpotQA")
	parser.add_argument("--task", type=str, default="all", choices=["musique", "hotpotqa", "all"])
	parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--musique-dataset-paths", type=str, default=",".join(DEFAULT_DATASET_PATHS["musique"]))
	parser.add_argument("--hotpotqa-dataset-paths", type=str, default=",".join(DEFAULT_DATASET_PATHS["hotpotqa"]))
	parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--limit", type=int, default=0)
	args = parser.parse_args()

	report = run_contradoc_timing(
		task=args.task,
		output_path=args.output_path,
		musique_dataset_paths=_split_paths(args.musique_dataset_paths),
		hotpotqa_dataset_paths=_split_paths(args.hotpotqa_dataset_paths),
		model_name=args.model_name,
		temperature=args.temperature,
		limit=(args.limit or None),
	)
	print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
