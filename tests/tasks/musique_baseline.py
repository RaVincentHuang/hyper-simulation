from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any

from langchain_ollama import ChatOllama
from tqdm import tqdm

from hyper_simulation.llm.chat_completion import get_invoke
from hyper_simulation.llm.prompt.musique import MUSIQUE_QA_BASE
from hyper_simulation.question_answer.vmdit.metrics import (
	exact_match_score,
	match,
	metric_max_over_ground_truths,
	qa_f1_score,
)
from hyper_simulation.baselines.CDIT import judge_similarity_batch
from hyper_simulation.baselines.contradoc import judge_contradiction_batch
from refine_hypergraph import load_dataset_index


DEFAULT_PROMPTS_DIR = "data/musique/prompts"
DEFAULT_OUTPUT_DIR = "data/baseline/musique"
DEFAULT_METHODS = ["bsim", "her", "sentli", "sparsecl", "cdit", "contradoc", "vanilla"]
DEFAULT_DATASET_PATHS = [
	"/home/vincent/.dataset/musique/sample1000/musique_answerable.jsonl",
	"/home/vincent/.dataset/musique/rest/musique_answerable.jsonl",
]
CONSISTENT_TAG = "[consistent]"
INCONSISTENT_TAG = "[inconsisitent]"


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
	tmp_path.replace(path)


def _row_key(instance_id: Any, question: str) -> str:
	inst = str(instance_id).strip() if instance_id is not None else ""
	q = str(question or "").strip()
	return inst if inst else q


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
		key = _row_key(one.get("instance_id"), str(one.get("question") or ""))
		if key:
			res[key] = one
	return res


def _load_context_cache_map(cache_file: Path) -> dict[str, dict[str, Any]]:
	if not cache_file.exists():
		return {}
	try:
		payload = json.loads(cache_file.read_text(encoding="utf-8"))
	except Exception:
		return {}

	rows = payload.get("rows", []) if isinstance(payload, dict) else []
	if not isinstance(rows, list):
		return {}

	cache: dict[str, dict[str, Any]] = {}
	for one in rows:
		if not isinstance(one, dict):
			continue
		key = _row_key(one.get("instance_id"), str(one.get("question") or ""))
		if key:
			cache[key] = one
	return cache


def _extract_context_text(raw_prompt: str) -> str:
	if not raw_prompt:
		return ""

	match_obj = re.search(r"###\s*Context:\s*\n(.*?)\n###\s*Question:\s*\n", raw_prompt, re.DOTALL | re.IGNORECASE)
	if match_obj:
		return match_obj.group(1).strip()

	# 兜底：无法严格匹配时，尽量截取到 Question 前。
	question_anchor = re.search(r"\n###\s*Question:\s*\n", raw_prompt, re.IGNORECASE)
	if question_anchor:
		return raw_prompt[: question_anchor.start()].strip()

	return raw_prompt.strip()


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


def _resolve_placeholders(question: str, answer_history: list[str]) -> str:
	def replace(match_obj: re.Match[str]) -> str:
		step_idx = int(match_obj.group(1)) - 1
		if 0 <= step_idx < len(answer_history):
			answer = (answer_history[step_idx] or "").strip()
			return answer if answer else match_obj.group(0)
		return match_obj.group(0)

	return re.sub(r"#(\d+)", replace, question)


def _build_subquestion_prompt(question: str, context_text: str, history_text: str) -> str:
	prompt = MUSIQUE_QA_BASE.format(context_text=context_text, question=question)
	return (
		prompt
		+ "\n\n### Additional instructions:\n"
		+ "This is one step in a multi-hop decomposition.\n"
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


def _evaluate_answer(prediction: str, reference_answer: list[str] | str) -> dict[str, float]:
	if isinstance(reference_answer, list):
		ground_truths = [gt for gt in reference_answer if isinstance(gt, str) and gt.strip()]
	else:
		ground_truths = [reference_answer] if isinstance(reference_answer, str) and reference_answer.strip() else []

	if not ground_truths:
		return {"exact_match": 0.0, "f1": 0.0, "match": 0.0, "hit": 0.0}

	em_score = float(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
	f1_score = max(qa_f1_score(prediction, gt) for gt in ground_truths)
	match_score = float(match(prediction, ground_truths))
	hit = em_score > 0.0 or f1_score > 0.0 or match_score > 0.0
	return {
		"exact_match": em_score,
		"f1": f1_score,
		"match": match_score,
		"hit": float(hit),
	}


def _load_jsonl(file_path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	if not file_path.exists():
		return rows

	with file_path.open("r", encoding="utf-8") as fp:
		for line in fp:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue
			if isinstance(obj, dict):
				rows.append(obj)
	return rows


def _load_dataset_items(dataset_paths: list[str], limit: int | None = None) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	seen_keys: set[str] = set()

	for one_path in dataset_paths:
		path = Path(one_path)
		if not path.exists():
			continue
		for item in _load_jsonl(path):
			question = str(item.get("question", "")).strip()
			if not question:
				continue
			qid = str(item.get("id") or item.get("instance_id") or question)
			key = qid if qid else question
			if key in seen_keys:
				continue
			seen_keys.add(key)
			rows.append(item)
			if limit is not None and limit > 0 and len(rows) >= limit:
				return rows

	return rows


def _load_dataset_index_multi(dataset_paths: list[str], target_ids: set[str]) -> dict[str, dict[str, Any]]:
	merged: dict[str, dict[str, Any]] = {}
	for path in dataset_paths:
		p = Path(path)
		if not p.exists():
			continue
		one = load_dataset_index(dataset_path=str(p), target_ids=target_ids)
		for key, value in one.items():
			if key not in merged:
				merged[key] = value
	return merged


def _extract_subquestions_from_item(item: dict[str, Any] | None) -> list[str]:
	if not isinstance(item, dict):
		return []

	decomposition = item.get("question_decomposition", []) or []
	if not isinstance(decomposition, list):
		return []

	sub_questions: list[str] = []
	for step in decomposition:
		if not isinstance(step, dict):
			continue
		q = (step.get("question") or "").strip()
		if q:
			sub_questions.append(q)
	return sub_questions


def _extract_docs_from_item(item: dict[str, Any] | None, max_docs: int = 20) -> list[str]:
	if not isinstance(item, dict):
		return []

	# 仅使用数据集中的 paragraphs 作为原始 context 来源。
	paragraphs = item.get("paragraphs", []) or []
	docs: list[str] = []
	if isinstance(paragraphs, list) and paragraphs:
		for paragraph in paragraphs:
			if not isinstance(paragraph, dict):
				continue
			title = (paragraph.get("title") or "").strip()
			text = (paragraph.get("paragraph_text") or paragraph.get("text") or "").strip()
			if not text:
				continue
			if title:
				docs.append(f"{title}\n{text}")
			else:
				docs.append(text)

	return docs[:max_docs]


def _build_marked_context_for_method(method: str, question: str, docs: list[str], model: ChatOllama) -> str:
	if not docs:
		return ""

	marked_docs: list[str] = []
	if method == "cdit":
		judgments = judge_similarity_batch(query_str=question, doc_list=docs, model=model)
		for idx, doc in enumerate(docs):
			is_similar = judgments[idx] if idx < len(judgments) else False
			tag = CONSISTENT_TAG if is_similar else INCONSISTENT_TAG
			marked_docs.append(f"{tag}\n{doc}")
	elif method == "contradoc":
		judgments = judge_contradiction_batch(doc_a_list=[question] * len(docs), doc_b_list=docs, model=model)
		for idx, doc in enumerate(docs):
			has_contradiction, evidence = judgments[idx] if idx < len(judgments) else (False, "")
			tag = INCONSISTENT_TAG if has_contradiction else CONSISTENT_TAG
			if has_contradiction and evidence:
				marked_docs.append(f"{tag}\n{doc}\nEvidence:\n{evidence}")
			else:
				marked_docs.append(f"{tag}\n{doc}")
	else:
		return "\n\n".join(docs)

	return "\n\n".join(marked_docs)


def _prepare_method_rows(
	method: str,
	prompts_root: Path,
	dataset_index: dict[str, dict[str, Any]],
	dataset_paths: list[str],
	model: ChatOllama,
	context_cache_file: Path | None = None,
	existing_result_map: dict[str, dict[str, Any]] | None = None,
	limit: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
	if method not in {"cdit", "contradoc", "vanilla"}:
		in_file = prompts_root / f"{method}_prompts.jsonl"
		rows = _load_jsonl(in_file)
		if limit is not None and limit > 0:
			rows = rows[:limit]
		return rows, str(in_file)

	# cdit / contradoc / vanilla: 完全基于原始 dataset 的 question + paragraphs 手工生成。
	dataset_rows = _load_dataset_items(dataset_paths=dataset_paths, limit=limit)
	generated_rows: list[dict[str, Any]] = []
	cache_map = _load_context_cache_map(context_cache_file) if context_cache_file else {}
	existing_result_map = existing_result_map or {}

	pbar = tqdm(dataset_rows, desc=f"build_context/{method}", unit="q")
	for item in pbar:
		question = (item.get("question") or "").strip()
		instance_id = item.get("id") or item.get("instance_id")
		key = _row_key(instance_id, question)
		pbar.set_postfix(instance_id=str(instance_id), refresh=False)
		if not question:
			continue

		if key in existing_result_map:
			generated_rows.append(
				{
					"instance_id": instance_id,
					"question": question,
					"reference_answer": existing_result_map[key].get("reference_answer", []),
					"sub_questions": [],
					"context_count": 0,
					"context_text": "",
				}
			)
			continue

		if key in cache_map:
			generated_rows.append(cache_map[key])
			continue

		sub_questions = _extract_subquestions_from_item(item)
		# 强保证：cdit/contradoc 必须有 sub-question，才进入多轮。
		if not sub_questions:
			continue

		docs = _extract_docs_from_item(item, max_docs=20)
		if not docs:
			# 严格要求：context 必须来自 dataset paragraphs。
			continue

		try:
			if method == "vanilla":
				context_text = "\n\n".join(docs)
			else:
				context_text = _build_marked_context_for_method(method=method, question=question, docs=docs, model=model)
		except Exception as exc:
			tqdm.write(
				f"[ERROR][build_context/{method}] instance_id={instance_id} question={question[:80]} err={type(exc).__name__}: {exc}"
			)
			continue
		answer = (item.get("answer") or "").strip()
		aliases = item.get("answer_alias", []) or item.get("answer_aliases", []) or []
		reference_answer = [answer] if answer else []
		for alias in aliases:
			if isinstance(alias, str) and alias.strip() and alias not in reference_answer:
				reference_answer.append(alias)

		new_row = {
			"instance_id": instance_id,
			"question": question,
			"reference_answer": reference_answer,
			"sub_questions": sub_questions,
			"context_count": len(docs),
			"context_text": context_text,
			"dataset_item": item,
		}
		generated_rows.append(new_row)
		cache_map[key] = new_row
		if context_cache_file:
			_safe_write_json(context_cache_file, {"method": method, "rows": list(cache_map.values())})

	return generated_rows, "generated_from_dataset_paragraphs"


def run_musique_baseline(
	prompts_dir: str = DEFAULT_PROMPTS_DIR,
	output_dir: str = DEFAULT_OUTPUT_DIR,
	methods: list[str] | None = None,
	dataset_paths: list[str] | None = None,
	model_name: str = "qwen3.5:9b",
	temperature: float = 0.1,
	limit: int | None = None,
) -> dict[str, Any]:
	methods = methods or list(DEFAULT_METHODS)
	dataset_paths = dataset_paths or list(DEFAULT_DATASET_PATHS)
	prompts_root = Path(prompts_dir)
	out_root = Path(output_dir)
	out_root.mkdir(parents=True, exist_ok=True)

	model = ChatOllama(model=model_name, temperature=temperature, top_p=1, reasoning=False, num_predict=8192)

	global_summary: dict[str, Any] = {
		"prompts_dir": str(prompts_root.resolve()),
		"output_dir": str(out_root.resolve()),
		"dataset_paths": dataset_paths,
		"methods": methods,
		"model_name": model_name,
		"temperature": temperature,
		"results": {},
	}

	# 统一收集 instance_id，加载分解所需的 question_decomposition。
	all_ids: set[str] = set()
	for method in methods:
		in_file = prompts_root / f"{method}_prompts.jsonl"
		for row in _load_jsonl(in_file):
			instance_id = row.get("instance_id")
			if isinstance(instance_id, str) and instance_id.strip():
				all_ids.add(instance_id.strip())

	dataset_index = _load_dataset_index_multi(dataset_paths=dataset_paths, target_ids=all_ids)

	for method in methods:
		out_file = out_root / f"musique_{method}.json"
		existing_result_map = _load_existing_results_map(out_file)
		context_cache_file = out_root / f"musique_{method}_contexts.json"

		rows, input_source = _prepare_method_rows(
			method=method,
			prompts_root=prompts_root,
			dataset_index=dataset_index,
			dataset_paths=dataset_paths,
			model=model,
			context_cache_file=context_cache_file,
			existing_result_map=existing_result_map,
			limit=limit,
		)

		if not rows:
			global_summary["results"][method] = {"status": "skipped", "reason": f"empty_or_missing: {input_source}"}
			_safe_write_json(out_root / "musique_baseline_summary.json", global_summary)
			continue

		method_results: list[dict[str, Any]] = []
		method_f1: list[float] = []
		method_em: list[float] = []
		method_match: list[float] = []
		method_hit: list[float] = []

		def _method_summary() -> dict[str, Any]:
			return {
				"method": method,
				"input_file": input_source,
				"total": len(method_results),
				"avg_f1": fmean(method_f1) if method_f1 else 0.0,
				"avg_exact_match": fmean(method_em) if method_em else 0.0,
				"avg_match": fmean(method_match) if method_match else 0.0,
				"avg_hit": fmean(method_hit) if method_hit else 0.0,
			}

		# 初始化方法文件，确保中断时也能看到开始状态。
		_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})

		pbar_exec = tqdm(rows, desc=f"baseline/{method}", unit="q")
		for row in pbar_exec:
			question = (row.get("question") or "").strip()
			if method in {"cdit", "contradoc", "vanilla"}:
				context_text = (row.get("context_text") or "").strip()
			else:
				context_text = _extract_context_text(row.get("prompt", ""))
			reference_answer = row.get("reference_answer", [])
			instance_id = row.get("instance_id")
			current_key = _row_key(instance_id, question)
			pbar_exec.set_postfix(instance_id=str(instance_id), refresh=False)

			if not question:
				tqdm.write(f"[ERROR][baseline/{method}] instance_id={instance_id} empty question, skip")
				# 即便跳过也同步写一次，体现实时进度。
				_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})
				global_summary["results"][method] = {
					"status": "running",
					"output_file": str(out_file.resolve()),
					**_method_summary(),
				}
				_safe_write_json(out_root / "musique_baseline_summary.json", global_summary)
				continue

			existing = existing_result_map.get(current_key)
			if isinstance(existing, dict) and "prediction" in existing and "metrics" in existing:
				metrics_exist = existing.get("metrics", {}) or {}
				try:
					method_f1.append(float(metrics_exist.get("f1", 0.0)))
					method_em.append(float(metrics_exist.get("exact_match", 0.0)))
					method_match.append(float(metrics_exist.get("match", 0.0)))
					method_hit.append(float(metrics_exist.get("hit", 0.0)))
				except Exception:
					pass
				method_results.append(existing)
				continue

			dataset_item = row.get("dataset_item") if method in {"cdit", "contradoc", "vanilla"} else None
			if method in {"cdit", "contradoc", "vanilla"}:
				sub_questions = row.get("sub_questions", []) or []
			else:
				if dataset_item is None and isinstance(instance_id, str):
					dataset_item = dataset_index.get(instance_id)
				sub_questions = _extract_subquestions_from_item(dataset_item)

			answer_history: list[str] = []
			subquestion_history: list[dict[str, str]] = []
			step_reports: list[dict[str, Any]] = []

			if sub_questions:
				try:
					for idx, sub_q in enumerate(sub_questions, start=1):
						resolved_question = _resolve_placeholders(sub_q, answer_history)

						history_text = ""
						if subquestion_history:
							history_lines = ["### Previous subquestion answers:"]
							for h_idx, turn in enumerate(subquestion_history, start=1):
								history_lines.append(f"{h_idx}. Q: {turn['question']}")
								history_lines.append(f"   A: {turn['answer']}")
							history_text = "\n".join(history_lines) + "\n\n"

						sub_prompt = _build_subquestion_prompt(resolved_question, context_text, history_text)
						sub_raw_answer = get_invoke(model, sub_prompt)
						sub_prediction = _normalize_answer(sub_raw_answer)

						answer_history.append(sub_prediction)
						subquestion_history.append({"question": resolved_question, "answer": sub_prediction})
						step_reports.append(
							{
								"index": idx,
								"raw_question": sub_q,
								"question": resolved_question,
								"prediction": sub_prediction,
							}
						)

					final_prompt = _build_final_prompt(question, subquestion_history)
					final_raw_answer = get_invoke(model, final_prompt)
					prediction = _normalize_answer(final_raw_answer)
				except Exception as exc:
					tqdm.write(
						f"[ERROR][baseline/{method}] instance_id={instance_id} question={question[:80]} err={type(exc).__name__}: {exc}"
					)
					continue
			else:
				if method in {"cdit", "contradoc", "vanilla"}:
					# 强保证：这两个方法无 sub-question 时不做单轮回退。
					tqdm.write(f"[ERROR][baseline/{method}] instance_id={instance_id} missing sub_questions, skip")
					_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})
					global_summary["results"][method] = {
						"status": "running",
						"output_file": str(out_file.resolve()),
						**_method_summary(),
					}
					_safe_write_json(out_root / "musique_baseline_summary.json", global_summary)
					continue
				# 找不到分解时，回退为单轮。
				try:
					prompt = MUSIQUE_QA_BASE.format(context_text=context_text, question=question)
					raw_answer = get_invoke(model, prompt)
					prediction = _normalize_answer(raw_answer)
				except Exception as exc:
					tqdm.write(
						f"[ERROR][baseline/{method}] instance_id={instance_id} question={question[:80]} err={type(exc).__name__}: {exc}"
					)
					continue

			metrics = _evaluate_answer(prediction, reference_answer)

			method_f1.append(metrics["f1"])
			method_em.append(metrics["exact_match"])
			method_match.append(metrics["match"])
			method_hit.append(metrics["hit"])

			method_results.append(
				{
					"instance_id": instance_id,
					"question": question,
					"subquestion_count": len(sub_questions),
					"subquestions": step_reports,
					"reference_answer": reference_answer,
					"prediction": prediction,
					"metrics": metrics,
				}
			)

			# 每处理完一个 question 立即落盘，防止崩溃/超时丢进度。
			_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})
			global_summary["results"][method] = {
				"status": "running",
				"output_file": str(out_file.resolve()),
				**_method_summary(),
			}
			_safe_write_json(out_root / "musique_baseline_summary.json", global_summary)

		method_summary = _method_summary()
		payload = {"summary": method_summary, "results": method_results}
		_safe_write_json(out_file, payload)

		global_summary["results"][method] = {
			"status": "ok",
			"output_file": str(out_file.resolve()),
			**method_summary,
		}
		_safe_write_json(out_root / "musique_baseline_summary.json", global_summary)

	overall_file = out_root / "musique_baseline_summary.json"
	_safe_write_json(overall_file, global_summary)
	global_summary["summary_file"] = str(overall_file.resolve())
	return global_summary


def main() -> None:
	parser = argparse.ArgumentParser(description="Run MuSiQue baseline on stored prompt jsonl files")
	parser.add_argument("--prompts-dir", type=str, default=DEFAULT_PROMPTS_DIR)
	parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument(
		"--dataset-paths",
		type=str,
		default=",".join(DEFAULT_DATASET_PATHS),
		help="comma-separated dataset jsonl paths for loading question_decomposition",
	)
	parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS), help="comma-separated methods")
	parser.add_argument("--model-name", type=str, default="qwen3.5:9b")
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--limit", type=int, default=0)
	args = parser.parse_args()

	methods = [one.strip() for one in args.methods.split(",") if one.strip()]
	dataset_paths = [one.strip() for one in args.dataset_paths.split(",") if one.strip()]
	report = run_musique_baseline(
		prompts_dir=args.prompts_dir,
		output_dir=args.output_dir,
		methods=methods,
		dataset_paths=dataset_paths,
		model_name=args.model_name,
		temperature=args.temperature,
		limit=(args.limit or None),
	)
	print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
