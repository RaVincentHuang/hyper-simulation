from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any

from langchain_ollama import ChatOllama
from tqdm import tqdm

from hyper_simulation.baselines.CDIT import judge_similarity_batch
from hyper_simulation.baselines.contradoc import judge_contradiction_batch
from hyper_simulation.llm.chat_completion import get_invoke
from hyper_simulation.llm.prompt.arc import ARC_BASE
from hyper_simulation.question_answer.vmdit.metrics import (
	exact_match_score,
	match,
	metric_max_over_ground_truths,
	qa_f1_score,
)


DEFAULT_PROMPTS_DIR = "data/ARC"
DEFAULT_OUTPUT_DIR = "data/baseline/arc"
DEFAULT_METHODS = ["bsim", "her", "sentli", "sparsecl", "cdit", "contradoc", "vanilla"]
DEFAULT_CHALLENGE_DATASET_PATHS = [
	"/home/vincent/.dataset/ARC/sample_ARC/ARC-Challenge-test-00000-of-00001.jsonl",
]
DIRECT_CONTEXT_METHODS = {"bsim", "her", "sentli", "sparsecl"}
SIMILARITY_METHODS = {"cdit"}
CONTRADICT_METHODS = {"contradoc"}
RAW_CONTEXT_METHODS = {"vanilla"}
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


def _normalize_question_for_match(question: str) -> str:
	q = str(question or "")
	q = q.split("\n\nOptions:")[0]
	q = re.sub(r"\s+", " ", q)
	return q.strip().lower()


def _coerce_reference_answers(value: Any) -> list[str]:
	if isinstance(value, str):
		v = value.strip()
		if not v or v == "[]":
			return []
		return [v]
	if isinstance(value, list):
		res: list[str] = []
		for one in value:
			if isinstance(one, str):
				v = one.strip()
				if v and v != "[]":
					res.append(v)
			elif isinstance(one, list):
				# 兼容 reference_answer=[[]] 这种脏数据
				continue
		return res
	return []


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

	m = re.match(r"^([A-Z])[\)\.:\-\s].*", v_upper)
	if m and m.group(1) in labels:
		return m.group(1)

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


def _iter_challenge_files(path: Path) -> list[Path]:
	if path.is_file():
		if "arc-challenge-test" in path.name.lower():
			return [path]
		return []
	files = sorted(path.glob("ARC-Challenge-test-*.jsonl"))
	if files:
		return files
	return sorted([p for p in path.glob("*.jsonl") if "arc-challenge-test" in p.name.lower()])


def _build_docs_from_choices(choices: dict[str, Any]) -> list[str]:
	if not isinstance(choices, dict):
		return []
	labels = choices.get("label", []) or []
	texts = choices.get("text", []) or []
	docs: list[str] = []
	for idx, text in enumerate(texts):
		label = str(labels[idx]).strip() if idx < len(labels) else str(idx + 1)
		t = str(text).strip()
		if t:
			docs.append(f"Option {label}: {t}")
	return docs


def _extract_question_with_options(question: str, choices: dict[str, Any] | None) -> str:
	base = (question or "").strip()
	if not base:
		return ""
	if "\n\nOptions:" in base:
		return base

	labels = []
	texts = []
	if isinstance(choices, dict):
		labels = choices.get("label", []) or []
		texts = choices.get("text", []) or []

	option_lines: list[str] = []
	for idx, text in enumerate(texts):
		label = labels[idx] if idx < len(labels) else str(idx + 1)
		option_lines.append(f"{label}) {str(text).strip()}")

	if not option_lines:
		return base
	return f"{base}\n\nOptions:\n" + "\n".join(option_lines)


def _extract_context_docs(item: dict[str, Any], max_docs: int = 20) -> list[str]:
	docs: list[str] = []

	paragraphs = item.get("paragraphs") or []
	if isinstance(paragraphs, list):
		for one in paragraphs:
			if not isinstance(one, dict):
				continue
			title = (one.get("title") or "").strip()
			text = (one.get("text") or "").strip()
			if not text:
				continue
			docs.append(f"{title}\n{text}" if title else text)

	ctxs = item.get("ctxs") or []
	if not docs and isinstance(ctxs, list):
		for one in ctxs:
			if not isinstance(one, dict):
				continue
			title = (one.get("title") or "").strip()
			text = (one.get("text") or "").strip()
			if not text:
				continue
			docs.append(f"{title}\n{text}" if title else text)

	context_docs = item.get("context_docs") or []
	if not docs and isinstance(context_docs, list):
		for one in context_docs:
			text = str(one).strip()
			if text:
				docs.append(text)

	context_text = (item.get("context_text") or "").strip()
	if not docs and context_text:
		docs.append(context_text)

	return docs[:max_docs]


def _build_marked_context_for_method(method: str, question: str, docs: list[str], model: ChatOllama) -> str:
	if not docs:
		return ""
	if method == "cdit":
		judgments = judge_similarity_batch(query_str=question, doc_list=docs, model=model)
		marked_docs: list[str] = []
		for idx, doc in enumerate(docs):
			is_similar = judgments[idx] if idx < len(judgments) else False
			tag = CONSISTENT_TAG if is_similar else INCONSISTENT_TAG
			marked_docs.append(f"{tag}\n{doc}")
		return "\n\n".join(marked_docs)
	if method == "contradoc":
		judgments = judge_contradiction_batch(doc_a_list=[question] * len(docs), doc_b_list=docs, model=model)
		marked_docs: list[str] = []
		for idx, doc in enumerate(docs):
			has_contradiction, evidence = judgments[idx] if idx < len(judgments) else (False, "")
			tag = INCONSISTENT_TAG if has_contradiction else CONSISTENT_TAG
			if has_contradiction and evidence:
				marked_docs.append(f"{tag}\n{doc}\nEvidence:\n{evidence}")
			else:
				marked_docs.append(f"{tag}\n{doc}")
		return "\n\n".join(marked_docs)
	return "\n\n".join(docs)


def _load_arc_challenge_items(dataset_paths: list[str], limit: int | None = None) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	seen_keys: set[str] = set()

	for one_path in dataset_paths:
		path = Path(one_path)
		if not path.exists():
			continue
		for file_path in _iter_challenge_files(path):
			for item in _load_jsonl(file_path):
				if not isinstance(item, dict):
					continue
				question = (item.get("question") or "").strip()
				if not question:
					continue
				instance_id = (item.get("id") or item.get("instance_id") or "").strip()
				key = instance_id if instance_id else _normalize_question_for_match(question)
				if key in seen_keys:
					continue
				seen_keys.add(key)
				rows.append(
					{
						"instance_id": instance_id,
						"question": question,
						"question_key": _normalize_question_for_match(question),
						"answerKey": (item.get("answerKey") or "").strip(),
						"choices": item.get("choices") or {},
					}
				)
				if limit is not None and limit > 0 and len(rows) >= limit:
					return rows

	return rows


def _load_direct_prompt_rows(
	method: str,
	prompts_root: Path,
	challenge_ids: set[str],
	challenge_qkeys: set[str],
	answer_by_qkey: dict[str, str],
	limit: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
	candidates = [
		prompts_root / f"ARC_{method}.jsonl",
		prompts_root / f"arc_{method}.jsonl",
	]
	input_file = next((path for path in candidates if path.exists()), candidates[0])
	rows: list[dict[str, Any]] = []

	for row in _load_jsonl(input_file):
		question = (row.get("question") or "").strip()
		if not question:
			continue
		instance_id = str(row.get("instance_id") or row.get("id") or "").strip()
		qkey = _normalize_question_for_match(question)
		if not ((instance_id and instance_id in challenge_ids) or qkey in challenge_qkeys):
			continue

		reference_answer = _coerce_reference_answers(row.get("reference_answer", []))
		if not reference_answer:
			mapped = answer_by_qkey.get(qkey, "")
			reference_answer = [mapped] if mapped else []

		rows.append(
			{
				"instance_id": instance_id,
				"question": question,
				"prompt": row.get("prompt", ""),
				"context_text": "",
				"reference_answer": reference_answer,
				"choices": row.get("choices") or {},
			}
		)

		if limit is not None and limit > 0 and len(rows) >= limit:
			break

	return rows, str(input_file)


def _prepare_live_rows(
	method: str,
	base_rows: list[dict[str, Any]],
	model: ChatOllama,
	context_cache_file: Path | None = None,
	existing_result_map: dict[str, dict[str, Any]] | None = None,
	limit: int | None = None,
) -> list[dict[str, Any]]:
	generated_rows: list[dict[str, Any]] = []
	cache_map = _load_context_cache_map(context_cache_file) if context_cache_file else {}
	existing_result_map = existing_result_map or {}

	pbar = tqdm(base_rows, desc=f"build_context/{method}", unit="q")
	for item in pbar:
		question = (item.get("question") or "").strip()
		instance_id = item.get("instance_id") or ""
		current_key = _row_key(instance_id, question)
		pbar.set_postfix(instance_id=str(instance_id), refresh=False)
		if not question:
			continue

		if current_key in existing_result_map:
			generated_rows.append(
				{
					"instance_id": instance_id,
					"question": question,
					"reference_answer": existing_result_map[current_key].get("reference_answer", []),
					"choices": item.get("choices") or {},
					"context_text": "",
					"prompt": "",
				}
			)
			continue

		if current_key in cache_map:
			generated_rows.append(cache_map[current_key])
			continue

		docs = item.get("docs", [])
		if not isinstance(docs, list) or not docs:
			continue

		if method == "vanilla":
			context_text = "\n\n".join(str(doc).strip() for doc in docs if str(doc).strip())
		else:
			context_text = _build_marked_context_for_method(method=method, question=question, docs=docs, model=model)

		question_with_options = _extract_question_with_options(question=question, choices=item.get("choices") or {})
		prompt = ARC_BASE.format(context_text=context_text, question=question_with_options)

		new_row = {
			"instance_id": instance_id,
			"question": question_with_options,
			"reference_answer": _coerce_reference_answers(item.get("reference_answer", [])),
			"choices": item.get("choices") or {},
			"context_count": len(docs),
			"context_text": context_text,
			"prompt": prompt,
		}
		generated_rows.append(new_row)
		cache_map[current_key] = new_row
		if context_cache_file:
			_safe_write_json(context_cache_file, {"method": method, "rows": list(cache_map.values())})

		if limit is not None and limit > 0 and len(generated_rows) >= limit:
			break

	return generated_rows


def run_arc_baseline(
	prompts_dir: str = DEFAULT_PROMPTS_DIR,
	output_dir: str = DEFAULT_OUTPUT_DIR,
	methods: list[str] | None = None,
	dataset_paths: list[str] | None = None,
	model_name: str = "qwen3.5:9b",
	temperature: float = 0.1,
	limit: int | None = None,
) -> dict[str, Any]:
	methods = methods or list(DEFAULT_METHODS)
	dataset_paths = dataset_paths or list(DEFAULT_CHALLENGE_DATASET_PATHS)
	prompts_root = Path(prompts_dir)
	out_root = Path(output_dir)
	out_root.mkdir(parents=True, exist_ok=True)

	challenge_items = _load_arc_challenge_items(dataset_paths=dataset_paths, limit=limit)
	challenge_ids = {str(one.get("instance_id") or "").strip() for one in challenge_items if str(one.get("instance_id") or "").strip()}
	challenge_qkeys = {str(one.get("question_key") or "").strip() for one in challenge_items if str(one.get("question_key") or "").strip()}
	answer_by_qkey = {
		str(one.get("question_key") or "").strip(): str(one.get("answerKey") or "").strip()
		for one in challenge_items
		if str(one.get("question_key") or "").strip() and str(one.get("answerKey") or "").strip()
	}

	model = ChatOllama(model=model_name, temperature=temperature, top_p=1, reasoning=False, num_predict=8192)
	global_summary: dict[str, Any] = {
		"prompts_dir": str(prompts_root.resolve()),
		"output_dir": str(out_root.resolve()),
		"dataset_paths": dataset_paths,
		"live_context_source": "ARC-Challenge-test choices",
		"methods": methods,
		"model_name": model_name,
		"temperature": temperature,
		"challenge_size": len(challenge_items),
		"results": {},
	}

	live_base_rows = [
		{
			"instance_id": one.get("instance_id", ""),
			"question": one.get("question", ""),
			"choices": one.get("choices") or {},
			"docs": _build_docs_from_choices(one.get("choices") or {}),
			"reference_answer": [str(one.get("answerKey") or "").strip()] if str(one.get("answerKey") or "").strip() else [],
		}
		for one in challenge_items
	]
	live_input_source = ",".join(dataset_paths)

	for method in methods:
		out_file = out_root / f"ARC_{method}.json"
		existing_result_map = _load_existing_results_map(out_file)

		if method in DIRECT_CONTEXT_METHODS:
			rows, input_source = _load_direct_prompt_rows(
				method=method,
				prompts_root=prompts_root,
				challenge_ids=challenge_ids,
				challenge_qkeys=challenge_qkeys,
				answer_by_qkey=answer_by_qkey,
				limit=limit,
			)
		else:
			input_source = live_input_source
			context_cache_file = out_root / f"ARC_{method}_contexts.json"
			rows = _prepare_live_rows(
				method=method,
				base_rows=live_base_rows,
				model=model,
				context_cache_file=context_cache_file,
				existing_result_map=existing_result_map,
				limit=limit,
			)

		if not rows:
			global_summary["results"][method] = {"status": "skipped", "reason": f"empty_or_missing: {input_source}"}
			_safe_write_json(out_root / "ARC_baseline_summary.json", global_summary)
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

		_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})

		pbar_exec = tqdm(rows, desc=f"baseline/{method}", unit="q")
		for row in pbar_exec:
			question = (row.get("question") or "").strip()
			instance_id = row.get("instance_id")
			current_key = _row_key(instance_id, question)
			pbar_exec.set_postfix(instance_id=str(instance_id), refresh=False)

			if not question:
				_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})
				global_summary["results"][method] = {"status": "running", "output_file": str(out_file.resolve()), **_method_summary()}
				_safe_write_json(out_root / "ARC_baseline_summary.json", global_summary)
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

			choices = row.get("choices") or {}
			choice_labels = choices.get("label", []) if isinstance(choices, dict) else []
			choice_texts = choices.get("text", []) if isinstance(choices, dict) else []

			prompt = str(row.get("prompt") or "").strip()
			if not prompt:
				context_text = (row.get("context_text") or "").strip()
				question_with_options = _extract_question_with_options(question=question, choices=choices)
				prompt = ARC_BASE.format(context_text=context_text, question=question_with_options)

			try:
				raw_answer = get_invoke(model, prompt)
				prediction = _normalize_arc_label(_normalize_answer(raw_answer), choice_labels=choice_labels, choice_texts=choice_texts)
			except Exception as exc:
				tqdm.write(
					f"[ERROR][baseline/{method}] instance_id={instance_id} question={question[:80]} err={type(exc).__name__}: {exc}"
				)
				continue

			reference_answer = _coerce_reference_answers(row.get("reference_answer", []))
			if not reference_answer:
				mapped = answer_by_qkey.get(_normalize_question_for_match(question), "")
				reference_answer = [mapped] if mapped else []

			metrics = _evaluate_answer(prediction, reference_answer)
			method_f1.append(metrics["f1"])
			method_em.append(metrics["exact_match"])
			method_match.append(metrics["match"])
			method_hit.append(metrics["hit"])

			method_results.append(
				{
					"instance_id": instance_id,
					"question": question,
					"reference_answer": reference_answer,
					"prediction": prediction,
					"metrics": metrics,
					"context_text": row.get("context_text", ""),
				}
			)

			_safe_write_json(out_file, {"summary": _method_summary(), "results": method_results})
			global_summary["results"][method] = {"status": "running", "output_file": str(out_file.resolve()), **_method_summary()}
			_safe_write_json(out_root / "ARC_baseline_summary.json", global_summary)

		method_summary = _method_summary()
		payload = {"summary": method_summary, "results": method_results}
		_safe_write_json(out_file, payload)
		global_summary["results"][method] = {"status": "ok", "output_file": str(out_file.resolve()), **method_summary}
		_safe_write_json(out_root / "ARC_baseline_summary.json", global_summary)

	overall_file = out_root / "ARC_baseline_summary.json"
	_safe_write_json(overall_file, global_summary)
	global_summary["summary_file"] = str(overall_file.resolve())
	return global_summary


def main() -> None:
	parser = argparse.ArgumentParser(description="Run ARC baseline (challenge-only)")
	parser.add_argument("--prompts-dir", type=str, default=DEFAULT_PROMPTS_DIR)
	parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--dataset-paths", type=str, default=",".join(DEFAULT_CHALLENGE_DATASET_PATHS))
	parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
	parser.add_argument("--model-name", type=str, default="qwen3.5:9b")
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--limit", type=int, default=0)
	args = parser.parse_args()

	methods = [one.strip() for one in args.methods.split(",") if one.strip()]
	dataset_paths = [one.strip() for one in args.dataset_paths.split(",") if one.strip()]
	report = run_arc_baseline(
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
