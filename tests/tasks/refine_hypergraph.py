from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

from hyper_simulation.hypergraph.entity import ENT
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex


DEFAULT_MODEL = "qwen3.5-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

TASK_DEFAULT_SOURCE_ROOTS = {
	"hotpotqa": Path("data/debug/hotpotqa"),
	"legalbench": Path("data/debug/legalbench"),
	"musique": Path("data/debug/musique"),
	"multihop": Path("data/debug/multihop"),
	"arc": Path("data/debug/arc"),
	"docnli": Path("data/debug/docnli/sample50"),
	"econ": Path("data/debug/econ/sample"),
	"contract_nli": Path("data/debug/contract_nli/sample65"),
	"control": Path("data/debug/control/sample80"),
}


def generate_instance_id(query: str) -> str:
	normalized = "".join(query.split()).lower()
	return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]


def _should_refine_vertex(vertex: Vertex) -> bool:
	return not (
		vertex.is_query()
		or vertex.is_verb()
		or vertex.is_virtual()
		or vertex.is_adjective()
		or vertex.is_adverb()
	)


def _ent_definitions() -> str:
	return "\n".join(
		[
			"PERSON: Human being, individual, or specific character.",
			"COUNTRY: A nation with its own government.",
			"LOC: Geographical location, natural region, body of water.",
			"ORG: Organization, institution, company, government body.",
			"FAC: Physical building, facility, structure.",
			"GPE: Geopolitical entity, such as cities, states, provinces (but not countries).",
			"NORP: Nationalities, religious or political groups.",
			"PRODUCT: Physical object, vehicle, device, manufactured good.",
			"WORK_OF_ART: Piece of art, publication, show.",
			"LAW: Legal document, binding agreement.",
			"LANGUAGE: Spoken or written human language.",
			"OCCUPATION: Job, profession, trade.",
			"EVENT: Phenomenon, historical event, sports match.",
			"TEMPORAL: Time period, specific date, unit of time.",
			"NUMBER: Mathematical number, quantity.",
			"CONCEPT: Abstract idea, theoretical concept.",
			"ORGANISM: Living being, such as animal, plant, or microorganism.",
			"FOOD: Edible substance, dish, or cuisine.",
			"MEDICAL: Medical condition, disease, symptom, or treatment.",
			"ANATOMY: Body part, organ, or anatomical structure.",
			"SUBSTANCE: Chemical element, compound, or material.",
			"ASTRO: Astronomical object, such as a star, planet, or galaxy.",
			"AWARD: Prize, honor, or recognition given to a person or organization.",
			"VEHICLE: Means of transportation, such as a car, airplane, or bicycle.",
			"THEORY: Scientific or philosophical theory, principle, or framework.",
			"GROUP: Collection of individuals like a family, team, class, or social group.",
			"FEATURE: Distinctive attribute, property, or characteristic of an entity or concept.",
			"ECONOMIC: Economic entity, such as a market, industry, or economic concept.",
			"SOCIOLOGY: Concepts related to society, culture, sociology, or social interactions.",
			"PHENOMENON: Natural or social phenomenon, such as climate change or cultural trend.",
			"ACTION: Action, behavior, or process not covered by the above categories.",
			"NOT_ENT: Use this if it does not fit any category above.",
		]
	)


def _normalize_response_content(raw_content: Any) -> str:
	if isinstance(raw_content, str):
		return raw_content
	if isinstance(raw_content, list):
		return "\n".join(
			item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
			for item in raw_content
		)
	return str(raw_content)


def _extract_json_payload(content: str) -> dict[str, Any] | None:
	try:
		parsed = json.loads(content)
		if isinstance(parsed, dict):
			return parsed
	except Exception:
		pass

	left = content.find("{")
	right = content.rfind("}")
	if left == -1 or right == -1 or right <= left:
		return None

	try:
		parsed = json.loads(content[left : right + 1])
		if isinstance(parsed, dict):
			return parsed
	except Exception:
		return None
	return None


def _sorted_index_from_name(path: Path) -> int:
	match = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
	if match is None:
		return 10**9
	return int(match.group(1))


def _data_file_index(path: Path) -> int:
	for pattern in [r"data_hypergraph(\d+)\.pkl", r"data_(\d+)\.pkl"]:
		match = re.fullmatch(pattern, path.name)
		if match is not None:
			return int(match.group(1))
	return 10**9


def _find_query_file(instance_dir: Path) -> Path | None:
	for name in ["query_hypergraph.pkl", "query.pkl"]:
		candidate = instance_dir / name
		if candidate.exists():
			return candidate
	return None


def _list_data_files(instance_dir: Path) -> list[Path]:
	all_files: dict[str, Path] = {}
	for pattern in ["data_hypergraph*.pkl", "data_*.pkl"]:
		for path in instance_dir.glob(pattern):
			all_files[path.name] = path
	return sorted(all_files.values(), key=lambda p: (_data_file_index(p), p.name))


def _normalize_output_instance_files(source_dir: Path, output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	query_src = _find_query_file(source_dir)
	if query_src is not None:
		shutil.copy2(query_src, output_dir / "query_hypergraph.pkl")

	for data_src in _list_data_files(source_dir):
		idx = _data_file_index(data_src)
		if idx >= 10**9:
			continue
		canonical_name = f"data_hypergraph{idx}.pkl"
		shutil.copy2(data_src, output_dir / canonical_name)

	metadata_src = source_dir / "metadata.json"
	if metadata_src.exists():
		shutil.copy2(metadata_src, output_dir / "metadata.json")


def _ensure_output_instance_canonical_names(output_dir: Path) -> None:
	query_new = output_dir / "query_hypergraph.pkl"
	query_old = output_dir / "query.pkl"
	if (not query_new.exists()) and query_old.exists():
		query_old.rename(query_new)

	for old_data in sorted(output_dir.glob("data_*.pkl")):
		idx = _data_file_index(old_data)
		if idx >= 10**9:
			continue
		new_data = output_dir / f"data_hypergraph{idx}.pkl"
		if not new_data.exists():
			old_data.rename(new_data)


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _write_progress_snapshot(progress_path: Path | None, payload: dict[str, Any]) -> None:
	if progress_path is None:
		return
	progress_path.parent.mkdir(parents=True, exist_ok=True)
	progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class RefinementProgressUI:
	def __init__(self, total_instances: int, progress_path: Path | None = None) -> None:
		self.progress_path = progress_path
		self.total_instances = total_instances
		self.processed_instances = 0
		self.skipped_instances = 0
		self.current_instance_id: str | None = None
		self.current_step: str = "idle"
		self.current_data_index: int | None = None
		self.current_data_file: str | None = None
		self.query_stats: dict[str, int] | None = None
		self.data_stats: dict[str, int] | None = None
		self.instance_bar = tqdm(total=total_instances, desc="instances", unit="inst", dynamic_ncols=True)
		self.data_bar: tqdm | None = None
		self._flush()

	def _flush(self) -> None:
		_write_progress_snapshot(
			self.progress_path,
			{
				"timestamp": _utc_now(),
				"total_instances": self.total_instances,
				"processed_instances": self.processed_instances,
				"skipped_instances": self.skipped_instances,
				"current_instance_id": self.current_instance_id,
				"current_step": self.current_step,
				"current_data_index": self.current_data_index,
				"current_data_file": self.current_data_file,
				"query_stats": self.query_stats,
				"data_stats": self.data_stats,
			},
		)

	def set_current_instance(self, instance_id: str, data_total: int) -> None:
		self.current_instance_id = instance_id
		self.current_step = "query"
		self.current_data_index = None
		self.current_data_file = None
		self.query_stats = None
		self.data_stats = None
		if self.data_bar is not None:
			self.data_bar.close()
		self.data_bar = tqdm(total=data_total, desc=f"data[{instance_id[:8]}]", unit="graph", leave=False, dynamic_ncols=True)
		self._flush()

	def mark_query_done(self, stats: dict[str, int]) -> None:
		self.current_step = "data"
		self.query_stats = stats
		if self.instance_bar is not None:
			self.instance_bar.set_postfix(query_fixed=stats.get("fixed", 0), query_filled=stats.get("filled", 0), refresh=False)
		self._flush()

	def mark_data_start(self, data_index: int, data_file: str) -> None:
		self.current_step = "data"
		self.current_data_index = data_index
		self.current_data_file = data_file
		self.data_stats = None
		self._flush()

	def mark_data_done(self, stats: dict[str, int]) -> None:
		self.data_stats = stats
		if self.data_bar is not None:
			self.data_bar.update(1)
			self.data_bar.set_postfix(fixed=stats.get("fixed", 0), filled=stats.get("filled", 0), refresh=False)
		self._flush()

	def finish_instance(self, skipped: bool = False) -> None:
		if skipped:
			self.skipped_instances += 1
		else:
			self.processed_instances += 1
		if self.instance_bar is not None:
			self.instance_bar.update(1)
			self.instance_bar.set_postfix(processed=self.processed_instances, skipped=self.skipped_instances, refresh=False)
		self.current_instance_id = None
		self.current_step = "idle"
		self.current_data_index = None
		self.current_data_file = None
		self.query_stats = None
		self.data_stats = None
		self._flush()

	def close(self) -> None:
		if self.data_bar is not None:
			self.data_bar.close()
			self.data_bar = None
		if self.instance_bar is not None:
			self.instance_bar.close()
			self.instance_bar = None
		self._flush()


class BatchProgressTracker:
	def __init__(self, total_instances: int, progress_path: Path) -> None:
		self.total_instances = total_instances
		self.progress_path = progress_path
		self.lock = Lock()
		self.started = 0
		self.completed = 0
		self.updated = 0
		self.skipped = 0
		self.failed = 0
		self.active_instances: set[str] = set()
		self.latest_instance: str | None = None
		self.latest_status: str | None = None
		self.latest_reason: str | None = None
		self._flush()

	def _snapshot(self) -> dict[str, Any]:
		return {
			"timestamp": _utc_now(),
			"total_instances": self.total_instances,
			"started": self.started,
			"completed": self.completed,
			"updated": self.updated,
			"skipped": self.skipped,
			"failed": self.failed,
			"active_instances": sorted(self.active_instances),
			"latest_instance": self.latest_instance,
			"latest_status": self.latest_status,
			"latest_reason": self.latest_reason,
		}

	def _flush(self) -> None:
		_write_progress_snapshot(self.progress_path, self._snapshot())

	def start(self, instance_id: str) -> None:
		with self.lock:
			self.started += 1
			self.active_instances.add(instance_id)
			self.latest_instance = instance_id
			self.latest_status = "running"
			self.latest_reason = None
			self._flush()

	def finish(self, instance_id: str, status: str, reason: str | None = None) -> None:
		with self.lock:
			self.completed += 1
			self.active_instances.discard(instance_id)
			self.latest_instance = instance_id
			self.latest_status = status
			self.latest_reason = reason
			if status == "updated":
				self.updated += 1
			elif status == "skipped":
				self.skipped += 1
			else:
				self.failed += 1
			self._flush()


def _coerce_int(value: Any) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except (TypeError, ValueError):
		return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
	items: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as fin:
		for line in fin:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			if isinstance(obj, dict):
				items.append(obj)
	return items


def _extract_question(item: dict[str, Any]) -> str:
	return ((item.get("question") or item.get("query") or item.get("hypothesis") or "").strip())


def _load_hotpotqa_items(dataset_path: str) -> list[dict[str, Any]]:
	path = Path(dataset_path)
	if path.is_file():
		files = [path]
	else:
		files = sorted(path.glob("hotpot_distractor*.jsonl"))
		if not files:
			files = sorted(path.glob("*.jsonl"))
	items: list[dict[str, Any]] = []
	for file_path in files:
		items.extend(_read_jsonl(file_path))
	return items


def _load_multihop_items(dataset_path: str) -> list[dict[str, Any]]:
	path = Path(dataset_path)
	files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
	items: list[dict[str, Any]] = []
	for file_path in files:
		items.extend(_read_jsonl(file_path))
	return items


def _load_musique_items(dataset_path: str) -> list[dict[str, Any]]:
	path = Path(dataset_path)
	if path.is_file():
		files = [path]
	else:
		files = sorted(path.glob("musique_answerable*.jsonl"))
		if not files:
			files = sorted(path.glob("*.jsonl"))

	items: list[dict[str, Any]] = []
	for file_path in files:
		items.extend(_read_jsonl(file_path))
	return items


def _load_arc_items(dataset_path: str) -> list[dict[str, Any]]:
	path = Path(dataset_path)
	if path.is_file():
		files = [path] if path.name.lower() == "arc-challenge-test-00000-of-00001.jsonl" else []
	else:
		files = sorted(path.glob("ARC-Challenge-test-00000-of-00001.jsonl"))
		if not files:
			files = sorted(path.glob("ARC-Challenge-test-*.jsonl"))

	items: list[dict[str, Any]] = []
	for file_path in files:
		items.extend(_read_jsonl(file_path))
	return items


def _load_legalbench_items(dataset_path: str) -> list[dict[str, Any]]:
	path = Path(dataset_path)
	files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
	items: list[dict[str, Any]] = []
	for file_path in files:
		for item in _read_jsonl(file_path):
			item = dict(item)
			item["_source_file"] = file_path.name
			items.append(item)
	return items


def _normalize_nli_item(task: str, raw: dict[str, Any], row_idx: int) -> dict[str, Any] | None:
	premise_raw = raw.get("premise")
	hypothesis = (raw.get("hypothesis") or "").strip()
	label = (raw.get("label") or "").strip()
	if not hypothesis:
		return None

	premise_chunks: list[str] = []
	if isinstance(premise_raw, str):
		text = premise_raw.strip()
		if text:
			premise_chunks = [text]
	elif isinstance(premise_raw, list):
		for chunk in premise_raw:
			if not isinstance(chunk, str):
				continue
			text = chunk.strip()
			if text:
				premise_chunks.append(text)

	if not premise_chunks:
		return None

	source_id = raw.get("id")
	if source_id is None:
		source_id = raw.get("uid")
	if source_id is None:
		source_id = f"row-{row_idx}"

	if task == "contract_nli":
		source_file = "contract_nli.split.jsonl"
		dataset_name = raw.get("dataset", "contract_nli")
	elif task == "control":
		source_file = "control.jsonl"
		dataset_name = raw.get("dataset", "control")
	else:
		source_file = "docnli.jsonl" if task == "docnli" else "econ_qa.jsonl"
		dataset_name = raw.get("dataset", task)

	return {
		"_id": str(source_id),
		"question": hypothesis,
		"answer": label,
		"context_docs": premise_chunks,
		"premise_chunks": premise_chunks,
		"premise": "\n\n".join(premise_chunks),
		"hypothesis": hypothesis,
		"label": label,
		"chunked": bool(raw.get("chunked", False) or len(premise_chunks) > 1),
		"num_chunks": len(premise_chunks),
		"dataset": dataset_name,
		"subset": raw.get("subset", "test"),
		"source_file": source_file,
	}


def _load_nli_items(dataset_path: str, task: str) -> list[dict[str, Any]]:
	path = Path(dataset_path)
	files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
	items: list[dict[str, Any]] = []
	for file_path in files:
		for idx, raw in enumerate(_read_jsonl(file_path)):
			normalized = _normalize_nli_item(task=task, raw=raw, row_idx=idx)
			if normalized is not None:
				items.append(normalized)
	return items


def load_task_dataset(task: str, dataset_path: str) -> list[dict[str, Any]]:
	if task == "hotpotqa":
		return _load_hotpotqa_items(dataset_path)
	if task == "legalbench":
		return _load_legalbench_items(dataset_path)
	if task == "musique":
		return _load_musique_items(dataset_path)
	if task == "multihop":
		return _load_multihop_items(dataset_path)
	if task == "arc" or task == "ARC":
		return _load_arc_items(dataset_path)
	if task in {"docnli", "econ", "contract_nli", "control"}:
		return _load_nli_items(dataset_path, task=task)
	raise ValueError(f"Unsupported task: {task}")


def _collect_source_instance_ids(source_root: Path) -> set[str]:
	return {path.name for path in source_root.iterdir() if path.is_dir()}


def _validate_source_alignment(source_root: Path, dataset_instance_ids: list[str], task: str) -> None:
	source_instance_ids = _collect_source_instance_ids(source_root)
	if not source_instance_ids:
		raise ValueError(f"No source instance directories found under: {source_root}")

	matched = [instance_id for instance_id in dataset_instance_ids if instance_id in source_instance_ids]
	if matched:
		return

	sample_dataset_ids = dataset_instance_ids[:5]
	sample_source_ids = sorted(source_instance_ids)[:5]
	raise ValueError(
		"Source hypergraph tree does not match the selected dataset. "
		f"task={task} source_root={source_root} dataset_instances={len(dataset_instance_ids)} "
		f"source_instances={len(source_instance_ids)} sample_dataset_ids={sample_dataset_ids} "
		f"sample_source_ids={sample_source_ids}. "
		"Regenerate the ARC source hypergraph tree from the same challenge-test split before running refine."
	)


def _resolve_task_source_root(task: str, source_root: Path | None) -> Path:
	if source_root is None:
		preferred = TASK_DEFAULT_SOURCE_ROOTS.get(task)
		if preferred is not None:
			return preferred
		raise ValueError(f"Unable to resolve source root for task: {task}")

	if _collect_source_instance_ids(source_root):
		return source_root

	preferred = TASK_DEFAULT_SOURCE_ROOTS.get(task)
	if preferred is not None and preferred.exists():
		return preferred

	if source_root.exists():
		for child in sorted(source_root.iterdir()):
			if not child.is_dir():
				continue
			if _collect_source_instance_ids(child):
				return child

	return source_root


def _build_data_entries(task: str, item: dict[str, Any]) -> list[dict[str, Any]]:
	entries: list[dict[str, Any]] = []

	if task == "hotpotqa":
		context = item.get("context") or {}
		titles = context.get("title", []) if isinstance(context, dict) else []
		sent_groups = context.get("sentences", []) if isinstance(context, dict) else []
		supporting_facts = item.get("supporting_facts", {}) if isinstance(item.get("supporting_facts"), dict) else {}
		support_titles = set(supporting_facts.get("title", []) or [])
		support_sent_ids = set(supporting_facts.get("sent_id", []) or [])
		flat_idx = 0
		for doc_idx, title in enumerate(titles):
			sentences = sent_groups[doc_idx] if doc_idx < len(sent_groups) and isinstance(sent_groups[doc_idx], list) else []
			for sent_idx, sentence in enumerate(sentences):
				text = str(sentence).strip()
				if not text:
					continue
				is_supporting = (title in support_titles) and (sent_idx in support_sent_ids)
				entries.append(
					{
						"index": flat_idx,
						"title": str(title),
						"text": text,
						"is_supporting": is_supporting,
						"source": {"doc_index": doc_idx, "sent_index": sent_idx},
					}
				)
				flat_idx += 1
		return entries

	if task == "multihop":
		for idx, evidence in enumerate(item.get("evidence_list", []) or []):
			if not isinstance(evidence, dict):
				continue
			text = (evidence.get("text") or evidence.get("fact") or "").strip()
			if not text:
				continue
			entries.append(
				{
					"index": idx,
					"title": (evidence.get("title") or "").strip(),
					"text": text,
					"is_supporting": True,
					"source": {"source": evidence.get("source"), "published_at": evidence.get("published_at")},
				}
			)
		return entries

	if task == "musique":
		paragraphs = item.get("paragraphs") or []
		if isinstance(paragraphs, list):
			for record in paragraphs:
				if not isinstance(record, dict):
					continue
				idx = _coerce_int(record.get("idx"))
				if idx is None:
					idx = len(entries)
				title = str(record.get("title") or "").strip()
				text = str(record.get("paragraph_text") or record.get("text") or "").strip()
				if not text:
					continue
				entries.append(
					{
						"index": idx,
						"title": title,
						"text": text,
						"is_supporting": bool(record.get("is_supporting", False)),
						"source": {"type": "paragraphs", "idx": idx},
					}
				)
		return entries

	if task == "legalbench":
		text = (item.get("text") or item.get("contract") or "").strip()
		if text:
			entries.append(
				{
					"index": 0,
					"title": item.get("_source_file", "legalbench"),
					"text": text,
					"is_supporting": True,
					"source": {"file": item.get("_source_file", "")},
				}
			)
		return entries

	if task in {"docnli", "econ", "contract_nli", "control"}:
		premise_chunks = item.get("premise_chunks") or item.get("context_docs") or []
		if isinstance(premise_chunks, list):
			for idx, premise_chunk in enumerate(premise_chunks):
				if not isinstance(premise_chunk, str):
					continue
				text = premise_chunk.strip()
				if not text:
					continue
				entries.append(
					{
						"index": idx,
						"title": f"premise_{idx}",
						"text": text,
						"is_supporting": True,
						"source": {"chunk_index": idx},
					}
				)
		return entries

	if task == "arc":
		context = item.get("context")
		if isinstance(context, list):
			for record in context:
				if isinstance(record, (list, tuple)) and len(record) >= 2:
					title = str(record[0] or "").strip()
					sentences = record[1]
					if not isinstance(sentences, list):
						sentences = []
					text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip()).strip()
					if text:
						entries.append({"index": len(entries), "title": title, "text": text, "is_supporting": True, "source": {"type": "context"}})
				elif isinstance(record, dict):
					title = str(record.get("title") or "").strip()
					text = str(record.get("text") or record.get("paragraph_text") or "").strip()
					if text:
						entries.append({"index": len(entries), "title": title, "text": text, "is_supporting": True, "source": {"type": "context"}})

		paragraphs = item.get("paragraphs") or []
		if not entries and isinstance(paragraphs, list):
			for record in paragraphs:
				if not isinstance(record, dict):
					continue
				title = str(record.get("title") or "").strip()
				text = str(record.get("text") or record.get("paragraph_text") or "").strip()
				if text:
					entries.append({"index": len(entries), "title": title, "text": text, "is_supporting": True, "source": {"type": "paragraphs"}})

		ctxs = item.get("ctxs") or []
		if not entries and isinstance(ctxs, list):
			for record in ctxs:
				if not isinstance(record, dict):
					continue
				title = str(record.get("title") or "").strip()
				text = str(record.get("text") or "").strip()
				if text:
					entries.append({"index": len(entries), "title": title, "text": text, "is_supporting": True, "source": {"type": "ctxs"}})

		context_docs = item.get("context_docs") or []
		if not entries and isinstance(context_docs, list):
			for record in context_docs:
				text = str(record or "").strip()
				if text:
					entries.append({"index": len(entries), "title": "", "text": text, "is_supporting": True, "source": {"type": "context_docs"}})

		context_text = str(item.get("context_text") or "").strip()
		if not entries and context_text:
			entries.append({"index": len(entries), "title": "", "text": context_text, "is_supporting": True, "source": {"type": "context_text"}})

		if not entries:
			question = _extract_question(item)
			choices = item.get("choices") or {}
			labels = choices.get("label", []) if isinstance(choices, dict) else []
			texts = choices.get("text", []) if isinstance(choices, dict) else []
			option_lines: list[str] = []
			for idx, text in enumerate(texts):
				label = str(labels[idx]).strip() if idx < len(labels) else chr(ord("A") + idx)
				option_text = str(text).strip()
				if option_text:
					option_lines.append(f"{label}) {option_text}")
			fallback_text = question
			if option_lines:
				fallback_text = f"{question}\n\nOptions:\n" + "\n".join(option_lines)
			if fallback_text.strip():
				entries.append({"index": 0, "title": "arc_question", "text": fallback_text.strip(), "is_supporting": True, "source": {"type": "question_options"}})

		return entries

	return entries


def _fallback_passage_from_entries(entries: list[dict[str, Any]]) -> str:
	if not entries:
		return ""
	return "\n\n".join((entry.get("text") or "").strip() for entry in entries if entry.get("text"))


def _normalize_decomposition(item: dict[str, Any]) -> list[dict[str, Any]]:
	raw_decomposition = item.get("question_decomposition", []) or []
	if not isinstance(raw_decomposition, list):
		return []

	if all(isinstance(step, dict) and "id" in step for step in raw_decomposition):
		raw_decomposition = sorted(raw_decomposition, key=lambda step: step.get("id"))

	normalized: list[dict[str, Any]] = []
	for step in raw_decomposition:
		if not isinstance(step, dict):
			continue
		normalized.append(
			{
				"id": step.get("id"),
				"question": (step.get("question") or "").strip(),
				"answer": (step.get("answer") or "").strip(),
				"paragraph_support_idx": _coerce_int(step.get("paragraph_support_idx")),
			}
		)
	return normalized


def _support_indices_from_decomposition(decomposition: list[dict[str, Any]]) -> list[int]:
	indices: set[int] = set()
	for step in decomposition:
		paragraph_idx = step.get("paragraph_support_idx")
		if paragraph_idx is None:
			continue
		coerced = _coerce_int(paragraph_idx)
		if coerced is not None:
			indices.add(coerced)
	return sorted(indices)


def _collect_refine_items(hypergraph: LocalHypergraph) -> tuple[list[dict[str, Any]], dict[int, ENT | None]]:
	items: list[dict[str, Any]] = []
	current_types: dict[int, ENT | None] = {}
	for idx, vertex in enumerate(hypergraph.vertices):
		if not _should_refine_vertex(vertex):
			continue
		old_type = vertex.type()
		current_types[idx] = old_type
		items.append(
			{
				"index": idx,
				"text": vertex.text(),
				"old_ent": old_type.name if old_type else None,
				"pos": [p.name for p in vertex.poses],
			}
		)
	return items, current_types


def _build_query_prompt(query_context: str, pending_items: list[dict[str, Any]]) -> str:
	return (
		"You are an expert entity-type refiner for query hypergraph vertices.\n"
		"Your task is to label only the query-side vertices.\n"
		"For each input vertex, assign exactly one ENT label.\n"
		"You must both fill missing labels and fix unreasonable existing labels when needed.\n"
		"Return strictly valid JSON only.\n\n"
		"Stage: query\n"
		"- Use the query context to infer the semantic role of each query vertex.\n"
		"- Do not rely on any data hypergraph information.\n\n"
		"Allowed ENT labels:\n"
		f"{_ent_definitions()}\n\n"
		"Query context:\n"
		f"{query_context}\n\n"
		"Query vertices to refine (JSON array):\n"
		f"{json.dumps(pending_items, ensure_ascii=False)}\n\n"
		"Output JSON schema:\n"
		"{\n"
		"  \"results\": [\n"
		"    {\"index\": 0, \"ent\": \"PERSON\"}\n"
		"  ]\n"
		"}\n"
		"Rules:\n"
		"1. Keep each index exactly from the input list.\n"
		"2. ent must be one of ENT enum names above.\n"
		"3. No extra commentary, markdown, or text outside JSON.\n"
		"4. If an existing type is wrong, output the corrected type."
	)


def _build_data_prompt(
	query_context: str,
	query_anchors: list[dict[str, Any]],
	data_context: str,
	pending_items: list[dict[str, Any]],
) -> str:
	anchor_texts = sorted(
		{
			str(anchor.get("text", "")).strip().lower(): str(anchor.get("text", "")).strip()
			for anchor in query_anchors
			if str(anchor.get("text", "")).strip()
		}.values()
	)
	return (
		"You are an expert entity-type refiner for data hypergraph vertices.\n"
		"Your task is to label only the data-side vertices.\n"
		"For each input vertex, assign exactly one ENT label.\n"
		"You must both fill missing labels and fix unreasonable existing labels when needed.\n"
		"The data hypergraph is meant to support the query. The query-side vertices are already refined and the data-side vertices should ideally align with the query when they refer to the same entity, event, or concept.\n"
		"Return strictly valid JSON only.\n\n"
		"Stage: data\n"
		"Use the already refined query-side types as the primary semantic reference.\n"
		"Prefer consistency with the query when the data node refers to the same entity, event, or concept.\n"
		"If the data context supports a different interpretation, choose the label that best fits the data context.\n\n"
		"Exact-text consistency rule:\n"
		f"If a data vertex text exactly matches any of these query texts, prefer the same ENT label unless the data context clearly contradicts it: {anchor_texts}\n\n"
		"Allowed ENT labels:\n"
		f"{_ent_definitions()}\n\n"
		"Query context:\n"
		f"{query_context}\n\n"
		"Query already labeled nodes (JSON array):\n"
		f"{json.dumps(query_anchors, ensure_ascii=False)}\n\n"
		"Data context:\n"
		f"{data_context}\n\n"
		"Data vertices to refine (JSON array):\n"
		f"{json.dumps(pending_items, ensure_ascii=False)}\n\n"
		"Output JSON schema:\n"
		"{\n"
		"  \"results\": [\n"
		"    {\"index\": 0, \"ent\": \"PERSON\"}\n"
		"  ]\n"
		"}\n"
		"Rules:\n"
		"1. Keep each index exactly from the input list.\n"
		"2. ent must be one of ENT enum names above.\n"
		"3. No extra commentary, markdown, or text outside JSON.\n"
		"4. If an existing type is wrong, output the corrected type.\n"
		"5. Use the query nodes only as context; do not copy their labels blindly when the data context conflicts.\n"
		"6. Align with the query-side types when there is a clear semantic match.\n"
		"7. If the same surface text appears in query and data, use the same ENT label unless the local data context strongly disagrees."
	)


def _canonical_query_type_map(query_anchors: list[dict[str, Any]]) -> dict[str, ENT]:
	canonical: dict[str, ENT] = {}
	for anchor in query_anchors:
		text = str(anchor.get("text", "")).strip().lower()
		ent_name = str(anchor.get("ent", "")).strip()
		if not text or not ent_name:
			continue
		try:
			canonical[text] = ENT.from_str(ent_name.upper())
		except Exception:
			continue
	return canonical


def _apply_canonical_text_types(hypergraph: LocalHypergraph, canonical_types: dict[str, ENT]) -> dict[int, ENT]:
	forced: dict[int, ENT] = {}
	if not canonical_types:
		return forced
	for idx, vertex in enumerate(hypergraph.vertices):
		if not _should_refine_vertex(vertex):
			continue
		text = vertex.text().strip().lower()
		if not text:
			continue
		canonical_ent = canonical_types.get(text)
		if canonical_ent is None:
			continue
		vertex.type_cache = canonical_ent
		forced[idx] = canonical_ent
	return forced


def _missing_refine_indices(items: list[dict[str, Any]], by_index: dict[int, ENT]) -> list[int]:
	return [int(item["index"]) for item in items if int(item["index"]) not in by_index]


def _build_retry_prompt(base_prompt: str, missing_indices: list[int], pending_items: list[dict[str, Any]]) -> str:
	missing_items = [item for item in pending_items if int(item["index"]) in missing_indices]
	return (
		base_prompt
		+ "\n\nSTRICT FOLLOW-UP:\n"
		+ "You omitted some required vertices in the previous answer. Return JSON for every input index.\n"
		+ f"Missing indices: {missing_indices}\n"
		+ f"Missing vertices: {json.dumps(missing_items, ensure_ascii=False)}\n"
		+ "Do not drop any item."
	)


class DashScopeChatClient:
	def __init__(self, model: str, base_url: str, api_key: str | None = None) -> None:
		resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY")
		if not resolved_key:
			raise EnvironmentError("Missing DASHSCOPE_API_KEY environment variable.")
		self._client = OpenAI(api_key=resolved_key, base_url=base_url)
		self.model = model

	def invoke(self, prompt: str) -> str:
		response = self._client.chat.completions.create(
			model=self.model,
			messages=[
				{"role": "system", "content": "You output only valid JSON."},
				{"role": "user", "content": prompt},
			],
			temperature=0.0,
			top_p=1.0,
			extra_body={"enable_thinking": False},
            # extra_headers={
            #     'X-DashScope-DataInspection': '{"input":"cip","output":"cip"}'
            # }
		)
		content = response.choices[0].message.content or ""
		if isinstance(content, list):
			return _normalize_response_content(content)
		return str(content)


def _is_data_inspection_failed(exc: Exception) -> bool:
	message = str(exc).lower()
	if "data_inspection_failed" in message:
		return True
	if "input text data may contain inappropriate content" in message:
		return True
	if getattr(exc, "code", None) == "data_inspection_failed":
		return True
	return False


def _is_bad_request_400(exc: Exception) -> bool:
	status_code = getattr(exc, "status_code", None)
	if status_code == 400:
		return True
	response = getattr(exc, "response", None)
	if getattr(response, "status_code", None) == 400:
		return True
	message = str(exc).lower()
	if "error code: 400" in message:
		return True
	if "<400>" in message:
		return True
	return False


def _llm_assign_types(
	client: DashScopeChatClient,
	prompt: str,
	pending_items: list[dict[str, Any]],
) -> dict[int, ENT] | None:
	if not pending_items:
		return {}

	try:
		content = _normalize_response_content(client.invoke(prompt))
	except Exception as exc:
		if _is_data_inspection_failed(exc) or _is_bad_request_400(exc):
			return None
		raise
	payload = _extract_json_payload(content)
	if payload is None:
		raise ValueError(f"LLM did not return valid JSON. raw={content[:500]}")

	by_index: dict[int, ENT] = {}
	for item in payload.get("results", []):
		if not isinstance(item, dict):
			continue
		raw_idx = item.get("index")
		raw_ent = item.get("ent")
		if not isinstance(raw_idx, (int, str)) or not isinstance(raw_ent, str):
			continue
		try:
			idx = int(raw_idx)
		except (TypeError, ValueError):
			continue
		by_index[idx] = ENT.from_str(raw_ent.strip().upper())
	return by_index


def _llm_assign_query_types(
	client: DashScopeChatClient,
	query_context: str,
	pending_items: list[dict[str, Any]],
) -> dict[int, ENT] | None:
	prompt = _build_query_prompt(query_context=query_context, pending_items=pending_items)
	by_index = _llm_assign_types(client=client, prompt=prompt, pending_items=pending_items)
	if by_index is None:
		return None
	missing_indices = _missing_refine_indices(pending_items, by_index)
	if missing_indices:
		retry_prompt = _build_retry_prompt(prompt, missing_indices, pending_items)
		retry_by_index = _llm_assign_types(client=client, prompt=retry_prompt, pending_items=pending_items)
		if retry_by_index is not None:
			by_index.update(retry_by_index)
	return by_index


def _llm_assign_data_types(
	client: DashScopeChatClient,
	query_context: str,
	query_anchors: list[dict[str, Any]],
	data_context: str,
	pending_items: list[dict[str, Any]],
) -> dict[int, ENT] | None:
	prompt = _build_data_prompt(
		query_context=query_context,
		query_anchors=query_anchors,
		data_context=data_context,
		pending_items=pending_items,
	)
	by_index = _llm_assign_types(client=client, prompt=prompt, pending_items=pending_items)
	if by_index is None:
		return None
	missing_indices = _missing_refine_indices(pending_items, by_index)
	if missing_indices:
		retry_prompt = _build_retry_prompt(prompt, missing_indices, pending_items)
		retry_by_index = _llm_assign_types(client=client, prompt=retry_prompt, pending_items=pending_items)
		if retry_by_index is not None:
			by_index.update(retry_by_index)
	return by_index


def refine_hypergraph_types(
	hypergraph: LocalHypergraph,
	passage: str,
	client: DashScopeChatClient,
	mode: str,
	query_context_text: str | None = None,
	query_anchors: list[dict[str, str]] | None = None,
) -> tuple[LocalHypergraph, dict[str, int]]:
	items, old_types = _collect_refine_items(hypergraph)
	if not items:
		return hypergraph, {"filled": 0, "fixed": 0, "unchanged": 0, "total": 0}

	if mode == "query":
		by_index = _llm_assign_query_types(
			client=client,
			query_context=passage,
			pending_items=items,
		)
	elif mode == "data":
		query_context = query_context_text or "\n".join(anchor.get("text", "") for anchor in query_anchors or [] if anchor.get("text"))
		by_index = _llm_assign_data_types(
			client=client,
			query_context=query_context,
			query_anchors=query_anchors or [],
			data_context=passage,
			pending_items=items,
		)
	else:
		raise ValueError(f"Unsupported refine mode: {mode}")
	if by_index is None:
		return hypergraph, {"filled": 0, "fixed": 0, "unchanged": 0, "total": len(items), "blocked": 1}

	if mode == "data" and query_anchors:
		canonical_types = _canonical_query_type_map(query_anchors)
		if canonical_types:
			forced = _apply_canonical_text_types(hypergraph, canonical_types)
			for idx, ent in forced.items():
				by_index[idx] = ent

	stats = {"filled": 0, "fixed": 0, "unchanged": 0, "total": len(items)}
	for item in items:
		idx = int(item["index"])
		old_type = old_types.get(idx)
		new_type = by_index.get(idx)
		if new_type is None:
			new_type = old_type or ENT.NOT_ENT
		vertex = hypergraph.vertices[idx]
		vertex.type_cache = new_type

		if old_type is None:
			stats["filled"] += 1
		elif old_type != new_type:
			stats["fixed"] += 1
		else:
			stats["unchanged"] += 1
	return hypergraph, stats


def build_query_anchors(hypergraph: LocalHypergraph) -> list[dict[str, str]]:
	anchors: list[dict[str, str]] = []
	for vertex in hypergraph.vertices:
		if not _should_refine_vertex(vertex):
			continue
		ent = vertex.type()
		if ent is None:
			continue
		anchors.append({"text": vertex.text(), "ent": ent.name})
	return anchors


def load_dataset_index(task: str, dataset_path: str) -> dict[str, dict[str, Any]]:
	items = load_task_dataset(task=task, dataset_path=dataset_path)
	result: dict[str, dict[str, Any]] = {}
	for item in items:
		if not isinstance(item, dict):
			continue
		question = _extract_question(item)
		if not question:
			continue
		if task in {"docnli", "econ", "contract_nli", "control"}:
			source_id = str(item.get("_id", "")).strip()
			instance_id = generate_instance_id(f"{source_id}:{question}")
		else:
			instance_id = generate_instance_id(question)
		if instance_id not in result:
			result[instance_id] = item
	return result


def _refine_one_instance(
	instance_dir: Path,
	item: dict[str, Any],
	task: str,
	dataset_path: str,
	model: str,
	base_url: str,
	max_data_graphs: int | None,
	save: bool,
) -> dict[str, Any]:
	client = DashScopeChatClient(model=model, base_url=base_url)
	return refine_instance_directory(
		instance_dir=instance_dir,
		item=item,
		task=task,
		client=client,
		model=model,
		base_url=base_url,
		dataset_path=dataset_path,
		max_data_graphs=max_data_graphs,
		save=save,
		progress_ui=None,
	)


def _normalize_metadata(
	task: str,
	item: dict[str, Any],
	instance_id: str,
	dataset_path: str,
	data_entries: list[dict[str, Any]],
) -> dict[str, Any]:
	premise_chunks = item.get("premise_chunks") or item.get("context_docs") or []
	if not isinstance(premise_chunks, list):
		premise_chunks = []
	return {
		"instance_id": instance_id,
		"task": task,
		"question": _extract_question(item),
		"answer": (item.get("answer") or "").strip(),
		"premise": (item.get("premise") or "").strip(),
		"premise_chunks": premise_chunks,
		"chunked": bool(item.get("chunked", False)),
		"num_chunks": item.get("num_chunks", len(premise_chunks)),
		"dataset_path": dataset_path,
		"data_entries": data_entries,
	}


def _build_data_refine_report(
	idx: int,
	data_path: Path,
	entry: dict[str, Any] | None,
	stats: dict[str, int],
) -> dict[str, Any]:
	entry = entry or {}
	return {
		"index": idx,
		"file": data_path.name,
		"text": (entry.get("text") or "").strip(),
		"title": (entry.get("title") or "").strip(),
		"is_supporting": bool(entry.get("is_supporting", False)),
		"source": entry.get("source", {}),
		"refine_stats": stats,
	}


def _update_metadata(
	task: str,
	metadata: dict[str, Any],
	item: dict[str, Any],
	instance_id: str,
	data_entries: list[dict[str, Any]],
	query_file: str,
	data_files: list[str],
	query_stats: dict[str, Any],
	data_reports: list[dict[str, Any]],
	query_anchor_count: int,
	model: str,
	base_url: str,
	dataset_path: str,
) -> dict[str, Any]:
	normalized = _normalize_metadata(
		task=task,
		item=item,
		instance_id=instance_id,
		dataset_path=dataset_path,
		data_entries=data_entries,
	)
	updated = dict(metadata)
	updated.update(normalized)
	updated["refined"] = True
	updated["refine_completed"] = True
	updated["refine_status"] = "completed"
	updated["refine_at"] = datetime.now(timezone.utc).isoformat()
	updated["refine_model"] = model
	updated["refine_base_url"] = base_url
	updated["refine_summary"] = {
		"query": query_stats,
		"data": data_reports,
		"query_anchor_count": query_anchor_count,
	}
	updated["data_hypergraphs"] = data_reports
	updated["saved_data_hypergraphs"] = len(data_reports)
	updated.setdefault("files", {})
	updated["files"]["query"] = query_file
	updated["files"]["data"] = data_files
	return updated


def refine_instance_directory(
	instance_dir: Path,
	item: dict[str, Any],
	task: str,
	client: DashScopeChatClient,
	model: str,
	base_url: str,
	dataset_path: str,
	max_data_graphs: int | None = None,
	save: bool = True,
	progress_ui: RefinementProgressUI | None = None,
) -> dict[str, Any]:
	query_path = _find_query_file(instance_dir)
	metadata_path = instance_dir / "metadata.json"
	if query_path is None:
		raise FileNotFoundError(f"Missing query pkl in: {instance_dir}")
	if not metadata_path.exists():
		raise FileNotFoundError(f"Missing metadata json: {metadata_path}")

	metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
	query_hg = LocalHypergraph.load(str(query_path))
	query_text = (_extract_question(item) or (metadata.get("question") or "").strip())
	data_paths = _list_data_files(instance_dir)
	if max_data_graphs is not None:
		data_paths = data_paths[:max_data_graphs]
	data_entries = _build_data_entries(task=task, item=item)
	fallback_passage = _fallback_passage_from_entries(data_entries)
	if progress_ui is not None:
		progress_ui.set_current_instance(instance_dir.name, len(data_paths))

	query_hg, query_stats = refine_hypergraph_types(
		hypergraph=query_hg,
		passage=query_text,
		client=client,
		mode="query",
		query_context_text=None,
		query_anchors=None,
	)
	if query_stats.get("blocked"):
		if progress_ui is not None:
			progress_ui.mark_query_done(query_stats)
		updated_metadata = _update_metadata(
			task=task,
			metadata=metadata,
			item=item,
			instance_id=instance_dir.name,
			data_entries=data_entries,
			query_file=query_path.name,
			data_files=[p.name for p in data_paths],
			query_stats={**query_stats, "error": "data_inspection_failed"},
			data_reports=[],
			query_anchor_count=0,
			model=model,
			base_url=base_url,
			dataset_path=dataset_path,
		)
		updated_metadata["refine_status"] = "skipped_by_content_inspection"
		updated_metadata["refine_error"] = "query_rejected_by_provider_content_inspection"
		updated_metadata["problematic_data_points"] = [
			{
				"type": "query",
				"text": query_text,
				"reason": "400_bad_request",
			}
		]
		if save:
			metadata_path.write_text(json.dumps(updated_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
		return updated_metadata
	if progress_ui is not None:
		progress_ui.mark_query_done(query_stats)
	query_anchors = build_query_anchors(query_hg)
	if save:
		query_hg.save(str(query_path))

	data_reports: list[dict[str, Any]] = []

	for data_path in data_paths:
		data_idx = _data_file_index(data_path)
		entry = data_entries[data_idx] if 0 <= data_idx < len(data_entries) else None
		passage = (entry or {}).get("text", "") if isinstance(entry, dict) else ""
		if not passage:
			passage = fallback_passage
		if progress_ui is not None:
			progress_ui.mark_data_start(data_idx, data_path.name)
		data_hg = LocalHypergraph.load(str(data_path))
		data_hg, data_stats = refine_hypergraph_types(
			hypergraph=data_hg,
			passage=passage,
			client=client,
			mode="data",
			query_context_text=query_text,
			query_anchors=query_anchors,
		)
		if data_stats.get("blocked"):
			data_stats = {"filled": 0, "fixed": 0, "unchanged": 0, "total": 0, "blocked": 1}
			data_reports.append(
				{
					"index": data_idx,
					"file": data_path.name,
					"text": passage,
					"title": (entry.get("title") if isinstance(entry, dict) else "") or "",
					"is_supporting": bool(entry.get("is_supporting", False)) if isinstance(entry, dict) else False,
					"source": entry.get("source", {}) if isinstance(entry, dict) else {},
					"refine_stats": {**data_stats, "error": "data_inspection_failed"},
					"skipped": True,
				}
			)
			metadata.setdefault("problematic_data_points", [])
			metadata["problematic_data_points"].append(
				{
					"type": "data",
					"index": data_idx,
					"file": data_path.name,
					"text": passage,
					"reason": "400_bad_request",
				}
			)
			if progress_ui is not None:
				progress_ui.mark_data_done(data_stats)
			continue
		data_reports.append(_build_data_refine_report(data_idx, data_path, entry if isinstance(entry, dict) else None, data_stats))
		if progress_ui is not None:
			progress_ui.mark_data_done(data_stats)
		if save:
			data_hg.save(str(data_path))

	updated_metadata = _update_metadata(
		task=task,
		metadata=metadata,
		item=item,
		instance_id=instance_dir.name,
		data_entries=data_entries,
		query_file=query_path.name,
		data_files=[p.name for p in data_paths],
		query_stats=query_stats,
		data_reports=data_reports,
		query_anchor_count=len(query_anchors),
		model=model,
		base_url=base_url,
		dataset_path=dataset_path,
	)
	if metadata.get("problematic_data_points"):
		updated_metadata["problematic_data_points"] = metadata["problematic_data_points"]
	if save:
		metadata_path.write_text(json.dumps(updated_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

	return updated_metadata


def refine_batch(
	task: str,
	source_root: str,
	output_root: str,
	dataset_path: str,
	model: str = DEFAULT_MODEL,
	base_url: str = DEFAULT_BASE_URL,
	max_data_graphs: int | None = None,
	save: bool = True,
	force: bool = False,
	progress_file: str | None = None,
	max_workers: int = 8,
) -> dict[str, Any]:
	source = Path(source_root)
	output = Path(output_root)
	if not source.exists():
		raise FileNotFoundError(f"Source root not found: {source}")
	output.mkdir(parents=True, exist_ok=True)

	dataset_index = load_dataset_index(task=task, dataset_path=dataset_path)
	instance_ids = sorted(dataset_index.keys())
	if not instance_ids:
		raise ValueError(f"No valid questions found from dataset: {dataset_path}")
	_validate_source_alignment(source_root=source, dataset_instance_ids=instance_ids, task=task)

	progress_path = Path(progress_file) if progress_file else output / f"refine_progress_{task}.json"
	tracker = BatchProgressTracker(total_instances=len(instance_ids), progress_path=progress_path)
	instance_bar = tqdm(total=len(instance_ids), desc="instances", unit="inst", dynamic_ncols=True)

	results: list[dict[str, Any]] = []
	def _submit_instance(instance_id: str) -> tuple[str, dict[str, Any], bool, str | None]:
		tracker.start(instance_id)
		item = dataset_index[instance_id]
		source_dir = source / instance_id
		if not source_dir.exists():
			return instance_id, {"instance_id": instance_id, "status": "skipped", "reason": "source_hypergraph_not_found"}, True, "source_hypergraph_not_found"

		output_dir = output / instance_id
		if not output_dir.exists():
			_normalize_output_instance_files(source_dir=source_dir, output_dir=output_dir)
		else:
			_ensure_output_instance_canonical_names(output_dir)

		metadata_path = output_dir / "metadata.json"
		if not metadata_path.exists():
			return instance_id, {"instance_id": instance_id, "status": "skipped", "reason": "missing_metadata"}, True, "missing_metadata"
		metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
		if metadata.get("refined") and not force:
			return instance_id, {"instance_id": instance_id, "status": "skipped", "reason": "already_refined"}, True, "already_refined"

		try:
			updated_metadata = _refine_one_instance(
				instance_dir=output_dir,
				item=item,
				task=task,
				dataset_path=dataset_path,
				model=model,
				base_url=base_url,
				max_data_graphs=max_data_graphs,
				save=save,
			)
		except Exception as exc:
			if _is_data_inspection_failed(exc) or _is_bad_request_400(exc):
				return instance_id, {"instance_id": instance_id, "status": "skipped", "reason": "bad_request_400"}, True, "bad_request_400"
			raise
		return (
			instance_id,
			{
				"instance_id": instance_id,
				"status": "updated",
				"query_refine": updated_metadata.get("refine_summary", {}).get("query", {}),
				"data_count": len(updated_metadata.get("data_hypergraphs", [])),
			},
			False,
			None,
		)

	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		future_map = {
			executor.submit(_submit_instance, instance_id): instance_id
			for instance_id in instance_ids
		}
		for future in as_completed(future_map):
			instance_id_ref = future_map[future]
			try:
				instance_id, result_item, skipped, reason = future.result()
			except Exception as exc:
				instance_id = instance_id_ref
				result_item = {
					"instance_id": instance_id,
					"status": "failed",
					"reason": f"{type(exc).__name__}: {exc}",
				}
				skipped = False
				reason = "failed"
			results.append(result_item)
			tracker.finish(instance_id, result_item["status"], reason)
			instance_bar.update(1)
			instance_bar.set_postfix(updated=tracker.updated, skipped=tracker.skipped, failed=tracker.failed, refresh=False)

	instance_bar.close()

	return {
		"task": task,
		"source_root": str(source.resolve()),
		"output_root": str(output.resolve()),
		"dataset_path": str(Path(dataset_path).resolve()) if Path(dataset_path).exists() else dataset_path,
		"model": model,
		"base_url": base_url,
		"progress_file": str(progress_path.resolve()),
		"max_workers": max_workers,
		"total_instances": len(instance_ids),
		"processed": len([item for item in results if item.get("status") == "updated"]),
		"skipped": len([item for item in results if item.get("status") == "skipped"]),
		"failed": len([item for item in results if item.get("status") == "failed"]),
		"results": results,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Refine hypergraph entity types with DashScope OpenAI-compatible API.")
	parser.add_argument("--task", type=str, required=True, choices=["hotpotqa", "legalbench", "musique", "multihop", "arc", "docnli", "econ", "contract_nli", "control"], help="Dataset task.")
	parser.add_argument("--source-root", type=str, default=None, help="Source hypergraph root. Defaults to data/debug/<task>.")
	parser.add_argument("--output-root", type=str, default=None, help="Output root. Defaults to source-root (in-place refine).")
	parser.add_argument("--dataset-path", type=str, required=True, help="Original dataset path (file or directory).")
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
	parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
	parser.add_argument("--max-data-graphs", type=int, default=None)
	parser.add_argument("--no-save", action="store_true", help="Run refinement without writing files back.")
	parser.add_argument("--force", action="store_true", help="Refine instances even if metadata already says refined.")
	parser.add_argument("--progress-file", type=str, default=None, help="Write live progress JSON here. Defaults to <output-root>/refine_progress_<task>.json.")
	parser.add_argument("--max-workers", type=int, default=8, help="Parallel instance workers to run at once.")
	args = parser.parse_args()

	default_task_root = TASK_DEFAULT_SOURCE_ROOTS.get(args.task, Path("data/debug") / args.task)
	source_root = Path(args.source_root) if args.source_root else default_task_root
	source_root = _resolve_task_source_root(args.task, source_root)

	output_root = Path(args.output_root) if args.output_root else source_root
	output_root = _resolve_task_source_root(args.task, output_root)

	summary = refine_batch(
		task=args.task,
		source_root=str(source_root),
		output_root=str(output_root),
		dataset_path=args.dataset_path,
		model=args.model,
		base_url=args.base_url,
		max_data_graphs=args.max_data_graphs,
		save=not args.no_save,
		force=args.force,
		progress_file=args.progress_file,
		max_workers=args.max_workers,
	)
	print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
