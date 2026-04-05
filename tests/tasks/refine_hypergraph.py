from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from hyper_simulation.hypergraph.entity import ENT
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex


DEFAULT_MODEL = "qwen3.5-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


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


def _coerce_int(value: Any) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except (TypeError, ValueError):
		return None


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


def _build_prompt(
	passage: str,
	pending_items: list[dict[str, Any]],
	mode: str,
	query_anchors: list[dict[str, str]] | None = None,
) -> str:
	anchor_block = ""
	if query_anchors:
		anchor_block = (
			"Query anchor entities and their refined types (for alignment):\n"
			f"{json.dumps(query_anchors, ensure_ascii=False)}\n\n"
		)

	return (
		"You are an expert entity-type refiner for hypergraph vertices.\n"
		"For each input vertex, assign exactly one ENT label.\n"
		"You must both fill missing labels and fix unreasonable existing labels when needed.\n"
		"Return strictly valid JSON only.\n\n"
		f"Mode: {mode}\n"
		"- mode=query: refine query hypergraph vertex types according to question intent.\n"
		"- mode=data: refine data hypergraph types and align with query anchors when entities are similar, conflicting, or in the same semantic scope.\n\n"
		"Allowed ENT labels:\n"
		f"{_ent_definitions()}\n\n"
		"Context passage:\n"
		f"{passage}\n\n"
		f"{anchor_block}"
		"Vertices to refine (JSON array):\n"
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
		"5. In data mode, prefer consistency with query anchors whenever semantically justified."
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
	passage: str,
	pending_items: list[dict[str, Any]],
	mode: str,
	query_anchors: list[dict[str, str]] | None = None,
) -> dict[int, ENT] | None:
	if not pending_items:
		return {}

	prompt = _build_prompt(
		passage=passage,
		pending_items=pending_items,
		mode=mode,
		query_anchors=query_anchors,
	)
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


def refine_hypergraph_types(
	hypergraph: LocalHypergraph,
	passage: str,
	client: DashScopeChatClient,
	mode: str,
	query_anchors: list[dict[str, str]] | None = None,
) -> tuple[LocalHypergraph, dict[str, int]]:
	items, old_types = _collect_refine_items(hypergraph)
	if not items:
		return hypergraph, {"filled": 0, "fixed": 0, "unchanged": 0, "total": 0}

	by_index = _llm_assign_types(
		client=client,
		passage=passage,
		pending_items=items,
		mode=mode,
		query_anchors=query_anchors,
	)
	if by_index is None:
		return hypergraph, {"filled": 0, "fixed": 0, "unchanged": 0, "total": len(items), "blocked": 1}

	stats = {"filled": 0, "fixed": 0, "unchanged": 0, "total": len(items)}
	for item in items:
		idx = int(item["index"])
		old_type = old_types.get(idx)
		new_type = by_index.get(idx, old_type or ENT.NOT_ENT)
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


def load_dataset_index(dataset_path: str, target_ids: set[str]) -> dict[str, dict[str, Any]]:
	path = Path(dataset_path)
	if not path.exists():
		raise FileNotFoundError(f"Dataset file not found: {path}")

	jsonl_paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
	if not jsonl_paths:
		raise FileNotFoundError(f"No jsonl files found under: {path}")

	result: dict[str, dict[str, Any]] = {}
	for jsonl_path in jsonl_paths:
		with jsonl_path.open("r", encoding="utf-8") as fin:
			for line in fin:
				line = line.strip()
				if not line:
					continue
				item = json.loads(line)
				if not isinstance(item, dict):
					continue
				question = (item.get("question") or "").strip()
				if not question:
					continue
				instance_id = generate_instance_id(question)
				if instance_id in target_ids:
					result[instance_id] = item
					if len(result) == len(target_ids):
						return result
	return result


def _normalize_metadata(item: dict[str, Any], instance_id: str, dataset_path: str) -> dict[str, Any]:
	paragraphs = item.get("paragraphs", []) or []
	decomposition = _normalize_decomposition(item)
	support_indices = _support_indices_from_decomposition(decomposition)
	answer = (item.get("answer") or "").strip()
	answer_aliases = item.get("answer_aliases", []) or item.get("answer_alias", []) or []
	if not isinstance(answer_aliases, list):
		answer_aliases = []

	normalized_paragraphs: list[dict[str, Any]] = []
	for idx, paragraph in enumerate(paragraphs):
		if not isinstance(paragraph, dict):
			continue
		paragraph_text = (paragraph.get("paragraph_text") or "").strip()
		paragraph_title = (paragraph.get("title") or "").strip()
		is_supporting = bool(paragraph.get("is_supporting", False))
		related_steps = [
			step
			for step in decomposition
			if _coerce_int(step.get("paragraph_support_idx")) == idx
		]
		normalized_paragraphs.append(
			{
				"index": idx,
				"title": paragraph_title,
				"text": paragraph_text,
				"is_supporting": is_supporting,
				"paragraph_support_idx": [idx] if idx in support_indices else [],
				"supporting_subquestions": [
					{
						"id": step.get("id"),
						"question": step.get("question", ""),
						"answer": step.get("answer", ""),
						"paragraph_support_idx": step.get("paragraph_support_idx"),
					}
					for step in related_steps
				],
			}
		)

	return {
		"instance_id": instance_id,
		"question": (item.get("question") or "").strip(),
		"final_answer": answer,
		"answer": answer,
		"answer_aliases": answer_aliases,
		"question_decomposition": decomposition,
		"question_decomposition_answers": [step.get("answer", "") for step in decomposition],
		"paragraph_support_idx": support_indices,
		"paragraphs": normalized_paragraphs,
		"dataset_path": dataset_path,
	}


def _build_data_refine_report(
	idx: int,
	data_path: Path,
	paragraph: dict[str, Any] | None,
	stats: dict[str, int],
) -> dict[str, Any]:
	paragraph = paragraph or {}
	return {
		"index": idx,
		"file": data_path.name,
		"text": (paragraph.get("paragraph_text") or paragraph.get("text") or "").strip(),
		"title": (paragraph.get("title") or "").strip(),
		"is_supporting": bool(paragraph.get("is_supporting", False)),
		"paragraph_support_idx": [idx] if bool(paragraph.get("is_supporting", False)) else [],
		"refine_stats": stats,
	}


def _update_metadata(
	metadata: dict[str, Any],
	item: dict[str, Any],
	instance_id: str,
	query_stats: dict[str, Any],
	data_reports: list[dict[str, Any]],
	query_anchor_count: int,
	model: str,
	base_url: str,
	dataset_path: str,
) -> dict[str, Any]:
	normalized = _normalize_metadata(item=item, instance_id=instance_id, dataset_path=dataset_path)
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
	updated["files"]["query"] = "query_hypergraph.pkl"
	updated["files"]["data"] = [report["file"] for report in data_reports]
	return updated


def refine_instance_directory(
	instance_dir: Path,
	item: dict[str, Any],
	client: DashScopeChatClient,
	model: str,
	base_url: str,
	dataset_path: str,
	max_data_graphs: int | None = None,
	save: bool = True,
	progress_ui: RefinementProgressUI | None = None,
) -> dict[str, Any]:
	query_path = instance_dir / "query_hypergraph.pkl"
	metadata_path = instance_dir / "metadata.json"
	if not query_path.exists():
		raise FileNotFoundError(f"Missing query pkl: {query_path}")
	if not metadata_path.exists():
		raise FileNotFoundError(f"Missing metadata json: {metadata_path}")

	metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
	query_hg = LocalHypergraph.load(str(query_path))
	query_text = (item.get("question") or metadata.get("question") or "").strip()
	data_paths = sorted(instance_dir.glob("data_hypergraph*.pkl"), key=_sorted_index_from_name)
	if max_data_graphs is not None:
		data_paths = data_paths[:max_data_graphs]
	if progress_ui is not None:
		progress_ui.set_current_instance(instance_dir.name, len(data_paths))

	query_hg, query_stats = refine_hypergraph_types(
		hypergraph=query_hg,
		passage=query_text,
		client=client,
		mode="query",
		query_anchors=None,
	)
	if query_stats.get("blocked"):
		if progress_ui is not None:
			progress_ui.mark_query_done(query_stats)
		updated_metadata = _update_metadata(
			metadata=metadata,
			item=item,
			instance_id=instance_dir.name,
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

	data_reports: list[dict[str, Any]] = []
	paragraphs = item.get("paragraphs", []) or []

	for data_path in data_paths:
		match = re.fullmatch(r"data_hypergraph(\d+)\.pkl", data_path.name)
		if match is None:
			continue
		data_idx = int(match.group(1))
		paragraph = paragraphs[data_idx] if data_idx < len(paragraphs) else {}
		passage = (paragraph.get("paragraph_text") if isinstance(paragraph, dict) else "") or ""
		if progress_ui is not None:
			progress_ui.mark_data_start(data_idx, data_path.name)
		data_hg = LocalHypergraph.load(str(data_path))
		data_hg, data_stats = refine_hypergraph_types(
			hypergraph=data_hg,
			passage=passage,
			client=client,
			mode="data",
			query_anchors=query_anchors,
		)
		if data_stats.get("blocked"):
			data_stats = {"filled": 0, "fixed": 0, "unchanged": 0, "total": 0, "blocked": 1}
			data_reports.append(
				{
					"index": data_idx,
					"file": data_path.name,
					"text": passage,
					"title": (paragraph.get("title") if isinstance(paragraph, dict) else "") or "",
					"is_supporting": bool(paragraph.get("is_supporting", False)) if isinstance(paragraph, dict) else False,
					"paragraph_support_idx": [data_idx] if isinstance(paragraph, dict) and bool(paragraph.get("is_supporting", False)) else [],
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
		data_reports.append(_build_data_refine_report(data_idx, data_path, paragraph if isinstance(paragraph, dict) else None, data_stats))
		if progress_ui is not None:
			progress_ui.mark_data_done(data_stats)
		if save:
			data_hg.save(str(data_path))

	updated_metadata = _update_metadata(
		metadata=metadata,
		item=item,
		instance_id=instance_dir.name,
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
	instances_root: str,
	dataset_path: str,
	model: str = DEFAULT_MODEL,
	base_url: str = DEFAULT_BASE_URL,
	max_data_graphs: int | None = None,
	save: bool = True,
	force: bool = False,
	progress_file: str | None = None,
) -> dict[str, Any]:
	root = Path(instances_root)
	if not root.exists():
		raise FileNotFoundError(f"Instances root not found: {root}")

	instance_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
	if not instance_dirs:
		raise FileNotFoundError(f"No instance directories found under: {root}")

	target_ids = {instance_dir.name for instance_dir in instance_dirs}
	dataset_index = load_dataset_index(dataset_path=dataset_path, target_ids=target_ids)
	client = DashScopeChatClient(model=model, base_url=base_url)
	progress_path = Path(progress_file) if progress_file else root / "refine_progress.json"
	progress_ui = RefinementProgressUI(total_instances=len(instance_dirs), progress_path=progress_path)

	results: list[dict[str, Any]] = []
	try:
		for instance_dir in instance_dirs:
			metadata_path = instance_dir / "metadata.json"
			query_path = instance_dir / "query_hypergraph.pkl"
			if not metadata_path.exists() or not query_path.exists():
				results.append(
					{
						"instance_id": instance_dir.name,
						"status": "skipped",
						"reason": "missing_files",
					}
				)
				progress_ui.finish_instance(skipped=True)
				continue

			metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
			if metadata.get("refined") and not force:
				results.append(
					{
						"instance_id": instance_dir.name,
						"status": "skipped",
						"reason": "already_refined",
					}
				)
				progress_ui.finish_instance(skipped=True)
				continue

			item = dataset_index.get(instance_dir.name)
			if item is None:
				question = (metadata.get("question") or "").strip()
				if question:
					fallback_id = generate_instance_id(question)
					item = dataset_index.get(fallback_id)
			if item is None:
				results.append(
					{
						"instance_id": instance_dir.name,
						"status": "skipped",
						"reason": "dataset_item_not_found",
					}
				)
				progress_ui.finish_instance(skipped=True)
				continue

			try:
				updated_metadata = refine_instance_directory(
					instance_dir=instance_dir,
					item=item,
					client=client,
					model=model,
					base_url=base_url,
					dataset_path=dataset_path,
					max_data_graphs=max_data_graphs,
					save=save,
					progress_ui=progress_ui,
				)
			except Exception as exc:
				if _is_data_inspection_failed(exc) or _is_bad_request_400(exc):
					results.append(
						{
							"instance_id": instance_dir.name,
							"status": "skipped",
							"reason": "bad_request_400",
						}
					)
					progress_ui.finish_instance(skipped=True)
					continue
				raise
			results.append(
				{
					"instance_id": instance_dir.name,
					"status": "updated",
					"query_refine": updated_metadata.get("refine_summary", {}).get("query", {}),
					"data_count": len(updated_metadata.get("data_hypergraphs", [])),
				}
			)
			progress_ui.finish_instance(skipped=False)
	finally:
		progress_ui.close()

	return {
		"instances_root": str(root.resolve()),
		"dataset_path": str(Path(dataset_path).resolve()) if Path(dataset_path).exists() else dataset_path,
		"model": model,
		"base_url": base_url,
		"progress_file": str(progress_path.resolve()),
		"total_instances": len(instance_dirs),
		"processed": len([item for item in results if item.get("status") == "updated"]),
		"skipped": len([item for item in results if item.get("status") == "skipped"]),
		"results": results,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Refine MuSiQue hypergraph types with DashScope OpenAI-compatible API.")
	parser.add_argument("--instances-root", type=str, required=True, help="Root directory that contains instance folders.")
	parser.add_argument("--dataset-path", type=str, required=True, help="Original dataset jsonl path.")
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
	parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
	parser.add_argument("--max-data-graphs", type=int, default=None)
	parser.add_argument("--no-save", action="store_true", help="Run refinement without writing files back.")
	parser.add_argument("--force", action="store_true", help="Refine instances even if metadata already says refined.")
	parser.add_argument("--progress-file", type=str, default=None, help="Write live progress JSON here. Defaults to <instances-root>/refine_progress.json.")
	args = parser.parse_args()

	summary = refine_batch(
		instances_root=args.instances_root,
		dataset_path=args.dataset_path,
		model=args.model,
		base_url=args.base_url,
		max_data_graphs=args.max_data_graphs,
		save=not args.no_save,
		force=args.force,
		progress_file=args.progress_file,
	)
	print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
