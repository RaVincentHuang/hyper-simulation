from pathlib import Path
from typing import Any

import json

from langchain_ollama import ChatOllama

from hyper_simulation.hypergraph.entity import ENT
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex


def _should_refine_vertex(vertex: Vertex) -> bool:
	return not (
		vertex.is_query()
		or vertex.is_verb()
		or vertex.is_virtual()
		or vertex.is_adjective()
		or vertex.is_adverb()
	)


def _ent_definitions() -> str:
	return "\n".join([
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
		"GROUP: Collection of individuals likes a family, team, class, or social group.",
		"FEATURE: Distinctive attribute, property, or characteristic of an entity or concept.",
		"ECONOMIC: Economic entity, such as a market, industry, or economic concept.",
		"SOCIOLOGY: Concepts related to society, culture, sociology, or social interactions.",
		"PHENOMENON: Natural or social phenomenon, such as climate change or cultural trend.",
		"ACTION: Action, behavior, or process not covered by the above categories.",
		"NOT_ENT: Use this if it does not fit any category above.",
	])


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
	# Prefer strict parse first.
	try:
		parsed = json.loads(content)
		if isinstance(parsed, dict):
			return parsed
	except Exception:
		pass

	# Fallback: extract first JSON object boundaries.
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


def _llm_assign_types(
	llm: ChatOllama,
	passage: str,
	pending_items: list[dict[str, Any]],
	mode: str,
	query_anchors: list[dict[str, str]] | None = None,
) -> dict[int, ENT]:
	if not pending_items:
		return {}

	anchor_block = ""
	if query_anchors:
		anchor_block = (
			"Query anchor entities and their refined types (for alignment):\n"
			f"{json.dumps(query_anchors, ensure_ascii=False)}\n\n"
		)

	prompt = (
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

	response = llm.invoke(prompt)
	raw_content = response.content if hasattr(response, "content") else str(response)
	content = _normalize_response_content(raw_content)
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


def refine_hypergraph_types(
	hypergraph: LocalHypergraph,
	passage: str,
	llm: ChatOllama,
	mode: str,
	query_anchors: list[dict[str, str]] | None = None,
) -> tuple[LocalHypergraph, dict[str, int]]:
	items, old_types = _collect_refine_items(hypergraph)
	if not items:
		return hypergraph, {"filled": 0, "fixed": 0, "unchanged": 0, "total": 0}

	by_index = _llm_assign_types(
		llm=llm,
		passage=passage,
		pending_items=items,
		mode=mode,
		query_anchors=query_anchors,
	)

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


def load_musique_case(json_path: str) -> tuple[str, list[str], list[int]]:
	path = Path(json_path)
	if not path.exists():
		raise FileNotFoundError(f"MuSiQue input file not found: {path}")

	raw = json.loads(path.read_text(encoding="utf-8"))
	item = raw[0] if isinstance(raw, list) else raw
	if not isinstance(item, dict):
		raise ValueError("Expected a JSON object (or a list whose first element is object).")

	query = item.get("question", "")
	paragraphs = item.get("paragraphs", [])
	dataset = [p.get("paragraph_text", "") for p in paragraphs if isinstance(p, dict)]

	supports_set: set[int] = set()
	for step in item.get("question_decomposition", []):
		if not isinstance(step, dict):
			continue
		paragraph_idx = step.get("paragraph_support_idx")
		if paragraph_idx is None:
			continue
		try:
			supports_set.add(int(paragraph_idx))
		except (TypeError, ValueError):
			continue

	supports = sorted(supports_set)
	return query, dataset, supports


def run_refine_pipeline(
	debug_dir: str,
	musique_json_path: str,
	max_data_graphs: int = 20,
	save: bool = True,
) -> None:
	base = Path(debug_dir)
	query_path = base / "query_hypergraph.pkl"
	if not query_path.exists():
		raise FileNotFoundError(f"Missing query pkl: {query_path}")

	query_hg = LocalHypergraph.load(str(query_path))
	query_text, data_texts, _ = load_musique_case(musique_json_path)

	print("[REFINE] Initializing LLM qwen3.5:9b...")
	llm = ChatOllama(model="qwen3.5:9b", top_p=0.95, reasoning=False, temperature=0.0)

	print("[REFINE] Step 1/2: refining query hypergraph (fill + fix)")
	query_hg, query_stats = refine_hypergraph_types(
		hypergraph=query_hg,
		passage=query_text,
		llm=llm,
		mode="query",
		query_anchors=None,
	)
	print(f"[REFINE][QUERY] stats={query_stats}")

	anchors = build_query_anchors(query_hg)
	print(f"[REFINE][QUERY] built {len(anchors)} type anchors for data alignment")

	if save:
		query_hg.save(str(query_path))
		print(f"[REFINE][QUERY] saved -> {query_path}")

	print("[REFINE] Step 2/2: refining each data hypergraph with query alignment")
	for i in range(max_data_graphs):
		data_path = base / f"data_hypergraph{i}.pkl"
		if not data_path.exists():
			print(f"[REFINE][DATA-{i}] missing, skip")
			continue

		data_hg = LocalHypergraph.load(str(data_path))
		passage = data_texts[i] if i < len(data_texts) else ""
		data_hg, stats = refine_hypergraph_types(
			hypergraph=data_hg,
			passage=passage,
			llm=llm,
			mode="data",
			query_anchors=anchors,
		)
		print(f"[REFINE][DATA-{i}] stats={stats}")

		if save:
			data_hg.save(str(data_path))
			print(f"[REFINE][DATA-{i}] saved -> {data_path}")


if __name__ == "__main__":
	run_refine_pipeline(
		debug_dir="logs/debugs",
		musique_json_path="/home/vincent/.dataset/musique/x.json",
		max_data_graphs=20,
		save=True,
	)
