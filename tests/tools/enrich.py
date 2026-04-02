import hashlib
from pathlib import Path
from typing import List, Tuple, Set, Dict

from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex, Hyperedge
from hyper_simulation.hypergraph.linguistic import Entity, Pos, Dep
from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.embedding import get_embedding_batch, cosine_similarity
from hyper_simulation.utils.log import getLogger
from tqdm import tqdm
from hyper_simulation.utils.log import current_query_id
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.component.nli import init_nli_model
from hyper_simulation.component.embedding import init_embedding_model
from hyper_simulation.hypergraph.entity import ENT
from langchain_ollama import ChatOllama
import json
import time

def enrich_hypergraph(hypergraph: LocalHypergraph, text: str) -> LocalHypergraph:
    """
    对超图进行丰富化处理，添加类型信息等
    """
    logger = getLogger(__name__)
    
    pending_vertices: list[Vertex] = []
    pending_items: list[dict[str, str | int]] = []
    vertex_by_idx: dict[int, Vertex] = {}  # 记录索引与顶点的映射

    for idx, vertex in enumerate(hypergraph.vertices):
        if vertex.is_query() or vertex.is_verb() or vertex.is_virtual() or vertex.is_adjective() or vertex.is_adverb():
            continue
        if vertex.type() is not None:
            continue
        pending_vertices.append(vertex)
        pending_items.append({"index": idx, "text": vertex.text()})
        vertex_by_idx[idx] = vertex

    if not pending_vertices:
        print("[ENRICH] No vertices need enrichment")
        return hypergraph
    
    print(f"[ENRICH] Found {len(pending_vertices)} vertices to enrich: {[v.text() for v in pending_vertices]}")

    ent_definitions = "\n".join([
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
        "SOCIOLOGY: Concepts related to society, culture, sociology, or social interactions, such as social media, social movement, or social issue.",
        "PHENOMENON: Natural or social phenomenon, such as climate change, pandemic, or cultural trend.",
        "ACTION: Action, behavior, or process, such as a specific activity, event, or process that is not covered by the above categories.",
        "NOT_ENT: Use this if it does not fit any category above.",
    ])

    prompt = (
        "You are an expert entity classifier for hypergraph vertices.\n"
        "Given one context passage and a list of vertex texts, assign exactly one ENT label to each vertex.\n"
        "Return strictly valid JSON only.\n\n"
        "Allowed ENT labels:\n"
        f"{ent_definitions}\n\n"
        "Context passage:\n"
        f"{text}\n\n"
        "Vertices to classify (JSON array):\n"
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
        "3. No extra commentary, markdown, or text outside JSON."
    )

    try:
        print("[ENRICH] Connecting to qwen3.5:9b LLM...")
        llm = ChatOllama(model="qwen3.5:9b", top_p=0.95, reasoning=False, temperature=0.0)
        response = llm.invoke(prompt)
        raw_content = response.content if hasattr(response, "content") else str(response)
        print(f"[ENRICH] LLM raw response type: {type(raw_content).__name__}, first 500 chars: {str(raw_content)[:500]}")
        
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            content = "\n".join(
                item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                for item in raw_content
            )
        else:
            content = str(raw_content)
        
        print(f"[ENRICH] Parsing JSON... content length: {len(content)}")
        payload = json.loads(content)
        print(f"[ENRICH] JSON payload keys: {list(payload.keys()) if isinstance(payload, dict) else 'not a dict'}")
    except Exception as exc:
        print(f"[ENRICH] ERROR: Failed to enrich vertex types with LLM: {exc}")
        import traceback
        traceback.print_exc()
        return hypergraph

    by_index: dict[int, ENT] = {}
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            print(f"[ENRICH] WARNING: Skipping non-dict result item: {item}")
            continue
        raw_idx = item.get("index")
        raw_ent = item.get("ent")
        if not isinstance(raw_idx, (int, str)):
            print(f"[ENRICH] WARNING: Skipping invalid index type: {type(raw_idx)} for {raw_ent}")
            continue
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            print(f"[ENRICH] WARNING: Could not convert index to int: {raw_idx}")
            continue
        if not isinstance(raw_ent, str):
            print(f"[ENRICH] WARNING: Skipping non-string ent: {type(raw_ent)} at index {idx}")
            continue
        ent = ENT.from_str(raw_ent.strip().upper())
        by_index[idx] = ent
        v = vertex_by_idx.get(idx)
        vtext = v.text() if v else "?"
        print(f"[ENRICH] Mapped index {idx} ('{vtext}') -> {ent.name}")
    
    print(f"[ENRICH] Successfully mapped {len(by_index)} vertices")

    for vertex_item in pending_items:
        idx = int(vertex_item["index"])
        ent = by_index.get(idx, ENT.NOT_ENT)
        vertex = hypergraph.vertices[idx]
        vertex.type_cache = ent
        print(f"[ENRICH] Set type_cache for vertex {idx} '{vertex.text()}' to {ent.name}")

    return hypergraph

def load_musique_case(json_path: str) -> tuple[str, list[str], list[int]]:
    """
    Load one MuSiQue item and map fields:
    - query <- question
    - dataset <- paragraphs[*].paragraph_text
    - supports <- question_decomposition[*].paragraph_support_idx
    """
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

if __name__ == "__main__":
    path: str = 'logs/debugs'
    query_path = f"{path}/query_hypergraph.pkl"
    query_hg = LocalHypergraph.load(query_path)
    data_hgs: list[LocalHypergraph | None] = []
    
    query, valid_texts, _ = load_musique_case(f"/home/vincent/.dataset/musique/x.json")
    
    for i in range(20):  # 假设最多有 20 个相关
        data_path = f"{path}/data_hypergraph{i}.pkl"
        if Path(data_path).exists():
            data_hgs.append(LocalHypergraph.load(data_path))
        else:
            data_hgs.append(None)
    
    # Dump query hypergraph
    print("Query Hypergraph:")
    print(query)
    print(f"Vertices ({len(query_hg.vertices)}):")
    # for i, v in enumerate(query_hg.vertices):
    #     if v.is_query():
    #         print(f"  - [{i}] '{v.text()}' TYPE: {v.query_type()}")
    #         continue
    #     print(f"  - [{i}] '{v.text()}' TYPE: {v.type()}")
    indexes = []
    for i, v in enumerate(query_hg.vertices):
        if v.is_query() or v.is_verb() or v.is_virtual() or v.is_adjective() or v.is_adverb():
            continue
        if not v.type():
            print(f"[{i}] '{v.text()}'")
            for node in v.nodes:
                print(f"    - '{node.text}' (pos={node.pos.name})")
            indexes.append(i)  # 只在v.type()为None时添加一次
    q_hd = enrich_hypergraph(query_hg, query)
    print(f"\nEnriched Vertices for Query Hypergraph:")
    for i in indexes:
        v = q_hd.vertices[i]
        print(f"[{i}] '{v.text()}' TYPE: {v.type()}")
    
    for idx, data_hg in enumerate(data_hgs):
        if data_hg is None:
            print(f"\nData Hypergraph {idx}: MISSING")
            continue
        print(f"\nData Hypergraph {idx}:")
        print(valid_texts[idx])
        print(f"Vertices ({len(data_hg.vertices)}):")
        # for i, v in enumerate(data_hg.vertices):
        #     print(f"  - [{i}] '{v.text()}' TYPE: {v.type()}")
        indexes = []
        for i, v in enumerate(data_hg.vertices):
            if v.is_query() or v.is_verb() or v.is_virtual() or v.is_adjective() or v.is_adverb():
                continue
            if not v.type():
                print(f"[{i}] '{v.text()}'")
                for node in v.nodes:
                    print(f"    - '{node.text}' (pos={node.pos.name})")
                indexes.append(i)  # 只在v.type()为None时添加一次
        hg = enrich_hypergraph(data_hg, valid_texts[idx])
        print(f"\nEnriched Vertices for Data Hypergraph {idx}:")
        for i in indexes:
            v = hg.vertices[i]
            print(f"[{i}] '{v.text()}' TYPE: {v.type()}")
