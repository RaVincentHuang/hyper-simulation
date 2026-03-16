from hyper_simulation.hypergraph.hypergraph import Hyperedge, Hypergraph, Node, Vertex, LocalDoc
from typing import Dict, List, Set, Tuple, Optional, Union
from hyper_simulation.component.nli import get_nli_labels_with_score_batch, get_nli_label
from hyper_simulation.hypergraph.dependency import Entity, QueryType
from hyper_simulation.hypergraph.hypergraph import Hyperedge, Hypergraph, Node, Vertex, LocalDoc
from hyper_simulation.hypergraph.dependency import Pos, Dep, Entity
from hyper_simulation.utils.log import getLogger

def is_not_denial_with_score_batch(vertices_pairs: list[tuple[Vertex, Vertex]]) -> List[Tuple[bool, float]]:

    text_pairs = [(v1.text(), v2.text()) for v1, v2 in vertices_pairs]
    
    labels_with_score = get_nli_labels_with_score_batch(text_pairs)
    
    results: list[tuple[bool, float]] = []
    for (label, score), (v1, v2) in zip(labels_with_score, vertices_pairs):
        if label == "entailment" or (label == "neutral" and v1.is_domain(v2)):
            results.append((True, score))
        else:
            results.append((False, score))
    
    return results

def get_matched_vertices(vertices1: list[Vertex], vertices2: list[Vertex]) -> dict[Vertex, set[Tuple[Vertex, float]]]:
    matched_vertices: dict[Vertex, set[Tuple[Vertex, float]]] = {}
    vertices_pairs: list[tuple[Vertex, Vertex]] = []
    for v1 in vertices1:
        if v1.is_virtual():
            continue
        for v2 in vertices2:
            if v2.is_virtual():
                continue
            vertices_pairs.append((v1, v2))
    match_with_score = is_not_denial_with_score_batch(vertices_pairs)
    for (v1, v2), (is_not_denial, score) in zip(vertices_pairs, match_with_score):
        if is_not_denial:
            if v1 not in matched_vertices:
                matched_vertices[v1] = set()
            matched_vertices[v1].add((v2, score))
    return matched_vertices

def get_top_k_matched_vertices(matched_vertices: dict[Vertex, set[Tuple[Vertex, float]]], k: int) -> dict[Vertex, set[Tuple[Vertex, float]]]:
    top_k_matched_vertices: dict[Vertex, set[Tuple[Vertex, float]]] = {}
    for v1, matches in matched_vertices.items():
        sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
        top_k_matches = sorted_matches[:k]
        top_k_matched_vertices[v1] = set(top_k_matches)
    return top_k_matched_vertices

def compute_allowed_pairs(
    query_vertices: Dict[int, Vertex],
    data_vertices: Dict[int, Vertex]
) -> Set[Tuple[int, int]]:
    """
    批量计算 allowed pairs，并输出详细日志（含未匹配的 Q/D 文本）
    """
    logger = getLogger("denial_comment")
    
    if not query_vertices or not data_vertices:
        if logger:
            logger.info("Empty query or data vertices. Returning empty allowed set.")
        return set()

    allowed: Set[Tuple[int, int]] = set()
    allowed_logs: List[str] = []
    contradicted_logs: List[str] = []

    for q_id, q_vertex in query_vertices.items():
        for d_id, d_vertex in data_vertices.items():
            is_allowed, reason = denial_comment(q_vertex, d_vertex)
            qt = q_vertex.text()
            dt = d_vertex.text()
            
            log_entry = f"Q{q_id}: '{qt}' vs D{d_id}: '{dt}' (reason: {reason})"
            if is_allowed:
                allowed.add((q_id, d_id))
                allowed_logs.append(log_entry)
            else:
                contradicted_logs.append(log_entry)

    # === 全量日志输出（无截断）===
    if logger is not None:
        total = len(query_vertices) * len(data_vertices)
        logger.info(f"Total Q-D pairs processed: {total}")
        logger.info(f"Allowed pairs count: {len(allowed_logs)}")
        logger.info(f"Contradicted pairs count: {len(contradicted_logs)}")

        if allowed_logs:
            logger.info("=== BEGIN ALLOWED PAIRS ===")
            for idx, log in enumerate(allowed_logs, start=1):
                logger.info(f"[ALLOWED {idx}] {log}")
            logger.info("=== END ALLOWED PAIRS ===")
        else:
            logger.info("No allowed pairs.")

        if contradicted_logs:
            logger.info("=== BEGIN CONTRADICTED PAIRS ===")
            for idx, log in enumerate(contradicted_logs, start=1):
                logger.info(f"[CONTRADICTED {idx}] {log}")
            logger.info("=== END CONTRADICTED PAIRS ===")
        else:
            logger.info("No contradicted pairs.")

    return allowed

def denial_comment(u: Vertex, v: Vertex) -> Tuple[bool, str]:
    """
    返回 (是否非冲突, 原因说明)
    """
    ut = u.text().strip()
    vt = v.text().strip()
    
    # 1. 空文本检查
    if not ut or not vt:
        return False, "Empty text"
    
    # 2. 基于 QueryType 的疑问词处理
    if getattr(u, 'is_query', False):
        query_types = {n.query_type for n in u.nodes if hasattr(n, 'query_type') and n.query_type}
        if query_types:
            matched_ent = []
            for qtype in query_types:
                if qtype == QueryType.PERSON and v.ent_range(Entity.PERSON):
                    matched_ent.append("PERSON")
                elif qtype == QueryType.TIME and (v.ent_range(Entity.DATE) or v.ent_range(Entity.TIME)):
                    matched_ent.append("DATE/TIME")
                elif qtype == QueryType.LOCATION and (v.ent_range(Entity.GPE) or v.ent_range(Entity.LOC)):
                    matched_ent.append("GPE/LOC")
                elif qtype == QueryType.NUMBER and (v.ent_range(Entity.CARDINAL) or v.ent_range(Entity.QUANTITY) or v.pos_range(Pos.NUM)):
                    matched_ent.append("CARDINAL/QUANTITY/NUM")
                elif qtype == QueryType.BELONGS and (v.ent_range(Entity.PERSON) or v.ent_range(Entity.ORG) or v.ent_range(Entity.GPE)):
                    matched_ent.append("PERSON/ORG/GPE")
                elif qtype in {QueryType.WHAT, QueryType.WHICH} and (any(e != Entity.NOT_ENTITY for e in v.ents) or v.pos_range(Pos.NOUN) or v.pos_range(Pos.PROPN)):
                    matched_ent.append("NON_EMPTY_ENTITY_OR_NOUN")
                elif qtype == QueryType.ATTRIBUTE and (v.pos_range(Pos.ADJ) or v.pos_range(Pos.ADV)):
                    matched_ent.append("ADJ/ADV")
                elif qtype == QueryType.REASON and not v.pos_equal(Pos.PUNCT):
                    matched_ent.append("NON_PUNCT")
            
            if matched_ent:
                qtype_names = [str(qt).split('.')[-1] for qt in query_types]
                return True, f"QueryType={qtype_names} → matched Data entity types: {matched_ent}"
            else:
                data_ents = [str(e).split('.')[-1] for e in v.ents if e != Entity.NOT_ENTITY]
                data_poses = [str(p).split('.')[-1] for p in v.poses]
                qtype_names = [str(qt).split('.')[-1] for qt in query_types]
                return False, f"QueryType={qtype_names} → Data has ents={data_ents}, poses={data_poses}"


    # 3. 同类型实体豁免
    u_has_ent = any(e != Entity.NOT_ENTITY for e in u.ents)
    v_has_ent = any(e != Entity.NOT_ENTITY for e in v.ents)
    if u_has_ent and v_has_ent:
        entity_groups = [
            ({Entity.PERSON, Entity.NORP}, "PERSON_GROUP"),
            ({Entity.GPE, Entity.LOC, Entity.FAC, Entity.ORG, Entity.NORP}, "LOCATION_ORG_GROUP"),
            ({Entity.DATE, Entity.TIME}, "TIME_GROUP"),
            ({Entity.PRODUCT, Entity.WORK_OF_ART}, "PRODUCT_GROUP"),
            ({Entity.MONEY, Entity.PERCENT, Entity.QUANTITY, Entity.CARDINAL}, "NUMBER_GROUP"),
            ({Entity.EVENT, Entity.LAW, Entity.LANGUAGE}, "EVENT_GROUP"),
        ]
        for group_entities, group_name in entity_groups:
            u_in_group = any(u.ent_range(e) for e in group_entities)
            v_in_group = any(v.ent_range(e) for e in group_entities)
            if u_in_group and v_in_group:
                return True, f"Entity-compatible: both in {group_name}"

        u_ents = [str(e).split('.')[-1] for e in u.ents if e != Entity.NOT_ENTITY]
        v_ents = [str(e).split('.')[-1] for e in v.ents if e != Entity.NOT_ENTITY]
        return False, f"Entity-mismatch: Query ents={u_ents} vs Data ents={v_ents}"

    # 4. NLI 矛盾检测
    label = get_nli_label(ut, vt)
    if label != "contradiction":
        return True, f"NLI={label} (non-contradiction)"
    else:
        return False, f"NLI={label} (contradiction)"