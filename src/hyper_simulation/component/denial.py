from hyper_simulation.hypergraph.hypergraph import Hyperedge, Hypergraph, Node, Vertex, LocalDoc
from typing import Dict, List, Set, Tuple, Optional, Union
from hyper_simulation.component.nli import get_nli_labels_with_score_batch, get_nli_label, get_nli_labels_batch
from hyper_simulation.hypergraph.linguistic import Entity, QueryType
from hyper_simulation.hypergraph.entity import ENT
from hyper_simulation.hypergraph.hypergraph import Hyperedge, Hypergraph, Node, Vertex, LocalDoc
from hyper_simulation.hypergraph.linguistic import Pos, Dep
from hyper_simulation.utils.log import getLogger


def _strict_type_group(vertex: Vertex) -> str:
    if vertex.is_verb():
        return "VERB_GROUP"

    ent_type = vertex.type()

    if ent_type in {ENT.LOC, ENT.ORG, ENT.FAC, ENT.GPE, ENT.NORP, ENT.GROUP, ENT.COUNTRY}:
        return "LOC_ORG_GROUP"
    if ent_type == ENT.PERSON:
        return "PERSON_GROUP"
    if ent_type in {ENT.PRODUCT, ENT.WORK_OF_ART}:
        return "PRODUCT_GROUP"
    if ent_type == ENT.OCCUPATION:
        return "OCCUPATION_GROUP"
    if ent_type == ENT.NUMBER:
        return "NUMBER_GROUP"
    if ent_type == ENT.TEMPORAL:
        return "TEMPORAL_GROUP"
    if ent_type in {ENT.CONCEPT, ENT.THEORY, ENT.FEATURE, ENT.PHENOMENON}:
        return "ABSTRACT_GROUP"
    if ent_type == ENT.EVENT:
        return "EVENT_GROUP"

    if ent_type is not None and ent_type != ENT.NOT_ENT:
        return f"ENT::{ent_type.name}"

    if vertex.is_adjective():
        return "NO_ENT_ADJ_GROUP"
    if vertex.is_adverb():
        return "NO_ENT_ADV_GROUP"

    return "NO_ENT_OTHER"


def _hard_type_match_only(u: Vertex, v: Vertex) -> Tuple[bool, str]:
    ut = u.text().strip()
    vt = v.text().strip()

    if not ut or not vt:
        return False, "Empty text"

    if u.is_virtual() or v.is_virtual():
        return False, "Virtual vertex is not matchable in hard mode"

    query_type = u.query_type()
    if u.is_query() and query_type:
        v_type = v.type()
        if query_type == QueryType.PERSON and v_type == ENT.PERSON:
            return True, "QueryType=PERSON → matched Data entity type: PERSON"
        if query_type == QueryType.TIME and v_type == ENT.TEMPORAL:
            return True, "QueryType=TIME → matched Data entity type: TEMPORAL"
        if query_type == QueryType.LOCATION and v_type in {ENT.GPE, ENT.LOC, ENT.FAC, ENT.ORG, ENT.COUNTRY}:
            return True, "QueryType=LOCATION → matched Data entity type: GPE/LOC/FAC/ORG/COUNTRY"
        if query_type == QueryType.NUMBER and v_type == ENT.NUMBER:
            return True, "QueryType=NUMBER → matched Data entity type: NUMBER"
        if query_type == QueryType.BELONGS and v_type in {ENT.PERSON, ENT.ORG, ENT.GPE, ENT.COUNTRY, ENT.GROUP}:
            return True, "QueryType=BELONGS → matched Data entity type: PERSON/ORG/GPE/COUNTRY/GROUP"
        if query_type in {QueryType.WHAT, QueryType.WHICH} and (
            (v_type is not None and v_type != ENT.NOT_ENT) or v.pos_range(Pos.NOUN) or v.pos_range(Pos.PROPN)
        ):
            return True, "QueryType=WHAT/WHICH → matched Data entity type: NON_EMPTY_TYPE_OR_NOUN"
        if query_type == QueryType.ATTRIBUTE and (v.pos_range(Pos.ADJ) or v.pos_range(Pos.ADV)):
            return True, "QueryType=ATTRIBUTE → matched Data entity type: ADJ/ADV"
        if query_type == QueryType.REASON and not v.pos_equal(Pos.PUNCT):
            return True, "QueryType=REASON → matched Data entity type: NON_PUNCT"

        return False, f"QueryType={query_type} → Data type={v_type} does not match hard rules"

    u_group = _strict_type_group(u)
    v_group = _strict_type_group(v)
    if u_group != v_group:
        return False, f"Hard type-group mismatch: {u_group} != {v_group}"

    return True, f"Hard type-group matched: {u_group}"

def is_not_denial_with_score_batch(vertices_pairs: list[tuple[Vertex, Vertex]]) -> List[Tuple[bool, float]]:

    text_pairs = [(v2.text(), v1.text()) for v1, v2 in vertices_pairs]
    
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

def compute_allowed_pairs_batch(
    query_vertices: Dict[int, Vertex],
    data_vertices: Dict[int, Vertex]
) -> Set[Tuple[int, int]]:
    """
    批量计算 allowed pairs，使用 get_nli_labels_batch 提升吞吐效率
    
    步骤：
    1. 构建所有 Q-D 顶点对和对应的文本对
    2. 批量调用 get_nli_labels_batch 获取所有 NLI 标签
    3. 使用 denial_comment_by_label 替代逐个调用 denial_comment
    4. 输出详细日志
    """
    logger = getLogger("denial_comment")
    
    if not query_vertices or not data_vertices:
        if logger:
            logger.info("Empty query or data vertices. Returning empty allowed set.")
        return set()

    # 步骤1：构建所有顶点对和文本对
    pairs_metadata: List[Tuple[int, Vertex, int, Vertex]] = []  # (q_id, q_vertex, d_id, d_vertex)
    text_pairs: List[Tuple[str, str]] = []

    for q_id, q_vertex in query_vertices.items():
        for d_id, d_vertex in data_vertices.items():
            pairs_metadata.append((q_id, q_vertex, d_id, d_vertex))
            text_pairs.append((d_vertex.text(), q_vertex.text()))

    # 步骤2：批量获取所有 NLI 标签
    nli_labels = get_nli_labels_batch(text_pairs)

    # 步骤3：使用标签批量处理每一对
    allowed: Set[Tuple[int, int]] = set()
    allowed_logs: List[str] = []
    contradicted_logs: List[str] = []

    for (q_id, q_vertex, d_id, d_vertex), nli_label in zip(pairs_metadata, nli_labels):
        # is_allowed, reason = denial_comment_by_label(q_vertex, d_vertex, nli_label)
        is_allowed, reason = denial_comment_by_label_hard(q_vertex, d_vertex, nli_label)  # 仍然调用原函数以保留 QueryType 和实体类型的判断逻辑
        qt = q_vertex.text()
        dt = d_vertex.text()
        if is_allowed:
            # print(f"✅ ALLOWED: {q_vertex.text()} <-> {d_vertex.text()}")
        
        log_entry = f"Q{q_id}: '{qt}' vs D{d_id}: '{dt}' (reason: {reason})"
        if is_allowed:
            allowed.add((q_id, d_id))
            allowed_logs.append(log_entry)
        else:
            contradicted_logs.append(log_entry)

    # 步骤4：全量日志输出（无截断）
    if logger is not None:
        total = len(pairs_metadata)
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
    
def denial_comment_by_label(u: Vertex, v: Vertex, label: str) -> Tuple[bool, str]:
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
            if u_in_group and v_in_group and label != "contradiction":
                return True, f"Entity-compatible: both in {group_name}"

        u_ents = [str(e).split('.')[-1] for e in u.ents if e != Entity.NOT_ENTITY]
        v_ents = [str(e).split('.')[-1] for e in v.ents if e != Entity.NOT_ENTITY]
        return False, f"Entity-mismatch: Query ents={u_ents} vs Data ents={v_ents}"

    # 4. NLI 矛盾检测
    if label != "contradiction":
        return True, f"NLI={label} (non-contradiction)"
    else:
        return False, f"NLI={label} (contradiction)"
    
    
def denial_comment_by_label_hard(u: Vertex, v: Vertex, label: str) -> Tuple[bool, str]:
    """
    返回 (是否匹配, 原因说明)

    hard 规则：
    1) 虚节点直接不匹配。
    2) Query 节点先按 QueryType 规则匹配。
    3) 非 Query 节点按严格类型分组匹配。
    4) 以上任意匹配路径都必须满足 NLI != contradiction 。
    """
    ut = u.text().strip()
    vt = v.text().strip()
    
    if ut == "contestant" and vt == "Chris Daughtry":
        # print(f"Debug: Special case - allowing 'contestant' [{u.type()}] <-> 'Chris Daughtry' [{v.type()}] regardless of NLI result")

    if not ut or not vt:
        return False, "Empty text"

    if u.is_virtual() or v.is_virtual():
        return False, "Virtual vertex is not matchable in hard mode"

    type_match, type_reason = _hard_type_match_only(u, v)
    if not type_match:
        return False, type_reason

    if label != "contradiction":
        return True, f"Hard mode requires not contradiction, got NLI={label}"

    return False, f"Hard mode blocked by NLI={label}"


def compute_allowed_pairs_batch_with_score(
    query_vertices: Dict[int, Vertex],
    data_vertices: Dict[int, Vertex]
) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    批量计算 allowed pairs，并返回每对的匹配置信度（基于 NLI 结果的 score）

    先按 hard 规则预筛类型，只对可能通过 `denial_comment_by_label_hard` 的 pair
    调用 `get_nli_labels_with_score_batch`，减少 NLI 计算开销。
    """
    logger = getLogger("denial_comment")
    
    if not query_vertices or not data_vertices:
        if logger:
            logger.info("Empty query or data vertices. Returning empty allowed set.")
        return set(), {}

    candidate_pairs_metadata: List[Tuple[int, Vertex, int, Vertex]] = []  # (q_id, q_vertex, d_id, d_vertex)
    candidate_text_pairs: List[Tuple[str, str]] = []
    filtered_by_type_count = 0

    for q_id, q_vertex in query_vertices.items():
        for d_id, d_vertex in data_vertices.items():
            type_match, _ = _hard_type_match_only(q_vertex, d_vertex)
            if not type_match:
                filtered_by_type_count += 1
                continue

            candidate_pairs_metadata.append((q_id, q_vertex, d_id, d_vertex))
            candidate_text_pairs.append((d_vertex.text(), q_vertex.text()))

    if not candidate_pairs_metadata:
        if logger is not None:
            total = len(query_vertices) * len(data_vertices)
            logger.info(f"Total Q-D pairs processed: {total}")
            logger.info(f"Type-filtered pairs count: {filtered_by_type_count}")
            logger.info("No candidate pairs survived hard type prefiltering.")
        return set(), {}

    # print(f"Debug: {len(candidate_text_pairs)} candidate pairs after hard type prefiltering, will call NLI batch API")
    nli_labels_with_score = get_nli_labels_with_score_batch(candidate_text_pairs)

    allowed: Set[Tuple[int, int]] = set()
    confidence_scores: Dict[Tuple[int, int], float] = {}
    contradicted_count = 0

    for (q_id, q_vertex, d_id, d_vertex), (nli_label, nli_score) in zip(candidate_pairs_metadata, nli_labels_with_score):
        is_allowed = nli_label != "contradiction"
        reason = f"NLI={nli_label} (non-contradiction)" if is_allowed else f"NLI={nli_label} (contradiction)"
        qt = q_vertex.text()
        dt = d_vertex.text()
        
        log_entry = f"Q{q_id}: '{qt}' vs D{d_id}: '{dt}' (reason: {reason}, NLI={nli_label}, score={nli_score:.4f})"
        if is_allowed:
            # print(f"✅ ALLOWED: {q_vertex.text()} <-> {d_vertex.text()} with confidence {nli_score:.4f}")
            allowed.add((q_id, d_id))
            confidence_scores[(q_id, d_id)] = nli_score
        else:
            contradicted_count += 1
            # print(f"❌ CONTRADICTED: {q_vertex.text()} <-> {d_vertex.text()} with confidence {nli_score:.4f}")

    # 全量日志输出
    if logger is not None:
        total = len(query_vertices) * len(data_vertices)
        logger.info(f"Total Q-D pairs processed: {total}")
        logger.info(f"Type-filtered pairs count: {filtered_by_type_count}")
        logger.info(f"Candidate pairs count for NLI: {len(candidate_pairs_metadata)}")
        logger.info(f"Allowed pairs count: {len(allowed)}")
        logger.info(f"Contradicted pairs count: {contradicted_count}")
        for idx, ((q_id, d_id), score) in enumerate(confidence_scores.items(), start=1):
            q_vertex = query_vertices[q_id]
            d_vertex = data_vertices[d_id]
            logger.info(f"[ALLOWED {idx}] Q{q_id}: '{q_vertex.text()}' <-> D{d_id}: '{d_vertex.text()}' with confidence {score:.4f}")
    
    return allowed, confidence_scores

def get_top_k_matched_vertices_by_scores(query_vertices: Dict[int, Vertex], data_vertices: Dict[int, Vertex], confidence_scores: Dict[Tuple[int, int], float], k: int) -> Dict[Vertex, Set[Tuple[Vertex, float]]]:
    """
    从 confidence_scores 中获取每个 query vertex 的 top-k 匹配 data vertices
    返回格式：{q_id: [(d_id, score), ...]}
    """
    top_k_matches: Dict[Vertex, List[Tuple[Vertex, float]]] = {}
    for (q_id, d_id), score in confidence_scores.items():
        q_vertex = query_vertices[q_id]
        d_vertex = data_vertices[d_id]
        if q_vertex not in top_k_matches:
            top_k_matches[q_vertex] = []
        top_k_matches[q_vertex].append((d_vertex, score))
    
    result: Dict[Vertex, Set[Tuple[Vertex, float]]] = {}
    
    for q_vertex, matches in top_k_matches.items():
        matches.sort(key=lambda x: x[1], reverse=True)  # 按 score 降序排序
        result[q_vertex] = set(matches[:k])  # 取 top-k
    
    return result