from hmac import new
from typing import List, Dict, Set, Tuple, Any, Optional
import itertools
from collections import defaultdict

from hyper_simulation.hypergraph.hypergraph import Hypergraph, Vertex, Hyperedge
from hyper_simulation.hypergraph.linguistic import QueryType, Pos, Tag, Dep, Entity
from hyper_simulation.component.nli import get_nli_labels_batch
from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.utils.log import getLogger
from hyper_simulation.component.postprocess import post_detection, get_simulation_slice
logger = getLogger(__name__)

class UnionFind:
    def __init__(self, elements):
        # 使用 Python 的 id() 函数作为键，避免 Vertex.__hash__ 冲突
        self.parent = {id(e): id(e) for e in elements}
        self.id_to_vertex = {id(e): e for e in elements}

    def find(self, item):
        item_id = id(item)
        if self.parent[item_id] == item_id:
            return item_id
        self.parent[item_id] = self.find(self.id_to_vertex[self.parent[item_id]])
        return self.parent[item_id]

    def union(self, item1, item2):
        root1_id = self.find(item1)
        root2_id = self.find(item2)
        if root1_id != root2_id:
            self.parent[root1_id] = root2_id
    
    def get_vertex(self, item_id):
        """通过 id 获取 vertex 对象"""
        return self.id_to_vertex.get(item_id)

class MultiHopFusion:
    def __init__(self):
        self.fusion_logger = getLogger("merge_hypergraph")
        self.consistent_logger = getLogger("consistency")

    def _are_entities_compatible(self, v1: Vertex, v2: Vertex) -> bool:
        """
        检查两个顶点是否可以进行 NLI 合并。
        """
        def get_main_ent(vertex: Vertex) -> Entity:
            # 优先返回非 NOT_ENTITY 的类型
            for e in vertex.ents:
                if e != Entity.NOT_ENTITY:
                    return e
            return Entity.NOT_ENTITY
        ent1 = get_main_ent(v1)
        ent2 = get_main_ent(v2)
        # 两个实体类型不一致都认为错误（包括一个是实体一个不是实体）
        if ent1 != ent2:
            return False
        # 两个实体类型一致并且都是实体
        if ent1 != Entity.NOT_ENTITY:
            return True
        # 两个都不是实体
        has_propn_1 = any(n.pos == Pos.PROPN for n in v1.nodes)
        has_propn_2 = any(n.pos == Pos.PROPN for n in v2.nodes)
        if has_propn_1 and has_propn_2:
            return True
        return False

    def _is_critical_node(self, vertex: Vertex) -> bool:
        """
        判断是否为需要追踪的关键 query 节点
        标准: 包含实体 (非 NOT_ENTITY) 或 是 NOUN/PROPN
        """
        return (
            any(e != Entity.NOT_ENTITY for e in vertex.ents) or
            any(n.pos in {Pos.NOUN, Pos.PROPN} for n in vertex.nodes)
        )

    def _estimate_match_confidence(self, q_v: Vertex, d_v: Vertex) -> float:
        """
        启发式估计匹配置信度 (0.0 ~ 1.0)
        
        评分规则:
        - 文本精确匹配: +0.5
        - 实体类型一致: +0.3
        - POS 兼容: +0.2
        """
        score = 0.0
        
        # 1. 文本精确匹配
        if q_v.text().strip() == d_v.text().strip():
            score += 0.5
        
        # 2. 实体类型一致
        if q_v.ent_same(d_v):
            score += 0.3
        
        # 3. POS 兼容
        if q_v.pos_same(d_v):
            score += 0.2
        
        return min(1.0, score)

    def _extract_supporting_span(self, merged_vertex: Vertex, evidence_hg: Hypergraph, evidence_text: str) -> str:
        """
        从原始 evidence 中提取与 merged_vertex 对应的具体文本跨度
        
        策略优先级:
        1. 如果 merged_vertex 的 node 直接来自该 evidence，返回 covered_sentence
        2. 在 evidence_text 中搜索 merged_vertex.text() 的子串，返回上下文
        3. 兜底: 返回 evidence_text 的前 100 字符
        """
        # 策略 1: 通过 node 的文档/句子引用定位 (如果数据结构支持)
        for node in merged_vertex.nodes:
            # 假设 Node 有 doc_id 或 sentence_id 能关联到原始 evidence
            if hasattr(node, 'doc_id') and hasattr(evidence_hg.doc, 'id') and node.doc_id == evidence_hg.doc.id:
                return node.covered_sentence or node.text or ""
        
        # 策略 2: 文本子串匹配 + 上下文提取
        merged_text = merged_vertex.text().strip()
        if merged_text and merged_text in evidence_text:
            idx = evidence_text.find(merged_text)
            # 提取 ±50 字符的上下文
            start = max(0, idx - 50)
            end = min(len(evidence_text), idx + len(merged_text) + 50)
            span = evidence_text[start:end].strip()
            # 限制长度，避免过长
            return span if len(span) <= 200 else span[:200] + "..."
        
        # 策略 3: 兜底返回摘要
        return evidence_text[:100] + "..." if len(evidence_text) > 100 else evidence_text

    def _reverse_trace_consistency(
        self,
        query_hg: Hypergraph,
        merged_hg: Hypergraph,
        evidence_hgs: List[Hypergraph],
        evidence_texts: List[str],
        q_map: Dict[int, Vertex],
        d_map: Dict[int, Vertex],
        mapping: Dict[int, Set[int]],
        provenance: Dict[int, Set[int]]
    ) -> Dict[str, List[Dict]]:
        """
        核心反向溯源: query → merged vertex → original evidence
        
        返回格式:
        {
            "query_text": [
                {
                    'matched': bool,
                    'evidence_idx': int | None,
                    'supporting_text': str | None,
                    'confidence': float,
                    'merged_vertex_text': str
                },
                ...
            ]
        }
        """
        # 辅助映射: merged_vertex.id → {evidence_indices}
        merged_to_sources = {v_id: srcs for v_id, srcs in provenance.items()}
        
        # 结果容器
        query_to_evidence_details: Dict[str, List[Dict]] = defaultdict(list)
        
        # 反向映射: query_vertex.id → sim_id
        q_vertex_to_sim_id = {v.id: k for k, v in q_map.items()}
        
        for q_v in query_hg.vertices:
            # 只追踪关键节点
            if not self._is_critical_node(q_v):
                continue
            
            q_text = q_v.text()
            sim_id = q_vertex_to_sim_id.get(q_v.id)
            d_sim_ids = mapping.get(sim_id, set()) if sim_id is not None else set()
            
            # 情况 1: 未匹配到任何 merged vertex
            if not d_sim_ids:
                query_to_evidence_details[q_text].append({
                    'matched': False,
                    'evidence_idx': None,
                    'supporting_text': None,
                    'confidence': 0.0,
                    'merged_vertex_text': None
                })
                continue
            
            # 情况 2: 已匹配，追溯每个 matched merged vertex 的来源
            for d_sim_id in d_sim_ids:
                d_v_merged = d_map.get(d_sim_id)
                if not d_v_merged:
                    continue
                
                # 🔑 关键: 通过 provenance 找到原始 evidence indices
                source_indices = merged_to_sources.get(d_v_merged.id, set())
                
                for src_idx in source_indices:
                    # 边界检查
                    if not (0 <= src_idx < len(evidence_hgs)) or evidence_hgs[src_idx] is None:
                        continue
                    
                    # 提取支撑文本
                    supporting_text = self._extract_supporting_span(
                        d_v_merged, 
                        evidence_hgs[src_idx], 
                        evidence_texts[src_idx]
                    )
                    
                    # 记录详情
                    query_to_evidence_details[q_text].append({
                        'matched': True,
                        'evidence_idx': src_idx,
                        'supporting_text': supporting_text,
                        'confidence': self._estimate_match_confidence(q_v, d_v_merged),
                        'merged_vertex_text': d_v_merged.text()
                    })
        
        return dict(query_to_evidence_details)

    def _build_structured_context(
        self,
        query_hg: Hypergraph,
        query_to_evidence_details: Dict[str, List[Dict]],
        evidence_texts: List[str],
        merged_hg: Hypergraph,
        q_map: Dict[int, Vertex],
        d_map: Dict[int, Vertex],
        mapping: Dict[int, Set[int]],
        provenance: Dict[int, Set[int]]
    ) -> str:
        """
        生成结构化上下文，聚焦"证据支撑详情"，移除整体一致性判断
        
        输出格式:
        ## Multi-hop Evidence Support Analysis
        ### 🔍 Evidence-Centric View:
        [0] 🔹 High Confidence (covers 2 query components)
             Preview: "Alice founded TechCorp..."
             Supports:
               • Q: 'who founded TechCorp' → "Alice founded TechCorp in 2010"
        
        ### ⚠️  Low-Confidence / Conflicting Info:
          • 'headquarters': [0] says "San Francisco", [2] says "New York"
        """
        lines = []
        
        # === Header: 覆盖统计（仅信息展示） ===
        total_q_nodes = len(query_to_evidence_details)
        covered_count = sum(
            1 for details in query_to_evidence_details.values()
            if any(d['matched'] for d in details)
        )
        
        lines.append("## Multi-hop Evidence Support Analysis")
        lines.append(f"**Coverage**: {covered_count}/{total_q_nodes} query components have evidence support")
        lines.append("")
        
        # === View 1: By Evidence (按置信度分组) ===
        lines.append("### 🔍 Evidence-Centric View:")
        
        # 按 evidence index 分组
        evidence_to_queries: Dict[int, List[Dict]] = defaultdict(list)
        for q_text, details in query_to_evidence_details.items():
            for d in details:
                if d['matched'] and d['evidence_idx'] is not None:
                    evidence_to_queries[d['evidence_idx']].append({
                        'query': q_text,
                        'supporting_text': d['supporting_text'],
                        'confidence': d['confidence']
                    })
        
        # 输出每个 evidence（按"平均置信度"排序，高置信度在前）
        evidence_stats = []
        for idx in range(len(evidence_texts)):
            queries = evidence_to_queries.get(idx, [])
            avg_conf = sum(q['confidence'] for q in queries) / len(queries) if queries else 0.0
            evidence_stats.append((idx, avg_conf, len(queries), evidence_texts[idx]))
        
        # 排序: 高置信度 + 多匹配 优先
        evidence_stats.sort(key=lambda x: (-x[1], -x[2]))
        
        for idx, avg_conf, match_count, text in evidence_stats:
            # 置信度图标
            if avg_conf >= 0.8:
                conf_label, conf_icon = "High Confidence", "🔹"
            elif avg_conf >= 0.5:
                conf_label, conf_icon = "Medium Confidence", "🔸"
            else:
                conf_label, conf_icon = "Low Confidence", "▫️"
            
            preview = text[:80] + "..." if len(text) > 80 else text
            
            lines.append(f"\n[{idx}] {conf_icon} {conf_label} ({match_count} matches, avg_conf={avg_conf:.2f})")
            lines.append(f"     Preview: {preview}")
            
            if evidence_to_queries.get(idx):
                lines.append("     Supports:")
                for item in sorted(evidence_to_queries[idx], key=lambda x: -x['confidence']):
                    lines.append(f"       • Q: '{item['query']}'")
                    if item['supporting_text']:
                        span_lines = item['supporting_text'].split('\n')
                        for sl in span_lines[:2]:
                            lines.append(f"          → {sl.strip()}")
        
        # === View 2: 冲突/低置信度信息（供 LLM 警惕） ===
        conflicting_info = []
        for q_text, details in query_to_evidence_details.items():
            matched = [d for d in details if d['matched'] and d['evidence_idx'] is not None]
            if len(matched) >= 2:
                # 检查是否有不同答案（简单：文本不同 + 置信度都>0.5）
                answers = [(d['merged_vertex_text'], d['evidence_idx']) for d in matched if d['confidence'] >= 0.5]
                unique_answers = set(a[0] for a in answers)
                if len(unique_answers) >= 2:
                    conflicting_info.append({
                        'query': q_text,
                        'candidates': list(unique_answers),
                        'sources': [a[1] for a in answers]
                    })
        
        if conflicting_info:
            lines.append("\n### ⚠️  Potential Conflicts / Low-Confidence Info:")
            for item in conflicting_info:
                candidates_str = ", ".join(f"'{c}'" for c in item['candidates'][:3])
                if len(item['candidates']) > 3:
                    candidates_str += f" (+{len(item['candidates'])-3} more)"
                lines.append(f"  • '{item['query']}': conflicting answers [{candidates_str}] from evidence {item['sources']}")
        
        # === View 3: 完全缺失的信息 ===
        missing_nodes = [
            q_text for q_text, details in query_to_evidence_details.items()
            if not any(d['matched'] for d in details)
        ]
        
        if missing_nodes:
            lines.append("\n### ❌ No Evidence Support:")
            for q_text in missing_nodes:
                lines.append(f"  • '{q_text}' (not found in any evidence)")
        
        # === Footer: LLM 使用指引 ===
        lines.append("\n---")
        lines.append("**How to use this context**:")
        lines.append("1. Prioritize [🔹 High Confidence] evidence for core facts")
        lines.append("2. If a query component has multiple answers, check for conflicts and use external knowledge")
        lines.append("3. If a component is in 'No Evidence Support', the answer may be 'unanswerable'")
        lines.append("4. Cite evidence: 'According to [0], ...' or 'Evidence [2] suggests ...'")
        
        return "\n".join(lines)

    def _is_valid_node(self, vertex: Vertex) -> bool:
        """
        Filter nodes for fusion candidates:
        - Must be NOUN or PROPN
        - MUST NOT be DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
        """
        valid_pos = {Pos.NOUN, Pos.PROPN}
        excluded_ent = {
            Entity.DATE, Entity.TIME, Entity.PERCENT, 
            Entity.MONEY, Entity.QUANTITY, Entity.ORDINAL, Entity.CARDINAL
        }
        
        for node in vertex.nodes:
            if node.pos not in valid_pos:
                continue
            if node.ent in excluded_ent:
                continue
            return True
            
        return False

    def merge_hypergraphs(self, evidence_hypergraphs: List[Hypergraph]) -> Tuple[Hypergraph, Dict[int, Set[int]]]:
        """
        Merge multiple evidence hypergraphs into a single hypergraph.
        """
        # 1. 尽量保守地去做合并；核心思路在于：严格通过滚动去看到底哪些类型的节点需要被合并；是考虑若干个文本，是考虑出现在不同文档中的同样的词；
        # 2. 在严格限定能参与合并的基础之上，就不用去考虑POS
        # 3. 重复的人名（A Bob;B Bob;）指代不同的人。一个名字中存在结构不同就不考虑；
        
        # 对于musique：优先级
            ## 1. 只处理那些可回答的问题
            ## 2. 对于可回答的query，将判定的结果和ground truth做对照
            ## 3. 可能需要针对”多跳回答类问题“做生成上的特殊处理

        self.fusion_logger.info("[Merge] Start merging evidence hypergraphs")
        self.fusion_logger.info(f"[Merge] Total evidence hypergraphs: {len(evidence_hypergraphs)}")
        
        # 1. Collect all vertices and edges
        all_vertex_ids = [] 
        all_hyperedges = []
        vertex_id_map: Dict[int, Vertex] = {}  # id(vertex) -> vertex object
        vertex_source_map: Dict[int, int] = {}  # id(vertex) -> source_index
        
        valid_vertex_ids = []  # 存储 valid vertex 的 id
        
        total_vertices_before = 0
        total_hyperedges_before = 0
        hyperedge_from_hypergraph: Dict[Hyperedge, int] = {}
        
        for i, hg in enumerate(evidence_hypergraphs):
            if hg is None:
                self.fusion_logger.warning(f"[Merge] Evidence [{i}] is None, skipped")
                continue
            
            total_vertices_before += len(hg.vertices)
            total_hyperedges_before += len(hg.hyperedges)
            
            self.fusion_logger.info(f"[Merge] Evidence [{i}]: {len(hg.vertices)} vertices, {len(hg.hyperedges)} edges")
            
            for vertex in hg.vertices:
                # 为节点写入来源，供 merged hyperedge.current_node 做同源匹配。
                for node in vertex.nodes:
                    node.source_id = i

                vid = id(vertex)  # 使用内存地址作为唯一 ID
                vertex_id_map[vid] = vertex
                vertex_source_map[vid] = i
                
                all_vertex_ids.append(vid)
                if self._is_valid_node(vertex):
                    valid_vertex_ids.append(vid)
            all_hyperedges.extend(hg.hyperedges)
            for he in hg.hyperedges:
                hyperedge_from_hypergraph[he] = i

        self.fusion_logger.info(f"[Merge] Total vertices before fusion: {total_vertices_before}")
        self.fusion_logger.info(f"[Merge] Valid vertices for fusion (after filtering): {len(valid_vertex_ids)}")

        # 2. Identify mergeable vertices using NLI
        uf = UnionFind(all_vertex_ids) 
        pairs_to_check = []
        exact_merge_count = 0
        skipped_type_mismatch = 0
        skipped_short_text = 0
        
        # 遍历 valid_vertex_ids（id 列表）
        for i in range(len(valid_vertex_ids)):
            for j in range(i + 1, len(valid_vertex_ids)):
                v1_id = valid_vertex_ids[i]
                v2_id = valid_vertex_ids[j]
                
                # 通过 id 获取 Vertex 对象
                v1 = vertex_id_map[v1_id]
                v2 = vertex_id_map[v2_id]
                
                # 使用 id 获取来源
                src1 = vertex_source_map.get(v1_id, -1)
                src2 = vertex_source_map.get(v2_id, -1)
                
                if src1 == src2:
                    continue
                
                # 精确文本匹配
                if v1.text() == v2.text():
                    uf.union(v1_id, v2_id)  # 使用 id 进行 union
                    exact_merge_count += 1
                    continue
                
                # 类型兼容性检查
                if not self._are_entities_compatible(v1, v2):
                    skipped_type_mismatch += 1
                    continue

                # 短词保护
                if len(v1.text()) < 4 or len(v2.text()) < 4:
                    skipped_short_text += 1
                    continue

                # 存储 id 对，而不是 Vertex 对象对
                pairs_to_check.append((v1_id, v2_id))

        self.fusion_logger.info(f"[Merge] Exact text matches merged: {exact_merge_count}")
        self.fusion_logger.info(f"[Merge] Skipped due to type mismatch: {skipped_type_mismatch}")
        self.fusion_logger.info(f"[Merge] Skipped due to short text: {skipped_short_text}")
        self.fusion_logger.info(f"[Merge] Pairs to check with NLI: {len(pairs_to_check)}")

        # Batch NLI
        nli_merge_count = 0
        if pairs_to_check:
            # 通过 id 获取 Vertex 对象来获取 text
            text_pairs = [(vertex_id_map[v1_id].text(), vertex_id_map[v2_id].text()) for v1_id, v2_id in pairs_to_check]
            try:
                labels = get_nli_labels_batch(text_pairs)
                for (v1_id, v2_id), label in zip(pairs_to_check, labels):
                    if label == 'entailment':
                        uf.union(v1_id, v2_id)  # 使用 id 进行 union
                        nli_merge_count += 1
            except Exception as e:
                self.fusion_logger.error(f"[Merge] NLI batch failed: {e}")
        
        self.fusion_logger.info(f"[Merge] NLI-based merges: {nli_merge_count}")
        
        # 3. Construct New Vertices
        old_to_new_vertex = {}  # id(vertex) -> new_vertex
        groups = defaultdict(list)
        
        # 遍历所有 vertex id 进行分组
        for vid in all_vertex_ids:
            root = uf.find(vid)
            groups[root].append(vid)
            
        new_vertices = []
        new_id_counter = 0
        new_vertex_provenance = defaultdict(set)
        
        for root, group_ids in groups.items():
            combined_nodes = []
            for vid in group_ids:
                v = vertex_id_map[vid]  # 通过 id 获取 Vertex 对象
                combined_nodes.extend(v.nodes)
                if vid in vertex_source_map:
                    new_vertex_provenance[new_id_counter].add(vertex_source_map[vid])
            
            type_cache = vertex_id_map[group_ids[0]].type_cache if group_ids else None
            new_v = Vertex(new_id_counter, combined_nodes)
            if type_cache:
                new_v.type_cache = type_cache
            new_v.set_provenance(new_vertex_provenance[new_id_counter])  # 设置来源信息
            new_vertices.append(new_v)
            
            for vid in group_ids:
                old_to_new_vertex[vid] = new_v
                
            new_id_counter += 1
        
        # 4. Construct New Hyperedges
        new_hyperedges = []
        for he in all_hyperedges:
            # 注意：he.root 和 he.vertices 是 Vertex 对象，需要转为 id 查找
            root_id = id(he.root)
            new_root = old_to_new_vertex.get(root_id)
            if not new_root:
                continue
                
            new_he_vertices = []
            for v in he.vertices:
                vid = id(v)
                nv = old_to_new_vertex.get(vid)
                if nv and nv not in new_he_vertices:
                    new_he_vertices.append(nv)
            new_he = Hyperedge(new_root, new_he_vertices, he.desc, he.full_desc, he.start, he.end)
            new_he.set_hypergraph_id(hyperedge_from_hypergraph.get(he, None))  # 记录来源超图 ID
            new_hyperedges.append(new_he)

        merged_hg = Hypergraph(new_vertices, new_hyperedges, None)
        
        # 日志：合并后超图信息
        self.fusion_logger.info(f"[Merge] === Merged Hypergraph Summary ===")
        if total_vertices_before > 0:
            reduction_rate = (1 - len(new_vertices)/total_vertices_before)*100
            self.fusion_logger.info(f"[Merge] Vertex reduction rate: {reduction_rate:.1f}%")
            if reduction_rate > 50.0:
                self.fusion_logger.warning(f"[Merge] High reduction rate detected! Check entity compatibility logic.")
        
        multi_source_count = sum(1 for srcs in new_vertex_provenance.values() if len(srcs) > 1)
        self.fusion_logger.info(f"[Merge] Vertices from multiple sources: {multi_source_count} / {len(new_vertices)} ({(multi_source_count/max(1,len(new_vertices)))*100:.1f}%)")
        
        merged_hg.log_summary(self.fusion_logger, level="INFO")
        
        if multi_source_count > 0:
            self.fusion_logger.info("[Merge] Sample Multi-source Vertices (Fusion Success Cases):")
            count = 0
            for v_id, sources in new_vertex_provenance.items():
                if len(sources) > 1 and count < 5:
                    v_text = new_vertices[v_id].text()
                    self.fusion_logger.info(f"  • Vertex [{v_id}] '{v_text}' <- Sources: {sorted(sources)}")
                    count += 1
        else:
            self.fusion_logger.warning("[Merge] No multi-source vertices found. Fusion might be too strict or data is disjoint.")

        return merged_hg, new_vertex_provenance

    def process(self, query_hg: Hypergraph, evidence_hgs: List[Hypergraph], evidence_texts: List[str]) -> Tuple[bool, str]:
        """
        Main pipeline: Merge → Simulate → Reverse Trace → Structured Context
        
        Returns:
            is_consistent: 固定返回 True（因为 merge 已融合所有信息，一致性判断交给 LLM）
            context: 结构化、按证据分组的支撑信息
        """
        # 日志：开始处理
        self.consistent_logger.info("[Multi-hop] Enter multi-hop consistency detection")
        self.consistent_logger.info(f"[Multi-hop] Query: '{query_hg.doc[:50] if query_hg.doc else 'N/A'}...'")
        self.consistent_logger.info(f"[Multi-hop] Evidence count: {len(evidence_hgs)}")
        
        # 1. Merge (保留 provenance)
        import time
        time1 = time.time()
        merged_hg, provenance = self.merge_hypergraphs(evidence_hgs)
        time2 = time.time()
        # print(f"Merge time: {time2 - time1:.2f} seconds")
        # 2. Global Simulation
        self.consistent_logger.info("[Multi-hop] Running Hyper Simulation...")
        mapping, q_map, d_map = compute_hyper_simulation(query_hg, merged_hg)
        # for q_id, d_ids in mapping.items():
        #     for d_id in d_ids:
        #         u = q_map[q_id]
        #         v = d_map[d_id]
        #         if u.is_verb() or v.is_verb():
        #             continue
        #         print(f"Hyper Simulation Match: {u.text()} <-> {v.text()}")
        
        simulation: list[Tuple[Vertex, Vertex]] = [(u, v) for q_id, d_ids in mapping.items() for d_id in d_ids for u in [q_map[q_id]] for v in [d_map[d_id]]]
        
        # time3 = time.time()
        # final = post_detection(query_hg, merged_hg, simulation)
        # time4 = time.time()
        simulations = get_simulation_slice(query_hg, merged_hg, simulation, len(evidence_hgs))
        for i, sim in enumerate(simulations):
            print(f"Simulation slice for evidence [{i + 1}]:")
            for u, v in sim:
                if u.is_verb() or v.is_verb() or u.is_adjective() or v.is_adjective() or u.is_adverb() or v.is_adverb():
                    continue
                print(f"  Match: {u.text()} <-> {v.text()}")
        # print(f"Post-processing time: {time4 - time3:.2f} seconds")
        # for u, v in final:
        #     if u.is_verb() or v.is_verb():
        #         continue
        #     print(f"Post-processed Match: {u.text()} <-> {v.text()} [{', '.join(str(id) for id in v.get_provenance())}]")
        
        self.consistent_logger.info(f"[Multi-hop] Simulation completed: {len(mapping)} query nodes mapped")
        
        # 3. Reverse Tracing: query → merged → original evidence
        query_to_evidence_details = self._reverse_trace_consistency(
            query_hg=query_hg,
            merged_hg=merged_hg,
            evidence_hgs=evidence_hgs,
            evidence_texts=evidence_texts,
            q_map=q_map,
            d_map=d_map,
            mapping=mapping,
            provenance=provenance
        )
        
        # 4. Generate structured context
        context = self._build_structured_context(
            query_hg=query_hg,
            query_to_evidence_details=query_to_evidence_details,
            evidence_texts=evidence_texts,
            merged_hg=merged_hg,
            q_map=q_map,
            d_map=d_map,
            mapping=mapping,
            provenance=provenance
        )
        # 日志：仅记录覆盖统计（用于监控，不影响返回）
        covered_count = sum(
            1 for details in query_to_evidence_details.values()
            if any(d['matched'] for d in details)
        )
        total_count = len(query_to_evidence_details)
        self.consistent_logger.info(
            f"[Multi-hop] Coverage stats: {covered_count}/{total_count} critical nodes have evidence support"
        )
        
        self.consistent_logger.info(f"[Multi-hop] Enhanced context generated, length: {len(context)} chars")
        self.consistent_logger.info(f"[Multi-hop] Context preview:\n{context}...")

        return context  # ✅ is_consistent 固定为 True
    
    
    
# query: [1, 2, 3, 4, 5, 6, 7]
# - #1: context: [1, 2, 3]
# - #2: context: [7, 8, 9]
# - #3: 
# - #4: 
