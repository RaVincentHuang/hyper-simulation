from hmac import new
from typing import List, Dict, Set, Tuple, Any, Optional
import itertools
from collections import defaultdict

from hyper_simulation.hypergraph.hypergraph import Hypergraph, Vertex, Hyperedge
from hyper_simulation.hypergraph.dependency import Node, Pos, Entity
from hyper_simulation.component.nli import get_nli_labels_batch
from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.utils.log import getLogger
import logging

logger = logging.getLogger(__name__)

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
        vertex_id_map = {}  # id(vertex) -> vertex object
        vertex_source_map = {}  # id(vertex) -> source_index
        
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
            
            new_v = Vertex(new_id_counter, combined_nodes)
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
            self.fusion_logger.debug("[Merge] Sample Multi-source Vertices (Fusion Success Cases):")
            count = 0
            for v_id, sources in new_vertex_provenance.items():
                if len(sources) > 1 and count < 5:
                    v_text = new_vertices[v_id].text()
                    self.fusion_logger.debug(f"  • Vertex [{v_id}] '{v_text}' <- Sources: {sorted(sources)}")
                    count += 1
        else:
            self.fusion_logger.warning("[Merge] No multi-source vertices found. Fusion might be too strict or data is disjoint.")

        return merged_hg, new_vertex_provenance
        
    def process(self, query_hg: Hypergraph, evidence_hgs: List[Hypergraph], evidence_texts: List[str]) -> Tuple[bool, str]:
        """
        Main pipeline: Merge -> Simulate -> Check Consistency -> Generate Context
        """
        # 日志：开始处理
        self.consistent_logger.info("[Multi-hop] Enter multi-hop consistency detection")
        self.consistent_logger.info(f"[Multi-hop] Query: '{query_hg.doc[:50] if query_hg.doc else 'N/A'}...'")
        self.consistent_logger.info(f"[Multi-hop] Evidence count: {len(evidence_hgs)}")
        
        # 1. Merge
        merged_hg, provenance = self.merge_hypergraphs(evidence_hgs)
        
        # 2. Simulation
        self.consistent_logger.debug("[Multi-hop] Running Hyper Simulation...")
        mapping, q_map, d_map = compute_hyper_simulation(query_hg, merged_hg)
        
        self.consistent_logger.info(f"[Multi-hop] Simulation completed: {len(mapping)} query nodes mapped")
        
        # 3. Map sim_id -> provenance
        # d_map maps sim_id -> Vertex object (from merged_hg)
        # provenance maps Vertex.id -> Set[source_id]
        sim_id_to_provenance = {}
        for sim_id, d_v in d_map.items():
             sim_id_to_provenance[sim_id] = provenance.get(d_v.id, set())

        # 3. Consistency Check (Reach Cover)
        critical_q_vertices = []
        for v in query_hg.vertices:
            if any(e != Entity.NOT_ENTITY for e in v.ents) or \
               any(n.pos in {Pos.NOUN, Pos.PROPN} for n in v.nodes):
                critical_q_vertices.append(v)
        
        self.consistent_logger.info(f"[Multi-hop] Critical query vertices to cover: {len(critical_q_vertices)}")
        for v in critical_q_vertices:
            self.consistent_logger.debug(f"[Multi-hop]   • Q{v.id}: '{v.text()}'")
        
        uncovered_critical_nodes = []
        q_node_matches = {}
        
        # Reverse map for convenience: q_vertex -> q_sim_id
        q_vertex_to_sim_id = {v.id: k for k, v in q_map.items()}
        
        for q_v in critical_q_vertices:
            sim_id = q_vertex_to_sim_id.get(q_v.id)
            d_sim_ids = mapping.get(sim_id, set()) if sim_id is not None else set()
            
            if not d_sim_ids:
                uncovered_critical_nodes.append(q_v)
            else:
                matches = []
                for d_sim_id in d_sim_ids:
                    d_v = d_map[d_sim_id]
                    sources = sorted(list(sim_id_to_provenance.get(d_sim_id, set())))
                    matches.append((d_v.text(), sources))
                q_node_matches[q_v.text()] = matches

        is_consistent = (len(uncovered_critical_nodes) == 0)
        
        # 日志：一致性结果
        self.consistent_logger.info(f"[Multi-hop] Consistency Result: {'CONSISTENT' if is_consistent else 'INCONSISTENT/PARTIAL'}")
        if not is_consistent:
            self.consistent_logger.warning(f"[Multi-hop] {len(uncovered_critical_nodes)} critical nodes missing")
            for v in uncovered_critical_nodes:
                self.consistent_logger.warning(f"[Multi-hop]   • Missing: '{v.text()}'")
        else:
            self.consistent_logger.info(f"[Multi-hop] All {len(critical_q_vertices)} critical nodes covered")
        
        # 4. Enhanced Context Construction
        lines = []
        lines.append("## Multi-hop Consistency Analysis")
        if is_consistent:
            lines.append("Status: CONSISTENT (All critical query nodes covered in merged evidence)")
        else:
            lines.append(f"Status: INCONSISTENT / PARTIAL ({len(uncovered_critical_nodes)} critical nodes missing)")
            
        lines.append("\n### Evidence Usage:")
        active_sources = set()
        for matches in q_node_matches.values():
            for _, srcs in matches:
                active_sources.update(srcs)
        
        # 日志：证据使用情况
        self.consistent_logger.info(f"[Multi-hop] Active evidence sources: {sorted(active_sources)}")
        self.consistent_logger.info(f"[Multi-hop] Unused evidence sources: {sorted(set(range(len(evidence_texts))) - active_sources)}")
                
        for i, text in enumerate(evidence_texts):
            status = "USED" if i in active_sources else "UNUSED"
            lines.append(f"[{i}] {status}: {text[:100]}..." if len(text) > 100 else f"[{i}] {status}: {text}")

        lines.append("\n### Reasoning Path:")
        for q_text, matches in q_node_matches.items():
            lines.append(f"- Query: '{q_text}'")
            for d_text, srcs in matches:
                lines.append(f"  -> Match: '{d_text}' (Sources: {srcs})")
        
        if uncovered_critical_nodes:
            lines.append("\n### Missing Information:")
            for v in uncovered_critical_nodes:
                lines.append(f"- Missing: '{v.text()}'")

        context = "\n".join(lines)
        
        # 日志：最终输出
        self.consistent_logger.info(f"[Multi-hop] Enhanced context generated, length: {len(context)} chars")
        self.consistent_logger.debug(f"[Multi-hop] Context preview:\n{context}")

        return is_consistent, context