"""
从 pkl 文件加载超图并运行 simulation
在 sc 环境中运行
"""
import sys
import os
import argparse
from pathlib import Path

# Reduce noisy logs from the Rust hyper-simulation lib if present
os.environ.setdefault("RUST_LOG", "error")
os.environ.setdefault("GRAPH_SIM_LOG", "error")
os.environ.setdefault("GRAPH_SIM_LOG_LEVEL", "error")

from hypergraph import Hypergraph as LocalHypergraph, Vertex as LocalVertex
from nli import get_nli_labels_batch
from simulation import Hypergraph as SimHypergraph, Hyperedge as SimHyperedge, Node as SimNode, Delta, DMatch

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def convert_local_to_sim(
    local_hg: LocalHypergraph,
) -> tuple[SimHypergraph, dict[int, str], dict[int, LocalVertex], dict[int, list[SimHyperedge]]]:
    """Translate a LocalHypergraph into simulation-alg structures while keeping node metadata."""
    sim_hg = SimHypergraph()
    vertex_id_map: dict[int, int] = {}
    node_text: dict[int, str] = {}
    sim_id_to_vertex: dict[int, LocalVertex] = {}
    node_to_edges: dict[int, list[SimHyperedge]] = {}
    for idx, vertex in enumerate(sorted(local_hg.vertices, key=lambda v: v.id)):
        sim_hg.add_node(vertex.text())
        vertex_id_map[vertex.id] = idx
        node_text[idx] = vertex.text()
        sim_id_to_vertex[idx] = vertex
    edge_id = 0
    for local_edge in local_hg.hyperedges:
        node_ids = {vertex_id_map[v.id] for v in local_edge.vertices if v.id in vertex_id_map}
        if not node_ids:
            continue
        sim_edge = SimHyperedge(node_ids, local_edge.desc, edge_id)
        sim_hg.add_hyperedge(sim_edge)
        for nid in node_ids:
            node_to_edges.setdefault(nid, []).append(sim_edge)
        edge_id += 1
    return sim_hg, node_text, sim_id_to_vertex, node_to_edges


def compute_allowed_pairs(
    query_vertices: dict[int, LocalVertex],
    data_vertices: dict[int, LocalVertex],
) -> set[tuple[int, int]]:
    """Use NLI + domain heuristics to decide which node IDs can share type."""
    text_pair_to_ids: dict[tuple[str, str], tuple[int, int, LocalVertex, LocalVertex]] = {}
    for q_id, q_vertex in query_vertices.items():
        for d_id, d_vertex in data_vertices.items():
            key = (q_vertex.text(), d_vertex.text())
            text_pair_to_ids[key] = (q_id, d_id, q_vertex, d_vertex)
    text_pairs = list(text_pair_to_ids.keys())
    if not text_pairs:
        return set()
    labels = get_nli_labels_batch(text_pairs)
    allowed: set[tuple[int, int]] = set()
    for (text1, text2), label in zip(text_pairs, labels):
        q_id, d_id, q_vertex, d_vertex = text_pair_to_ids[(text1, text2)]
        if label == "entailment" or (label == "neutral" and q_vertex.is_domain(d_vertex)):
            allowed.add((q_id, d_id))
    return allowed


def build_delta_and_dmatch(
    query: SimHypergraph,
    data: SimHypergraph,
    query_texts: dict[int, str],
    data_texts: dict[int, str],
    query_node_edges: dict[int, list[SimHyperedge]],
    data_node_edges: dict[int, list[SimHyperedge]],
    allowed_pairs: set[tuple[int, int]],
) -> tuple[Delta, DMatch]:
    """Register each allowed node pair as a semantic cluster pair and build DMatch."""
    delta = Delta()
    d_delta_matches: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for q_id, d_id in sorted(allowed_pairs):
        cluster_u = query_node_edges.get(q_id, [])
        cluster_v = data_node_edges.get(d_id, [])
        u_node = SimNode(q_id, query_texts.get(q_id, ""))
        v_node = SimNode(d_id, data_texts.get(d_id, ""))
        sc_id = delta.add_sematic_cluster_pair(u_node, v_node, cluster_u, cluster_v)
        d_delta_matches[(sc_id, sc_id)] = {(q_id, d_id)}
    return delta, DMatch.from_dict(d_delta_matches)


def format_result_to_string(
    title: str,
    result: dict[int, set[int]],
    query_texts: dict[int, str],
    data_texts: dict[int, str],
) -> str:
    """Format the simulation result as a string."""
    lines = [title]
    for q_id, vs in sorted(result.items()):
        q = query_texts.get(q_id, "")
        targets = ", ".join(data_texts[v] for v in sorted(vs)) if vs else "-"
        lines.append(f"  [{q_id}] {q} -> {targets}")
    lines.append("--------------------------------")
    lines.append(f"result: {result}")
    return "\n".join(lines)


def _process_item(
    item_dir_name: str,
    hypergraph_dir: str,
    output_dir: str,
    limit_docs: int,
    log_prefix: str = "",
) -> tuple[str, str, str, bool, str | None, str | None]:
    """
    Process a single item directory:
    - load query and doc hypergraphs
    - run simulation
    - save result file

    Returns: (item_dir_name, dataset_name, item_idx, ok, error, output_file)
    """
    item_path = os.path.join(hypergraph_dir, item_dir_name)
    dataset_name, item_idx = _parse_item_dir(item_dir_name)

    query_file = os.path.join(item_path, "query.pkl")
    if not os.path.exists(query_file):
        return (item_dir_name, dataset_name, item_idx, False, f"query.pkl not found at {query_file}", None)

    try:
        query_hg = LocalHypergraph.load(query_file)
    except Exception as e:
        return (item_dir_name, dataset_name, item_idx, False, f"Error loading query hypergraph: {e}", None)

    # Collect doc files
    try:
        all_files = os.listdir(item_path)
        doc_files = [f for f in all_files if f.startswith("doc_") and f.endswith(".pkl")]
        doc_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]) if x.split("_")[1].split(".")[0].isdigit() else 0)
        if limit_docs > 0:
            doc_files = doc_files[:limit_docs]
    except Exception as e:
        return (item_dir_name, dataset_name, item_idx, False, f"Error reading directory {item_path}: {e}", None)

    if not doc_files:
        return (item_dir_name, dataset_name, item_idx, False, f"No document hypergraphs found in {item_dir_name}", None)

    result_lines = []
    result_lines.append(f"Dataset: {dataset_name}")
    result_lines.append(f"Item: {item_idx}")
    result_lines.append(f"Number of documents: {len(doc_files)}")
    result_lines.append("="*80)

    for doc_file_name in doc_files:
        doc_file_path = os.path.join(item_path, doc_file_name)
        doc_idx = doc_file_name.split("_")[1].split(".")[0] if "_" in doc_file_name else "unknown"
        try:
            data_hg = LocalHypergraph.load(doc_file_path)
            result_lines.append(f"\n--- Document {doc_idx} ---")
            result_lines.append(f"Document file: {doc_file_name}")
            result_lines.append("")
            pair_result = process_query_document_pair(query_hg, data_hg)
            result_lines.append(pair_result)
            result_lines.append("")
            print(f"{log_prefix}doc {doc_idx} done ({doc_file_name})")
        except Exception as e:
            import traceback
            result_lines.append(f"\n--- Document {doc_idx} ---")
            result_lines.append(f"Error: {str(e)}")
            result_lines.append(f"File path: {doc_file_path}")
            result_lines.append("")
            return (
                item_dir_name,
                dataset_name,
                item_idx,
                False,
                f"Error processing {doc_file_name}: {e}\n{traceback.format_exc()}",
                None,
            )

    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    output_file = os.path.join(dataset_output_dir, f"item_{item_idx}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))

    return (item_dir_name, dataset_name, item_idx, True, None, output_file)


def _parse_item_dir(item_dir_name: str) -> tuple[str, str]:
    if "_item_" in item_dir_name:
        parts = item_dir_name.split("_item_")
        dataset_name = parts[0]
        item_idx = parts[1] if len(parts) > 1 else "unknown"
    else:
        dataset_name = "unknown"
        item_idx = "unknown"
    return dataset_name, item_idx


def process_query_document_pair(query_hg: LocalHypergraph, data_hg: LocalHypergraph) -> str:
    """Process a single (query, document) pair and return formatted result string."""
    try:
        # Convert to simulation structures
        query, query_texts, query_vertices, query_node_edges = convert_local_to_sim(query_hg)
        data, data_texts, data_vertices, data_node_edges = convert_local_to_sim(data_hg)
        
        # Compute allowed pairs
        allowed_pairs = compute_allowed_pairs(query_vertices, data_vertices)
        
        # Set type_same_fn
        def type_same_fn(x_id: int, y_id: int) -> bool:
            return (x_id, y_id) in allowed_pairs
        
        query.set_type_same_fn(type_same_fn)
        data.set_type_same_fn(type_same_fn)
        
        # Build delta and d_match
        delta, d_match = build_delta_and_dmatch(
            query,
            data,
            query_texts,
            data_texts,
            query_node_edges,
            data_node_edges,
            allowed_pairs,
        )
        
        # Run simulation
        sim_result = SimHypergraph.get_hyper_simulation(query, data, delta, d_match)
        
        # Format result
        result_str = format_result_to_string("get_hyper_simulation:", sim_result, query_texts, data_texts)
        return result_str
    except Exception as e:
        import traceback
        return f"Error processing pair: {str(e)}\n{type(e).__name__}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description="Run hypergraph simulation from saved pkl files.")
    parser.add_argument("--hypergraph_dir", default=os.path.join(ROOT, "hypergraphs"), help="Directory of saved hypergraphs")
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "output_data"), help="Directory to save simulation results")
    parser.add_argument("--start_item", type=int, default=int(os.environ.get("BATCH_START_ITEM", "0")), help="Start item index (default from env BATCH_START_ITEM or 0)")
    parser.add_argument("--limit_items", type=int, default=int(os.environ.get("BATCH_LIMIT_ITEMS", "0")), help="Limit number of items (0 for all)")
    parser.add_argument("--limit_docs", type=int, default=int(os.environ.get("BATCH_LIMIT_DOCS", "0")), help="Limit docs per item (0 for all)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("BATCH_WORKERS", "1")), help="Process pool size (>1 enables parallel)")
    args = parser.parse_args()

    hypergraph_dir = args.hypergraph_dir
    output_dir = args.output_dir
    limit_items = args.limit_items
    limit_docs = args.limit_docs
    workers = args.workers
    start_item = args.start_item
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all item directories
    if not os.path.exists(hypergraph_dir):
        print(f"Error: Hypergraph directory {hypergraph_dir} does not exist!")
        print("Please run batch_convert_hypergraph.py first to generate hypergraphs.")
        return
    
    # Find directories matching pattern: {dataset_name}_item_{item_idx}
    item_dirs = []
    try:
        for d in os.listdir(hypergraph_dir):
            dir_path = os.path.join(hypergraph_dir, d)
            if os.path.isdir(dir_path) and "_item_" in d:
                item_dirs.append(d)
    except Exception as e:
        print(f"Error reading hypergraph directory: {e}")
        return
    
    if not item_dirs:
        print(f"Warning: No item directories found in {hypergraph_dir}")
        print("Expected format: {dataset_name}_item_{item_idx}")
        return
    
    # Sort by dataset name and item index
    def sort_key(x):
        parts = x.split("_item_")
        if len(parts) == 2:
            dataset_name = parts[0]
            try:
                item_idx = int(parts[1])
                return (dataset_name, item_idx)
            except ValueError:
                return (dataset_name, 0)
        return (x, 0)
    
    item_dirs.sort(key=sort_key)
    if start_item > 0:
        item_dirs = item_dirs[start_item:]
        print(f"[Info] Starting from item index {start_item}")
    if limit_items > 0:
        item_dirs = item_dirs[:limit_items]
        print(f"[Info] Limiting items to first {limit_items}")
    
    print(f"Found {len(item_dirs)} item directories in {hypergraph_dir}")
    print(f"Directories: {item_dirs[:5]}{'...' if len(item_dirs) > 5 else ''}")
    total_items = len(item_dirs)

    # Skip items whose output already exists
    filtered_dirs: list[str] = []
    for item_dir_name in item_dirs:
        dataset_name, item_idx = _parse_item_dir(item_dir_name)
        output_file = os.path.join(output_dir, dataset_name, f"item_{item_idx}.txt")
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"[Skip] {item_dir_name} already processed -> {output_file}")
            continue
        filtered_dirs.append(item_dir_name)

    item_dirs = filtered_dirs
    total_items = len(item_dirs)
    if total_items == 0:
        print("No pending items to process. All outputs already exist.")
        return
    
    if workers <= 1:
        for idx, item_dir_name in enumerate(item_dirs, 1):
            log_prefix = f"[{idx}/{total_items}] "
            res = _process_item(item_dir_name, hypergraph_dir, output_dir, limit_docs, log_prefix=log_prefix)
            item_dir_name, dataset_name, item_idx, ok, err, output_file = res
            if ok:
                print(f"[{idx}/{total_items}] Saved result to {output_file}")
            else:
                print(f"[{idx}/{total_items}] [Error] {item_dir_name}: {err}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"[Info] Parallel mode with {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(_process_item, item_dir_name, hypergraph_dir, output_dir, limit_docs): item_dir_name
                for item_dir_name in item_dirs
            }
            done = 0
            for future in as_completed(future_to_item):
                item_dir_name = future_to_item[future]
                done += 1
                try:
                    log_prefix = f"[{done}/{total_items}] "
                    res = future.result()
                    _, _, _, ok, err, output_file = res
                    if ok:
                        print(f"[{done}/{total_items}] Saved result to {output_file}")
                    else:
                        print(f"[{done}/{total_items}] [Error] {item_dir_name}: {err}")
                except Exception as e:
                    print(f"[{done}/{total_items}] [Error] {item_dir_name}: {e}")
    
    print(f"\n{'='*80}")
    print(f"Simulation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

# 判断label是否准确的判定；100个
# 用1：3左右的q:d的比例，去跑simulation；用q和d发现的节点喂给大模型判断其一致性，输出他认为所匹配的节点和冲突的节点，**match的定义**

# 1. 该匹配的没匹配上
# 2. 不该匹配（冲突的）的匹配上了
# 3. 没匹配但不冲突的匹配上了