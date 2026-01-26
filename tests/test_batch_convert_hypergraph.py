"""
批量将文本转换为超图并保存为 pkl 文件
在 spacy-1 环境中运行
"""
import sys
import os
import glob
import json
from pathlib import Path

import coreferee
import spacy
from fastcoref import spacy_component

from combine import combine, calc_correfs_str
from dependency import Dependency, Node, LocalDoc
from hypergraph import Hypergraph as LocalHypergraph

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Initialize spaCy with coreference resolution (device auto-selected by fastcoref)
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe(
    'fastcoref',
    config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref'}
)


def load_data_from_dir(json_dir: str) -> list[tuple[str, dict]]:
    """Load data from JSON/JSONL files in the directory.
    
    Returns:
        List of tuples: (dataset_name, data_dict), where data_dict contains 'q' (str) and 'd' (list[str])
    """
    data_items = []
    for pattern in ("*.json", "*.jsonl"):
        for path in glob.glob(os.path.join(json_dir, pattern)):
            # Extract dataset name from file path (without extension)
            dataset_name = os.path.splitext(os.path.basename(path))[0]
            
            with open(path, "r", encoding="utf-8") as f:
                if path.lower().endswith(".jsonl"):
                    # Process JSONL line by line
                    for line in f:
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            if isinstance(obj, dict) and "q" in obj and "d" in obj:
                                data_items.append((dataset_name, obj))
                else:
                    # Process JSON file
                    obj = json.load(f)
                    if isinstance(obj, dict) and "q" in obj and "d" in obj:
                        data_items.append((dataset_name, obj))
                    elif isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, dict) and "q" in item and "d" in item:
                                data_items.append((dataset_name, item))
    return data_items


def text_to_hypergraph(nlp, text: str) -> LocalHypergraph:
    """Convert a text string to a LocalHypergraph."""
    doc = nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
    
    correfs = calc_correfs_str(doc)
    spans_to_merge = combine(doc, correfs)
    with doc.retokenize() as retokenizer:
        for span in spans_to_merge:
            retokenizer.merge(span)
    
    nodes, roots = Node.from_doc(doc)
    local_doc = LocalDoc(doc)
    dep = Dependency(nodes, roots, local_doc)
    vertices, rel, id_map = dep.solve_conjunctions().mark_pronoun_antecedents().mark_prefixes().mark_vertex().compress_dependencies().calc_relationships()
    
    # Convert to Hypergraph
    hypergraph = LocalHypergraph.from_rels(vertices, rel, id_map, local_doc)
    return hypergraph


def main():
    # Configuration
    select_data_dir = os.path.join(ROOT, "select_data")
    hypergraph_dir = os.path.join(ROOT, "hypergraphs")  # 保存超图的目录
    # Optional limits for quick tests
    try:
        limit_items = int(os.environ.get("BATCH_LIMIT_ITEMS", "0"))
    except Exception:
        limit_items = 0
    try:
        limit_docs = int(os.environ.get("BATCH_LIMIT_DOCS", "0"))
    except Exception:
        limit_docs = 0
    try:
        start_item = int(os.environ.get("BATCH_START_ITEM", "0"))
    except Exception:
        start_item = 0
    
    # Create output directory if it doesn't exist
    os.makedirs(hypergraph_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {select_data_dir}...")
    data_items = load_data_from_dir(select_data_dir)
    print(f"Loaded {len(data_items)} data items")
    if limit_items > 0:
        data_items = data_items[:limit_items]
        print(f"[Info] Limiting items to first {limit_items}")
    
    # Process each data item
    if start_item > 0:
        data_items = data_items[start_item:]
        print(f"[Info] Starting from item index {start_item}")

    base_offset = start_item

    for local_idx, (dataset_name, item) in enumerate(data_items):
        item_idx = base_offset + local_idx
        q_text = item.get("q", "")
        d_list = item.get("d", [])
        if limit_docs > 0:
            d_list = d_list[:limit_docs]
        
        if not q_text or not d_list:
            print(f"Skipping item {item_idx} from {dataset_name}: missing q or d")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing item {item_idx} from dataset: {dataset_name}")
        print(f"Query: {q_text[:100]}...")
        print(f"Number of documents: {len(d_list)}")
        
        # Create directory for this item with dataset name
        item_dir = os.path.join(hypergraph_dir, f"{dataset_name}_item_{item_idx}")
        os.makedirs(item_dir, exist_ok=True)
        
        # Convert query to hypergraph and save
        query_file = os.path.join(item_dir, "query.pkl")
        if os.path.exists(query_file):
            print(f"  Query hypergraph already exists, skipping...")
        else:
            try:
                query_hg = text_to_hypergraph(nlp, q_text)
                query_hg.save(query_file)
                print(f"  Saved query hypergraph")
            except Exception as e:
                print(f"  Error converting query: {e}")
                continue
        
        # Convert each document to hypergraph and save
        for d_idx, d_text in enumerate(d_list):
            doc_file = os.path.join(item_dir, f"doc_{d_idx}.pkl")
            if os.path.exists(doc_file):
                print(f"  Document {d_idx + 1}/{len(d_list)} already exists, skipping...")
                continue
            try:
                doc_hg = text_to_hypergraph(nlp, d_text)
                doc_hg.save(doc_file)
                print(f"  Saved document {d_idx + 1}/{len(d_list)}")
            except Exception as e:
                print(f"  Error converting document {d_idx}: {e}")
                continue
        
        print(f"Completed item {item_idx} from {dataset_name}")
    
    print(f"\n{'='*80}")
    print(f"Conversion complete! Hypergraphs saved to {hypergraph_dir}")
    print(f"Structure: hypergraphs/{'{dataset_name}'}_item_{'{item_idx}'}/query.pkl, doc_0.pkl, doc_1.pkl, ...")


if __name__ == "__main__":
    main()

