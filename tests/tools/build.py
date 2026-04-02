from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Iterable, Set

import spacy
from spacy import displacy
from fastcoref import spacy_component

from spacy.tokens import Doc
from spacy.symbols import ORTH

from hyper_simulation.hypergraph.combine import combine, calc_correfs_str, combine_links
from hyper_simulation.hypergraph.dependency import Node, LocalDoc, Dependency
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex, Node, Hyperedge
from hyper_simulation.utils.clean import clean_text_for_spacy
from hyper_simulation.hypergraph.corref import CorrefCluster, mark_corref
from hyper_simulation.hypergraph.abstraction import TokenEntityAdder

import time



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

def get_nlp() -> spacy.Language:
    nlp = spacy.load("en_core_web_trf")
    if "fastcoref" not in nlp.pipe_names:
        nlp.add_pipe(
            "fastcoref",
            config={
                "model_architecture": "LingMessCoref",
                "model_path": "biu-nlp/lingmess-coref",
                "device": "cpu",
            },
        )
        
    nlp.tokenizer.add_special_case("I.", [{ORTH: "I"}, {ORTH: "."}])
    ROMAN_NUMERALS = ["II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII"]
    for numeral in ROMAN_NUMERALS:
        nlp.tokenizer.add_special_case(f"{numeral}.", [{ORTH: numeral}, {ORTH: "."}])
    return nlp

nlp = get_nlp()

def print_dep_tokens(doc: Doc, title: str) -> None:
    print(f"\n[{title}] Tokens / Lemma / Dep / Head / Ent / POS / TAG")
    for token in doc:
        print(
            "[{i}]: '{text}', Lemma: '{lemma}', Dep: {dep} ['{head}'], Ent: {ent}, POS: {pos}, TAG: {tag}".format(
                i=token.i,
                text=token.text,
                lemma=token.lemma_,
                dep=token.dep_,
                head=token.head.text,
                ent=token.ent_type_,
                pos=token.pos_,
                tag=token.tag_,
            )
        )


def render_dep_html(doc: Doc, output_path: Path, title: str) -> None:
    html = displacy.render(doc, style="dep", jupyter=False, page=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"[{title}] Dep visualization saved: {output_path}")


def format_vertex(vertex: Vertex) -> str:
    nodes = "\n".join(
        f"    - '{node.text}' (pos={node.pos.name}, dep={node.dep.name}, ent={node.ent.name}, ENT={node.entity.name if node.entity else 'None'})"
        for node in vertex.nodes
    )
    return f"[{vertex.id}] '{vertex.text()}'\n{nodes}"


def _parse_steps(steps: str) -> Set[int]:
    if not steps or steps.strip().lower() == "all":
        return {1, 2, 3, 4}
    if steps.strip().lower() == "no":
        return set()
    parsed: Set[int] = set()
    for part in steps.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(
                f"Invalid step '{part}'. Use comma-separated numbers like 1,3,4 or 'all'."
            )
        step = int(part)
        if step not in (1, 2, 3, 4):
            raise ValueError(f"Invalid step '{step}'. Valid steps are 1, 2, 3, 4.")
        parsed.add(step)
    if not parsed:
        raise ValueError(
            "No valid steps provided. Use comma-separated numbers like 1,3,4 or 'all'."
        )
    return parsed


def debug_text_to_hypergraph(
    text: str, output_dir: str = "logs/dep_debug", steps: Iterable[int] | None = None, is_query: bool = False
) -> LocalHypergraph:
    time0 = time.time()
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    steps_set = set(steps) if steps is not None else {1, 2, 3, 4}
    text = clean_text_for_spacy(text)
    # nlp = get_nlp()
    cfg = {"fastcoref": {"resolve_text": True}} if "fastcoref" in nlp.pipe_names else {}
    time15 = time.time()
    doc = nlp(text, component_cfg=cfg)

    # print(f"\n[Input Text]:\n {text}")
    time1 = time.time()
    # 1) 原始依存分析
    if 1 in steps_set:
        print_dep_tokens(doc, "Step 1 - Raw (before combine)")
        for ent in doc.ents:
            print(f"Entity: '{ent.text}' ({ent.label_}), id: [{ent.start}, {ent.end})")
        render_dep_html(doc, output_base / f"{'query_' if is_query else 'data_'}step1_raw_dep.html", "Step 1 - Raw (before combine)")
    abstracter = TokenEntityAdder("qwen_ontology_mapping.json")
    # combine + retokenize
    correfs = calc_correfs_str(doc) if hasattr(doc._, "coref_clusters") else set()
    links_to_merge = combine_links(doc)
    with doc.retokenize() as retokenizer:
        for link in links_to_merge:
            # print(f"Merging link: '{link.text}' (start={link.start}, end={link.end}, label={link.label_})")
            retokenizer.merge(link)
    # print(f"Correfs : {correfs}")
    corref_clusters = CorrefCluster.from_doc(doc)
    spans_to_merge = combine(doc, correfs, is_query=is_query, corefs_clusters=corref_clusters)
    
    abstracter.set_entity_from_spans(spans_to_merge, doc)
    
    with doc.retokenize() as retokenizer:
        for span in spans_to_merge:
            # print(f"Merging span: '{span.text}' (start={span.start}, end={span.end}, label={span.label_})")
            retokenizer.merge(span)
    corref_clusters = CorrefCluster.update_by_doc(corref_clusters, doc)
    # 2) combine 后依存分析
    time2 = time.time()
    if 2 in steps_set:
        print_dep_tokens(doc, "Step 2 - Combined (after retokenize)")
        render_dep_html(
            doc, output_base / f"{'query_' if is_query else 'data_'}step2_combined_dep.html", "Step 2 - Combined (after retokenize)"
        )

        # 3) 指代消解 + vertices
        print(f"Coreference clusters")
        # print the coref_clusters by texts
        for idx, cluster in enumerate(corref_clusters):
            
            if cluster.is_dropped():
                continue
            
            mention_text = ", ".join([mention.text for mention in cluster.mentions])
            primary_text = ", ".join(sorted({mention.text for mention in cluster.is_primary_mention})) if cluster.is_primary_mention else "None"
            print(f"Cluster [{idx}]: [{mention_text}], primary={primary_text}")
        print(f"Resolved Text: {getattr(doc._, 'resolved_text', None)}")
    nodes, roots = Node.from_doc(doc, abstracter)
    nodes = mark_corref(nodes, corref_clusters)
    local_doc = LocalDoc(doc)
    dep = Dependency(nodes, roots, local_doc, is_query=is_query)
    vertices, rels, id_map = (
        dep.solve_conjunctions()
        .mark_pronoun_antecedents()
        .mark_prefixes()
        .mark_vertex()
        .compress_dependencies()
        .calc_relationships()
    )
    time3 = time.time()
    if 3 in steps_set:
        print("\n[Step 3 - Vertices] (after coreference & vertex construction)")
        # for k, v in id_map.items():
        #     print(f"    Node '{k.text}' in [{v}]")
        vertex_objs = Vertex.from_nodes(vertices, id_map)
        for vertex in sorted(vertex_objs, key=lambda v: v.id):
            print(format_vertex(vertex))
    
    time4 = time.time()
    # 4) hyperedges
    hypergraph = LocalHypergraph.from_rels(vertices, rels, id_map, local_doc)
    hypergraph.original_text = text
    if 4 in steps_set:
        print("\n[Step 4 - Hyperedges]")
        for idx, edge in enumerate(hypergraph.hyperedges):
            root_text = edge.root.text()
            vertices = [v.text() for v in edge.vertices]
            print(
                f"[{idx}]  '{root_text}'({','.join(vertices)}); '{edge.text()}'"
            )
    time5 = time.time()
    print(f"\n⏱️ Timing (seconds): Step 1={time1-time0:.2f}({(time1-time0)/(time5-time0)*100:.1f}%) while loading takes {time15-time0:.2f}({(time15-time0)/(time5-time0)*100:.1f}%), Step 2={time2-time1:.2f}({(time2-time1)/(time5-time0)*100:.1f}%), Step 3={time3-time2:.2f}({(time3-time2)/(time5-time0)*100:.1f}%), Step 4={time4-time3:.2f}({(time4-time3)/(time5-time0)*100:.1f}%), Total={time5-time0:.2f}")
    return hypergraph

# The first expedition to reach the geographic South Pole was led by the Norwegian explorer Roald Amundsen. He and four others arrived at the pole on 14 December 1911, five weeks ahead of a British party led by Robert Falcon Scott as part of the Terra Nova Expedition. Amundsen and his team returned safely to their base, and later learned that Scott and his four companions had died on their return journey.

query, dataset, supports = load_musique_case("/home/vincent/.dataset/musique/x.json")


if __name__ == "__main__":

    parser = ArgumentParser(description="Debug spaCy dependency pipeline steps.")
    parser.add_argument("--output-dir", type=str, default="logs/debugs")
    parser.add_argument(
        "--steps",
        type=str,
        # default="1,2,3,4",
        default="no",
        help="Comma-separated steps to output: 1,2,3,4 or 'all' or `no`.",
    )
    args = parser.parse_args()
    steps = _parse_steps(args.steps)
    query_hypergraph = debug_text_to_hypergraph(query, output_dir=args.output_dir, steps=steps, is_query=True)
    query_hypergraph.save(args.output_dir + "/query_hypergraph.pkl")
    for i, data in enumerate(dataset):
        data_hypergraph = debug_text_to_hypergraph(data, output_dir=args.output_dir, steps=steps, is_query=False)
        data_hypergraph.save(args.output_dir + f"/data_hypergraph{i}.pkl")
    print(f"Supports: {supports}")
