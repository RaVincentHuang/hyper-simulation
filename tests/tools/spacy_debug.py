from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Set

import spacy
from spacy import displacy
from fastcoref import spacy_component

from spacy.tokens import Doc

from hyper_simulation.hypergraph.combine import combine, calc_correfs_str
from hyper_simulation.hypergraph.dependency import Node, LocalDoc, Dependency
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex, Node, Hyperedge


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
    return nlp


def print_dep_tokens(doc: Doc, title: str) -> None:
    print(f"\n[{title}] Tokens / Lemma / Dep / Head / Ent / POS / TAG")
    for token in doc:
        print(
            "Token: '{text}', Lemma: '{lemma}', Dep: {dep} ['{head}'], Ent: {ent}, POS: {pos}, TAG: {tag}".format(
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
        f"    - '{node.text}' (pos={node.pos.name}, dep={node.dep.name}, ent={node.ent.name})"
        for node in vertex.nodes
    )
    return f"[{vertex.id}] '{vertex.text()}'\n{nodes}"


def _parse_steps(steps: str) -> Set[int]:
    if not steps or steps.strip().lower() == "all":
        return {1, 2, 3, 4}
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
    text: str, output_dir: str = "logs/dep_debug", steps: Iterable[int] | None = None
) -> LocalHypergraph:
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    steps_set = set(steps) if steps is not None else {1, 2, 3, 4}

    nlp = get_nlp()
    cfg = {"fastcoref": {"resolve_text": True}} if "fastcoref" in nlp.pipe_names else {}
    doc = nlp(text, component_cfg=cfg)

    print(f"\n[Input Text]:\n {text}")
    
    # 1) 原始依存分析
    if 1 in steps_set:
        print_dep_tokens(doc, "Step 1 - Raw (before combine)")
        render_dep_html(doc, output_base / "step1_raw_dep.html", "Step 1 - Raw (before combine)")

    # combine + retokenize
    correfs = calc_correfs_str(doc) if hasattr(doc._, "coref_clusters") else set()
    spans_to_merge = combine(doc, correfs)
    with doc.retokenize() as retokenizer:
        for span in spans_to_merge:
            retokenizer.merge(span)

    # 2) combine 后依存分析
    if 2 in steps_set:
        print_dep_tokens(doc, "Step 2 - Combined (after retokenize)")
        render_dep_html(
            doc, output_base / "step2_combined_dep.html", "Step 2 - Combined (after retokenize)"
        )

    # 3) 指代消解 + vertices
    nodes, roots = Node.from_doc(doc)
    local_doc = LocalDoc(doc)
    dep = Dependency(nodes, roots, local_doc)
    vertices, rels, id_map = (
        dep.solve_conjunctions()
        .mark_pronoun_antecedents()
        .mark_prefixes()
        .mark_vertex()
        .compress_dependencies()
        .calc_relationships()
    )
    if 3 in steps_set:
        print("\n[Step 3 - Vertices] (after coreference & vertex construction)")
        vertex_objs = Vertex.from_nodes(vertices, id_map)
        for vertex in sorted(vertex_objs, key=lambda v: v.id):
            print(format_vertex(vertex))

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

    return hypergraph


if __name__ == "__main__":
    text = "What is the title of the 13th episode of the American fantasy drama television series that premiered on October 23, 2011, on ABC?"
    
    parser = ArgumentParser(description="Debug spaCy dependency pipeline steps.")
    parser.add_argument("--output-dir", type=str, default="logs/dep_debug")
    parser.add_argument(
        "--steps",
        type=str,
        default="1,2,3,4",
        help="Comma-separated steps to output: 1,2,3,4 or 'all'.",
    )
    args = parser.parse_args()
    steps = _parse_steps(args.steps)
    debug_text_to_hypergraph(text, output_dir=args.output_dir, steps=steps)
