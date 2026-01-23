import coreferee, spacy
from fastcoref import spacy_component
from combine import combine, calc_correfs_str
from dependency import Dependency, Node, LocalDoc
import os, glob, json

nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('fastcoref', 
            config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})

text = "a new software's function is to this process data. The goal of this process is the optimization."

# Give all node that share the same ID a single string.
def load_ctx_texts(json_dir):
    texts = []
    def _extract_from_obj(obj):
        if not isinstance(obj, dict):
            return
        q = obj.get("q")
        if isinstance(q, str):
            texts.append(q)
        d = obj.get("d")
        if isinstance(d, list):
            for item in d:
                if isinstance(item, str):
                    texts.append(item)
    for pattern in ("*.json", "*.jsonl"):
        for path in glob.glob(os.path.join(json_dir, pattern)):
            with open(path, "r", encoding="utf-8") as f:
                if path.lower().endswith(".jsonl"):
                    # 逐行处理 JSONL
                    for line in f:
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            _extract_from_obj(obj)
                else:
                    obj = json.load(f)
                    _extract_from_obj(obj)
    # print(texts)
    return texts

def get_rel(nlp, text):
    doc = nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
    print(f"Resolved Text: {doc._.resolved_text}")
    correfs = calc_correfs_str(doc)
    spans_to_merge = combine(doc, correfs)
    with doc.retokenize() as retokenizer:
        for span in spans_to_merge:
            retokenizer.merge(span)

    # for token in doc:
    #     print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")

    nodes, roots = Node.from_doc(doc)
# print(f"Dependency Tree Nodes:{nodes}")
# print(f"Dependency Tree Roots:{roots}")
    local_doc = LocalDoc(doc)
    dep = Dependency(nodes, roots, local_doc)
    vertices, rel, id_map = dep.solve_conjunctions().mark_pronoun_antecedents().mark_prefixes().mark_vertex().compress_dependencies().calc_relationships()
    return vertices, rel, id_map
#     print(f"Dependency Relationships:\n")
#     for r in rel:
#         print(r)

# Example usage: set json_dir to the folder containing your JSON files
json_dir = f"D:\\desktop\\experiment\\spacy\\test_data"
target_dir = f"D:\\desktop\\experiment\\spacy\\output_data"
ctx_texts = load_ctx_texts(json_dir)
cnt = 0
now = 25
for text in ctx_texts:
    print(f"{cnt}: {text}")
    # if cnt < now: # 这个是为啥？
        # cnt += 1
        # continue
    # if cnt > now:
    #     continue
    vertices, rel, id_map = get_rel(nlp, text)
    res = ""
    for r in rel:
        res += f"{r}\n"
    single_name: dict[int, str] = {}
    for vertex, vid in id_map.items():
        if vid not in single_name:
            single_name[vid] = vertex.text

    for vertex in vertices:
        res += f"Vertex: {vertex.text}, ID: {id_map[vertex]} ['{single_name[id_map[vertex]]}']\n"

    res += f"Squeezed rate is {len(single_name)}/{len(vertices)} = {len(single_name)/len(vertices):.2%}"
    with open(os.path.join(target_dir, f"{cnt}.txt"), "w", encoding="utf-8") as f:
        f.write(res)
    cnt += 1
