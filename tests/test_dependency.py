import coreferee, spacy
from fastcoref import spacy_component
from hyper_simulation.hypergraph.combine import combine, calc_correfs_str
from hyper_simulation.hypergraph.dependency import Dependency, Node, LocalDoc
from hyper_simulation.hypergraph.hypergraph import Hypergraph


nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('fastcoref', 
            config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})
filename = "data/hypergraph/query_hypergraph.pkl"
text = "Scholar Nilsson delivered a keynote at Stockholmsmässan on August. He also participated in roundtable discussions. That day, the venue hosted an AI ethics seminar, which featured his keynote and discussions."
# text = "At Stockholmsmässan, Lars Nilsson gave a presentation on Aug. 15. He was also engaged in debates. A seminar on quantum computing was held there that same day, which included his presentation and debates."
# text = "EcoTech developed an AI application that analyzes consumption patterns to optimize power usage."
# text = "A technology company, has created a new software. This software utilizes artificial intelligence. Its function is to analyze data from users. The goal of this process is the optimization of energy consumption."
# text = "In the process, María has become ostracized from the villagers: a group led by a man named Carlos calls her 'la bruja' ('the witch'). In addition, Travis must silence an American TV reporter named Patty Clark who has been exposing the pollution of the lake. In Chimayo, children Andrea and Glen Anderson observe strange ripples in the lake. Glen claims they belong to an animal he has been sighting in the lake for some time. Their father Pete (Anthony Eisley), administrator of the Durado plant, is introduced to Mayor Montero and his daughter, helicopter pilot Juanita."
# text = "High-pressure systems stop air from rising into the colder regions of the atmosphere where water can condense. What will most likely result if a high-pressure system remains in an area for a long period of time?"
# text = "I love apple, which make me happy."
# text = "Thus, in are equal numbers of molecules evaporating from the water as there are condensing back into the water. If the relative humidity becomes greater than 100%, it is called supersaturated. Supersaturation occurs in the absence of condensation nuclei. Since the saturation vapor pressure is proportional to temperature, cold air has a lower saturation point than warm air. The difference between these values is the basis for the formation of clouds. When saturated air cools, it can no longer contain the same amount of water vapor. If the conditions are right, the excess water will condense out of the air until the the airfoil and flight control surfaces."
doc = nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
correfs = calc_correfs_str(doc)
for token in doc:
    print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")
spans_to_merge = combine(doc, correfs)
with doc.retokenize() as retokenizer:
    for span in spans_to_merge:
        retokenizer.merge(span)

for token in doc:
    print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")

nodes, roots = Node.from_doc(doc)
# print(f"Dependency Tree Nodes:{nodes}")
# print(f"Dependency Tree Roots:{roots}")
# Print coreference information for nodes
print(f"\nCoreference Information for Nodes:")
for node in nodes:
    if node.correfence_id is not None:
        primary_status = "PRIMARY" if node.is_correfence_primary else "REFERENCE"
        print(f"  Node '{node.text}' (index={node.index}): correfence_id={node.correfence_id}, is_primary={node.is_correfence_primary} [{primary_status}]")
local_doc = LocalDoc(doc)
dep = Dependency(nodes, roots, local_doc)
vertices, rel, id_map = dep.solve_conjunctions().mark_pronoun_antecedents().mark_prefixes().mark_vertex().compress_dependencies().calc_relationships()
print(f"Dependency Relationships:\n")
for r in rel:
    print(r)
    
# Print coreference map information
if hasattr(dep, 'correfence_map') and dep.correfence_map:
    print(f"\nCoreference Map (Reference -> Primary):")
    for ref_node, primary_node in dep.correfence_map.items():
        print(f"  '{ref_node.text}' (correfence_id={ref_node.correfence_id}) -> '{primary_node.text}' (correfence_id={primary_node.correfence_id}, PRIMARY)")
else:
    print(f"\nNo coreference map found or empty.")

# Give all node that share the same ID a single string.
single_name: dict[int, str] = {}
for vertex, vid in id_map.items():
    if vid not in single_name:
        single_name[vid] = vertex.text

for vertex in dep.vertexes:
    corref_info = ""
    if vertex.correfence_id is not None:
        primary_status = "PRIMARY" if vertex.is_correfence_primary else "REF"
        corref_info = f", CorefID={vertex.correfence_id} [{primary_status}]"
    print(f"Vertex: {vertex.text}, ID: {id_map[vertex]} ['{single_name[id_map[vertex]]}']{corref_info}")

print(f"Squeezed rate is {len(single_name)}/{len(dep.vertexes)} = {len(single_name)/len(dep.vertexes):.2%}")

local_doc = LocalDoc(doc)

hypergraph = Hypergraph.from_rels(vertices, rel, id_map, local_doc)

hypergraph.save(filename)

