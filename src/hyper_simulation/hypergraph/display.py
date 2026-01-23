from math import pi
import spacy
from spacy import displacy
import coreferee, spacy
from fastcoref import spacy_component
from hyper_simulation.hypergraph.combine import combine, calc_correfs_str

nlp0 = spacy.load('en_core_web_trf')
nlp0.add_pipe('fastcoref', 
            config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})

nlp1 = spacy.load('en_core_web_trf')

text = "Scholar Nilsson delivered a keynote at Stockholmsmässan on August. He also participated in roundtable discussions. That day, the venue hosted an AI ethics seminar, which featured his keynote and discussions."
doc1 = nlp0(text, component_cfg={"fastcoref": {'resolve_text': True}})
print(f"Resolved Text: {doc1._.resolved_text}")
doc2 = nlp1(doc1._.resolved_text)
for token in doc2:
    print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")
print("\nMerging spans...\n")

correfs = calc_correfs_str(doc1)
spans_to_merge = combine(doc2, correfs)
with doc2.retokenize() as retokenizer:
    for span in spans_to_merge:
        retokenizer.merge(span)

for token in doc2:
    print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")  

displacy.serve(doc2, style="dep")
