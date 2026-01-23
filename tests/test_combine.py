from math import pi
import spacy
from spacy import displacy
import coreferee, spacy
from fastcoref import spacy_component
from combine import combine
nlp = spacy.load('en_core_web_trf')

text = "When Marry was in school, I like her book."

doc = nlp(text)

for token in doc:
    print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")  


spans_to_merge = combine(doc)
with doc.retokenize() as retokenizer:
    for span in spans_to_merge:
        retokenizer.merge(span)
    

print("\nAfter merging:\n")
for token in doc:
    print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")

displacy.serve(doc, style="dep")
