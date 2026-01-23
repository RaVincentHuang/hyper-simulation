from spacy.tokens import Doc, Span, Token

def get_level_order(doc: Doc, reversed=False) -> list[Token]:
    levels: dict[int, list[Token]] = {}
    max_level = 0
    for token in doc:
        level = 0
        current = token
        while current.head != current:
            level += 1
            current = current.head
        if level not in levels:
            levels[level] = []
        levels[level].append(token)
        if level > max_level:
            max_level = level
    ordered_tokens: list[Token] = []
    # if reversed==True, then root to leaf, else leaf to root
    if reversed:
        for level in range(max_level, -1, -1):
            ordered_tokens.extend(levels.get(level, []))
    else:
        for level in range(0, max_level + 1):
            ordered_tokens.extend(levels.get(level, []))
    return ordered_tokens

def calc_correfs_str(doc: Doc) -> set[str]:
    correfs: set[str] = set()
    clusters = getattr(doc._, "coref_clusters", None)
    if not clusters:
        return correfs
    text = doc.text
    for cluster in clusters:
        for (start, end) in cluster:
            correfs.add(text[start:end])
    correfs.update(_build_coref_span_map(doc).values())
    return correfs


def _build_coref_span_map(doc: Doc) -> dict[tuple[int, int], str]:
    span_map: dict[tuple[int, int], str] = {}
    clusters = getattr(doc._, "coref_clusters", None)
    if not clusters:
        return span_map
    for cluster in clusters:
        if not cluster:
            continue
        canonical_span = doc.char_span(*cluster[0], alignment_mode="expand")
        if canonical_span is None:
            continue
        canonical_text = canonical_span.text
        for start_char, end_char in cluster:
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            if span is None:
                continue
            span_map[(span.start, span.end)] = canonical_text
    return span_map

def _calc_same_tokens(doc: Doc, correfs: set[str]) -> dict[str, list[tuple[int, int]]]:
    token_map: dict[str, list[tuple[int, int]]] = {}
    # calc the that doc[i:j].text == doc[k:l].text, save all (i, j) (k, l) in token_map[doc[i:j].text]
    # print(f"Correfs: {correfs}")
    resolved_span_map = _build_coref_span_map(doc)
    # print(f"Resolved span map: {resolved_span_map}")
    for i in range(len(doc)):
        for j in range(i + 1, len(doc) + 1):
            span = doc[i:j]
            span_text = resolved_span_map.get((span.start, span.end), span.text)
            if len(correfs) > 0 and span_text not in correfs:
                continue
            token_map.setdefault(span_text, []).append((span.start, span.end))
            
    # if j = i + 1, or len(token_map[span_text]) == 1, remove it
    # print(f"Token map: {token_map}")
    token_map = {k: v for k, v in token_map.items() if len(v) > 1 and (v[0][1] - v[0][0]) > 1}
    # print(f"Token map after filter: {token_map}")
    return token_map
    

def combine(doc: Doc, correfs: set[str]=set()) -> list[Span]:
    spans_to_merge = []
    ent_token_idxs: set[int] = set()

    # Correferences
    token_map = _calc_same_tokens(doc, correfs)
    # print(f"Token map: {token_map}")
    for span_text, positions in token_map.items():
        # print(f"Considering coreference span: {span_text}")
        for start, end in positions:
            span = doc[start:end]
            if ent_token_idxs.intersection(range(span.start, span.end)):
                continue
            spans_to_merge.append(span)
            ent_token_idxs.update(range(span.start, span.end))
    
    # Entities
    for ent in doc.ents:
        if ent_token_idxs.intersection(range(ent.start, ent.end)):
            continue
        # print(f"Merging entity span: {ent.text}")
        spans_to_merge.append(ent)
        ent_token_idxs.update(range(ent.start, ent.end))

    # Noun phrases
    not_naive_dets = {"all", "both", "every", "each", "either", "neither"}
    
    # for chunk in doc.noun_chunks:
    #     span_start = chunk.start
    #     span_end = chunk.end
    #     for left in reversed(list(chunk.lefts)):
    #         if left.dep_ in {"amod", "advmod", "neg", "nummod", "quantmod", "npadvmod"} or (left.dep_ == "det" and left.text.lower() not in not_naive_dets):
    #                 span_start = left.i
    #         else:
    #             break
    #     for right in chunk.rights:
    #         if right.dep_ in {"amod", "advmod", "neg", "nummod", "quantmod", "npadvmod"}:
    #             span_end = right.i + 1
    #         else:
    #             break

    #     if span_start + 1 == span_end:
    #             continue 
    #     span = doc[span_start:span_end]
    #     print(f"Considering noun phrase span: {span}, from chunk: {chunk}")
    #     if ent_token_idxs.intersection(range(span.start, span.end)):
    #         continue
        
    #     spans_to_merge.append(span)
    #     ent_token_idxs.update(range(span.start, span.end))

    # Add [`amod`, `advmod`, `neg`, `nummod`, `quantmod`, `npadvmod`] modifiers to noun phrases
    noun_token_idxs: set[int] = set()
    max_span_tokens = 4
    spans_to_merge_on_noun: dict[tuple[int,int], Span] = {}
    # Leaf-to-root traversal
    doc_by_level = list(doc)
    # reorder to leaf-to-root
    for token in doc:
        if token.pos_ == "NOUN":
            span_start = token.i
            span_end = token.i + 1
            
            for left in reversed(list(token.lefts)):
                if left.i != span_start - 1:
                    break
                # {"advmod", "neg", "nummod", "quantmod", "npadvmod", "amod", "compound"} or {"advmod", "neg", "nummod", "quantmod", "npadvmod"}
                if left.dep_ in {"advmod", "neg", "nummod", "quantmod", "npadvmod", "amod", "compound"} or (left.dep_ == "det" and left.text.lower() not in not_naive_dets):
                    span_start = left.i
                else:
                    break

            for right in token.rights:
                if right.i != span_end:
                    break
                if right.dep_ in {"case", "advmod", "neg", "nummod", "quantmod", "npadvmod"}:
                    span_end = right.i + 1
                else:
                    break
                
            if span_start + 1 == span_end or (span_end - span_start) > max_span_tokens:
                continue
            span = doc[span_start:span_end]
            # print(f"Considering noun phrase span: {span}")
            
            if noun_token_idxs.intersection(range(span.start, span.end)):
                for start, end in spans_to_merge_on_noun.keys():
                    if not (span.end <= start or span.start >= end):
                        # merge spans
                        new_start = min(span.start, start)
                        new_end = max(span.end, end)
                        if new_end - new_start > max_span_tokens:
                            break
                        new_span = doc[new_start:new_end]
                        spans_to_merge_on_noun.pop((start, end))
                        spans_to_merge_on_noun[(new_start, new_end)] = new_span
                        break
                continue
            spans_to_merge_on_noun[(span.start, span.end)] = span
            noun_token_idxs.update(range(span.start, span.end))
    
    for span in spans_to_merge_on_noun.values():
        if ent_token_idxs.intersection(range(span.start, span.end)):
            continue
        spans_to_merge.append(span)
        ent_token_idxs.update(range(span.start, span.end))

    # Verbal phrases
    for token in doc:
        if token.pos_ == "VERB":
            span_start = token.i
            span_end = token.i + 1

            for left in reversed(list(token.lefts)):
                if left.i != span_start - 1:
                    break
                if left.dep_ in {"aux", "auxpass", "neg", "advmod"}:
                    span_start = left.i
                else:
                    break
            
            for right in token.rights:
                # if right is not near the verb, break
                if right.i != span_end:
                    break
                if right.dep_ in {"prt", "advmod", "acomp", "xcomp", "ccomp"}:
                    span_end = right.i + 1
                else:
                    break
            if span_start + 1 == span_end:
                continue 
            span = doc[span_start:span_end]
            if ent_token_idxs.intersection(range(span.start, span.end)):
                continue
            
            # print(f"Verbal phrase span: {span}")
            spans_to_merge.append(span)
            ent_token_idxs.update(range(span.start, span.end))

    # Adjectival phrases
    for token in doc:
        if token.pos_ == "ADJ" and token.dep_ in {"amod"}:
            span_start = token.i
            span_end = token.i + 1

            for left in reversed(list(token.lefts)):
                if left.i != span_start - 1:
                    break
                if left.dep_ in {"advmod", "neg"}:
                    span_start = left.i
                else:
                    break

            for right in token.rights:
                if right.i != span_end:
                    break
                if right.dep_ in {"advmod", "acomp", "prep", "conj", "cc", "det"}:
                    span_end = right.i + 1
                else:
                    break
            if span_start + 1 == span_end:
                continue 
            span = doc[span_start:span_end]
            if ent_token_idxs.intersection(range(span.start, span.end)):
                continue
            
            # print("Adjectival phrase span: ", span)
            spans_to_merge.append(span)
            ent_token_idxs.update(range(span.start, span.end))

    spans_to_merge = sorted(spans_to_merge, key=lambda s: s.start, reverse=True)
    return spans_to_merge