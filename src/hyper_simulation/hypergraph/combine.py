from spacy.tokens import Doc, Span, Token
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

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

def _restrict_correfs(clusters: list[list[tuple[int, int]]], level: int=0) -> list[list[tuple[int, int]]]:
    restricted = []
    for cluster in clusters:
        # Level 0: Do not restrict
        # Level 1: All spans in a cluster can not be sub of another span in the same cluster
        # Level 2: All spans in a cluster can not have intersection with another span in the same cluster
        if level == 0:
            restricted.append(cluster)
        elif level == 1:
            is_sub = False
            for span in cluster:
                if is_sub:
                    break
                for other_span in cluster:
                    if span == other_span:
                        continue
                    if span[0] >= other_span[0] and span[1] <= other_span[1]:
                        is_sub = True
                        break
            if not is_sub:
                restricted.append(cluster)
        elif level == 2:
            has_intersection = False
            for span in cluster:
                if has_intersection:
                    break
                for other_span in cluster:
                    if span == other_span:
                        continue
                    if not (span[1] <= other_span[0] or span[0] >= other_span[1]):
                        has_intersection = True
                        break
            if not has_intersection:
                restricted.append(cluster)
    return restricted

def calc_correfs_str(doc: Doc) -> set[str]:
    correfs: set[str] = set()
    clusters = getattr(doc._, "coref_clusters", None)
    if not clusters:
        return correfs
    text = doc.text
    clusters = _restrict_correfs(clusters, level=1)
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
    clusters = _restrict_correfs(clusters, level=1)
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
    token_map: dict[str, set[tuple[int, int]]] = {}
    resolved_span_map = _build_coref_span_map(doc)
    n = len(doc)

    for i in range(n):
        for k in range(i + 1, n):
            # longest common span starting at i and k
            max_len = 0
            while i + max_len < n and k + max_len < n:
                if doc[i + max_len].text != doc[k + max_len].text:
                    break
                max_len += 1
            if max_len <= 1:
                continue

            # trim edge SPACE/PUNCT tokens from both sides
            left_trim = 0
            right_trim = 0
            while left_trim < max_len and (doc[i + left_trim].pos_ in {"SPACE", "PUNCT"} or doc[k + left_trim].pos_ in {"SPACE", "PUNCT"}):
                left_trim += 1
            while right_trim < max_len - left_trim and (doc[i + max_len - 1 - right_trim].pos_ in {"SPACE", "PUNCT"} or doc[k + max_len - 1 - right_trim].pos_ in {"SPACE", "PUNCT"}):
                right_trim += 1

            span_len = max_len - left_trim - right_trim
            if span_len <= 1:
                continue

            start_i = i + left_trim
            end_i = start_i + span_len
            start_k = k + left_trim
            end_k = start_k + span_len

            span_i = doc[start_i:end_i]
            span_k = doc[start_k:end_k]
            span_text_i = resolved_span_map.get((span_i.start, span_i.end), span_i.text)
            span_text_k = resolved_span_map.get((span_k.start, span_k.end), span_k.text)
            if span_text_i != span_text_k:
                continue
            if len(correfs) > 0 and span_text_i not in correfs:
                continue

            token_map.setdefault(span_text_i, set()).add((span_i.start, span_i.end))
            token_map.setdefault(span_text_i, set()).add((span_k.start, span_k.end))

    token_map_filtered: dict[str, list[tuple[int, int]]] = {}
    for text, positions in token_map.items():
        pos_list = sorted(positions)
        if len(pos_list) > 1 and (pos_list[0][1] - pos_list[0][0]) > 1:
            token_map_filtered[text] = pos_list
    return token_map_filtered


def _calc_bigram_likelihood_scores(doc: Doc) -> dict[tuple[str, str], float]:
    tokens = [token.text.lower() for token in doc]
    if len(tokens) < 2:
        return {}
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(BigramAssocMeasures.likelihood_ratio)
    return {pair: score for pair, score in scored}
    

def combine(doc: Doc, correfs: set[str]=set(), is_query: bool = False) -> list[Span]:
    spans_to_merge = []
    ent_token_idxs: set[int] = set()
    bigram_lr_scores = _calc_bigram_likelihood_scores(doc)
    lr_threshold = 8.0

    # Correferences
    token_map = _calc_same_tokens(doc, correfs)
    # print(f"Correfs: {correfs}")
    # print(f"Token map: {token_map}")
    for span_text, positions in token_map.items():
        # print(f"Considering coreference span: {span_text}")
        for start, end in positions:
            span = doc[start:end]
            if ent_token_idxs.intersection(range(span.start, span.end)):
                continue
            # print(f"Considering coreference span: {span.text}")
            spans_to_merge.append(span)
            ent_token_idxs.update(range(span.start, span.end))
    
    # - linked phrases
    for token in doc:
        # if token equals '-' alone, combine with left and right
        if token.pos_ == "PUNCT" and token.text == "-" :
            if token.i - 1 >= 0 and token.i + 1 < len(doc):
                span = doc[token.i - 1:token.i + 2]
                if ent_token_idxs.intersection(range(span.start, span.end)):
                    continue
                spans_to_merge.append(span)
                ent_token_idxs.update(range(span.start, span.end))
        
        # if token end with '-' then combine with the right
        elif token.text.endswith("-") and token.i + 1 < len(doc):
            span = doc[token.i:token.i + 2]
            if ent_token_idxs.intersection(range(span.start, span.end)):
                continue
            spans_to_merge.append(span)
            ent_token_idxs.update(range(span.start, span.end))
        
        # if token start with '-' then combine with the left
        elif token.text.startswith("-") and token.i - 1 >= 0:
            span = doc[token.i - 1:token.i + 1]
            if ent_token_idxs.intersection(range(span.start, span.end)):
                continue
            spans_to_merge.append(span)
            ent_token_idxs.update(range(span.start, span.end))
            
    # Entities
    for ent in doc.ents:
        if ent.label_ in {"ORDINAL", "CARDINAL"} and len(ent) == 1:
            continue
        if ent_token_idxs.intersection(range(ent.start, ent.end)):
            continue
        # print(f"Merging entity span: {ent.text}")
        spans_to_merge.append(ent)
        ent_token_idxs.update(range(ent.start, ent.end))

    # Noun phrases
    not_naive_dets = {"all", "both", "every", "each", "either", "neither", "whichever", "whatever"}
    wh_dets = {"what", "which", "whose", "whichever", "whatever"}

    # Add [`amod`, `advmod`, `neg`, `nummod`, `quantmod`, `npadvmod`] modifiers to noun phrases
    noun_token_idxs: set[int] = set()
    max_span_tokens = 5
    spans_to_merge_on_noun: dict[tuple[int,int], Span] = {}
    # Leaf-to-root traversal
    doc_by_level = list(doc)
    # reorder to leaf-to-root
    for token in doc_by_level:
        if token.pos_ == "NOUN":
            span_start = token.i
            span_end = token.i + 1
            for left in reversed(list(token.lefts)):
                # print(f"Left token: {left}, dep: {left.dep_}, token: {token}: {left.i != span_start - 1}")
                if left.i != span_start - 1:
                    break
                # {"advmod", "neg", "nummod", "quantmod", "npadvmod", "compound"} or {"advmod", "neg", "nummod", "quantmod", "npadvmod"}
                if left.dep_ == "amod":
                    pair = (left.text.lower(), left.head.text.lower())
                    score = bigram_lr_scores.get(pair, 0.0)
                    # print(f"Bigram LR score for {pair}: {score} - {score >= lr_threshold}")
                    if score >= lr_threshold:
                        span_start = left.i
                    else:
                        break
                elif left.dep_ in {"advmod", "neg", "nummod", "quantmod", "npadvmod", "compound"} or (left.dep_ == "det" and left.text.lower() not in not_naive_dets):
                    if is_query and left.dep_ == "det" and left.text.lower() in wh_dets:
                        break
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
            # print(f"Considering noun phrase span: {span}: {[doc[token] for token in noun_token_idxs.intersection(range(span.start, span.end))]}")
            
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
                        noun_token_idxs.update(range(new_start, new_end)) # WARNING: may over-add tokens
                        break
                continue
            spans_to_merge_on_noun[(span.start, span.end)] = span
            noun_token_idxs.update(range(span.start, span.end))
    
    for span in spans_to_merge_on_noun.values():
        if ent_token_idxs.intersection(range(span.start, span.end)):
            continue
        # print(f"Merging noun phrase span: {span.text}")
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
        if token.pos_ == "ADJ" and token.dep_ in {"amod", "acomp"}: # WARN: acomp may meet bad cases
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
    
    # Adverbs phrases
    for token in doc:
        if token.pos_ == "ADV" and token.dep_ in {"advmod"}:
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
                if right.dep_ in {"advmod", "prep", "conj", "cc", "det"}:
                    span_end = right.i + 1
                else:
                    break
            if span_start + 1 == span_end:
                continue 
            span = doc[span_start:span_end]
            if ent_token_idxs.intersection(range(span.start, span.end)):
                continue
            
            # print("Adverb phrase span: ", span)
            spans_to_merge.append(span)
            ent_token_idxs.update(range(span.start, span.end))

    spans_to_merge = sorted(spans_to_merge, key=lambda s: s.start, reverse=True)
    return spans_to_merge