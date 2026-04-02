from spacy.tokens import Doc, Span, Token
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from spacy.util import filter_spans

from hyper_simulation.hypergraph.corref import CorrefCluster

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
        # ✅ 修复 1: 跳过 None 的 cluster
        if cluster is None:
            continue
        
        # ✅ 修复 2: 过滤掉 cluster 中的 None span
        cluster = [span for span in cluster if span is not None and isinstance(span, (list, tuple)) and len(span) == 2]
        
        if not cluster:  # 过滤后为空的 cluster 也跳过
            continue
        
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

def _left_descendants(token: Token) -> list[Token]:
    # Calculate left descendants of a token, they MUST be continuous and end at token.i - 1
    descendants = []
    for left in reversed(list(token.lefts)):
        if left.i != token.i - 1:
            break
        descendants.append(left)
        descendants.extend(_left_descendants(left))
    return descendants

def _right_descendants(token: Token) -> list[Token]:
    # Calculate right descendants of a token, they MUST be continuous and start at token.i + 1
    descendants = []
    for right in token.rights:
        if right.i != token.i + 1:
            break
        descendants.append(right)
        descendants.extend(_right_descendants(right))
    return descendants

def calc_correfs_str(doc: Doc) -> set[str]:
    correfs: set[str] = set()
    clusters = getattr(doc._, "coref_clusters", None)
    if not clusters:
        return correfs
    
    # ✅ 修复：过滤掉 None 的 cluster
    clusters = [c for c in clusters if c is not None]
    if not clusters:
        return correfs
    
    text = doc.text
    clusters = _restrict_correfs(clusters, level=1)
    for cluster in clusters:
        for (start, end) in cluster:
            # ✅ 防御性：跳过无效的 span
            if start is None or end is None or start < 0 or end > len(text):
                continue
            correfs.add(text[start:end])
    correfs.update(_build_coref_span_map(doc).values())
    return correfs

def _build_coref_span_map(doc: Doc) -> dict[tuple[int, int], str]:
    span_map: dict[tuple[int, int], str] = {}
    clusters = getattr(doc._, "coref_clusters", None)
    if not clusters:
        return span_map
    
    # ✅ 修复：过滤掉 None 的 cluster
    clusters = [c for c in clusters if c is not None]
    if not clusters:
        return span_map
    
    clusters = _restrict_correfs(clusters, level=1)
    for cluster in clusters:
        if not cluster:
            continue
        # ✅ 防御性：检查 cluster[0] 是否有效
        if len(cluster) == 0 or cluster[0] is None or len(cluster[0]) != 2:
            continue
        canonical_span = doc.char_span(*cluster[0], alignment_mode="expand")
        if canonical_span is None:
            continue
        canonical_text = canonical_span.text
        for start_char, end_char in cluster:
            # ✅ 防御性：跳过无效的 span
            if start_char is None or end_char is None:
                continue
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
    
def combine_links(doc: Doc) -> list[Span]:
    links_to_merge: list[Span] = []
    links_token_idxs: set[int] = set()
    
    for token in doc:
        # if token equals '-' alone, combine with left and right 
        if token.text == "-" :
            if token.i - 1 >= 0 and token.i + 1 < len(doc):
                left_token, right_token = doc[token.i - 1], doc[token.i + 1]
                if left_token.dep_ == "conj" or right_token.dep_ == "conj":
                    continue
                span = doc[token.i - 1:token.i + 2]
                if links_token_idxs.intersection(range(span.start, span.end)):
                    continue
                links_to_merge.append(span)
                links_token_idxs.update(range(span.start, span.end))
        
        # if token end with '-' then combine with the right
        elif token.text.endswith("-") and token.i + 1 < len(doc):
            span = doc[token.i:token.i + 2]
            if links_token_idxs.intersection(range(span.start, span.end)):
                continue
            links_to_merge.append(span)
            links_token_idxs.update(range(span.start, span.end))
        
        # if token start with '-' then combine with the left
        elif token.text.startswith("-") and token.i - 1 >= 0:
            span = doc[token.i - 1:token.i + 1]
            if links_token_idxs.intersection(range(span.start, span.end)):
                continue
            links_to_merge.append(span)
            links_token_idxs.update(range(span.start, span.end))
            
    # Update the doc's entities to include the merged links if they overlap with existing entities
    new_ents = []
    for ent in doc.ents:
        span_start = ent.start
        span_end = ent.end
        
        # Extend entity by links if [span_start:span_end] has intersection with links_to_merge
        for link_span in links_to_merge:
            if not (link_span.end <= span_start or link_span.start >= span_end):
                # merge spans
                new_start = min(span_start, link_span.start)
                new_end = max(span_end, link_span.end)
                span_start = new_start
                span_end = new_end
        
        new_ent = Span(doc, span_start, span_end, label=ent.label_)
        new_ents.append(new_ent)
    
    doc.set_ents(filter_spans(new_ents), default="unmodified")
    
    return links_to_merge

def combine(doc: Doc, correfs: set[str]=set(), is_query: bool = False, corefs_clusters: list[CorrefCluster] = []) -> list[Span]:
    spans_to_merge: list[Span] = []
    ent_token_idxs: set[int] = set()
    bigram_lr_scores = _calc_bigram_likelihood_scores(doc)
    lr_threshold = 7.5
    
    new_ents = []
    
    # Entities
    for ent in doc.ents:
        if ent.label_ in {"ORDINAL", "CARDINAL"} and len(ent) == 1:
            continue
        if ent_token_idxs.intersection(range(ent.start, ent.end)):
            continue
        # print(f"Merging entity span: {ent.text}")
        
        # Extend entity span to include compound deps
        span_start = ent.start
        span_end = ent.end
        
        def is_compound_to_ent(token: Token, ent: Span):
            # True if token is a compound to a token in the entity, or a token in the entity is a compound to it
            if token.dep_ == "compound" and token.head in ent: # token -compound-> ent
                return True
            # for child in token.children: # ent -compound-> token
            #     if child.dep_ == "compound" and child in ent:
            #         return True
            return False
        
        # Extend left, if the left token is a compound to a token in the ent, or a token in ent is a compound to it.
        for i in range(ent.start - 1, -1, -1):
            if is_compound_to_ent(doc[i], ent) and not (is_query and doc[i].pos_ == "PROPN"):
                # Set the doc[i]'s entity type to ent_type
                span_start = i
            else:
                break
        # Extend right, if the right token is a compound to a token in the ent, or a token in ent is a compound to it.
        for i in range(ent.end, len(doc)):
            if is_compound_to_ent(doc[i], ent) and not (is_query and doc[i].pos_ == "PROPN"):
                span_end = i + 1
            else:
                break
            
        # Extend entity by links if [span_start:span_end] has intersection with links_to_merge
        # update the span_start and span_end to include the whole link span
        # Last, delete the link span in links_to_merge and links_token_idxs if it is merged
        # deleted_links = []
        # if links_token_idxs.intersection(range(span_start, span_end)):
        #     for link_span in links_to_merge:
        #         if not (link_span.end <= span_start or link_span.start >= span_end):
        #             # merge spans
        #             new_start = min(span_start, link_span.start)
        #             new_end = max(span_end, link_span.end)
        #             span_start = new_start
        #             span_end = new_end
        #             deleted_links.append(link_span)
        #     for link_span in deleted_links:
        #         if link_span in links_to_merge:
        #             links_to_merge.remove(link_span)
        #             for i in range(link_span.start, link_span.end):
        #                 if i in links_token_idxs:
        #                     links_token_idxs.remove(i)

        if ent_token_idxs.intersection(range(span_start, span_end)) or (span_start == ent.start and span_end == ent.end): # if the extended span intersects with existing spans, skip merging this entity
            spans_to_merge.append(ent)
            ent_token_idxs.update(range(ent.start, ent.end))
            continue
        
        new_ent = Span(doc, span_start, span_end, label=ent.label_)
        # doc.set_ents([new_ent], default="unmodified")
        new_ents.append(new_ent)

        spans_to_merge.append(new_ent)
        ent_token_idxs.update(range(span_start, span_end))

    doc.set_ents(filter_spans(new_ents), default="unmodified")
    
    # Noun phrases
    naive_dets = { "the", "a", "an" }
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
                # WARN: we remove the combine of `det` 
                # elif left.dep_ in {"advmod", "neg", "nummod", "quantmod", "npadvmod", "compound"} or (left.dep_ == "det" and left.text.lower() not in not_naive_dets):
                elif left.dep_ in {"advmod", "neg", "nummod", "quantmod", "npadvmod", "compound"} or (left.dep_ == "det" and  left.text.lower() not in naive_dets):
                    if is_query and left.dep_ == "compound" and left.pos_ == "PROPN":
                        break
                    if is_query and left.dep_ == "det" and left.tag_ == "WDT":
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
            # print(f"{token}'s lefts: {_left_descendants(token)}, rights: {_right_descendants(token)}")

            # for left in reversed(list(token.lefts)):
            for left in _left_descendants(token):
                if left.i != span_start - 1:
                    break
                if left.dep_ in {"aux", "auxpass", "neg", "advmod"}:
                    span_start = left.i
                else:
                    break
            
            # for right in token.rights:
            for right in _right_descendants(token):
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
                if right.dep_ in {"advmod", "acomp", "prep", "det"}: # WARN: considering conj/cc may meet bad cases
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
                if right.dep_ in {"advmod", "prep", "det"}:  # WARN: considering conj/cc may meet bad cases
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
            
    # Union the links_to_merge and spans_to_merge
    # where if they have intersection, merge them into a bigger span
    # if links_to_merge subset to spans_to_merge, do nothing
    
    
    # for link_span in links_to_merge:
    #     # print(f"Considering link span: {link_span}")
    #     has_intersection = False
    #     for span in spans_to_merge:
    #         if not (link_span.end <= span.start or link_span.start >= span.end):
                
    #             # if the link_span subset of span, break
    #             if link_span.start >= span.start and link_span.end <= span.end:
    #                 has_intersection = True
    #                 break
                
    #             # merge spans
    #             new_start = min(link_span.start, span.start)
    #             new_end = max(link_span.end, span.end)
    #             new_span = doc[new_start:new_end]
    #             spans_to_merge.remove(span)
    #             spans_to_merge.append(new_span)
    #             has_intersection = True
    #             break
    #     if not has_intersection:
    #         spans_to_merge.append(link_span)
            
    corefs_clusters, corref_to_merge = CorrefCluster.fixup_clusters(corefs_clusters, spans_to_merge)
    
    spans_to_merge.extend(corref_to_merge)

    spans_to_merge = sorted(spans_to_merge, key=lambda s: s.start, reverse=True)
    
    return spans_to_merge