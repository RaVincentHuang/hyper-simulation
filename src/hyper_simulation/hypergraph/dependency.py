from enum import Enum, IntEnum
from thefuzz import process

from hyper_simulation.hypergraph.abstraction import TokenAbstractor

class QueryType(Enum):
    BELONGS = 1
    WHAT = 2
    WHICH = 3
    PERSON = 4
    ATTRIBUTE = 5
    NUMBER = 6
    TIME = 7
    LOCATION = 8
    REASON = 9

class Pos(IntEnum):
    ADP = 1
    ADV = 2
    ADJ = 3
    AUX = 4
    CCONJ = 5
    DET = 6
    INTJ = 7
    NOUN = 8
    NUM = 9
    PART = 10
    PRON = 11
    PROPN = 12
    PUNCT = 13
    SCONJ = 14
    SYM = 15
    VERB = 16
    X = 17
    SPACE = 18

class Tag(IntEnum):
    CC = 1
    CD = 2
    DT = 3
    EX = 4
    FW = 5
    IN = 6
    JJ = 7
    JJR = 8
    JJS = 9
    MD = 10
    NN = 11
    NNS = 12
    NNP = 13
    NNPS = 14
    POS = 15
    PRP = 16
    PRPD = 17
    RB = 18
    RBR = 19
    RBS = 20
    RP = 21
    TO = 22
    UH = 23
    VB = 24
    VBZ = 25
    VBP = 26
    VBD = 27
    VBN = 28
    VBG = 29
    WP = 30
    WPD = 31
    WRB = 32
    _SP = 33
    HYPH = 34
    ADD = 35
    WDT = 36
    PDT = 37
    XX = 38
    NFP = 39
    SYM = 40
    LS = 41
    
    WILDCARD = 99

class Dep(IntEnum):
    nsubj = 1
    nsubjpass = 2
    csubj = 3
    csubjpass = 4
    dobj = 5
    iobj = 6
    pobj = 7
    dative = 8
    
    amod = 9
    advmod = 10
    nummod = 11
    quantmod = 12
    npadvmod = 13
    neg = 14
    
    acl = 15
    advcl = 16
    ccomp = 17
    xcomp = 18
    relcl = 19
    mark = 20
    
    prep = 21
    agent = 22
    cc = 23
    conj = 24
    case = 25
    prt = 26
    
    appos = 27
    attr = 28
    acomp = 29
    oprd = 30
    aux = 31
    auxpass = 32
    expl = 33
    parataxis = 34
    meta = 35
    det = 36
    poss = 37
    predet = 38
    preconj = 39
    intj = 40
    punct = 41
    dep = 42
    
    compound = 43
    pcomp = 44
    nmod = 45

    ROOT = 46

class Entity(Enum):
    PERSON = 1
    NORP = 2
    FAC = 3
    ORG = 4
    GPE = 5
    LOC = 6
    PRODUCT = 7
    EVENT = 8
    WORK_OF_ART = 9
    LAW = 10
    LANGUAGE = 11
    DATE = 12
    TIME = 13
    PERCENT = 14
    MONEY = 15
    QUANTITY = 16
    ORDINAL = 17
    CARDINAL = 18
    NOT_ENTITY = 99

dead_dep = {Dep.dative, Dep.prt, Dep.parataxis}
solved_dep = {Dep.meta, Dep.poss, Dep.det, Dep.predet, Dep.intj}


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

class Node:
    def __init__(self, text: str, pos: Pos, tag: Tag, dep: Dep, ent: Entity, lemma: str, index: int) -> None:
        self.text = text
        self.original_text = text
        self.pos: Pos = pos
        self.tag: Tag = tag
        self.dep: Dep = dep
        self.ent: Entity = ent
        self.lemma: str = lemma
        self.sentence: str = text
        self.covered_sentence: str = text
        self.sentence_start: int = -1
        self.sentence_end: int = -1
        self.index = index
        
        self.is_query = False
        self.query_type: QueryType | None = None
        self.query_attribute: str | None = None
        
        self.is_vertex = False
        self.former_nodes: list[Node] = []
        
        self.dominator = False
        
        self.pronoun_antecedent: Node | None = None
        
        self.prefix_prep: str | None = None
        self.suffix_prep: str | None = None
        self.prefix_agent: str | None = None
        self.suffix_agent: str | None = None
        self.prefix_index: int | None = None
        self.suffix_index: int | None = None
        
        self.correfence_id: int | None = None
        self.is_correfence_primary: bool = False
        self.coref_primary: Node | None = None
        self.resolved_text: str | None = None

        self.head: Node | None = None
        self.children: list[Node] = []
        self.lefts: list[Node] = []
        self.rights: list[Node] = []
        
        # WordNet 抽象信息（在 from_doc 时预计算）
        self.wn_abstraction: str | None = None  # 抽象类型，如 "AI_Model", "Person"
        self.wn_hypernym_path: list[str] = []   # 上位词路径
        
        # Wikidata 标签信息（在 from_doc 时预计算）
        self.wd_tags: dict[str, str] = {}  # 如 {"WD:InstanceOf": "software", "WD:FieldOfWork": "AI"}
        
    def set_sentence(self, sentence: str, start: int, end: int) -> None:
        self.sentence = sentence
        self.covered_sentence = sentence
        self.sentence_start = start
        self.sentence_end = end
        
    @staticmethod
    def from_doc(doc) -> tuple[list['Node'], list['Node']]:
        nodes: list[Node] = []
        node_map: dict[int, Node] = {}
        def _coref_primary_rank(node: 'Node') -> tuple[int, int, int, int]:
            """Rank nodes for primary selection: higher score = better candidate.
            Priority: VERB/AUX > NOUN/PROPN > ADJ > NUM > others > PRON"""
            ent_score = 1 if node.ent != Entity.NOT_ENTITY else 0
            pos_priority: dict[Pos, int] = {
                Pos.VERB: 10,      # Highest priority: verbs
                Pos.AUX: 10,       # Highest priority: auxiliaries
                Pos.NOUN: 8,       # Second: nouns
                Pos.PROPN: 8,      # Second: proper nouns
                Pos.ADJ: 6,        # Third: adjectives
                Pos.NUM: 5,        # Fourth: numbers
                Pos.ADV: 4,        # Fifth: adverbs
                Pos.ADP: 3,        # Sixth: adpositions
                Pos.PART: 2,       # Seventh: particles
                Pos.PRON: 0,       # Lowest: pronouns
            }
            pos_score = pos_priority.get(node.pos, 1)  # Default priority for other POS
            length_score = len(node.text)
            return (ent_score, pos_score, length_score, -node.index)
        wildcard_tags = {',', '.', '-LRB-', '-RRB-', '``', ':', "''", 'PRP$', 'WP$', '$'}
        for token in doc:
            # print(f"Token: '{token.text}', Lemma: '{token.lemma_}', Dep: {token.dep_} ['{token.head.text}'], Ent: {token.ent_type_}, POS: {token.pos_}, TAG: {token.tag_}")
            pos = token.pos_
            if pos in {"SPACE", "PUNCT"}:
                continue
            tag = "WILDCARD" if token.tag_ in wildcard_tags else token.tag_
            dep = token.dep_
            ent = token.ent_type_ if token.ent_type_ else "NOT_ENTITY"
            sentence = doc[token.left_edge.i : token.right_edge.i + 1].text
            # print(f"Node '{token.text}': sentence span [{token.left_edge.i}, {token.right_edge.i + 1}): '{sentence}'")
            node = Node(
                text=token.text,
                pos=Pos[pos],
                tag=Tag[tag],
                dep=Dep[dep],
                ent=Entity[ent],
                lemma=token.lemma_,
                index=token.i,
            )
            node.set_sentence(sentence, token.left_edge.i, token.right_edge.i + 1)
            # print(f"Set sentence for Node '{node.text}' :> {token.left_edge.text} ({node.sentence_start}), {token.right_edge.text} ({node.sentence_end}): \n\t'{node.sentence}'")
            node_map[token.i] = node
            nodes.append(node)

        for token in doc:
            node = node_map.get(token.i)
            if not node:
                continue
            if token.head.i != token.i and token.head.i in node_map:
                node.head = node_map[token.head.i]
                node_map[token.head.i].children.append(node)
            for left in token.lefts:
                left_node = node_map.get(left.i)
                if left_node:
                    node.lefts.append(left_node)
            for right in token.rights:
                right_node = node_map.get(right.i)
                if right_node:
                    node.rights.append(right_node)
        
        # 向node里面标记指代
        if doc._.coref_clusters is not None and doc._.resolved_text is not None:
            text, resolved_text, coref_clusters = doc.text, doc._.resolved_text, doc._.coref_clusters
            # print(f"\n[Coreference Processing] Found {len(coref_clusters)} coreference cluster(s)")
            
            coref_clusters = _restrict_correfs(coref_clusters, level=1)
            cluster_id = 0
            for cluster in coref_clusters:
                cluster_tokens: list[Node] = []
                cluster_token_set: set[Node] = set()
                cluster_texts: list[str] = []
                primary_candidates: list[Node] = []
                for start, end in cluster:
                    span_text = text[start:end]
                    cluster_texts.append(span_text)
                    # print(f"    - Span [{start}:{end}]: '{span_text}'") # 找到这轮指代中的span
                    span = doc.char_span(start, end)
                    if span is None:
                        iterable_tokens = (token for token in doc)
                    else:
                        iterable_tokens = span
                    for token in iterable_tokens:
                        token_start = token.idx
                        token_end = token.idx + len(token.text)
                        if token_end <= start or token_start >= end:
                            continue
                        node = node_map.get(token.i)
                        if not node:
                            continue
                        if node not in cluster_token_set:
                            cluster_tokens.append(node)
                            cluster_token_set.add(node)
                
                if len(cluster_tokens) > 1:
                    # Mark all cluster tokens with correfence_id
                    for node in cluster_tokens:
                        node.correfence_id = cluster_id
                    
                    # Skip if less than 2 nodes remain (coreference requires at least 2 entities)
                    if len(cluster_tokens) < 2:
                        # print(f"    Skipping cluster {cluster_id}: only {len(cluster_tokens)} token node(s)")
                        cluster_id += 1
                        continue
                    
                    # 确定主节点：找到在resolved_text中相同位置出现的文本
                    primary_text = None
                    primary_start = None
                    primary_end = None
                    for start, end in cluster:
                        span_text = text[start:end]
                        if span_text in resolved_text:
                            primary_text = span_text
                            primary_start = start
                            primary_end = end
                            break
                    
                    # Select primary node from all cluster_tokens based on POS priority:
                    # VERB/AUX > NOUN/PROPN > ADJ > NUM > others > PRON
                    # Primary span is used as a tiebreaker, not a filter
                    primary_node: Node | None = None
                    
                    # Calculate primary span info for tiebreaking
                    primary_span_nodes = set()
                    if primary_start is not None and primary_end is not None:
                        for node in cluster_tokens:
                            token = doc[node.index]
                            token_start = token.idx
                            token_end = token.idx + len(token.text)
                            if not (token_end <= primary_start or token_start >= primary_end):
                                primary_span_nodes.add(node)
                    
                    # Select primary node: prioritize by POS, use primary span as tiebreaker
                    # Filter out particles (like "to") that are not meaningful as primary
                    def _rank_with_span_preference(node: 'Node') -> tuple:
                        base_rank = _coref_primary_rank(node)
                        # Add bonus if in primary span (as tiebreaker)
                        in_span_bonus = 1 if node in primary_span_nodes else 0
                        # Penalize particles (like "to") that shouldn't be primary
                        particle_penalty = -100 if node.pos == Pos.PART and node.text.lower() == "to" else 0
                        # Return: (ent_score, pos_score + particle_penalty, in_span_bonus, length_score, -index)
                        return (base_rank[0], base_rank[1] + particle_penalty, in_span_bonus, base_rank[2], base_rank[3])
                    
                    primary_node = max(cluster_tokens, key=_rank_with_span_preference)
                    primary_node.is_correfence_primary = True
                    
                    # Set primary_text_for_replacement: 
                    # For verbs, prefer lemma (e.g., "analyze" from "to analyze")
                    # For other POS, use text
                    if primary_node.pos in {Pos.VERB, Pos.AUX}:
                        primary_text_for_replacement = primary_node.lemma
                    else:
                        primary_text_for_replacement = primary_node.text
                    
                    # Extract verb from merged tokens like "to analyze" -> "to analyze" (keep "to" for infinitive)
                    def extract_verb_text(node: Node, doc) -> str:
                        """Extract the verb from a merged token, keeping 'to' for infinitives."""
                        if node.pos not in {Pos.VERB, Pos.AUX}:
                            return node.text
                        text = node.text.strip()
                        # Handle "to <verb>" pattern: keep "to analyze" for infinitives
                        if text.lower().startswith("to ") and len(text) > 3:
                            # Keep the full "to analyze" form for infinitives
                            return text
                        # For other cases, try to use lemma if it's meaningful
                        if node.lemma and node.lemma.lower() not in {"to", node.text.lower()}:
                            return node.lemma
                        return node.text
                    
                    # Set resolved_text and coref_primary for all cluster tokens
                    for node in cluster_tokens:
                        if node.is_correfence_primary:
                            # For verbs, extract the actual verb (e.g., "analyze" from "to analyze")
                            if node.pos in {Pos.VERB, Pos.AUX}:
                                node.resolved_text = extract_verb_text(node, doc)
                            else:
                                node.resolved_text = node.text
                        elif node.pos == Pos.PRON and primary_text_for_replacement:
                            node.resolved_text = primary_text_for_replacement
                        
                        if primary_node:
                            node.coref_primary = primary_node
                            if node.pos == Pos.PRON and not node.pronoun_antecedent:
                                node.pronoun_antecedent = primary_node
                                # print(f"【*】set pronoun antecedent for '{node.text}' --> '{primary_node.text}'")
                        elif node.pos == Pos.PRON and not node.pronoun_antecedent:
                            node.pronoun_antecedent = None
                    
                    # Ensure primary node has resolved_text
                    if primary_node and not primary_node.resolved_text:
                        if primary_node.pos in {Pos.VERB, Pos.AUX}:
                            primary_node.resolved_text = extract_verb_text(primary_node, doc)
                        else:
                            primary_node.resolved_text = primary_node.text
                    
                    cluster_id += 1
                else:
                    # print(f"    Skipping cluster {cluster_id} (only {len(cluster_tokens)} token node, need > 1)")
                    cluster_id += 1
            # print(f"\n[Coreference Processing] Completed. Total clusters processed: {cluster_id}\n")
        
        # for node in nodes:
        #     if node.coref_primary:
        #         print(f"【&】Node '{node.text}' coref primary: '{node.coref_primary.text}'")
        roots = [node for node in nodes if node.head is None]
        for root in roots:
            assert root.dep == Dep.ROOT or root.dep == Dep.dep, f"Root node dep should be ROOT or _SP, got {root.dep.name}"
        
        # 预计算 WordNet 抽象信息
        abstractor = TokenAbstractor()
        for token in doc:
            node = node_map.get(token.i)
            if not node:
                continue
            node.wn_abstraction = abstractor.get_abstraction(token, doc)
            node.wn_hypernym_path = abstractor.get_abstraction_path(token, doc)
        
        return nodes, roots
    
    def __format__(self, format_spec: str) -> str:
        return f"Node(text='{self.text}', pos={self.pos.name}, tag={self.tag.name}, dep={self.dep.name}, ent={self.ent.name}, sentence='{self.sentence}')"
    def __repr__(self) -> str:
        return self.__format__('')
    def __display__(self) -> str:
        return self.__format__('')
    def __str__(self) -> str:
        return self.__format__('')

class Relationship:
    def __init__(self, entities: list[Node], sentence: str, relationship_sentence: str) -> None:
        self.root = entities[0]
        self.entities = entities
        self.sentence = sentence
        self.relationship_sentence = relationship_sentence
        # start, end are the range of entities' indices
        start, end = entities[0].index, entities[0].index
        for entity in entities[1:]:
            if entity.index < start:
                start = entity.index
            if entity.index > end:
                end = entity.index
        self.start = start
        self.end = end + 1
        self.father: Relationship | None = None

    def node_text(self, node: Node) -> str:
        # Import Vertex here to avoid circular import
        from hyper_simulation.hypergraph.hypergraph import Vertex
        res = Vertex.resolved_text(node)

        determiner_children: list[Node] = []
        for child in node.lefts:
            if child.dep in {Dep.det, Dep.poss, Dep.predet}:
                determiner_children.append(child)
        determiner_children.sort(key=lambda n: n.index)
        if determiner_children:
            prefix = " ".join(Vertex.resolved_text(child) for child in determiner_children)
            res = f"{prefix} {res}"

        return res
    
    def relationship_text_simple(self) -> str:
        
        # `self.relation_sentence` is a part of `self.sentence`
        # We need firstly record the prefix and suffix of `self.relation_sentence` to `self.sentence` 
        # after `sentence` calculation, we need cancel the prefix and suffix in `sentence`
        
        # 1. calc the prefix and suffix
        def calc_prefix_suffix():
            rel_start = self.sentence.find(self.relationship_sentence)
            if rel_start != -1:
                prefix = self.sentence[:rel_start].strip()
                suffix = self.sentence[rel_start + len(self.relationship_sentence):].strip()
            else:
                prefix = ""
                suffix = ""
            return prefix, suffix
        
        prefix, suffix = calc_prefix_suffix()
        
        # Import Vertex here to avoid circular import
        from hyper_simulation.hypergraph.hypergraph import Vertex
        sentence = str(self.sentence)
        for entity in self.entities[1:]:
            new_text = self.node_text(entity)
            old_candidates = [entity.sentence, Vertex.resolved_text(entity)]
            # print(f"Replacing entity in relationship sentence: '{entity.sentence}' --> '{new_text}' (candidates: {old_candidates})")
            for old in old_candidates:
                if old and old in sentence:
                    sentence = sentence.replace(old, new_text, 1)
                    break
        
        sentence = sentence.replace(prefix, "").replace(suffix, "").strip()
        
        return sentence

    def __format__(self, format_spec: str) -> str:
        return f"[root: {self.node_text(self.root)}] ({', '.join([self.node_text(entity) for entity in self.entities])})\n\tIn Sentence: '{self.sentence}'\n\tSimple: '{self.relationship_text_simple()}'"
    def __repr__(self) -> str:
        return self.__format__('')
    def __str__(self) -> str:
        return self.__format__('')
    def __display__(self) -> str:
        return self.__format__('')

class LocalDoc:
    def __init__(self, doc) -> None:
        self.tokens = [token.text for token in doc]

    def __getitem__(self, index) -> str:
        if isinstance(index, slice):
            return ' '.join(self.tokens[index])
        else:
            return self.tokens[index]

class Dependency:
    def __init__(self, nodes: list[Node], roots: list[Node], doc: LocalDoc, is_query: bool=False) -> None:
        self.nodes = nodes
        self.roots = roots
        self.doc = doc
        self.vertexes: list[Node] = []
        self.links_succ: dict[Node, list[Node]] = {}
        self.links_pred: dict[Node, Node] = {}
        self.relationship_sentences: dict[Node, str] = {}
        self.correfence_map: dict[Node, Node] = {}
        self.is_query = is_query
    
    def _fixup_lefts_rights_sentences(self, node: Node) -> None:
        node.children.sort(key=lambda n: n.index)
        node.lefts = [child for child in node.children if child.index < node.index]
        node.rights = [child for child in node.children if child.index > node.index]
        node.sentence_start = node.lefts[0].index if node.lefts else node.index
        node.sentence_end = node.rights[-1].index + 1 if node.rights else node.index + 1
        # print(f"Sentence of Node '{node.text}' [{node.sentence_start}, {node.sentence_end}): \n\t'{node.sentence}'")
        node.sentence = self.doc[node.sentence_start : node.sentence_end]
        # print(f"Reset sentence of [{node.sentence_start}, {node.sentence_end}): \n\t'{node.sentence}'")
    
    def _calc_relationship_sentence(self, root: Node):
        left_edge = right_edge = root.index
        for succ in self.links_succ.get(root, []):
            if succ.index < left_edge:
                left_edge = succ.index
            if succ.index > right_edge:
                right_edge = succ.index
        return self.doc[left_edge : right_edge + 1]
    
    # PASS 1: Solve all the Conjunction dependencies
    # Solve the `conj`, `cc`, `appos` and `preconj` dependencies by change the conjunct children to the children of the head;
    # e.g., in "Alice and Bob went to the store", "Bob" will get the same children as "Alice"
    # Execute this pass by Level-Order Traversal
    def solve_conjunctions(self):
        # Firstly, solve query for det wh-words
        if self.is_query:
            wh_dets = {"what", "which"}
            for node in self.nodes:
                if node.dep == Dep.det and node.text.lower() in wh_dets and node.head:
                    node.head.is_query = True
                    for child in node.head.children:
                        if child.dep == Dep.conj and child.head == node.head:
                            child.is_query = True
        
        # print("Solving conjunctions...\n")
        queue = self.roots.copy()
        next_level: list[Node] = []
        while queue:
            node = queue.pop(0)
            remove_children: list[Node] = []
            for child in node.children:
                if child.dep == Dep.conj or (child.dep == Dep.appos and node.head):
                    child.dep = node.dep
                    child.head = node.head
                    remove_children.append(child)
                    queue.append(child)
                else:
                    next_level.append(child)
            
            for child in remove_children:
                node.children.remove(child)
                if node.head:
                    node.head.children.append(child)
            if not queue:
                queue = next_level
                next_level = []
            
            if not remove_children:
                continue
            
            self._fixup_lefts_rights_sentences(node)
            if node.head:
                self._fixup_lefts_rights_sentences(node.head)
        
        self.roots = [node for node in self.nodes if node.head is None]
        print("Conjunctions solved. Resulting Nodes:")
        for node in self.nodes:
            print(f"{node}, head is '{node.head.text if node.head else 'ROOT'}'")
        return self
    
    # PASS 2: Mark all the antecedent of pronouns.
    # check all the `relcl` dependencies, uses the relative clause to find the antecedent of pronouns.
    # We split all `relcl` conditions to two dependencies tree.
    def mark_pronoun_antecedents(self):
        
        # relcl: relative clause
        for node in self.nodes:
            if node.dep != Dep.relcl or not node.head:
                continue
            antecedent = node.head
            for child in node.children:
                if child.pos == Pos.PRON:
                    child.pronoun_antecedent = antecedent
                    # print(f"【*】set pronoun antecedent for '{child.text}' --> '{antecedent.text}' via relcl")
            
            node.head.children.remove(node)
            self._fixup_lefts_rights_sentences(node.head)

            node.dep = Dep.ROOT
            node.head = None
            
        # ccomp: We solve formal complement clauses to find pronoun antecedents
        # For formal object patterns like "find it + ccomp", set pronoun antecedent to the ccomp clause head.
        if not self.is_query:
            return self
        
        for node in self.nodes:
            if node.dep != Dep.ccomp or not node.head:
                continue
            head = node.head
            # head and node all are verbs/auxiliary verbs
            # find head and node's nsubj & PRON children
            for child in head.children:
                if child.dep not in {Dep.nsubj, Dep.nsubjpass}:
                    continue
                if child.pos != Pos.PRON:
                    continue
                for ccomp_child in node.children:
                    if (ccomp_child.dep in {Dep.nsubj, Dep.nsubjpass}) and (ccomp_child.pos in {Pos.PRON}):
                        child.pronoun_antecedent = ccomp_child
                        print(f"【*】set pronoun antecedent for '{child.text}' --> '{ccomp_child.text}' via ccomp")
                        
        # TODO: solve the acl dependencies if necessary
        # e.g., I can not believe the fact that consistency is maintained.
        
        # TODO: solve the ccomp dependencies if necessary
        # e.g., It is true that simulation works.
        
        return self
    
    # PASS 3: Mark the prefixes for prepositions and agents
    def mark_prefixes(self):
        for node in self.nodes:
            if node.dep == Dep.agent and node.head:
                if node.index < node.head.index:
                    node.head.prefix_agent = node.text
                    node.head.prefix_index = node.index
                else:
                    node.head.suffix_agent = node.text
                    node.head.suffix_index = node.index
            if node.dep == Dep.prep and node.head:
                if node.index < node.head.index:
                    node.head.prefix_prep = node.text
                    node.head.prefix_index = node.index
                else:
                    node.head.suffix_prep = node.text
                    node.head.suffix_index = node.index
            if node.dep == Dep.pobj and node.head:
                if node.index > node.head.index:
                    node.prefix_prep = node.head.text
                    node.prefix_index = node.head.index
                else:
                    node.suffix_prep = node.head.text
                    node.suffix_index = node.head.index
        return self
    
    # PASS 4: Mark all the vertex, that all the nodes should be a Vertex i.f.f. statisfy:
    # 0. All root nodes
    # 1. holds a entity, i.e., ent != NOT_ENTITY
    # 2. to be a noun or proper noun, i.e., pos in {NOUN, PROPN}
    # 3. to be a verb or auxiliary verb, i.e., pos in {VERB, AUX}
    # 4. to be an adjective, i.e., pos == ADJ
    # 5. to be a numeric, i.e., pos == NUM
    # 6. to be a pronoun, i.e., pos == PRON
    # Merge coreference nodes: nodes with same correfence_id share the same vertex (primary node)
    def mark_vertex(self): #
        for node in self.nodes:
            if node.dep in {Dep.nsubj, Dep.nsubjpass, Dep.csubj, Dep.csubjpass}:
                node.dominator = True

        correfence_primary_map: dict[int, Node] = {}
        for node in self.nodes:
            if node.correfence_id is not None and node.is_correfence_primary:
                correfence_primary_map[node.correfence_id] = node

        self.correfence_map = {
            node: correfence_primary_map[node.correfence_id]
            for node in self.nodes
            if node.correfence_id is not None
            and not node.is_correfence_primary
            and node.correfence_id in correfence_primary_map
        }

        pronoun_antecedent_map: dict[Node, Node] = {}
        for node in self.nodes:
            if node.pos == Pos.PRON and node.pronoun_antecedent:
                antecedent = node.pronoun_antecedent
                while antecedent.coref_primary and antecedent != antecedent.coref_primary:
                    antecedent = antecedent.coref_primary
                pronoun_antecedent_map[node] = antecedent
        for node, antecedent in pronoun_antecedent_map.items():
            if node not in self.correfence_map:
                self.correfence_map[node] = antecedent
            if not node.coref_primary:
                node.coref_primary = antecedent

        qualifying_pos = {Pos.NOUN, Pos.PROPN, Pos.VERB, Pos.AUX, Pos.ADJ, Pos.NUM, Pos.PRON, Pos.ADV}
        self.vertexes = []
        for node in self.nodes:
            node.is_vertex = False
            if node.pos in {Pos.SPACE, Pos.PUNCT}:
                continue
            if self.is_query and node.pos == Pos.AUX and node.dep == Dep.aux and node.head and node.head.pos == Pos.VERB:
                continue
            if self.is_query and node.pos == Pos.PRON and node.dep == Dep.det:
                continue
            if self.is_query:
                normalized_how = node.text.lower().replace("-", " ").strip(" \t\n\r\f\v.,?!;:")
                normalized_how = " ".join(normalized_how.split())
                if normalized_how.startswith("how"):
                    parts = normalized_how.split()
                    how_tail = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
                    how_quantifiers = {"many", "much", "few", "little", "long", "often", "far"}
                    if how_tail in how_quantifiers:
                        node.is_query = True
                        node.query_type = QueryType.NUMBER
                    elif node.pos in {Pos.ADJ, Pos.ADV}:
                        node.is_query = True
                        node.query_type = QueryType.ATTRIBUTE
                        node.query_attribute = how_tail or None
            if self.is_query and node.pos == Pos.PRON:
                wh_pronouns = {"what", "which", "who", "whom", "whose"}
                if node.text.lower() in wh_pronouns:
                    node.is_query = True
                    pronoun = node.text.lower()
                    if pronoun == "which":
                        node.query_type = QueryType.WHICH
                    elif pronoun == "what":
                        node.query_type = QueryType.WHAT
                    elif pronoun in {"who", "whom"}:
                        node.query_type = QueryType.PERSON

            if self.is_query and node.pos == Pos.ADV:
                wh_adverbs = {"when", "where", "why"}
                if node.text.lower() in wh_adverbs:
                    node.is_query = True
                    adverb = node.text.lower()
                    if adverb == "when":
                        node.query_type = QueryType.TIME
                    elif adverb == "where":
                        node.query_type = QueryType.LOCATION
                    elif adverb == "why":
                        node.query_type = QueryType.REASON

            if self.is_query and node.pos == Pos.DET and node.dep == Dep.poss and node.text.lower() == "whose":
                node.is_vertex = True
                node.is_query = True
                node.query_type = QueryType.BELONGS
                self.vertexes.append(node)
                continue
            # if node.pos == Pos.PRON and node.pronoun_antecedent and not node.is_query:
            #     continue
            if (
                node.head is None
                or node.ent != Entity.NOT_ENTITY
                or node.pos in qualifying_pos
            ):
                node.is_vertex = True
                self.vertexes.append(node)
        
        return self
    
    # PASS 5: Compress dependencies that only links vertex nodes.
    # We use links to record the compressed dependencies.
    # collect all the pred non-vertex nodes of a node between vertexes into `former_nodes`.
    def compress_dependencies(self):
        # Preserve direct grammatical dependencies (dobj, nsubj, etc.) from coreference mapping
        direct_grammatical_deps = {Dep.nsubj, Dep.dobj, Dep.pobj, Dep.attr, Dep.oprd, Dep.iobj}
        for node in self.nodes:
            if node.head and node.head in self.correfence_map and node.dep not in direct_grammatical_deps:
                node.head = self.correfence_map[node.head]
        for node in self.vertexes:
            if not node.head:
                continue
            pred = node.head
            while pred and not pred.is_vertex:
                if pred in self.correfence_map:
                    pred = self.correfence_map[pred]
                else:
                    node.former_nodes.insert(0, pred)  # 插入到头部
                    pred = pred.head
            if pred:
                self.links_pred[node] = pred
                if pred not in self.links_succ:
                    self.links_succ[pred] = []
                self.links_succ[pred].append(node)
        return self

    # PASS 6: Calculate all the relationships.
    # For each non-leaf vertex, it and its successors form a relationship.
    # We temporarily using the roots's sentence as the relationship sentence.
    # Then we use the `thefuzz` to get all vertex a id, and calculate a map.
    def calc_relationships(self) -> tuple[list[Node], list[Relationship], dict[Node, int]]:
        def _match_same(best_match, score, node: Node, choices_map: dict[str, int], pos_map: dict[int, Pos]) -> bool:
            # if best_match's POS is ​​PRON​​ or node's POS is ​​PRON​​, we do not consider it a match
            if pos_map[choices_map[best_match]] == Pos.PRON or node.pos == Pos.PRON:
                return False
            elif score >= 90 and (pos_map[choices_map[best_match]] == node.pos):
                return True
            elif score == 100:
                return True
            return False
        
        # for node in self.vertexes:
        #     print(f"Vertex Node: {node}")
        
        saved_rels: set[tuple[str, str]] = set()
        relationships: list[Relationship] = []
        vertex_id_map: dict[Node, int] = {}
        root_to_relationship: dict[Node, Relationship] = {}
        for node in self.vertexes:
            if node in self.links_succ:
                node_key_text = (node.resolved_text or node.text)
                if (node_key_text, node.sentence) in saved_rels:
                    continue
                relational_sentence = self._calc_relationship_sentence(node)
                saved_rels.add((node_key_text, node.sentence))
                rel = Relationship(entities=[node] + self.links_succ[node], sentence=node.sentence, relationship_sentence=relational_sentence)
                root_to_relationship[node] = rel
                relationships.append(rel)
        
        relationship_trees: dict[Relationship, Relationship] = {} # mapping relationship to its father relationship
        for rel in relationships:
            node = rel.root
            for succ in self.links_succ.get(node, []):
                if succ in root_to_relationship:
                    child_rel = root_to_relationship[succ]
                    # Only set father relationship if the child's root is within the parent's sentence range
                    # This prevents cross-sentence or semantically incorrect father relationships
                    # caused by coreference mapping issues
                    if (succ.sentence_start >= node.sentence_start and succ.sentence_end <= node.sentence_end):
                        relationship_trees[child_rel] = rel
        
        for rel, father_rel in relationship_trees.items():
            rel.father = father_rel
        
        choices = []
        choices_map: dict[str, int] = {}
        pos_map: dict[int, Pos] = {}
        cnt = 1
        deferred_coref_nodes: list[Node] = []
        for node in self.vertexes:
            if node.coref_primary:
                deferred_coref_nodes.append(node)
                continue
            base_text = node.resolved_text or node.text
            text = base_text.lower()
            extraction = process.extractOne(text, choices) if choices else None
            match extraction:
                case (best_match, score) if _match_same(best_match, score, node, choices_map, pos_map):
                    vertex_id_map[node] = choices_map[best_match]
                    pos_map[vertex_id_map[node]] = node.pos
                case _:
                    choices.append(text)
                    choices_map[text] = cnt
                    vertex_id_map[node] = cnt
                    pos_map[cnt] = node.pos
                    cnt += 1
        print(f"【……】Deferred coreference nodes to process: {len(deferred_coref_nodes)}")
        for node in deferred_coref_nodes:
            print(f"    - Node '{node.text}' (resolved: '{node.resolved_text}') with coref primary '{node.coref_primary.text if node.coref_primary else 'None'}'")
        for node in deferred_coref_nodes:
            primary: Node | None = node.coref_primary
            if primary and primary in vertex_id_map:
                vertex_id_map[node] = vertex_id_map[primary]
                pos_map[vertex_id_map[node]] = primary.pos
                continue
            base_text = node.resolved_text or node.text
            text = base_text.lower()
            extraction = process.extractOne(text, choices) if choices else None
            match extraction:
                case (best_match, score) if _match_same(best_match, score, node, choices_map, pos_map):
                    vertex_id_map[node] = choices_map[best_match]
                    pos_map[vertex_id_map[node]] = node.pos
                case _:
                    choices.append(text)
                    choices_map[text] = cnt
                    vertex_id_map[node] = cnt
                    pos_map[cnt] = node.pos
                    cnt += 1
        print(f"【……】Deferred coreference nodes processed.")
        print("Final Vertex ID Map:")
        for node, vid in vertex_id_map.items():
            print(f"    - Node '{node.text}' (resolved: '{node.resolved_text}') --> Vertex ID {vid}")
        return self.vertexes, relationships, vertex_id_map

