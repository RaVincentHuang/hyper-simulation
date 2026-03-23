from enum import Enum, IntEnum

class QueryType(Enum):
    BELONGS = 1 # whose
    WHAT = 2 # what / which
    WHICH = 3 # what / which
    PERSON = 4 # who
    ATTRIBUTE = 5 # how *, *: str
    NUMBER = 6 # how many / how much / how fast
    TIME = 7 # when
    LOCATION = 8 # where
    REASON = 9 # why

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