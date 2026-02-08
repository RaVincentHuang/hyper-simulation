# spacy

## Token 性质

### 屈折性

*Lemma*: [`lemma`] 一个token的原始形式（我们保存为节点）

```python
token.lemma_, token.lemma
```

*Morph Analysis*: [`morph`]
- `Case`: 格 Nom（主格）, Acc（宾格）, Gen（所有格）
- `Number`: 单复数 Sing（单数）, Plur（复数）
- `Gender`: 性 	Masc, Fem, Neut （阳性/阴性/中性）
- `Tense` : 时态 	Pres 现在时, Past 过去时, Fut 将来时
- `Person` : 人称 1, 2, 3
- `VerbForm` : 动词形式 Fin 限定形式, Inf 不定式, Part 分词, Ger 动名词
- `Mood` : 语气 Ind（陈述）, Sub（虚拟）, Imp（祈使语气）
- `Definite` : 限定性 Def（定冠词）, Ind（不定）
- `Degree` : 形容词等级 Pos（原级）, Cmp（比较级）, Sup（最高级）
- `PronType` : Prs （人称代词）, Dem（指示代词）, Int（疑问代词）, Rel（关系代词）, Art （冠词）, Neg（否定代词）
- `Aspect` : 体 Perf （完成）, Prog（进行）
- `Poss`: Yes 物主形式
- `Reflex`: Yes 反身形式
- `Polarity` : Neg 否定形式
- `NumType` : Card 基数词, Ord 序数词, Frac 分数
- `Abbr` : Yes 缩写形式
- `​​Foreign​​` : Yes 外来词
- `Typo` : Yes 拼写错误

```python
token.morph.to_dict()
token.morph.get(.)
```

### 词性标签

*Part-of-Speech* [`pos`] 通用词性标签（粗粒度）

| POS标签 | 全称 | 中文 | 描述 | 英语示例 |
| -------------| -------------| ------------| ---------------- | ----------------- |
`​​ADP​​` | Adposition | 介词/后置词 | 表示名词与其他词关系的词 | in, on, at, of | 
`​​ADV​​` | Adverb | 副词 | 修饰动词、形容词或其他副词的词 | quickly, very, well |
`​​ADJ​​` | Adjective | 形容词 | 描述或修饰名词的词 | big, happy, blue | 
`​​AUX​​` | Auxiliary | 助动词 | 帮助构成时态、语态等的动词 | is, have, will, can |
`​​CCONJ​​` | Coordinating Conjunction | 并列连词 | 连接同等语法地位的词或短语 |and, but, or |
`​​DET​​` | Determiner | 限定词 | 限定名词的词 | the, a, this, some
`​​INTJ​​` |  Interjection | 感叹词 | 表达情感的词 | oh, hello, wow
`​​NOUN​​` | Noun | 名词 | 表示人、事物、地点、概念等 | cat, book, London
`​​NUM​​` | Numeral | 数词 | 表示数量的词 | one, 100, first
`​​PART​​` | Particle | 小品词 | 功能词，无完整词汇意义 | 's(所有格), not
`​​PRON​​` | Pronoun | 代词 | 代替名词的词 | I, you, he, it
`​​PROPN​​` | Proper Noun | 专有名词 | 特定名称（人名、地名等）| John, Paris, Google
`​​PUNCT​​` | Punctuation | 标点符号 | 标点符号 | ., ,, !, ?
`​​SCONJ​​` | Subordinating Conjunction | 从属连词 | 引导从句的连词 | that, if, because
`​​SYM​​` | Symbol | 符号 | 符号、特殊字符 | $, %, &
`​​VERB​​` | Verb | 动词 | 表示动作或状态的词 | run, eat, think
`​​X​​` | Other | 其他 | 无法归类的词 | etc., ...
`​​SPACE​​` | Space | 空格 | 空格字符 | (空格)

*Penn Treebank* [`tag`] 标签集（细粒度）

| 标签 | 全称 | 描述 | 示例 |
| ------ | ------- | -------| ------- |
`CC​​` | Coordinating Conjunction | 并列连词 | and, but, or
`​CD​​` | Cardinal Number | 基数词 | one, two, 100 
`​DT​​` | Determiner | 限定词 | the, a, this 
`​EX​​` | Existential there | 存在句中的 "there" | Thereis a cat.
`​​FW​​` | Foreign Word | 外来词 | bonjour, siesta 
`​IN​​` | Preposition/Sub. Conjunction | 介词或从属连词 | in, on, if, because
`JJ​​` | Adjective | 形容词 | big, happy 
`​JJR​​` | Adjective, Comparative | 形容词比较级 | bigger, happier 
`​​JJS​​` | Adjective, Superlative | 形容词最高级 | biggest, happiest
`​​MD​​` | Modal | 情态动词 | can, should, will 
`​​NN​​` | Noun, Singular or Mass | 单数名词或不可数名词 | cat, water 
`​​NNS​​` | Noun, Plural | 复数名词 | cats, books 
`​​NNP​​` | Proper Noun, Singular | 单数专有名词 | London, John 
`​NNPS​​` | Proper Noun, Plural | 复数专有名词 | The Americas | ​​PDT​​ | Predeterminer | 前位限定词 | allthe kids, bothsides 
`​POS​​` | Possessive Ending | 所有格结尾 | 's(如 in John's) 
`​PRP​​` | Personal Pronoun | 人称代词 | I, you, he, it 
`​​PRP$`​​ | Possessive Pronoun | 物主代词 | my, your, his 
`​​RB​​` | Adverb | 副词 | quickly, very 
`​​RBR​​` | Adverb, Comparative | 副词比较级 | faster, better 
`​RBS​​` | Adverb, Superlative | 副词最高级 | fastest, best 
`​RP​​` | Particle | 小品词 | give​​up​​ 
`​​TO​​` | to | 不定式符号 "to" | torun 
`​​UH​​` | Interjection | 感叹词 | hello, oh, wow 
`​​VB​​` | Verb, Base Form | 动词原形 | run, eat 
`​VBZ​​` | Verb, 3rd person Sing. Present | 动词第三人称单数现在式 | he​​runs​​ 
`​​VBP​​` | Verb, Non-3rd person Sing. Present | 动词非第三人称单数现在式 | I/you/we​​run​​ 
`​​VBD​​` | Verb, Past Tense | 动词过去式 | ran, ate 
`​​VBN​​` | Verb, Past Participle | 动词过去分词 | has​​run​​, was​​eaten​​ 
`​​VBG​​` | Verb, Gerund/Present Participle | 动词动名词/现在分词 | is​​running​​ `​WDT​​` | Wh-determiner | Wh-限定词 | which, what, whose 
`​​WP​​` | Wh-pronoun | Wh-代词 | who, what 
`​​WP$​`​ | Possessive Wh-pronoun | 所有格 Wh-代词 | whose 
`​​WRB​​` | Wh-adverb | Wh-副词 | when, where, why

## Dependency

*Noun chunks* 名词性短语
```python
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
```
- `Text` chunk的原始文本
- `Root text` 将名词短语与句法分析其余部分连接起来的词的原始文本。
- `Root dep` 连接根节点与其头节点的依赖关系。
- `Root head text` 跟节点的头节点

*Navigating the parse tree* 
- `child: children` 表示当前token的儿子们
- `head` 表示当前token的父亲
- `lefts` 当前左侧的儿子 `n_lefts = len(lefts)`
- `rights` 当前右侧的儿子 `n_rights = len(rights)`
- `dep` 连接当前节点与其头节点的依赖关系
- `head text` 头节点的文本
- `head pos` 头节点的粗粒度类型

```python
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])
```
- `subtree` 按照文本顺序排列的所有的后代构成的集合
- `ancestors` 是一个序列，顺着当前的token向上溯，`a.is_ancestor(b)`表示 a 是 b 的祖先
```python
roots = [token for token in doc if token.head == token]
print(f"\nRoot of the sentence: {[root.text for root in roots]}")
for root in roots:
    for descendant in root.subtree:
        print(f"Descendant of root '{root.text}': {descendant.text}")
        for ancestor in descendant.ancestors:
            print(f"\tAncestor of '{descendant.text}': {ancestor.text}")
```
- `left_edge` 子树最左侧的token `i` 为token的下标
- `right_edge` 子树最右侧的token
```python
print(doc[4].left_edge, doc[4].right_edge)
span = doc[doc[4].left_edge.i : doc[4].right_edge.i+1]
with doc.retokenize() as retokenizer:
    retokenizer.merge(span)
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)
```
将token拼接为全新的token

### Dependency Label
**核心关系**

| 标签 | 全称 | 描述 | 示例 |
| ---- | ---- | ---- | ---- |
`ROOT​​` | Root | 句子的核心谓语动词（全句的根节点） | She ​​ate​​ an apple.→ ate

**主宾关系**
| 标签 | 全称 | 描述 | 示例 |
| ---- | ---- | ---- | ---- |
`​nsubj​​` | Nominal Subject | 名词性主语（主动句） | ​​She​​ ate an apple. |
`​nsubjpass​​` | Nominal Subject (Passive) | 名词性主语（被动句） | ​​The apple​​ was eaten. |
`​csubj​​` | Clausal Subject | 从句主语 | ​​That she left​​ surprised us. |
`​​csubjpass​​` | Clausal Subject (Passive) | 从句主语（被动） | ​​That she left​​was known. |
`​dobj​​` | Direct Object | 直接宾语 | She ate ​​an apple​​. |
`​​iobj​​` | Indirect Object | 间接宾语 | She gave ​​him​​ a gift. |
`​pobj​​` | Prepositional Object | 介词宾语 | She sat ​​on​​ ​​the chair​​.(chair是on的宾语) |
`​​dative​​` | Dative | 与格（类似间接宾语） | He spoke ​​to me​​.(me是to的宾语) |
**修饰关系**

| 标签 | 全称 | 描述 | 示例 |
| ---- | ---- | ---- | ---- |
`​amod​​` | Adjectival Modifier | 形容词修饰名词 | a ​​red​​ apple | ​
`​advmod​​` | Adverbial Modifier | 副词修饰动词/形容词 | She ran ​​quickly​​. |
`​nummod​​` | Numeric Modifier | 数词修饰名词 | ​​three​​ apples |
`​quantmod​​` | Quantifier Modifier | 数量修饰 | ​​All​​ students passed. |
`​​npadvmod​​` | Noun Phrase Adverbial Modifier | 名词短语作状语 | She arrived​​this morning​​. |
`​neg​​` | Negation Modifier | 否定修饰 | She ​​did not​​ go.(not修饰go) | 

**从句关系**
| 标签 | 全称 | 描述 | 示例 |
| ---- | ---- | ---- | ---- |
`acl​​` | Clausal Modifier | 从句修饰名词 | the man ​​who left​​ |
`​​advcl​​` | Adverbial Clause | 状语从句 | She cried ​​because she was sad​​. |
`​ccomp​​` | Clausal Complement | 从句作补语（需主语） | She said ​​he left​​. |
`​xcomp​​` | Open Clausal Complement | 从句作补语（无主语） | She wants ​​to leave​​. |
`​​relcl​​` | Relative Clause | 关系从句 | the book ​​that I read​​ | 
`​​mark​​` | Marker | 从句引导词 | She left ​​before​​ he arrived. | 

**介词与连接词​**
| 标签 | 全称 | 描述 | 示例 |
| ---- | ---- | ---- | ---- |
`prep​​` | Preposition | 介词 | She sat ​​on​​ the chair. | ​
`​agent​​` | Agent | 被动句施事 | eaten ​​by​​ the wolf | 
`​​cc​​` | Coordinating Conjunction | 并列连词 | apples ​​and​​ oranges | 
`​​conj​​` | Conjunct | 并列成分 | She bought ​​apples​​ and ​​pears​​. | ​
`​case​​` | Case Marker | 格标记（介词/所有格） | the cover ​​of​​ the book | ​
`​prt​​` | Particle | 动词小品词 | She ​​gave up​​.(up是gave的小品词) | 

**其他功能​**
| 标签 | 全称 | 描述 | 示例 |
| ---- | ---- | ---- | ---- |
`appos​​` | Apposition | 同位语 | Tom, ​​my brother​​, left. | 
`​​attr​​` | Attribute | 表语（系动词后） | She is ​​a doctor​​. | 
`​​acomp​​` | Adjectival Complement | 形容词补语 | She seems ​​happy​​. | 
`​​oprd​​` | Object Predicate | 宾语补足语 | They elected her ​​president​​. | ​
`​aux​​` | Auxiliary | 助动词 | She ​​will​​ go. | ​
`​auxpass​​` | Auxiliary (Passive) | 被动助动词 | It ​​was​​ eaten. | ​
`​expl​​` | Expletive | 形式主语（如 there） | ​​There​​ is a cat. | 
`​​parataxis​​` | Parataxis | 并列句（松散连接） | ​​He left​​, she cried. | 
`​​meta​​` | Meta Modifier | 元修饰语（如邮件主题） | ​​Subject​​: Hello | 
`​​det​​` | Determiner | 限定词 | ​​the​​ apple | ​
`​poss​​` | Possessive Modifier | 所有格 | ​​her​​ book | ​
`​predet​​` | Predeterminer | 前位限定词 | ​​All​​ the students | 
`​​preconj​​` | Preconjunction | 前置连词 | ​​Both​​ cats and dogs | ​
`​intj​​` | Interjection | 感叹词 | ​​Oh​​, I see! | ​
`​punct​​` | Punctuation | 标点符号 | She left ​​.​​ |
`compound` | Compound | 复合词 | ​​toothbrush​​, ​​New York​​ |
`pcomp` | Prepositional Complement | 介词补语 | She is fond ​​of music​​. |
`nmod` | Nominal Modifier | 名词修饰语 | the​​ roof​​ of the house |
`​​dep​​` | Unclassified Dependency | 无法分类的关系（罕见） | （备用标签） |


## NER
对于实体`doc.ent`，它们可以包含类型标签`ent.label_`

- `token.ent_iob` 描述了token当前的状态 `I` 在一个实体中, `O` 在实体之外, `B` 是一个实体的开头 （`L` 实体的最后一个token，`U` 单token实体）

*Setting entity annotations* 为当前的文本添加新的 entity
```python
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
doc = nlp("fb is hiring a new vice president of global policy")
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print('Before', ents)
# The model didn't recognize "fb" as an entity :(

# Create a span for the new entity
fb_ent = Span(doc, 0, 1, label="ORG")
orig_ents = list(doc.ents)

# Option 1: Modify the provided entity spans, leaving the rest unmodified
doc.set_ents([fb_ent], default="unmodified")

# Option 2: Assign a complete list of ents to doc.ents
doc.ents = orig_ents + [fb_ent]

ents = [(e.text, e.start, e.end, e.label_) for e in doc.ents]
print('After', ents)
# [('fb', 0, 1, 'ORG')] 🎉
```

*Setting entity annotations from array* 
```python
import numpy
import spacy
from spacy.attrs import ENT_IOB, ENT_TYPE

nlp = spacy.load("en_core_web_sm")
doc = nlp.make_doc("London is a big city in the United Kingdom.")
print("Before", doc.ents)  # []

header = [ENT_IOB, ENT_TYPE]
attr_array = numpy.zeros((len(doc), len(header)), dtype="uint64")
attr_array[0, 0] = 3  # B
attr_array[0, 1] = doc.vocab.strings["GPE"]
doc.from_array(header, attr_array)
print("After", doc.ents)  # [London]
```
- `spacy.explain` 解释label的意思

**NER 标签**
| 标签 | 全称 | 中文 | 描述 | 示例 |
| ---- | ---- | ---- | ---- | ---- |
`PERSON​​` | Person | 人物 | 真实或虚构的人物姓名 | Barack Obama, Sherlock Holmes | ​
`​NORP​​` | Nationalities/Religious/Political Groups | 民族/宗教/政治团体 | 民族、宗教、政治团体 | American, Christian, Republican | ​​
`FAC​​` | Facility | 设施 | 建筑、机场、高速公路等 | Golden Gate Bridge, Heathrow Airport | ​​
`ORG​​` | Organization | 组织机构 | 公司、机构、组织等 | Google, United Nations, NASA | ​
`​GPE​​` | Geo-Political Entity | 地缘政治实体 | 国家、城市、州等行政区域 | China, New York, California | 
`​​LOC​​` | Location | 地理位置 | 非GPE的地理位置（山脉、水体等） | Mount Everest, Pacific Ocean | 
`​​PRODUCT​​` | Product | 产品 | 商品、产品、作品等 | iPhone, Toyota Camry, Windows 10 | ​
`EVENT​​` | Event | 事件 | 命名的事件（历史、体育、自然等） | Olympic Games, World War II, Hurricane Katrina | ​
`​WORK_OF_ART​​` | Work of Art | 艺术品 | 书籍、歌曲、电影、绘画等 | Mona Lisa, Harry Potter, Bohemian Rhapsody | ​
`​LAW​​` | Law | 法律 | 命名的法律、法规、条约等 | First Amendment, Paris Agreement | ​
`​LANGUAGE​​` | Language | 语言 | 任何命名的人类语言 | English, Mandarin, Spanish | ​
`​DATE​​` | Date | 日期 | 绝对或相对日期、期间 | January 1st, yesterday, next week | ​
`​TIME​​` | Time | 时间 | 一天内的时间、持续时间 | 3:00 PM, two hours, noon | ​
`​PERCENT​​` | Percent | 百分比 | 百分比数值（含"%"符号） | 50%, twenty percent | ​
`​MONEY​​` | Money | 货币金额 | 货币价值、金额 | $100, 50 euros, ¥1000 | ​
`​QUANTITY​​` | Quantity | 数量 | 度量、重量、距离等 | 10 kilometers, 5 pounds, 2 liters | ​
`​ORDINAL​​` | Ordinal | 序数词 | 表示顺序的词语 | first, second, third | ​
`​CARDINAL​​` | Cardinal | 基数词 | 数值、计数 | one, 100, a dozen | 
## Matcher


## Pipes
`nlp.add_pipe(xxx)`

*Merge entities* 将一个相同的实体合并为一个token 

*Merge noun chunks* 将一个名词性短语合并为一个token

*Sentencizer* 使用规则来切分句子

## Rule
```python
import spacy

nlp = spacy.load("en_core_web_sm")
# Merge noun phrases and entities for easier analysis
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")

TEXTS = [
    "Net income was $9.4 million compared to the prior year of $2.7 million.",
    "Revenue exceeded twelve billion dollars, with a loss of $1b.",
]
for doc in nlp.pipe(TEXTS):
    for token in doc:
        if token.ent_type_ == "MONEY":
            # We have an attribute and direct object, so check for subject
            if token.dep_ in ("attr", "dobj"):
                subj = [w for w in token.head.lefts if w.dep_ == "nsubj"]
                if subj:
                    print(subj[0], "|-->", token)
            # We have a prepositional object with a preposition
            elif token.dep_ == "pobj" and token.head.dep_ == "prep":
                print(token.head.head, "-->", token)
```