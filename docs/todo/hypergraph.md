# Hypergraph Generation

## Vertices
1. 特殊处理用"-"链接的实体，例如 "200-lop", "300-lop" 这种，作为单独的vertex处理，而不是拆分成 "200" 和 "lop" 两个vertex。 *
6. #59-Q, #38-Q,  超长的节点 *
4. 处理好括号
8. 重复的节点
9. 错误的节点指代
5. NAME 三段式的NAME会被截断 #23-9
2. what 作为det的时候特殊处理。
([11] held, [12] what city, [13] state) -> ([11] held, [12] what city?, [13] what state?)
在问句中的并列成分的提取要不要split？

## Hyperedges
1. 那种复合疑问句中，很可能会包含多个relation，需要拆分成多个hyperedge。
2. 重整一下 sentence 的逻辑

## Query
对于疑问句的特殊处理