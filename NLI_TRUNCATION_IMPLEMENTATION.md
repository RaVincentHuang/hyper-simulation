# NLI 矛盾判断改进实现

## 概述
在构造 (v, v') 路径描述时，新增"截断构造"方法，并改进 NLI 矛盾判断逻辑，使判断更加严格。

## 修改地点
文件：`src/hyper_simulation/component/postprocess.py`  
函数：`post_detection()` 中 NLI 检查部分（第 600-790 行）

## 实现细节

### 1. 截断函数 `_truncate_desc_between_vertices`
```python
def _truncate_desc_between_vertices(desc: str, v: Vertex, v_prime: Vertex) -> str | None
```

**逻辑：**
- 在描述文本中查找 v 和 v_prime 对应的文本
- 若两者都存在，保留从 v **第一次出现** 到 v_prime **最后一次出现** 的部分
- 删去 v_prime 最后一次出现位置右侧的所有文本
- 目的：减少不相关的背景文本，降低噪声对 NLI 的影响

**示例：**
```
原描述：    "The contestant A American Idol B candidate C"
v = candidate, v_prime = American Idol
截断后：   "American Idol B candidate"
```

### 2. NLI 检查扩展（4 种组合）

对于每个四元组 (u, u', v, v')，现在调用 **4 种** NLI 检查：

| 检查类型 | 描述 A | 描述 B | NLI 方向 |
|---------|-------|-------|---------|
| 原构造→ | v' 原完整路径 | u' 查询路径 | A→B |
| 原构造← | u' 查询路径 | v' 原完整路径 | B→A |
| 截断构造→ | v' 截断路径 | u' 查询路径 | A→B |
| 截断构造← | u' 查询路径 | v' 截断路径 | B→A |

### 3. 矛盾判断规则（严格化）

**原规则（v1.0）：**
```
是否矛盾 = (原AB == 'contradiction') AND (原BA == 'contradiction')
```
只需两个方向都返回矛盾，就认为是矛盾。

**新规则（v2.0）：**
```
是否矛盾 = (原AB == 'contradiction') AND (原BA == 'contradiction') 
          AND (截断AB == 'contradiction') AND (截断BA == 'contradiction')
```
需要 4 种都返回矛盾，才认为是矛盾。

**结果：**
- 只要有一个方向不是矛盾，就认为该对是非矛盾的，保留在匹配中
- 判断更严格，保留更多边界情况

## 实现位置详解

| 部分 | 行号 | 功能 |
|------|------|------|
| 截断函数定义 | 607-640 | 实现 v 和 v_prime 的文本级截断 |
| 原/截断描述生成 | 655-670 | 为每个 quad 生成两个版本的 v' 描述 |
| NLI pair 构造 | 675-690 | 构建 4 个 NLI 输入对 |
| 标签聚合 | 703-707 | 将 4 个 NLI 标签按 quad 聚合 |
| 矛盾判断 | 710-740 | 检查 4 种都是 contradiction |
| 规则应用 | 743-766 | 更新 uu_to_vv_match（只要不是全矛盾就保留） |

## 测试要点

1. **截断场景测试：**
   - v 和 v_prime 都在描述中出现 → 应该截断
   - v 或 v_prime 不在描述中 → 保留原描述
   - v 在 v_prime 之后 → 保留原描述（无法截断）

2. **NLI 判断测试：**
   - 原构造全矛盾 + 截断构造非矛盾 → **保留**（新规则）
   - 4 种都矛盾 → **删除**（两规则一致）
   - 任一方向非矛盾 → **保留**（两规则一致）

3. **回归测试：**
   - 确保 Phase 2（worklist）逻辑不受影响
   - 确保 quad_evidence 中记录 4 个标签的组合结果

## 调试方式

启用调试输出查看 4 种 NLI 结果：
```bash
POSTPROCESS_DEBUG=1 python -m your_script
```

日志示例：
```
[NLI 4-way contradiction] (state, North Carolina) <-> (contestant, candidate)
detail=original_ab=contradiction|original_ba=contradiction|truncated_ab=contradiction|truncated_ba=contradiction

[POSTPROCESS DEBUG] NLI ok (4-way): (state, North Carolina) <-> (contestant, candidate)
detail=original_ab=contradiction|original_ba=non_contradiction|truncated_ab=contradiction|truncated_ba=contradiction
```

## 性能影响

- **NLI 调用增加 4 倍：** 每个 quad 从 2 个调用增加到 4 个
- **内存增加轻微：** 只在 nli_pairs_list 中额外存储截断描述（长度通常 <10% 增长）
- **矛盾判定更严格：** 预期保留的对数增加（更少的级联删除）

