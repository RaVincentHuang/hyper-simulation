# Hyper Simulation

## 项目主体结构
```
src/hyper_simulation
├── __init__.py
├── graph_generator
│   ├── __init__.py
│   ├── build.py # 建图主要在这里
│   └── ontology.py # 节点的类型标签可以参考这里的定义
├── llm
│   ├── __init__.py
│   ├── chat_completion.py # 使用Chat模式下的langchain接口的代码
│   ├── ll.py # 测试代码，可以忽略
│   ├── prompt # 各类prompt集中在这里
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── question.py
│   │   └── vmdit.py
│   ├── text_completion.py # 使用LLM模式下的langchain接口的代码
│   └── time_cost.py # 计算时间开销的代码
└── question_answer
    ├── __init__.py
    ├── base_line_lm.py
    ├── generate_passage_embedding.py
    ├── get_trim.py 
    ├── retrival
    │   ├── __init__.py
    │   └── analysis.py # 当前转图的代码
    ├── utils
    │   ├── __init__.py
    │   └── show_task.py # 一个展示当前任务的工具
    └── vmdit # 现有的RAG模块，我没怎么做修改
        ├── __init__.py
        ├── metrics.py
        ├── relation.py
        ├── retrieval.py
        ├── rewrite.py
        ├── trim.py
        └── utils.py
```

## 子节点

### Graph generator
现在在用的是`build_graph_batch`和`fresh_records`，前面的是调用，后面的是验证。

难理解的地方可能是`fresh_records`中
```python
match = re.match(r'"entity",*(.+),*(.+),*(.+)', record)
if match:
    name, type, desc = match.groups()
    name, type, desc = name.strip().rstrip().strip('"').rstrip('"'), typstrip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"'rstrip('"')
    entities_list.append((name, type, desc))
match = re.match(r'"relationship",(.+),(.+),(.+),(.+)', record)
if match:
    src, dst, type, desc = match.groups()
    src, dst, type, desc = src.strip().rstrip().strip('"').rstrip('"'), dsstrip().rstrip().strip('"').rstrip('"'), type.strip().rstrip().strip('"'rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"')
    relations_list.append((src, dst, type, desc))
match = re.match(r'"attribute",(.+),(.+),(.+),(.+)', record)
if match:
    key, value, desc, entity = match.groups()
    key, value, desc, entity = key.strip().rstrip().strip('"').rstrip('"')value.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().str('"').rstrip('"'), entity.strip().rstrip().strip('"').rstrip('"')
    attributes_list.append((key, value, desc, entity))
```
其实就是做正则表达式匹配。

被我注释掉的
```python
@retry(stop=stop_after_attempt(5))
```
是一个装饰器，来自`tenacity`库，如果捕获到了crash，它会重新执行，可以结合`try`来使用。

**async/await** 我体感上不建议用，但是调用API可以尝试，通过异步逻辑可以提升吞吐量。

函数定义形如
```python
async def func() -> T:
```
接受返回时需要
```python
res = await func()
```
程序会在`func`未执行完成之前执行后面的代码。

最上层需要
```python
import asyncio
asyncio.run()
```
来执行。

如果需要控制最大并发数，可以用信号量来控制。
```python
pool_cnt = 4
semaphore = asyncio.Semaphore(1)

with semaphore:
    res = await func()
```

具体可以看`src/hyper_simulation/question_answer/retrival/analysis.py`中注释掉的。

另外`save_graph`, `save_graph`是存取图的方法。

### Analysis
重点是
```python
class SolvedTask:
```
现在能用的接口是`add_task()`当添加的任务大于`self.task_pool_cnt`时批量执行。

### Show task
`src/hyper_simulation/question_answer/utils/show_task.py`下
```
pixi run show
```
可以显示当前的检索问题答案和上下文。