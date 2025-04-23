
from ast import mod
from itertools import chain
import json
import re
from tracemalloc import start
from hyper_simulation.graph_generator.ontology import general_entity, general_relation
from hyper_simulation.llm import prompt
from hyper_simulation.llm.prompt.graph import graph_building, simple_graph_building, graph_records
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_community.chat_models.moonshot import MoonshotChat
from pydantic import BaseModel, Field
import networkx as nx
from tenacity import retry, wait_random_exponential, stop_after_attempt

import time
from tqdm import tqdm

"""
{
    "grahp_name": "graph_name",
    "graph_description": "graph_description",
    "entities": [
        {
            "name": "entity_name",
            "type": "entity_type",
            "desc": "description"
        },
    ],
    "relations": [
        {
            "src": "src_entity",
            "dst": "dst_entity",
            "type": "relation_type",
            "desc": "description"
        },
    ]
}
"""


def build_graph(model: BaseLLM, prompt_list: list[dict], task='popqa'):
    
    
    global graph_building
    
    start = time.time()
    
    match task:
        case 'popqa':
            prompt = graph_records.partial(
                entity_types=",".join(general_entity),
                relation_types=",".join(general_relation)
            )
        case _:
            raise ValueError(f"Unknown task: {task}")
    
    # json_output_parser = JsonOutputParser(pydantic_object=Graph)
    
    chain = prompt | model
    
    if len(prompt_list) == 1:
        res_list =   [chain.invoke(prompt_list[0])]
    else:
        res_list = chain.batch(prompt_list)
    # res_list = [res.content for res in res_list]
    # res = chain.invoke({
    #     "input_title": title,
    #     "input_text": text,
    #     "input_prop": prop
    # })
    
    
    
    # print(type(res_list))
    
    end = time.time()
    print(f"Time cost per call: {(end - start) / len(res_list)} ({end - start}/{len(res_list)})")
    
    # print(f"Response: {res_list}")
    # exit()
    
def test_batch(build_graph, file_path, model, top_k):
    with open(file_path, "r") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            question_id = example['id']
            prop = example['prop']
            # cnt += len(example['ctxs'])
            
            prompt_list = []
            for ctx in example['ctxs'][:top_k]:
                title = ctx['title']
                text = ctx['text']
                id = ctx['id']
                prompt_list.append({
                    # "input_title": title,
                    "input_text": text,
                    # "input_prop": prop
                })

            build_graph(model, prompt_list, task='popqa')
            return

if __name__ == '__main__':
    file_path = 'data/retr_result/popqa_longtail_w_gs.jsonl'
    model = OllamaLLM(model="qwen2.5:32b")

    # model = ChatOpenAI(model='moonshot-v1-auto', base_url="https://api.moonshot.cn/v1")
    
    # start_time = time.time()
    # for _ in tqdm(range(100), desc="Processing"):
    #     model.invoke("What is the capital of France?")
    # end_time = time.time()
    # print(f"Time cost: {(end_time - start_time) / 100}")
    
    # read_retrival_data(file_path)
    # cnt = 0
    top_k_list = [1, 2, 4, 8, 10, 15, 20]
    # top_k = 1
    for top_k in top_k_list:
        print(f"Top K: {top_k}")
        test_batch(build_graph, file_path, model, top_k)
    
    # print(f"Total number of ctxs: {cnt}")
