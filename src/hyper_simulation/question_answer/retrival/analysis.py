import json
import re

from hyper_simulation.graph_generator.build import build_graph, save_graph, build_graph_batch, build_graph_step_by_step, load_graph
from hyper_simulation.llm import prompt
from langchain_ollama import OllamaLLM, ChatOllama
import os
from tqdm import tqdm
from colorama import Fore, Back, Style, init
import time
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt
import networkx as nx
import logging

def read_retrival_data(file_path):
    data = []
    with open(file_path, "r") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            # print(example['prop'])
            data.append(example)
    return data

class SolvedTask:
    def __init__(self, source_path, target_path, model, top_k=15):
        self.source_path = source_path
        self.target_dir = target_path
        self.task_need_solve: dict[int, set[int]] = {}
        self.task_solved: dict[int, set[int]] = {}
        self.current_tasks: list[tuple[int, int, str, str, str]] = []
        self.model = model
        self.task_pool_cnt = 4
        total_cnt = 0
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(f'logs/{__name__}.log')
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        with open(source_path, "r") as file:
            for example in file:
                example = json.loads(example)
                task_id = example['id']
                if task_id not in self.task_need_solve:
                    self.task_need_solve[task_id] = set()
                total_cnt += len(example['ctxs'][:top_k])
                for ctx in example['ctxs'][:top_k]:
                    self.task_need_solve[task_id].add(ctx['id'])
        self.total_cnt = total_cnt
        self.target_files = []
        solved_task_cnt = 0
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.endswith(".json") and "_" in file:
                    if self._clean_up(os.path.join(root, file)):
                        continue
                    question_id, id = file.rsplit("_", 1)
                    id = id.removesuffix(".json")
                    question_id, id = int(question_id), int(id)
                    if question_id not in self.task_solved:
                        self.task_solved[question_id] = set()
                    self.task_solved[question_id].add(id)
                    solved_task_cnt += 1
        self.solved_task_cnt = solved_task_cnt
        self._display_current_status()
        self.pbar = tqdm(total=total_cnt, initial=solved_task_cnt, desc="Processing ctxs")
    
    def _display_current_status(self):
        init(autoreset=True)
        print(f"Solving file {self.source_path}", end=', ')
        print(f"solved rates: {self.solved_task_cnt / self.total_cnt * 100}% [{self.solved_task_cnt}/{self.total_cnt}]")
        self.logger.info(f"Solving file {self.source_path}, solved rates: {self.solved_task_cnt / self.total_cnt * 100}% [{self.solved_task_cnt}/{self.total_cnt}]")
    
    async def build_graph(self, semaphore, question_id: int, id: int, title: str, text: str, prop: str) -> nx.DiGraph | None:
        if self.is_solved(question_id, id):
            return
        # return await build_graph_step_by_step(self.model, title, text, prop, task='popqa')
        async with semaphore:
            return await build_graph(self.model, title, text, prop, task='popqa')
    
    def _clean_up(self, json_path) -> bool:
        graph = load_graph(json_path)
        if len(graph.nodes) == 0 or len(graph.edges) == 0:
            os.remove(json_path)
            self.logger.warning(f"Removed empty graph: {json_path}")
            return True
        return False
        
    
    def add_task(self, question_id: int, id: int, title: str, text: str, prop: str):
        if self.is_solved(question_id, id):
            return
        self.current_tasks.append((question_id, id, title, text, prop))
        if len(self.current_tasks) >= self.task_pool_cnt:
            prompt_list: list[dict] = []
            for question_id, id, title, text, prop in self.current_tasks:
                prompt_list.append({
                    # "input_title": title,
                    "input_text": text,
                    # "input_prop": prop
                })
            current_str = ", ".join(map(lambda x: f"[Q: {x[0]}, ID: {x[1]}]", self.current_tasks))
            # tqdm.write(f"Current task: {current_str}")
            self.logger.info(f"Current task: {current_str}")
            strat_time = time.time()
            graphs = build_graph_batch(self.model, prompt_list, task='popqa')
            end_time = time.time()
            # tqdm.write(f"[Task pool: {self.task_pool_cnt}]Time cost: {end_time - strat_time} ({end_time - strat_time}/{len(graphs)})")
            self.logger.info(f"[Task pool: {self.task_pool_cnt}]Time cost: {end_time - strat_time} ({end_time - strat_time}/{len(graphs)})")
            for i, graph in enumerate(graphs):
                question_id, id, title, text, prop = self.current_tasks[i]
                graph.graph['title'] = title
                graph.graph['text'] = text
                graph.graph['prop'] = prop
                save_graph(graph, f"{self.target_dir}/{question_id}_{id}.json")
                self.logger.info(f"Saved graph: {self.target_dir}/{question_id}_{id}.json")
                self.register_solved(question_id, id)
                self.pbar.update(1)
            self.current_tasks = []
    
    def is_solved(self, question_id: int, id: int):
        if question_id in self.task_solved:
            if id in self.task_solved[question_id]:
                return True
        return False
    
    def register_solved(self, question_id, id):
        if question_id not in self.task_solved:
            self.task_solved[question_id] = set()
        self.task_solved[question_id].add(id)
    

# @retry(stop=stop_after_attempt(114514))
def task_seqs(file_path, target_dir, top_k):
    model = OllamaLLM(model="qwen2.5:32b")
    # model = OllamaLLM(model="qwen2.5:72b")
    solved_task = SolvedTask(file_path, target_dir, model, top_k=top_k)
    # read_retrival_data(file_path)
    with open(file_path, "r") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            question_id = example['id']
            question_id = int(question_id)
            prop = example['prop']
            
            # prompt_list: list[dict] = []
            # id_list: list[int] = []
            for ctx in example['ctxs'][:top_k]:
                title = ctx['title']
                text = ctx['text']
                id = ctx['id']
                id = int(id)
                solved_task.add_task(question_id, id, title, text, prop)
                # graph = await solved_task.build_graph(semaphore, question_id, id, title, text, prop)
                # if graph:
                #     save_graph(graph, f"{target_dir}/{question_id}_{id}.json")
                #     solved_task.register_solved(question_id, id)
                #     solved_task.pbar.update(1)


if __name__ == '__main__':
    file_path = 'data/retr_result/popqa_longtail_w_gs.jsonl'
    target_dir = "data/graph/popqa"
    top_k = 15
    # semaphore = asyncio.Semaphore(1)
    # asyncio.run(task_seqs(file_path, target_dir, top_k, semaphore), )
    task_seqs(file_path, target_dir, top_k)
                # if solved_task.is_solved(question_id, id):
                #     # print(f"Already solved: {question_id}_{id}")
                #     continue  
                # prompt_list.append({
                #     "input_title": title,
                #     "input_text": text,
                #     "input_prop": prop
                # })
                # id_list.append(id)
                # graph = build_graph(model, title, text, prop, task='popqa')
                # print(f"graph: {graph}")
                # save_graph(graph, f"data/graph/popqa/{question_id}_{id}.json")
            
            # graphs = build_graph_batch(model, prompt_list, task='popqa')
            
            # for i, graph in enumerate(graphs):
            #     save_graph(graph, f"{target_dir}/{question_id}_{id_list[i]}.json")
            #     solved_task.register_solved(question_id, id_list[i])
            