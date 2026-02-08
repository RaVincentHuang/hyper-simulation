from ast import mod
from html import entities
from itertools import chain
import json
import re
from venv import logger
from hyper_simulation.llm import prompt
from hyper_simulation.question_answer.vmdit import relation
from sympy import content
from hyper_simulation.graph_generator.ontology import general_entity, general_relation
from hyper_simulation.llm.prompt.graph import graph_building, simple_graph_building, graph_building_without_type, graph_records
from hyper_simulation.llm.prompt.graph import graph_entity_records_msg, graph_relation_records_msg, graph_attributes_records_msg
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from pydantic import BaseModel, Field
import networkx as nx
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
import re
import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(f'logs/{__name__}.log')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

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
class Graph(BaseModel):
        class Node(BaseModel):
            name: str = Field(..., description="The name of the entity")
            type: str = Field(..., description="The type of the entity")
            desc: str = Field(..., description="A description of the entity")
        class Edge(BaseModel):
            src: str = Field(..., description="The source entity")
            dst: str = Field(..., description="The destination entity")
            type: str = Field(..., description="The type of the relation")
            desc: str = Field(..., description="A description of the relation")
        graph_name: str = Field(..., description="The name of the graph")
        graph_description: str = Field(..., description="The description of the graph")
        entities: list[Node] = Field(..., description="The list of entities")
        relations: list[Edge] = Field(..., description="The list of relations")



# @retry(stop=stop_after_attempt(5))
# def build_graph(model: BaseLLM, title, text, prop, task='popqa') -> nx.DiGraph:
    
#     global graph_building
    
#     match task:
#         case 'popqa':
#             prompt = graph_building.partial(
#                 entity_types=", ".join(general_entity),
#                 relation_types=", ".join(general_relation)
#             )
#             # prompt = graph_building_without_type
#         case _:
#             raise ValueError(f"Unknown task: {task}")
    
#     json_output_parser = JsonOutputParser(pydantic_object=Graph)
    
#     chain = prompt | model | json_output_parser
    
#     # print(f"Title: {title}")
#     # print(f"Text: {text}")
#     # print(f"Prop: {prop}")
    
#     res = chain.invoke({
#         "input_title": title,
#         "input_text": text,
#         "input_prop": prop
#     })
    
#     if isinstance(res, str):
#         res = json.loads(res)
#     elif isinstance(res, dict):
#         res = Graph(**res)
#     elif isinstance(res, Graph):
#         res = json.loads(res.model_dump_json())
#     else:
#         raise ValueError(f"Unknown type: {type(res)}")
    
#     graph = nx.DiGraph()
#     graph.graph['name'] = res.graph_name
#     graph.graph['description'] = res.graph_description
#     graph.graph['title'] = title
#     graph.graph['text'] = text
#     graph.graph['prop'] = prop
#     graph.graph['task'] = task
#     for node in res.entities:
#         graph.add_node(node.name, type=node.type, desc=node.desc)
#     for edge in res.relations:
#         graph.add_edge(edge.src, edge.dst, type=edge.type, desc=edge.desc)
#     return graph

# @retry(stop=stop_after_attempt(5))
# def build_graph_batch(model: BaseLLM, prompt_list: list[dict], task='popqa') -> list[nx.DiGraph]:
#     """
#     Build a batch of graphs from a list of prompts.
    
#     Args:
#         model (BaseLLM): The language model to use for graph generation.
#         prompt_list (list[dict]): A list of prompts for graph generation.
#         task (str): The task type for graph generation.
    
#     Returns:
#         list[nx.DiGraph]: A list of generated graphs.
#     """
    
#     global graph_building
    
#     match task:
#         case 'popqa':
#             # prompt = graph_building.partial(
#             #     entity_types=", ".join(general_entity),
#             #     relation_types=", ".join(general_relation)
#             # )
#             # prompt = graph_building_without_type
#             prompt = graph_records.partial(
#                 entity_types=", ".join(general_entity),
#             )
#         case _:
#             raise ValueError(f"Unknown task: {task}")
    
#     # tqdm.write(f"Prompt: {prompt}")
    
#     # json_output_parser = JsonOutputParser(pydantic_object=Graph)
    
#     # chain = prompt | model | json_output_parser
#     chain = prompt | model
    
#     res_list = chain.batch(prompt_list)
    
#     tqdm.write(f"Result: {res_list}")
#     exit(0)
    
#     graphs = []
#     for res in res_list:
#         if isinstance(res, str):
#             res = json.loads(res)
#         elif isinstance(res, dict):
#             res = Graph(**res)
#         elif isinstance(res, Graph):
#             res = json.loads(res.model_dump_json())
#         else:
#             raise ValueError(f"Unknown type: {type(res)}")
        
#         graph = nx.DiGraph()
#         graph.graph['name'] = res.graph_name
#         graph.graph['description'] = res.graph_description
#         for node in res.entities:
#             graph.add_node(node.name, type=node.type, desc=node.desc)
#         for edge in res.relations:
#             graph.add_edge(edge.src, edge.dst, type=edge.type, desc=edge.desc)
        
#         graphs.append(graph)
    
#     return graphs

def fresh_entity_records(msg: AIMessage)-> tuple[str, list[tuple[str, str, str]]]:
    # print(msg.content)
    if not isinstance(msg.content, str):
        raise ValueError(f"Expected str")
    content = msg.content
    
    # print(f"Content: {content}")
    entities = []
    entities_list = []
    for record in content.split("\n"):
        if "<END_OUTPUT>" in record:
            break
        # print(f"Record: {record}")
        match = re.match(r'"entity",*(.+),*(.+),*(.+)', record)
        if match:
            name, type, desc = match.groups()
            name, type, desc = name.strip().rstrip().strip('"').rstrip('"'), type.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"')
            entities.append(f'("entity", "{name.strip().rstrip()}", "{type.strip().rstrip()}", "{desc.strip().rstrip()}");')
    outs = "\n".join(entities)
    if not len(entities):
        raise ValueError(f"Expected entities")
    return outs, entities_list

def fresh_relation_records(msg: AIMessage)-> tuple[str, list[tuple[str, str, str, str]]]:
    if not isinstance(msg.content, str):
        raise ValueError(f"Expected str")
    content = msg.content
    print(f"Relation: {content}")
    relations = []
    relations_list = []
    for record in content.split("\n"):
        if "<END_OUTPUT>" in record:
            break
        match = re.match(r'"relationship",(.+),(.+),(.+),(.+)', record)
        if match:
            src, dst, type, desc = match.groups()
            src, dst, type, desc = src.strip().rstrip().strip('"').rstrip('"'), dst.strip().rstrip().strip('"').rstrip('"'), type.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"')
            relations.append(f'("relationship", "{src}", "{dst}", "{type}", "{desc}");')
            relations_list.append((src, dst, type, desc))
    outs = "\n".join(relations)
    if not len(relations_list):
        raise ValueError(f"Expected relations")
    return outs, relations_list

def fresh_attribute_records(msg: AIMessage)-> tuple[str, list[tuple[str, str, str, str]]]:
    # print(msg.content)
    if not isinstance(msg.content, str):
        raise ValueError(f"Expected str")
    content = msg.content
    print(f"Attribute: {content}")
    # tqdm.write(f"Relation: {content}")
    attributes = []
    attributes_list = []
    for record in content.split("\n"):
        if "<END_OUTPUT>" in record:
            break
        match = re.match(r'"attribute",(.+),(.+),(.+),(.+)', record)
        if match:
            key, value, desc, entity = match.groups()
            key, value, desc, entity = key.strip().rstrip().strip('"').rstrip('"'), value.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"'), entity.strip().rstrip().strip('"').rstrip('"')
            attributes.append(f'("attribute", "{key}", "{value}", "{desc}", "{entity}");')
            attributes_list.append((key, value, desc, entity))
    outs = "\n".join(attributes)
    if not len(attributes_list):
        raise ValueError(f"Expected attributes")
    return outs, attributes_list

@retry(stop=stop_after_attempt(5))
async def build_graph_step_by_step(model: BaseChatModel, title, text, prop, task='popqa') -> nx.DiGraph:
    # Step 1: Ask for entity records
    msgs = [
        graph_entity_records_msg.format(entity_types=", ".join(general_entity), input_text=text),
    ]
    merger = merge_message_runs()
    chain = merger | model
    msg = await chain.ainvoke(msgs)
    entities_msg, entities_list = fresh_entity_records(msg)
    # msgs.append(entities_msg)
    
    # Step 2: Ask for relation records
    msgs = [
        graph_relation_records_msg.format(relation_types=", ".join(general_relation), input_text=text, entities=entities_msg),
    ]
    
    msg = await chain.ainvoke(msgs)
    relations_msg, relations_list = fresh_relation_records(msg)
    
    # Step 3: Ask for attribute records
    msgs = [
        graph_attributes_records_msg.format(input_text=text, relations=relations_msg),
    ]
    msg = await chain.ainvoke(msgs)
    attributes_msg, attributes_list = fresh_attribute_records(msg)
    # print(f"Entities: {entities_msg}")
    # print(f"Relations: {relations_msg}")
    # print(f"Attributes: {attributes_msg}")
    
    # Step 4: Build the graph
    graph = nx.DiGraph()
    graph.graph['title'] = title
    graph.graph['text'] = text
    graph.graph['prop'] = prop
    graph.graph['task'] = task
    
    for node in entities_list:
        name, type, desc = node
        graph.add_node(name, type=type, desc=desc)
    
    for edge in relations_list:
        src, dst, type, desc = edge
        graph.add_edge(src, dst, type=type, desc=desc)
    
    for node in attributes_list:
        key, value, desc, entity = node
        graph.add_node(f"{key}={value}", type='Fact', desc=desc)
        if not graph.has_node(entity):
            graph.add_node(entity, type=key, desc=desc)

        graph.add_edge(entity, key, type='attribute', desc=desc)
    
    return graph

def fresh_records(content: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    entities_list: list[tuple[str, str, str]] = []
    relations_list: list[tuple[str, str, str, str]]= []
    attributes_list: list[tuple[str, str, str, str]] = []
    def _out_bars(record: str):
        match1 = re.match(r'\((.*)\);', record)
        match2 = re.match(r'\((.*)\)', record)
        if match1:
            res: str = match1.group(1)
            return res
        elif match2:
            res: str = match2.group(1)
            return res
        else:
            return record
    for record in content.split("\n"):
        if "<END_OUTPUT>" in record:
            break
        # print(f"Record: {record}")
        record = _out_bars(record)
        match = re.match(r'"entity",*(.+),*(.+),*(.+)', record)
        if match:
            name, type, desc = match.groups()
            name, type, desc = name.strip().rstrip().strip('"').rstrip('"'), type.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"')
            entities_list.append((name, type, desc))
        match = re.match(r'"relationship",(.+),(.+),(.+),(.+)', record)
        if match:
            src, dst, type, desc = match.groups()
            src, dst, type, desc = src.strip().rstrip().strip('"').rstrip('"'), dst.strip().rstrip().strip('"').rstrip('"'), type.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"')
            relations_list.append((src, dst, type, desc))
        match = re.match(r'"attribute",(.+),(.+),(.+),(.+)', record)
        if match:
            key, value, desc, entity = match.groups()
            key, value, desc, entity = key.strip().rstrip().strip('"').rstrip('"'), value.strip().rstrip().strip('"').rstrip('"'), desc.strip().rstrip().strip('"').rstrip('"'), entity.strip().rstrip().strip('"').rstrip('"')
            attributes_list.append((key, value, desc, entity))
    if not len(entities_list):
        logger.warning(f"[No entities]: {content}")
    if not len(relations_list):
        logger.warning(f"[No relations]: {content}")
    if not len(attributes_list):
        logger.warning(f"[No attributes]: {content}")
    return entities_list, relations_list, attributes_list

# @retry(stop=stop_after_attempt(5))
async def build_graph(model: BaseLLM, title, text, prop, task='popqa') -> nx.DiGraph:
    # Step 1: Ask for entity records
    
    prompt = graph_records.partial(
        entity_types=", ".join(general_entity),
        relation_types=", ".join(general_relation)
    )
    # tqdm.write(f"Prompt: {prompt}")
    chain = prompt | model
    ans = await chain.ainvoke({
        "input_text": text,
    })
    
    entities_list, relations_list, attributes_list = fresh_records(ans)
    
    # Step 4: Build the graph
    graph = nx.DiGraph()
    graph.graph['title'] = title
    graph.graph['text'] = text
    graph.graph['prop'] = prop
    graph.graph['task'] = task
    
    for node in entities_list:
        name, type, desc = node
        graph.add_node(name, type=type, desc=desc)
    
    for edge in relations_list:
        src, dst, type, desc = edge
        graph.add_edge(src, dst, type=type, desc=desc)
    
    for node in attributes_list:
        key, value, desc, entity = node
        graph.add_node(f"{key}={value}", type='Fact', desc=desc)
        if not graph.has_node(entity):
            graph.add_node(entity, type=key, desc=desc)

        graph.add_edge(entity, key, type='attribute', desc=desc)
    
    return graph

# @retry(stop=stop_after_attempt(5))
def build_graph_batch(model: BaseLLM, prompt_list: list[dict], task='popqa') -> list[nx.DiGraph]:
    """
    Build a batch of graphs from a list of prompts.
    
    Args:
        model (BaseLLM): The language model to use for graph generation.
        prompt_list (list[dict]): A list of prompts for graph generation.
        task (str): The task type for graph generation.
    
    Returns:
        list[nx.DiGraph]: A list of generated graphs.
    """
    
    global graph_building
    
    match task:
        case 'popqa':
            # prompt = graph_building.partial(
            #     entity_types=", ".join(general_entity),
            #     relation_types=", ".join(general_relation)
            # )
            # prompt = graph_building_without_type
            prompt = graph_records.partial(
                entity_types=", ".join(general_entity),
                relation_types=", ".join(general_relation)
            )
        case _:
            raise ValueError(f"Unknown task: {task}")
    
    # tqdm.write(f"Prompt: {prompt}")
    
    # json_output_parser = JsonOutputParser(pydantic_object=Graph)
    
    # chain = prompt | model | json_output_parser
    chain = prompt | model
    
    res_list = chain.batch(prompt_list)
    
    # tqdm.write(f"Result: {res_list}")
    # exit(0)
    
    graphs = []
    for res in res_list:
        entities_list, relations_list, attributes_list = fresh_records(res)
    
    # Step 4: Build the graph
        graph = nx.DiGraph()
        graph.graph['task'] = task
        
        for node in entities_list:
            name, type, desc = node
            graph.add_node(name, type=type, desc=desc)
        
        for edge in relations_list:
            src, dst, type, desc = edge
            graph.add_edge(src, dst, type=type, desc=desc)
        
        for node in attributes_list:
            key, value, desc, entity = node
            graph.add_node(f"{key}={value}", type='Fact', desc=desc)
            if not graph.has_node(entity):
                graph.add_node(entity, type=key, desc=desc)

            graph.add_edge(entity, key, type='attribute', desc=desc)
        
        graphs.append(graph)
    
    return graphs

def save_graph(graph: nx.DiGraph, path: str) -> None:
    """
    Save the graph to a file.
    
    Args:
        graph (nx.DiGraph): The graph to save.
        path (str): The path to save the graph to.
    """
    with open(path, 'w') as f:
        data = nx.node_link_data(graph)
        json.dump(data, f, indent=4)

def load_graph(path: str) -> nx.DiGraph:
    """
    Load the graph from a file.
    
    Args:
        path (str): The path to load the graph from.
    
    Returns:
        nx.DiGraph: The loaded graph.
    """
    with open(path, 'r') as f:
        data = json.load(f)
        graph = nx.node_link_graph(data)
    return graph