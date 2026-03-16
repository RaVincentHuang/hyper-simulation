import os
import random

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left')
# model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B')

_model_cache = {}
# _embedding_cache: dict[str, np.ndarray] = {}

max_length = 8192

_DEFAULT_SEED = int(os.environ.get("SC_SEED", "42"))
random.seed(_DEFAULT_SEED)
np.random.seed(_DEFAULT_SEED)
torch.manual_seed(_DEFAULT_SEED)
torch.cuda.manual_seed_all(_DEFAULT_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

def _get_sentence_transformer() -> SentenceTransformer:
    if "Qwen/Qwen3-Embedding-0.6B" not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
        # model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
        model = SentenceTransformer(local_model_path)
        model.eval()
        _model_cache["Qwen/Qwen3-Embedding-0.6B"] = model
    return _model_cache["Qwen/Qwen3-Embedding-0.6B"]


def get_embedding_batch(texts: list[str], batch_size: int=256, cache: None | dict[str, np.ndarray]=None) -> list[np.ndarray]:
    model = _get_sentence_transformer()
    
    if cache is None:
        cache = {}
        
    unique_texts = list(set(texts))
    missing_texts = [t for t in unique_texts if t not in cache]

    if missing_texts:
        new_embeddings = model.encode(
            missing_texts,
            batch_size=batch_size, # 让模型内部去处理分批
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        cache.update(zip(missing_texts, new_embeddings))

    return [cache[t] for t in texts]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))

def get_similarity_batch(query: list[str], data: list[str], N: int=8) -> list[float]:
    query_embeddings = get_embedding_batch(query, N)
    data_embeddings = get_embedding_batch(data, N)
    pairs = [(q_emb, d_emb) for q_emb in query_embeddings for d_emb in data_embeddings]
    similarities = get_cosine_similarity_batch(pairs, is_normalized=True)
    return similarities

def get_similarity(text1: str, text2: str) -> float:
    emb1 = get_embedding_batch([text1])[0]
    emb2 = get_embedding_batch([text2])[0]
    return cosine_similarity(emb1, emb2)

def get_cosine_similarity_batch(pairs: list[tuple[np.ndarray, np.ndarray]], is_normalized: bool=False) -> list[float]:
    """
    计算 One-to-One 的批量余弦相似度
    :param pairs: 输入格式为 [(vecA_1, vecB_1), (vecA_2, vecB_2), ...]
    :return: 对应的相似度得分列表 [score_1, score_2, ...]
    """
    if not pairs:
        return []

    # 1. 解包：将 list of tuples 拆解为两个 list
    A_list, B_list = zip(*pairs)
    
    # 2. 堆叠：将 1D 向量列表转换为连续内存的 2D 矩阵 (N, D)
    A = np.stack(A_list)
    B = np.stack(B_list)
    
    if is_normalized:
        # 如果输入已经是归一化的向量，则直接计算点积
        similarities = np.einsum('ij,ij->i', A, B)
        return similarities.tolist()
    
    # 3. 批量计算范数并归一化
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)
    
    # 避免除以零
    A_norms[A_norms == 0] = 1e-10
    B_norms[B_norms == 0] = 1e-10
    
    A_normalized = A / A_norms
    B_normalized = B / B_norms
    
    # 4. 批量点积：使用 einsum 计算每行的内积，结果形状为 (N,)
    similarities = np.einsum('ij,ij->i', A_normalized, B_normalized)
    
    # 将 NumPy 数组转回 Python 的 float 列表返回
    return similarities.tolist()

