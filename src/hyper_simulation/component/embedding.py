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


def get_embedding_batch(texts: list[str], N: int=8, cache: None | dict[str, np.ndarray]=None) -> list[np.ndarray]:
    model = _get_sentence_transformer()

    if not cache:
        ans: list[np.ndarray]  = []
        for i in range(0, len(texts), N):
            batch_texts = texts[i:i+N]
            if not batch_texts:
                continue
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            
            for emb in batch_embeddings:
                ans.append(emb)
        
        return ans
    
    ans_map = {}
    missing_texts = []
    
    # 1. 查表：找出哪些需要计算
    for text in texts:
        if text in cache:
            ans_map[text] = cache[text]
        elif text not in ans_map:  # 避免当前输入列表中有重复项
            missing_texts.append(text)

    # 2. 批处理：仅计算缺失部分
    for i in range(0, len(missing_texts), N):
        batch_texts = missing_texts[i:i+N]
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # 更新中间 Map 和 外部 Cache
        for text, emb in zip(batch_texts, batch_embeddings):
            ans_map[text] = emb
            cache[text] = emb  # 实时更新缓存供下次使用

    # 3. 重组：按照原始文本顺序返回结果
    return [ans_map[text] for text in texts]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))

def get_similarity_batch(query: list[str], data: list[str], N: int=8) -> list[float]:
    query_embeddings = get_embedding_batch(query, N)
    data_embeddings = get_embedding_batch(data, N)
    similarities = []
    for q_emb in query_embeddings:
        for d_emb in data_embeddings:
            sim = cosine_similarity(q_emb, d_emb)
            similarities.append(sim)
    return similarities

def get_similarity(text1: str, text2: str) -> float:
    emb1 = get_embedding_batch([text1])[0]
    emb2 = get_embedding_batch([text2])[0]
    return cosine_similarity(emb1, emb2)