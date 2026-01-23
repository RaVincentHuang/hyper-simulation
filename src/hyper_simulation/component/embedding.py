import os
import random

from sentence_transformers import SentenceTransformer
from modelscope import AutoModel, AutoTokenizer, AutoModelForCausalLM
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
_embedding_cache: dict[str, np.ndarray] = {}

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

def get_embedding_batch_old(texts: list[str]) -> list[np.ndarray]:
    if 'Qwen3-Embedding-4B' not in _model_cache:
        _model_cache['Qwen3-Embedding-4B'] = (AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left'), AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B'))

    tokenizer, model = _model_cache['Qwen3-Embedding-4B']
    
    batch_dict = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    batch_dict.to(model.device)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu().detach().numpy()
    embedding_list = embeddings.tolist()
    return embedding_list

def _get_sentence_transformer() -> SentenceTransformer:
    if "Qwen/Qwen3-Embedding-0.6B" not in _model_cache:
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
        model.eval()
        _model_cache["Qwen/Qwen3-Embedding-0.6B"] = model
    return _model_cache["Qwen/Qwen3-Embedding-0.6B"]


def get_embedding_batch(texts: list[str], N: int=8) -> list[np.ndarray]:
    model = _get_sentence_transformer()
    uncached: list[str] = [t for t in texts if t not in _embedding_cache]
    for i in range(0, len(uncached), N):
        batch_texts = uncached[i:i+N]
        if not batch_texts:
            continue
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for text, emb in zip(batch_texts, batch_embeddings):
            _embedding_cache[text] = emb
    return [_embedding_cache[text] for text in texts]

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