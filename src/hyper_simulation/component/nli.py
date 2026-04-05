import os
import torch
from sentence_transformers import CrossEncoder

_model_cache = {}

BATCH_SIZE = 1024

def init_nli_model():
    _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base')

def get_nli_labels_batch(pairs: list[tuple[str, str]]) -> list[str]:
    if 'nli-deberta-v3-base' not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
        # _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    return labels

def get_nli_labels_with_score_batch(pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
    if 'nli-deberta-v3-base' not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
        # _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base')

    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    entailment_index = 1
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels_with_score = [(label_mapping[score_max], scores[i][entailment_index]) for i, score_max in enumerate(scores.argmax(axis=1))]
    return labels_with_score

def get_nli_label(text1: str, text2: str) -> str:
    labels = get_nli_labels_batch([(text1, text2)])
    return labels[0]

def get_nli_entailment_score_batch(pairs: list[tuple[str, str]]) -> list[float]:
    if 'nli-deberta-v3-base' not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
        # _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    entailment_scores = [score[1] for score in scores]
    return entailment_scores

def get_nli_contradiction_score_batch(pairs: list[tuple[str, str]]) -> list[float]:
    if 'nli-deberta-v3-base' not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        # _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base', device="cpu")
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
        # _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    contradiction_scores = [score[0] for score in scores]
    return contradiction_scores

def get_nli_remix_score_batch(pairs: list[tuple[str, str]], to_refine: bool = False) -> list[float]:
    if 'nli-deberta-v3-base' not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
        # _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    if not pairs:
        return []

    model = _model_cache['nli-deberta-v3-base']
    logits = model.predict(pairs, convert_to_tensor=True, batch_size=BATCH_SIZE)
    probs = torch.softmax(logits, dim=1)

    labels_idx = torch.argmax(probs, dim=1)
    entailment_probs = probs[:, 1]

    # If `to_refine` is True and the pair is predicted as contradiction, force score to 0.
    if to_refine:
        pair_scores = torch.where(labels_idx == 0, torch.zeros_like(entailment_probs), entailment_probs)
    else:
        pair_scores = entailment_probs
    return pair_scores.detach().cpu().tolist()

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base')

    features = tokenizer(['A man is eating pizza', 'A black race car starts up in front of a crowd of people.'], ['A man eats something', 'A man is driving down a lonely road.'],  padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        label_mapping = ['contradiction', 'entailment', 'neutral']
        labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
        # print(labels)
