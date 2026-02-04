import os


_model_cache = {}

def get_nli_labels_batch(pairs: list[tuple[str, str]]) -> list[str]:
    if 'nli-deberta-v3-base' not in _model_cache:
        from sentence_transformers import CrossEncoder
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        # _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base', device="cpu")
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs)
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    return labels

def get_nli_label(text1: str, text2: str) -> str:
    labels = get_nli_labels_batch([(text1, text2)])
    return labels[0]

def get_nli_entailment_score_batch(pairs: list[tuple[str, str]]) -> list[float]:
    if 'nli-deberta-v3-base' not in _model_cache:
        from sentence_transformers import CrossEncoder
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        # _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base', device="cpu")
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs)
    entailment_scores = [score[1] for score in scores]
    return entailment_scores

def get_nli_contradiction_score_batch(pairs: list[tuple[str, str]]) -> list[float]:
    if 'nli-deberta-v3-base' not in _model_cache:
        from sentence_transformers import CrossEncoder
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        # _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base', device="cpu")
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs)
    contradiction_scores = [score[0] for score in scores]
    return contradiction_scores

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
        print(labels)
