# import os
# import sys
# import sysconfig

# # If a local dependency folder (e.g. `.deps`) exists, restrict third-party
# # imports to that folder only. This prevents this script from picking up
# # packages from the active conda environment's site-packages while still
# # allowing imports from the Python standard library and the script folder.
# _HERE = os.path.dirname(__file__)
# _DEPS_DIR = os.path.join(_HERE, ".deps")
# if os.path.isdir(_DEPS_DIR):
#     # Keep a copy of the original sys.path to detect runtime-specific
#     # directories (like lib-dynload) which are required for C-extension
#     # standard-library modules (e.g. _struct). We'll exclude third-party
#     # site-packages but preserve stdlib and lib-dynload paths.
#     _orig_sys_path = sys.path.copy()

#     allowed = [_DEPS_DIR, _HERE, ""]
#     try:
#         stdlib = sysconfig.get_path("stdlib")
#         platstdlib = sysconfig.get_path("platstdlib")
#     except Exception:
#         stdlib = None
#         platstdlib = None
#     for p in (stdlib, platstdlib):
#         if p and p not in allowed:
#             allowed.append(p)

#     # Include any original sys.path entries that look like interpreter
#     # internal library locations (e.g. lib-dynload) but skip site-packages so
#     # that third-party packages from the conda env are not visible.
#     for p in _orig_sys_path:
#         if not p:
#             continue
#         lp = p.lower()
#         if "site-packages" in lp:
#             # intentionally skip third-party site-packages
#             continue
#         if "lib-dynload" in lp or lp.startswith(sys.exec_prefix) or lp.startswith(sys.base_prefix):
#             if os.path.isdir(p) and p not in allowed:
#                 allowed.append(p)

#     # Replace sys.path in-place so subsequent imports only search allowed paths
#     sys.path[:] = allowed
    


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

_model_cache = {}

def get_nli_labels_batch(pairs: list[tuple[str, str]]) -> list[str]:
    if 'nli-deberta-v3-base' not in _model_cache:
        from sentence_transformers import CrossEncoder
        _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base', device="cpu")
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
        _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base', device="cpu")
    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs)
    entailment_scores = [score[1] for score in scores]
    return entailment_scores

def get_nli_contradiction_score_batch(pairs: list[tuple[str, str]]) -> list[float]:
    if 'nli-deberta-v3-base' not in _model_cache:
        from sentence_transformers import CrossEncoder
        _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base', device="cpu")
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
