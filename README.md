# hyper-simulation


### Build 50
```shell
pixi run -e hypergraph build --rebuild --data_path /home/vincent/.dataset/HotpotQA/sample20distractor --task hotpotqa ; \
pixi run -e hypergraph build --rebuild --data_path /home/vincent/.dataset/musique/sample50/musique.jsonl --task musique
```

### RUN 50
```shell

pixi run -e simulation hyper_simulation --data_path /home/vincent/.dataset/HotpotQA/sample20distractor --task hotpotqa ; \
pixi run -e simulation hyper_simulation --data_path /home/vincent/.dataset/musique/sample50/musique.jsonl --task musique

```

```
pixi run -e hypergraph build  --data_path /home/vincent/.dataset/HotpotQA/sample20distractor --task hotpotqa ; \
pixi run -e hypergraph build  --data_path /home/vincent/.dataset/musique/sample50/musique.jsonl --task musique; \
pixi run -e simulation hyper_simulation --data_path /home/vincent/.dataset/HotpotQA/sample20distractor --task hotpotqa ; \
pixi run -e simulation hyper_simulation --data_path /home/vincent/.dataset/musique/sample50/musique.jsonl --task musique
```

### Musique
```shell

pixi run -e simulation rag_no_retrival --data_path /home/vincent/.dataset/musique/sample50/musique.jsonl --output_path data/baseline/contradoc/musique --method contradoc --task musique

```

### TEST BUILD
```shell
pixi run -e hypergraph test_build --data_path /home/vincent/.dataset/HotpotQA/sample20distractor --output_dir data/playground/hypergraph --task hotpotqa ; \
pixi run -e hypergraph test_build --data_path /home/vincent/.dataset/musique/sample50/musique_answerable.jsonl --output_dir data/playground/hypergraph --task musique; \
pixi run -e hypergraph test_build --data_path /home/vincent/.dataset/MultiHop/sample50/multihop_rag.jsonl --output_dir data/playground/hypergraph --task multihop
```

### SpaCy Debugging
```shell
pixi run -e hypergraph display --steps 1
```

### Local Model
```shell
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
```

pixi run -e simulation remote --task docnli --dataset-path data/nli/docnli_50.jsonl --source-root data/debug/docnli/sample50 --max-workers 8 && \
pixi run -e simulation remote --task econ --dataset-path data/nli/econ_qa.jsonl --source-root data/debug/econ/sample --max-workers 8 && \
pixi run -e simulation remote --task contract_nli --dataset-path data/nli/contract_nli_split_sample65.jsonl --source-root data/debug/contract_nli/sample65 --max-workers 8
