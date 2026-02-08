# hyper-simulation


### Musique
```shell

pixi run -e simulation rag_no_retrival --data_path /home/vincent/.dataset/musique/sample50/musique_answerable.jsonl --output_path data/baseline/contradoc/musique --method contradoc --task musique

```

### TEST BUILD
```shell
pixi run -e hypergraph test_build --data_path /home/vincent/.dataset/HotpotQA/sample20distractor --output_dir data/playground/hypergraph --task hotpotqa

pixi run -e hypergraph test_build --data_path /home/vincent/.dataset/musique/sample50/musique_answerable.jsonl --output_dir data/playground/hypergraph --task musique

pixi run -e hypergraph test_build --data_path /home/vincent/.dataset/MultiHop/sample50/multihop_rag.jsonl --output_dir data/playground/hypergraph --task multihop
```

### SpaCy Debugging
```shell
pixi run -e hypergraph display --steps 1
```