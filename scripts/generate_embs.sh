python src/hyper_simulation/question_answer/generate_passage_embedding.py \
    --model_name_or_path "models/contriever-msmarco" \
    --passages data/eval_data/popqa_longtail.jsonl \
    --output_dir data/ex_embeddings \
    --shard_id 0 \
    --num_shards 1 \
    --per_gpu_batch_size 500
