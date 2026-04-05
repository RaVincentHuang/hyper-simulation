pixi run -e simulation ARC --output_path /home/vincent/hyper-simulation/data/baseline/vanilla --method vanilla
pixi run -e simulation ARC --output_path /home/vincent/hyper-simulation/data/baseline/cdit --method cdit
pixi run -e simulation ARC --output_path /home/vincent/hyper-simulation/data/baseline/contradoc --method contradoc
pixi run -e simulation ARC --output_path /home/vincent/hyper-simulation/data/mid_result --method sparsecl --save_prompts_only
pixi run -e simulation ARC --output_path /home/vincent/hyper-simulation/data/mid_result --method sentli --save_prompts_only
pixi run -e simulation ARC --output_path /home/vincent/hyper-simulation/data/baseline/sparsecl --method sparsecl --load_prompts /home/vincent/hyper-simulation/data/mid_result/sparsecl/ARC.jsonl
pixi run -e simulation ARC --output_path /home/vincent/hyper-simulation/data/baseline/sentli --method sentli --load_prompts /home/vincent/hyper-simulation/data/mid_result/sentli/ARC.jsonl
