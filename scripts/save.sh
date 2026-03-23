pixi run -e simulation Musique --output_path data/baseline/contradoc --method contradoc
pixi run -e simulation HotpotQA --output_path mid_result --method sparsecl --save_prompts_only
pixi run -e simulation Musique --output_path mid_result --method sparsecl --save_prompts_only
pixi run -e simulation MultiHop --output_path mid_result --method sparsecl --save_prompts_only
pixi run -e simulation ARC --output_path mid_result --method sparsecl --save_prompts_only
pixi run -e simulation LegalBench --output_path mid_result --method sparsecl --save_prompts_only
LD_LIBRARY_PATH="/home/vincent/hyper-simulation/.pixi/envs/simulation/lib" pixi run -e simulation HotpotQA --output_path mid_result --method sentli --save_prompts_only
LD_LIBRARY_PATH="/home/vincent/hyper-simulation/.pixi/envs/simulation/lib" pixi run -e simulation Musique --output_path mid_result --method sentli --save_prompts_only
LD_LIBRARY_PATH="/home/vincent/hyper-simulation/.pixi/envs/simulation/lib" pixi run -e simulation MultiHop --output_path mid_result --method sentli --save_prompts_only
LD_LIBRARY_PATH="/home/vincent/hyper-simulation/.pixi/envs/simulation/lib" pixi run -e simulation ARC --output_path mid_result --method sentli --save_prompts_only
LD_LIBRARY_PATH="/home/vincent/hyper-simulation/.pixi/envs/simulation/lib" pixi run -e simulation LegalBench --output_path mid_result --method sentli --save_prompts_only