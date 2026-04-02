cd /home/vincent/HER
pixi run Musique
cd /home/vincent/hyper-simulation
cp /home/vincent/HER/data/HER/musique.jsonl /home/vincent/hyper-simulation/data/mid_result/her/
pixi run Musique --output_path /home/vincent/hyper-simulation/data/baseline/her --method vanilla --load_prompts /home/vincent/hyper-simulation/data/mid_result/her/musique.jsonl
cd /home/vincent/HER
pixi run LegalBench
cd /home/vincent/hyper-simulation
cp /home/vincent/HER/data/HER/legalbench.jsonl /home/vincent/hyper-simulation/data/baseline/her/
pixi run LegalBench --output_path /home/vincent/hyper-simulation/data/baseline/her --method vanilla --load_prompts /home/vincent/hyper-simulation/data/mid_result/her/legalbench.jsonl