import jsonlines

orig_data = {}
with jsonlines.open('/home/vincent/hyper-simulation/data/eval_data/arc_challenge_processed.jsonl') as reader:
    for item in reader:
        orig_data[item['question']] = {
            'answerKey': item.get('answerKey', ''),
            'choices': item.get('choices', {})
        }

fixed_data = []
with jsonlines.open('/home/vincent/hyper-simulation/data/retr_result/arc/arc_with_context.jsonl') as reader:
    for item in reader:
        q_full = item['question']
        # Extract original question by removing \n\nOptions:
        q = q_full.split('\n\nOptions:\n')[0]
        if q in orig_data:
            item['answerKey'] = orig_data[q]['answerKey']
            item['choices'] = orig_data[q]['choices']
        fixed_data.append(item)

with jsonlines.open('/home/vincent/hyper-simulation/data/retr_result/arc/arc_with_context.jsonl', 'w') as writer:
    writer.write_all(fixed_data)
print("Fixed arc_with_context.jsonl")
