import jsonlines

orig_data = []
with jsonlines.open('/home/vincent/hyper-simulation/data/eval_data/arc_challenge_processed.jsonl') as reader:
    for item in reader:
        orig_data.append({
            'q': item['question'].strip(),
            'answerKey': item.get('answerKey', ''),
            'choices': item.get('choices', {})
        })

fixed_data = []
with jsonlines.open('/home/vincent/hyper-simulation/data/retr_result/arc/arc_with_context.jsonl') as reader:
    for item in reader:
        q_full = item['question']
        
        # Try to find the matching question
        matched = False
        for od in orig_data:
            if q_full.startswith(od['q']):
                item['answerKey'] = od['answerKey']
                item['choices'] = od['choices']
                matched = True
                break
        
        if not matched:
            print("Failed to match:", repr(q_full[:50]))
            
        fixed_data.append(item)

with jsonlines.open('/home/vincent/hyper-simulation/data/retr_result/arc/arc_with_context.jsonl', 'w') as writer:
    writer.write_all(fixed_data)
print("Fixed arc_with_context.jsonl")
