import jsonlines
empty_count = 0
total_count = 0
with jsonlines.open('/home/vincent/hyper-simulation/data/retr_result/arc/arc_with_context.jsonl') as reader:
    for item in reader:
        total_count += 1
        if not item.get('answerKey') or item.get('answerKey') == []:
            empty_count += 1
print(f"Total: {total_count}, Empty: {empty_count}")
