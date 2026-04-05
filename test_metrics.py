from hyper_simulation.question_answer.vmdit.metrics import exact_match_score, match
print("exact_match_score('C', 'C'):", exact_match_score('C', 'C'))
print("exact_match_score('C', '[]'):", exact_match_score('C', '[]'))
print("match('C', ['C']):", match('C', ['C']))
print("match('C', ['[]']):", match('C', ['[]']))
