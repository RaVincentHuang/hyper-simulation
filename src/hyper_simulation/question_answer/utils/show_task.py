import json
from prompt_toolkit.key_binding import KeyBindings
bindings = KeyBindings()

class Questions:
    def __init__(self, path: str):
        

    def __repr__(self):
        return f"Question: {self.question}, Answer: {self.answer}"

def show_line(path):
    hop_4 = []
    with open(path) as file:
        for line in file:
            task = json.loads(line)
            task = json.dumps(task, indent=4)
            print(task)
            exit(0)
    
def show_retrieval(path):
    hop_4 = []
    with open(path) as file:
        for line in file:
            task = json.loads(line)
            task = json.dumps(task, indent=4)
            hop_4.append(task)
    return hop_4



if __name__ == "__main__":
    path = "data/eval_data/musique_4hop.jsonl"
    show_line(path)