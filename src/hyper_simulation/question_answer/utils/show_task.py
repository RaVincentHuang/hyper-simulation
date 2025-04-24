import json
from huggingface_hub import QuestionAnsweringInput
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit import Application
from prompt_toolkit.layout.containers import VSplit, Window
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import HSplit, Window, Layout
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.application import get_app
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import clear
bindings = KeyBindings()

top_k = 15

class Task:
    class Question:
        def __init__(self, question_id: str, question: str, answer: str):
            self.question_id = question_id
            self.question = question
            self.answer = answer
            self.contexts = []
        
        def add_context(self, context: str):
            self.contexts.append(context)
    
    def __init__(self, path: str):
        self.questions: list[Task.Question] = []
        with open(path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                question_id = example['id']
                question_id = question_id
                # print(example)
                # exit()
                question = Task.Question(question_id, example['question'], example['answers'])             
                # prompt_list: list[dict] = []
                # id_list: list[int] = []
                for ctx in example['ctxs'][:top_k]:
                    text = ctx['text']
                    question.add_context(text)
                self.questions.append(question)
        self.current_question = 0
    
    def show_current(self):
        new_container = VSplit([
            HSplit([
            Window(content=FormattedTextControl(text=lambda: f"[Question] ({task.current_question + 1}/{len(task.questions)}):\n{task.questions[task.current_question].question}"), height=3, style="bg:#555555", wrap_lines=True),
            Window(content=FormattedTextControl(text=lambda: f"[Answer]:\n{task.questions[task.current_question].answer}"), height=3, style="bg:#555555", wrap_lines=True),
            ], width=lambda: int(get_app().output.get_size().columns * 1 / 3)),  # Left side width is 1/3 of the total terminal width
            Window(content=FormattedTextControl(text=lambda: "\n".join([f"[{i + 1}]. {ctx}" for i, ctx in enumerate(task.questions[task.current_question].contexts)])), style="bg:#333333", wrap_lines=True),
        ])
        return new_container
    
    def shift_right(self):
        if self.current_question < len(self.questions) - 1:
            self.current_question += 1
    
    def shift_left(self):
        if self.current_question > 0:
            self.current_question -= 1
            
    def jump(self, id):
        if id < len(self.questions):
            self.current_question = id




if __name__ == "__main__":
    path = "data/retr_result/popqa_longtail_w_gs.jsonl"
    task = Task(path)
    
    buffer1 = Buffer()  # Editable buffer.

    root_container = task.show_current()
    
    
    @bindings.add("right")
    def _(event):
        task.shift_right()
        event.app.layout.container = task.show_current()
    
    @bindings.add("left")
    def _(event):
        task.shift_left()
        event.app.layout.container = task.show_current()
    
    @bindings.add(":")
    def _(event):
        id = prompt("Jump to question ID: ")
        try:
            id = int(id)
            task.jump(id)
            event.app.layout.container = task.show_current()
        except ValueError:
            print("Invalid ID. Please enter a number.")
    
    @bindings.add("q")
    def _(event):
        event.app.exit()

    layout = Layout(root_container)
    
    app = Application(layout=layout, key_bindings=bindings, full_screen=True)
    app.run()
    task.show_current()