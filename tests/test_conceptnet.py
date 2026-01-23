import conceptnet_lite
conceptnet_lite.connect("conceptnet.db") # 第一次运行会自动下载数据库文件

from conceptnet_lite import Label

try:
    apple = Label.get(text='apple', language='en').concepts[0]
    print(f"Concept: {apple.text}")
    for relation in apple.relations:
         print(f"  - {relation.start.text} -> {relation.uri} -> {relation.end.text}")
except Exception as e:
    print(f"Error: {e}")