import langchain 
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen2.5:72b", )

x= llm.invoke("What is the capital of France?")
print(x)
