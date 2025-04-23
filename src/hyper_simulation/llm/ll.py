import langchain 
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen2.5:72b", )

# x= llm.invoke("What is the capital of France?")
# print(x)

def get_invoke_response(prompt):
    """
    Get the response from the LLM using the invoke method.
    
    Args:
        prompt (str): The input prompt for the LLM.
    
    Returns:
        str: The response from the LLM.
    """
    response = llm.invoke(prompt)
    return response

if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = get_invoke_response(prompt)
