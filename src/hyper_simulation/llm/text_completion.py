from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen2.5:72b",)

def get_invoke(text, **args) -> str:
    """
    Get the response from the LLM using the invoke method.
    
    Args:
        text (str): The input prompt for the LLM.
    
    Returns:
        str: The response from the LLM.
    """
    response = llm.invoke(text, **args)
    return response

def get_invoke_prompt(msg: dict[str, str], prompt: ChatPromptTemplate, **args) -> str:
    """
    Get the response from the LLM using the invoke method.
    
    Args:
        msg (dict[str, str]): The input message for the LLM.
        prompt (ChatPromptTemplate): The prompt template for the LLM.
    
    Returns:
        str: The response from the LLM.
    """
    # Convert the message to a string
    
    chain = prompt | llm
    
    response = chain.invoke(msg, **args)
    
    return response