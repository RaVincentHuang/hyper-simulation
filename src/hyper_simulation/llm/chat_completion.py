from langchain_core import messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ChatMessage, BaseMessage
from langchain_ollama import ChatOllama

# llm_json = ChatOllama(model="qwen3.5:9b", format="json")

def get_generate(prompts: list[str], model: ChatOllama) -> list:
    """
    Get the response from the LLM using the generate method.
    
    Args:
        prompts (list[str]): The input prompts for the LLM.
        model (ChatOllama): The LLM model to use.
    
    Returns:
        list: The responses from the LLM.
    """
    messages_list: list[list[BaseMessage]] = [
        [HumanMessage(content=prompt)] for prompt in prompts
    ]
    # print("generated_answer")
    responses = model.generate(messages_list)
    res = [generate[0].text for generate in responses.generations]
    # print(f"answer: {res}")
    return res

def get_invoke(model: ChatOllama, text: str, **args):
    """
    Get the response from the LLM using the invoke method.
    
    Args:
        text (str): The input prompt for the LLM.
    
    Returns:
        str: The response from the LLM.
    """
    response = model.invoke(text, **args)
    
    return response.content

def get_stream(model: ChatOllama, text: str, **args):
    
    response = model.stream(text, **args)
    
    return response

def get_invoke_prompt(msg: dict[str, str], prompt: ChatPromptTemplate, **args):
    """
    Get the response from the LLM using the invoke method.
    
    Args:
        msg (dict[str, str]): The input message for the LLM.
        prompt (ChatPromptTemplate): The prompt template for the LLM.
    
    Returns:
        str: The response from the LLM.
    """
    # Convert the message to a string
    llm = ChatOllama(model="qwen3.5:9b", **args)
    chain = prompt | llm
    
    response = chain.invoke(msg, **args)
    
    return response.content

def get_next_msg(msg: AIMessage, **args):
    """
    Get the next message from the LLM using the invoke method.
    
    Args:
        msg (AIMessage): The input message for the LLM.
    
    Returns:
        str: The response from the LLM.
    """
    llm = ChatOllama(model="qwen3.5:9b", **args)
    response = llm.invoke(msg.content)
    return response
