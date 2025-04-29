import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from langchain_core.messages import HumanMessage

def get_openai_llm():
    return LangchainLLMWrapper(
        langchain_llm=ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    )

def get_openai_embeddings():
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def generate_answer(llm_wrapper, question, context):
    combined_prompt = f"質問: {question}\nコンテキスト: {context}\n\n回答:"
    messages = [HumanMessage(content=combined_prompt)]
    llm_result = llm_wrapper.langchain_llm.generate([messages])
    return llm_result.generations[0][0].text.strip()

def read_context_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        context = file.read()
    return context