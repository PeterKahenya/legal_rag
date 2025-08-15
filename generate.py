import argparse
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings() # use default model for now
vector_store = Chroma(
    collection_name="legal_rag",
    embedding_function=embeddings,
    persist_directory = "./legal_rag_vectorstore"
)
PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
chain = prompt | llm

def retreive(question: str)-> List[Document]:
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke(question)
    return docs


def generate(context: List[Document], question: str):
    answer = chain.invoke({"context":context,"question":question})
    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="legal_rag_generator",
        description="Generate legal answer for a given question",
        usage="""python generate.py --question 'what is kenya?'"""
    )
    parser.add_argument("--question")
    args = parser.parse_args()

    if not args.question:
        raise Exception("Please supply a question via --question argument")
    
    context = retreive(question=args.question)
    answer = generate(context=context, question=args.question)
    print(f"Question: {args.question} \nAnswer: {answer.content}")
