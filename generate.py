import argparse
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

from query_translation.multi_query import multiquery_prompt, get_unique_union
from query_translation.rag_fusion import reciprocal_rank_fusion
from query_translation.decomposition import query_decomposition_prompt, decomposition_prompt, format_qa_pair
from query_translation.step_back import stepback_question_prompt, stepback_response_prompt
from query_translation.hyde import hypothetical_doc_prompt, hyde_prompt

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
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retreive(question: str)-> List[Document]:
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke(question)
    return docs

def generate(context: List[Document], question: str):
    chain =  prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": format_docs(context), "question": question})
    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="legal_rag_generator",
        description="Generate legal answer for a given question",
        usage="""python generate.py --question 'what is kenya?'"""
    )
    parser.add_argument("--question")
    parser.add_argument("--translation",choices=("multi","fusion","decomposition","stepback","hyde"))
    args = parser.parse_args()

    if not args.question:
        raise Exception("Please supply a question via --question argument")
    
    match args.translation:
        case "multi":
            generate_multiple_questions = multiquery_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
            retrieve_chain =  generate_multiple_questions | retriever.map() | get_unique_union
            context = retrieve_chain.invoke({"question": args.question})
            answer = generate(context=context, question=args.question)
        case "fusion":
            generate_multiple_questions = multiquery_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
            retrieve_chain = generate_multiple_questions | retriever.map() | reciprocal_rank_fusion
            context = retrieve_chain.invoke({"question": args.question})
            answer = generate(context=context, question=args.question)
        case "decomposition":
            generate_queries_decomposition =  query_decomposition_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
            questions = generate_queries_decomposition.invoke({"question":args.question})
            q_a_pairs = ""
            for q in questions:
                rag_chain = (
                    {"context": itemgetter("question") | retriever, 
                    "question": itemgetter("question"),
                    "q_a_pairs": itemgetter("q_a_pairs")} 
                    | decomposition_prompt | llm | StrOutputParser()
                    | llm
                    | StrOutputParser()
                    )
                answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
                q_a_pair = format_qa_pair(q,answer)
                q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
        case "stepback":
            stepback_chain = (
                {
                    "original_context": RunnableLambda(lambda x: x["question"]) | retriever,
                    "step_back_context": stepback_question_prompt | llm | StrOutputParser() | retriever,
                    "question": lambda x: x["question"],
                } 
                | stepback_response_prompt
                | llm
                | StrOutputParser()
            )
            answer = stepback_chain.invoke({"question": args.question})
        case "hyde":
            hypothetical_doc_chain = hypothetical_doc_prompt | llm | StrOutputParser() | retriever
            context = hypothetical_doc_chain.invoke({"question": args.question})
            hyde_chain = hyde_prompt | llm | StrOutputParser()
            answer = hyde_chain.invoke({"question":args.question, "context": context})
        case _:
            context = retreive(question=args.question)
            answer = generate(context=context, question=args.question)

    print(f"\n\n ==== {args.translation} ===")
    print(f"Question: {args.question} \nAnswer: {answer}")
