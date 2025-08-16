
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

def constitutional_vectorstore_query_structure(question:str):
    system = """You are an expert at converting user legal questions into vector database queries. \
    You have access to a vector database containing sections of Kenya's constitution \
    Given a question, return a database query optimized to retrieve the most relevant results.

    If there are acronyms or words you are not familiar with, do not try to rephrase them."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    class ConstitutionSearch(BaseModel):
        content_search: str = Field(
            ...,
            description="Similarity search to apply to section text",
        )
        heading_search: str = Field(
            ...,
            description="Search to apply to heading of section"
        )
    structured_llm = llm.with_structured_output(ConstitutionSearch)
    query_analyzer = prompt | structured_llm
    filters = query_analyzer.invoke(question)
    return filters


filters = constitutional_vectorstore_query_structure("what is kenya?")
print(filters)