from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class RouteQuery(BaseModel):
    datasource: Literal["constitution", "case_law", "acts"] = Field(...,description="Given a user question choose which datasource would be most relevant for answering their question",)


llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)
llm = llm.with_structured_output(RouteQuery)

system = """
You are a legal assistant and researcher responsible for researching precendence and legal justifications to questions.
Given a question, you are required to select an appropriate datasource from those provided for search to proceed.
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
logical_router = prompt | llm

constitutional_law_prompt_template = """You are a very skilled Kenyan constitutional law researcher. \
You are great at answering questions about constitutional provisions, interpretation, and case law in a concise and easy to understand manner. \
You carefully explain how different articles of the constitution apply, and connect them to relevant judicial decisions when appropriate. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

caselaw_search_prompt_template = """You are an excellent legal analyst. You are great at answering questions about judicial precedents. \
You are so good because you are able to break down complex legal issues into their component parts, \
explain the component parts, and then put them together to answer the broader legal question.

Here is a question:
{query}"""

legislative_acts_search_prompt_template = """You are an excellent legislative researcher. You are great at answering questions about statutory law and legislative acts. \
You are so good because you are able to break down complex statutory provisions into their component parts, \
explain each part, and then synthesize them to answer the broader legal question.

Here is a question:
{query}"""

embeddings = OpenAIEmbeddings()
prompt_templates = [constitutional_law_prompt_template, caselaw_search_prompt_template, legislative_acts_search_prompt_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    return PromptTemplate.from_template(most_similar)


semantic_router = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatOpenAI()
    | StrOutputParser()
)