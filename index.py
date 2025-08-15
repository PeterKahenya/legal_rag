import os
import json
import argparse
from typing import List
import uuid
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


load_dotenv()


embeddings = OpenAIEmbeddings() # use default model for now
vector_store = Chroma(
    collection_name="legal_rag",
    embedding_function=embeddings,
)

# Load
def load_documents(path: str) -> List[Document]:
    with open(path,"r", encoding="utf-8") as f:
        docs = json.load(f)
    docs = [
            Document(
                page_content=document["text"],
                metadata={"heading": document["heading"]},
                id=id,
            )
            for id,document in enumerate(docs)
        ]
    return docs

# Chunk
def chunk_documents(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

# Embed and Index
def index_chunks(docs: List[Document]) -> List[str]:
    uuids = [str(uuid.uuid4()) for _ in range(len(docs))]
    return vector_store.add_documents(documents=docs, ids=uuids, )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="index",
        description="indexes the json docs into the vector db for further querying",
        epilog="supports json only for now"
    )
    parser.add_argument("--jsonfile")
    args = parser.parse_args()

    if not os.path.exists(args.jsonfile):
        raise Exception(f"JSON file not found: {args.jsonfile}")
    
    docs = load_documents(path=args.jsonfile)
    chunks = chunk_documents(docs=docs)
    chunk_ids = index_chunks(docs=chunks)

    ### TEMP TEST WITH RETRIEVE ###
    retriever = vector_store.as_retriever(search_kwargs={"k": 1}) # k is the number of neighbor documents
    docs_r = retriever.invoke("What is Kenya?")
    print(docs_r)