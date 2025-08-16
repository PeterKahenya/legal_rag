import os
import json
import argparse
from typing import List
import uuid
import bs4
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import requests


load_dotenv()
embeddings = OpenAIEmbeddings() # use default model for now

# Load
def load_documents(path: str) -> List[Document]:
    with open(path,"r", encoding="utf-8") as f:
        docs = json.load(f)
    docs = [
            Document(
                page_content=document["text"],
                metadata={"heading": document["heading"]},
                id=id
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
def index_chunks(docs: List[Document], collection: str) -> List[str]:
    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory = "./legal_rag_vectorstore"
    )
    uuids = [str(uuid.uuid4()) for _ in range(len(docs))]
    return vector_store.add_documents(documents=docs, ids=uuids)

def load_case(url: str):
    case_loader = WebBaseLoader(
        web_path=url,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_ = ("case_content")
            )
        )
    )
    doc = case_loader.load()[0]
    bs = bs4.BeautifulSoup(requests.get(url).content, features="html.parser")
    table = bs.find("table", {"class": "meta_info"})
    rows = table.find_all("tr")[1:-1]
    metadata = {}
    for row in rows:
        th = row.find("th")
        td = row.find("td")
        if th and td:
            key = th.get_text(strip=True).rstrip(":")  # remove trailing colon
            value = td.get_text(strip=True)
            metadata[key] = value
    doc.metadata = {**doc.metadata,**metadata}
    return doc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="index",
        description="indexes the json docs into the vector db for further querying",
        epilog="supports json only for now",
        usage="python index.py --jsonfile datasets/TheConstitutionOfKenya.json --collection constitution"
    )
    parser.add_argument("--jsonfile")
    parser.add_argument("--url")
    parser.add_argument("--collection",default="case_law",choices=("constitution","case_law","acts"))
    args = parser.parse_args()

    if args.jsonfile and os.path.exists(args.jsonfile):    
        docs = load_documents(path=args.jsonfile)
    elif args.url:
        doc = load_case(url=args.url)
        docs = [doc]
    else:
        raise Exception("Either a url or jsonfile should be provided")
    chunks = chunk_documents(docs=docs)
    chunk_ids = index_chunks(docs=chunks, collection=args.collection)