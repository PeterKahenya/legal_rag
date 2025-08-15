import uuid
import chromadb
from dotenv import load_dotenv
import os
import json

load_dotenv()
CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY")

client  = chromadb.Client() # or PersistentClient(path=".") or HttpClient
# client  = chromadb.CloudClient(
#     tenant=os.environ.get("CHROMA_TENANT"),
#     api_key=os.environ.get("CHROMA_API_KEY"),
#     database=os.environ.get("CHROMA_DATABASE"),
# )

collection = client.get_or_create_collection(name="legal_rag") # or get_or_create_collection

with open("TheConstitutionOfKenya.json","r", encoding="utf-8") as f:
    chapters = json.load(f)

collection.add(
    ids=[str(uuid.uuid4()) for _ in chapters],
    documents=[chapter["text"] for chapter in chapters],
    metadatas=[{"heading": chapter["heading"], "chapter": id} for id,chapter in enumerate(chapters)]
)

results = collection.query(
    query_texts=[
        "what is devolution?",
        "what is the judiciary?"
    ],
    n_results=1
)

for i, query_results in enumerate(results["metadatas"]):
    print(f"\nQuery {i} Results:")
    print("\n".join([q["heading"] for q in query_results]))
    print("\n=======\n")