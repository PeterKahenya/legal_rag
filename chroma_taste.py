import uuid
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()
CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY")

# client  = chromadb.Client() # or PersistentClient(path=".") or HttpClient
client  = chromadb.CloudClient(
    tenant=os.environ.get("CHROMA_TENANT"),
    api_key=os.environ.get("CHROMA_API_KEY"),
    database=os.environ.get("CHROMA_DATABASE"),
)

collection = client.get_or_create_collection(name="legal_rag") # or get_or_create_collection

with open("driving_license_instructions.txt","r", encoding="utf-8") as f:
    instructions = f.read().splitlines()

collection.add(
    ids=[str(uuid.uuid4()) for _ in instructions],
    documents=[i.strip().replace("\n","") for i in instructions],
    metadatas=[{"line": line} for line in range(len(instructions))]
)

results = collection.query(
    query_texts=[
        "how do i create an account",
        "what is the cost?"
    ],
    n_results=2
)

for i, query_results in enumerate(results["documents"]):
    print(f"\nQuery {i} Results:")
    print("\n".join(query_results))
    print("\n=======\n")