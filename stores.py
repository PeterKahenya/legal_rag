from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore, LocalFileStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings()

constitution_retriever = MultiVectorRetriever(
    vectorstore = Chroma(
        collection_name="constitution",
        embedding_function=embeddings,
        persist_directory = "./legal_rag_vectorstore"
    ),
    byte_store=LocalFileStore(root_path="legal_rag_filestore"),
    id_key="doc_id"
)

caselaw_retriever = MultiVectorRetriever(
    vectorstore = Chroma(
        collection_name="caselaw",
        embedding_function=embeddings,
        persist_directory = "./legal_rag_vectorstore"
    ),
    byte_store=LocalFileStore(root_path="legal_rag_filestore"),
    id_key="doc_id"
)

acts_retriever = MultiVectorRetriever(
    vectorstore = Chroma(
        collection_name="acts",
        embedding_function=embeddings,
        persist_directory = "./legal_rag_vectorstore"
    ),
    byte_store=LocalFileStore(root_path="legal_rag_filestore"),
    id_key="doc_id"
)
