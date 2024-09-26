from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from constants import APIKEY
import os

os.environ["OPENAI_API_KEY"] = APIKEY


# Khai bao bien
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

def create_vector_db():
    # Load data
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(pdf_data_path, glob="*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embedding
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # Create vector store
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

create_vector_db()