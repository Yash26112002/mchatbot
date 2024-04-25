from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
print("yo")
docs_chunks = [t.page_content for t in text_chunks]
print("yoo")
index_name="mchat"
vectorstore_from_texts = PineconeVectorStore.from_texts(
        docs_chunks,
        index_name=index_name,
        embedding=embeddings
    )
print("yooo")
