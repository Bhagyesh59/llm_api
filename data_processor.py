import pymongo
import uuid
from typing import List, Union
import hashlib
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

model = SentenceTransformer('all-MiniLM-L6-v2')

# remove this in prod ----------------------------------------------------------------
MONGO_URI = "mongodb+srv://LLM_User:User_llm01@clusterllm.saneb.mongodb.net/?retryWrites=true&w=majority&appName=ClusterLLM"
# remove this in prod----------------------------------------------------------------



def load_doc(doc_name):
    loader = PyPDFLoader(doc_name)
    pages = loader.load()
    return pages

  
def mongo_setup(mongo_uri):
    client = pymongo.MongoClient(mongo_uri)
    db = client['LLM_Embeddings_Jobs']
    collection = db['test_collection01']
    return collection


def deterministic_uuid(content: Union[str, bytes]) -> str:
    """Creates deterministic UUID on hash value of string or byte content.

    Args:
        content: String or byte representation of data.

    Returns:
        UUID of the content.
    """
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        content_bytes = content
    else:
        raise ValueError(f"Content type {type(content)} not supported!")

    hash_object = hashlib.sha256(content_bytes)
    hash_hex = hash_object.hexdigest()
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    content_uuid = str(uuid.uuid5(namespace, hash_hex))

    return content_uuid

def doc_embedor(chunks: List, jd_name: str):
    embedded_docs = []
    for doc in chunks:
        # Assuming `model.encode` generates embeddings for the document
        embedding = model.encode(doc.page_content)  
        embedding = embedding.tolist()

        # Create a deterministic UUID for the document content
        id = deterministic_uuid(doc.page_content)+"-"+ jd_name

        # Append the document data and embedding to the embedded_docs list
        embedded_docs.append({"jd_id": id,"data": doc.page_content, "embedding": embedding})

    return embedded_docs


def get_chunked_data(data,chunk_size,chunk_overlap):
    
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap  = chunk_overlap,
    length_function = len,
  )
  return text_splitter.split_documents(data)


def run(doc_name,jd_name):
    pages=load_doc(doc_name)
    chunks = get_chunked_data(pages, 200, 50)
    ready_data=doc_embedor(chunks,jd_name)
    collection=mongo_setup(MONGO_URI)
    collection.insert_many(ready_data)
    return "data processing completed"
    