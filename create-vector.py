import os
from pymongo import MongoClient
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch

# Define the URL of the PDF MongoDB Atlas Best Practices document to be processed
pdf_url = "https://query.prod.cms.rt.microsoft.com/cms/api/am/binary/RE4HkJP"

# Retrieve environment variables for sensitive information
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

ATLAS_CONNECTION_STRING = os.getenv('ATLAS_CONNECTION_STRING')
if not ATLAS_CONNECTION_STRING:
    raise ValueError("The ATLAS_CONNECTION_STRING environment variable is not set.")

# Connect to MongoDB Atlas cluster using the connection string
cluster = MongoClient(ATLAS_CONNECTION_STRING)

# Define the MongoDB database and collection names
DB_NAME = "langchain"
COLLECTION_NAME = "vectorSearch"

# Connect to the specific collection in the database
MONGODB_COLLECTION = cluster[DB_NAME][COLLECTION_NAME]

# Initialize the PDF loader with the defined URL
loader = PyPDFLoader(pdf_url)

# Load the PDF document's data
data = loader.load()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# Split the document into manageable segments
docs = text_splitter.split_documents(data)

# Initialize MongoDB Atlas vector search with the document segments
vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    collection=MONGODB_COLLECTION,
    index_name="default"  # Use a predefined index name
)

# At this point, 'docs' are split and indexed in MongoDB Atlas, enabling text search capabilities.
