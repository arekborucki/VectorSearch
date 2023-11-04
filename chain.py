import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Set DB
if os.environ.get("MONGO_URI", None) is None:
    raise Exception("Missing `MONGO_URI` environment variable.")
MONGO_URI = os.environ["MONGO_URI"]

DB_NAME = "langchain"
COLLECTION_NAME = "vectorSearch"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "default"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
MONGODB_COLLECTION = db[COLLECTION_NAME]

# Read from MongoDB Atlas Vector Search
vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    DB_NAME + "." + COLLECTION_NAME,
    OpenAIEmbeddings(),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 100, "post_filter_pipeline": [{"$limit": 1}]}
    )


# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613") 
chain = (
    RunnableParallel({"context": retriever,"question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)
# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
