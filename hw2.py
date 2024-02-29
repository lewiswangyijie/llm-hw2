from openai import AzureOpenAI
from dotenv import load_dotenv
import pandas as pd
import tiktoken
import os

env_path = os.getenv("HOME") + "/Documents/src/openai/.env"
load_dotenv(dotenv_path=env_path, verbose=True)

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://pvg-azure-openai-uk-south.openai.azure.com"

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2023-05-15"
)

db_path = "data/irm-help"
input_path = "data/IRM-Help.pdf"

from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS


loader = PyPDFLoader(file_path=input_path)
data = loader.load()

db = FAISS.from_documents(data, AzureOpenAIEmbeddings())

db.save_local(db_path)
new_db = FAISS.load_local(db_path, AzureOpenAIEmbeddings())

query = "Managing Return Reasons"
answer_list = new_db.similarity_search(query)
for ans in answer_list:
    print(ans.page_content + "\n")

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)
docs = retriever.get_relevant_documents("Managing Return Reasons")
for doc in docs:
    print(doc.page_content + "\n")

from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI

llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.3)
qa_chain = RetrievalQA.from_chain_type(llm,
             retriever=new_db.as_retriever(search_type="similarity_score_threshold",
               search_kwargs={"score_threshold": 0.7}))
qa_chain.combine_documents_chain.verbose = True
qa_chain.return_source_documents = True

qa_chain({"query": "Managing Return Reasons"})
