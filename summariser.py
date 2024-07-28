#! /usr/bin/env python3

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

import os
import sys

local_path = sys.argv[1]
if os.path.exists(local_path):
  loader = UnstructuredPDFLoader(file_path=local_path)
  data = loader.load()
else:
  print("Upload a PDF file")
  exit(1)

data[0].page_content

text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=False),
    collection_name="send-rag"
)

local_model = "mistral"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI architecture expert. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#print("Enter your question:")
#print(chain.invoke(input()))

prompt = """
We have a multi-agent infrastructure where agents can communicate
with each other to retrieve concise information. We have an orchestrator, who's
job it is to delegate tasks to other agents. The orchestrator is responsible for
managing the communication between agents and the user and each agent is
responsible for a specific task.

The purpose of this system is to help users find relevant answers from a large
pool of data coming from students, teachers, local authorities, and other
stakeholders.

We're keeping user related documentation within the block-chain
and retrieve the information when it is required. We should also be able to
store business domain relevant data within a vector database.

When we require information from a user we temporarily store it in a vector
database dedicated to the individual user. We also need to make sure that this
information is only accessible to users that are authorised to access it.

We want to create an AI architecture that can support the finding of relevant
documents in a vector database. What are the key components?
"""

print(chain.invoke(prompt))
