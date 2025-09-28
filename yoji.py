



# !pip install --upgrade --quiet  langchain langchain-huggingface sentence_transformers
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from transformers import AutoModel

model = AutoModel.from_pretrained("google-bert/bert-base-cased")

# Push the model to your namespace with the name "my-finetuned-bert".
# model.push_to_hub("my-finetuned-bert")
# type(model)
# Push the model to an organization with the name "my-finetuned-bert".
# model.push_to_hub("huggingface/my-finetuned-bert")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs={'device': 'cpu'} )

# !pip install -U langchain-community

import streamlit as st

# Commented out IPython magic to ensure Python compatibility.
# %pip install -qU langchain-openai
import chromadb
import os
from openai import OpenAI
os.environ["GITHUB_TOKEN"] = "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"
token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
    
)

CHROMA_PATH = "/chroma"
DATA_PATH = "/content/sample_data/data"
# "/content/sample_data/chroma"

# pip install langchain.vectorstores

import argparse
from langchain.schema.document import Document

# import chromadb
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
# from langchain_mistralai import ChatMistralAI
# from get_embedding_function import get_embedding_function

# CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""



cliento = chromadb.PersistentClient(path="./chroma")



import argparse
from langchain.schema.document import Document

# import chromadb
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""





def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = embeddings
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

#     model = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=0,
#     max_retries=2,
#     # other params...
# )
    response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": f"You are an expert assistant. Please answer the user's question based only on the following context:\n\n---\n{context_text}\n---"

        },
        {
            "role": "user",
            "content": query_text,


        }
    ],
    temperature=1,
    top_p=1,
    model=model
)

    response_text = response.choices[0].message.content
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}" #\nSources: {sources}#
    response = {"query_text":query_text,"answer":formatted_response}
    return formatted_response
text = "what is rainwater harvesting"
# query_rag(text)
final_model = query_rag(text)
print(final_model)

# from langchain_openai.chat_models import ChatOpenAI

st.title("🦜🔗 Quickstart App")

# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


def generate_response(input_text):
    # model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
    st.info(query_rag(input_text))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted :
        generate_response(text)
# !pip install --upgrade --force-reinstall chromadb opentelemetry-api opentelemetry-sdk

# !zip -r vector_db.zip /content/sample_data/chroma/
