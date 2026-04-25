import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

app = Flask(__name__)

CHROMA_DIR = "chroma_db"
KB_DIR = "knowledge_base"


def get_api_key():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY is missing")
    return api_key


def get_llm():
    return ChatCohere(
        cohere_api_key=get_api_key(),
        model="command-r-08-2024",
        temperature=0.3
    )


def get_embeddings():
    return CohereEmbeddings(
        cohere_api_key=get_api_key(),
        model="embed-english-v3.0"
    )


def load_documents():
    documents = []

    for filename in os.listdir(KB_DIR):
        if filename.endswith(".txt"):
            loader = TextLoader(
                os.path.join(KB_DIR, filename),
                encoding="utf-8"
            )
            documents.extend(loader.load())

    return documents


def get_vector_db():
    embeddings = get_embeddings()

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )


def answer_from_knowledgebase(message):
    vector_db = get_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever
    )

    result = qa.invoke({"query": message})
    return result["result"]


def search_knowledgebase(message):
    vector_db = get_vector_db()
    docs = vector_db.similarity_search(message, k=3)

    if not docs:
        return "No matching results found."

    results = []
    for i, doc in enumerate(docs, start=1):
        results.append(f"{i}. {doc.page_content}")

    return "\n\n".join(results)


def answer_as_chatbot(message):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are ThinkBot, a helpful chatbot. Answer clearly and simply."),
        ("human", "{message}")
    ])

    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"message": message})


@app.route("/kbanswer", methods=["POST"])
def kbanswer():
    message = request.json["message"]
    response_message = answer_from_knowledgebase(message)
    return jsonify({"message": response_message}), 200


@app.route("/search", methods=["POST"])
def search():
    message = request.json["message"]
    response_message = search_knowledgebase(message)
    return jsonify({"message": response_message}), 200


@app.route("/answer", methods=["POST"])
def answer():
    message = request.json["message"]
    response_message = answer_as_chatbot(message)
    return jsonify({"message": response_message}), 200


@app.route("/")
def index():
    return render_template("index.html", title="ThinkBot")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)