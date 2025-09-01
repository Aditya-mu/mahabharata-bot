import faiss
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import random
import os

# ----------------------------------------------------
# Load and chunk the text
# ----------------------------------------------------
with open("mahabharata.txt", "r", encoding="utf-8") as f:
    full_text = f.read()


def chunk_text(text, max_words=300):
    """Split text into word chunks (default: 300 words each)."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


chunks = chunk_text(full_text)

# ----------------------------------------------------
# Embeddings + FAISS Index
# ----------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ----------------------------------------------------
# Load LLM
# ----------------------------------------------------
llm = Llama(model_path="mistral.gguf", n_ctx=8192)

# ----------------------------------------------------
# Flask App
# ----------------------------------------------------
app = Flask(__name__)
CORS(app)


# Helper: Humanize answers a little
def humanize(text):
    fillers = [
        "Here’s what it’s saying:",
        "So basically,",
        "In other words,",
        "The way it puts it is this:",
        "To put it simply,"
    ]
    text = text.strip()
    if len(text.split()) < 40 and not text.lower().startswith(("yes", "no")):
        return f"{random.choice(fillers)} {text}"
    return text


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Could you try asking a full question?"})

    # 1) Embed the question and search for relevant chunks
    question_embedding = embedder.encode(question)
    D, I = index.search(np.array([question_embedding]), k=3)
    retrieved_chunks = [chunks[i] for i in I[0]]

    filtered_chunks = list(dict.fromkeys([c.strip() for c in retrieved_chunks if c.strip()]))

    context = "\n\n".join(filtered_chunks)

    # 2) Build prompt
    prompt = f"""
    <|system|>
    You are an insightful, down-to-earth teacher.
    Answer naturally, like you’re explaining to a friend: warm, conversational, and easy to follow.
    Use only the context provided. If the answer isn’t there, just say you don’t know.

    <|user|>
    Context:
    \"\"\"{context}\"\"\"

    Question: {question}

    <|assistant|>
    """

    # 3) Get model response
    response = llm(prompt=prompt, max_tokens=512, stop=["<|user|>"])
    answer = response["choices"][0]["text"].strip()

    # 4) Humanize style
    answer = humanize(answer)

    return jsonify({"answer": answer})


# ----------------------------------------------------
# Run Server
# ----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
