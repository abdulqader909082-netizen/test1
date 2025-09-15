import os
import time
import json
import faiss
import fitz  
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
set_background("default.jpg")    


INDEX_PATH = "relativity_index.faiss"
META_PATH = "relativity_meta.json"
CHUNKS_PATH = "relativity_chunks.json"  
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/mixtral-8x7b-instruct"  

@st.cache_resource
def load_index_and_models():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedder = SentenceTransformer(EMBED_MODEL)

    # API Key من Streamlit Secrets
    
    api = st.secrets["OPENROUTER_API_KEY"]
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api
    )

    return index, meta, chunks, embedder, client


# Initialize once
index, chunk_meta, chunks, embedder, client = load_index_and_models()


def search(query, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({
            "text": chunks[idx],
            "meta": chunk_meta[idx],
            "score": float(score)
        })
    return results

def build_prompt(query, retrieved):
    context = ""
    for r in retrieved:
        src = os.path.basename(r["meta"].get("source", "unknown"))
        page = r["meta"].get("page", "?")
        context += f"[Source: {src}, page {page}]\n{r['text']}\n\n---\n\n"

    prompt = f"""
You are a helpful assistant. Answer the following question using ONLY the provided Context.
If the answer cannot be found, say "NOT IN DOCUMENTS".

Context:
{context}

Question: {query}
Answer:
"""
    return prompt

def ask_question(query):
    start = time.time()
    retrieved = search(query, top_k=3)
    prompt = build_prompt(query, retrieved)

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = completion.choices[0].message.content
    latency = time.time() - start

    return {
        "query": query,
        "answer": answer,
        "retrieved": retrieved,
        "latency_sec": round(latency, 3)
    }

st.title("📄 Special Relativity Q&A (RAG)")
st.markdown("Ask questions about Special Relativity based on the provided documents.")
st.write("API Key Loaded:", "OPENROUTER_API_KEY" in st.secrets)


query = st.text_input("Write your question here:", "")

if st.button("Get Answer") and query.strip():
    with st.spinner("⏳ Finiding answer..."):
        result = ask_question(query)

    st.subheader("💡 Answer")
    st.write(result["answer"])

    st.subheader("📑 Retrieved Context")
    for r in result["retrieved"]:
        st.markdown(f"**Page {r['meta']['page']}**: {r['text'][:200]}...")

    st.caption(f" Latency: {result['latency_sec']}s")


