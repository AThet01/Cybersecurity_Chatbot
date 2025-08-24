# 🔐 Cybersecurity Chatbot

An interactive AI-powered chatbot built with **Streamlit** and **Google Gemini** (Generative AI), designed to answer questions about cybersecurity topics such as phishing, spam, and data leaks using a custom knowledge base including chat history.

---

## ✨ Features

- 💬 Conversational UI with styled chat bubbles (user + bot)
- ⚡ Powered by Google Gemini (`gemini-1.5-flash`) via `langchain`
- 📚 Searches a knowledge base (`facts.txt`) using vector embeddings
- 🔎 Retrieval-based QA (RAG) using `FAISS` for document search
- 🚀 Easy to run locally using Streamlit

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – Web app framework
- [LangChain](https://www.langchain.com/) – LLM orchestration
- [Google Generative AI](https://ai.google.dev/) – LLM and Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) – Vector database

---

## 🧠 How It Works

1. Loads a `facts.txt` document using LangChain.
2. Splits text into manageable chunks and creates embeddings with `embedding-001`.
3. Stores and retrieves similar text chunks using FAISS.
4. Uses `gemini-1.5-flash` to answer questions with reference to the knowledge base.
5. Presents the conversation with styled left/right chat bubbles.

---

## ▶️ Getting Started

### Clone the Repo

```bash
git clone https://github.com/AThet01/Cybersecurity_Chatbot
cd cybersecurity-chatbot
```

### Installation

```bash
pip install -r requirements.txt
```





