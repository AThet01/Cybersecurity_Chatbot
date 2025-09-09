# 🔐 Cybersecurity Chatbot

An interactive AI-powered chatbot built with **Streamlit** and **Google Gemini** (Generative AI), designed to answer questions about cybersecurity topics such as phishing, spam, and data leaks using a custom knowledge base including chat history.Updated Code is "mainCsecChatbot.py".

---

# ✨ Features

Cybersecurity Q&A: Ask any cybersecurity-related question and get a grounded answer using your knowledge base (facts.txt) + Gemini LLM.

Knowledge Base Retrieval: Uses FAISS vector search + Gemini embeddings to fetch the most relevant chunks from facts.txt.

Conversational Memory: Remembers previous questions and answers for contextual follow-ups.

Conversation Manager (in the sidebar):

➕ New Chat: Start a fresh conversation anytime.

✏️ Rename Chat: Update the title of a chat for easier organization.

🗑️ Delete Chat: Permanently remove unwanted conversations.

🧹 Clear Chat: Reset the current chat history while keeping the conversation entry.

⬇️ Export Chat: Download the full conversation (messages + history) as a JSON file.

Persistent Storage: Conversations are saved locally under conversations/, with timestamps and an index for quick access.

Session Restore: The app automatically reloads your last active conversation when reopened.

UI Enhancements:

Custom styled chat bubbles for user vs. bot.

Sidebar selector to switch between multiple saved chats.

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
### To run the updated code (mainCsecChatbot.py)
```bash
$env:GEMINI_API_KEY="your_api_key_here"
streamlit run mainCsecChatbot.py

```




