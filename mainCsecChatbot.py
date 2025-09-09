import os
import json
import uuid
from datetime import datetime

import streamlit as st
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader  # ← community import to avoid deprecation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage  # ← for proper chat_history

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="🔐 Cybersecurity Chatbot", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def get_gemini_key() -> str:
    # 1) Prefer environment (safe even if secrets.toml doesn't exist)
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    # 2) Try Streamlit secrets safely
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""

def safe_rerun():
    # Streamlit >= 1.27: st.rerun; older: st.experimental_rerun
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            pass

def _now_iso():
    return datetime.now().isoformat(timespec="seconds")

# Convert persisted history (list of [user, bot]) -> LangChain messages
def to_langchain_messages(history):
    msgs = []
    for pair in history:
        if isinstance(pair, (tuple, list)) and len(pair) == 2:
            human, ai = pair
            msgs.append(HumanMessage(content=str(human)))
            msgs.append(AIMessage(content=str(ai)))
    return msgs

# -----------------------------
# API key (secrets or env)
# -----------------------------
GEMINI_KEY = get_gemini_key()
if not GEMINI_KEY:
    st.error(
        "❌ No Gemini API key found. Set GEMINI_API_KEY in your environment, "
        "or create .streamlit/secrets.toml with GEMINI_API_KEY."
    )
    st.stop()

genai.configure(api_key=GEMINI_KEY)

# -----------------------------
# Simple local persistence
# -----------------------------
CONV_DIR = "conversations"
INDEX_PATH = os.path.join(CONV_DIR, "_index.json")
os.makedirs(CONV_DIR, exist_ok=True)

def load_index():
    if not os.path.exists(INDEX_PATH):
        return []
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_index(index):
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def new_conversation(title="New chat"):
    cid = str(uuid.uuid4())[:8]
    convo = {
        "id": cid,
        "title": title,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    index = load_index()
    index.insert(0, convo)  # newest first
    save_index(index)
    save_conversation(cid, [], [])  # messages, chat_history
    return cid

def conv_path(cid):
    return os.path.join(CONV_DIR, f"{cid}.json")

def save_conversation(cid, messages, chat_history):
    payload = {
        "id": cid,
        "messages": messages,
        "chat_history": chat_history,
        "updated_at": _now_iso(),
    }
    with open(conv_path(cid), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # update index timestamp
    index = load_index()
    for item in index:
        if item["id"] == cid:
            item["updated_at"] = payload["updated_at"]
            break
    save_index(index)

def load_conversation(cid):
    path = conv_path(cid)
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("messages", []), data.get("chat_history", [])

def rename_conversation(cid, new_title):
    index = load_index()
    for item in index:
        if item["id"] == cid:
            item["title"] = new_title
            item["updated_at"] = _now_iso()
            break
    save_index(index)

def delete_conversation(cid):
    try:
        os.remove(conv_path(cid))
    except FileNotFoundError:
        pass
    index = [x for x in load_index() if x["id"] != cid]
    save_index(index)

# -----------------------------
# Build retriever (cached)
# -----------------------------
@st.cache_resource(show_spinner="Indexing knowledge base…")
def load_retriever():
    # Fail early if facts.txt is missing
    if not os.path.exists("facts.txt"):
        st.error("facts.txt not found in working directory.")
        st.stop()

    loader = TextLoader("facts.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_KEY,
    )
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever(search_kwargs={"k": 4})

retriever = load_retriever()

# -----------------------------
# LLM + Conversational QA chain
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_KEY,
    system_instruction=(
        "You are a helpful cybersecurity assistant. Use the retrieved knowledge base first. "
        "If asked about learning cybersecurity, recommend resources from the KB. "
        "If the KB lacks info, still provide a concise, practical answer with safe best practices."
    ),
    temperature=0.3,
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
)

# -----------------------------
# Session state bootstrap
# -----------------------------
if "current_cid" not in st.session_state:
    idx = load_index()
    if idx:
        st.session_state.current_cid = idx[0]["id"]
    else:
        st.session_state.current_cid = new_conversation("First chat")

if "messages" not in st.session_state or "chat_history" not in st.session_state:
    msgs, hist = load_conversation(st.session_state.current_cid)
    st.session_state.messages = msgs
    st.session_state.chat_history = hist

# -----------------------------
# UI helpers
# -----------------------------
def display_message(message, is_user):
    if is_user:
        st.markdown(
            f"<div style='background:#E9E9E9;padding:10px 15px;border-radius:10px;text-align:right;max-width:70%;margin-left:auto;font-size:16px;color:#000'>{message}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:#6B46C1;padding:10px 15px;border-radius:10px;text-align:left;max-width:70%;font-size:16px;color:#FFF'>{message}</div>",
            unsafe_allow_html=True,
        )

# -----------------------------
# Sidebar: Chat history manager
# -----------------------------
with st.sidebar:
    st.subheader("💬 Conversations")

    index_list = load_index()
    ids = [item["id"] for item in index_list]

    # Select conversation
    if ids:
        selected = st.selectbox(
            "Select a chat",
            options=ids,
            format_func=lambda cid: next(i['title'] for i in index_list if i['id'] == cid),
            index=0 if st.session_state.current_cid not in ids else ids.index(st.session_state.current_cid),
        )
        if selected != st.session_state.current_cid:
            st.session_state.current_cid = selected
            st.session_state.messages, st.session_state.chat_history = load_conversation(selected)
            safe_rerun()

    # New conversation
    if st.button("➕ New chat"):
        new_id = new_conversation("New chat")
        st.session_state.current_cid = new_id
        st.session_state.messages, st.session_state.chat_history = [], []
        safe_rerun()

    # Rename
    with st.expander("✏️ Rename chat"):
        current_title = next((i["title"] for i in index_list if i["id"] == st.session_state.current_cid), "")
        new_title = st.text_input("New title", value=current_title)
        if st.button("Save title"):
            rename_conversation(st.session_state.current_cid, new_title.strip() or "Untitled")
            safe_rerun()

    # Delete
    with st.expander("🗑️ Delete chat"):
        st.warning("Deleting is permanent.")
        if st.button("Delete this chat"):
            delete_conversation(st.session_state.current_cid)
            idx2 = load_index()
            if idx2:
                st.session_state.current_cid = idx2[0]["id"]
                st.session_state.messages, st.session_state.chat_history = load_conversation(st.session_state.current_cid)
            else:
                st.session_state.current_cid = new_conversation("First chat")
                st.session_state.messages, st.session_state.chat_history = [], []
            safe_rerun()

    # Download chat
    with st.expander("⬇️ Export chat"):
        msgs, hist = load_conversation(st.session_state.current_cid)
        export_payload = {
            "id": st.session_state.current_cid,
            "title": next((i["title"] for i in index_list if i["id"] == st.session_state.current_cid), "Untitled"),
            "messages": msgs,
            "chat_history": hist,
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2),
            file_name=f"chat_{st.session_state.current_cid}.json",
            mime="application/json",
        )

    # Clear current chat only (keeps others)
    if st.button("🧹 Clear current chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        save_conversation(st.session_state.current_cid, [], [])
        safe_rerun()

# -----------------------------
# Main UI
# -----------------------------
st.title("🔐 Cybersecurity Chatbot")
st.caption("Grounded in your knowledge base with full conversation history.")

# Display chat
for msg in st.session_state.messages:
    display_message(msg["content"], is_user=(msg["role"] == "user"))

# Chat input
user_input = st.chat_input("Type your message here…")
if user_input:
    # Append user
    st.session_state.messages.append({"role": "user", "content": user_input})
    display_message(user_input, is_user=True)

    # Answer with memory
    with st.spinner("🔎 Searching knowledge base and reasoning with chat history…"):
        result = qa_chain({
            "question": user_input,
            # ↓↓↓ FIX: convert persisted pairs -> LangChain message objects
            "chat_history": to_langchain_messages(st.session_state.chat_history),
        })
        bot_response = result["answer"]

    # Record exchange (persist as simple pairs; converter handles tuple/list)
    st.session_state.chat_history.append((user_input, bot_response))
    st.session_state.messages.append({"role": "bot", "content": bot_response})

    # Persist conversation
    save_conversation(st.session_state.current_cid, st.session_state.messages, st.session_state.chat_history)

    # Show response
    display_message(bot_response, is_user=False)
