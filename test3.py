import os
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Set page config FIRST
st.set_page_config(page_title="Cybersecurity Chatbot", layout="wide")

GEMINI_KEY = "AIzaSyCARantSHqTfDQZSlXJ7BMxGu8b8BOy0_U"
genai.configure(api_key=GEMINI_KEY)

@st.cache_resource
def load_retriever():
    loader = TextLoader("facts.txt")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_KEY
    )
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever()

retriever = load_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_KEY,
        system_instruction=(
    "You are a helpful cybersecurity assistant. "
    "Answer questions using the provided knowledge base. "
    "If the question is about where to learn cybersecurity, "
    "recommend learning resources listed in the knowledge base. "
    "If you don't find info in the knowledge base, still provide a general helpful answer."
)

    ),
    retriever=retriever,
    return_source_documents=False
)

# Store conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts {"role": "user"|"bot", "content": text}
def display_message(message, is_user):
    if is_user:
        # User message (light grey, right-aligned)
        left_col, right_col = st.columns([1, 4])
        with left_col:
            st.write("")
        with right_col:
            st.markdown(
                f"""
                <div style='
                    background-color:#E9E9E9;
                    padding:10px 15px;
                    border-radius:10px;
                    text-align:right;
                    max-width:70%;
                    margin-left:auto;
                    font-size:16px;
                    color:#000000;'>
                    {message}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        # Bot message (purple, left-aligned)
        left_col, right_col = st.columns([4, 1])
        with left_col:
            st.markdown(
                f"""
                <div style='
                    background-color:#C153A3;
                    padding:10px 15px;
                    border-radius:10px;
                    text-align:left;
                    max-width:70%;
                    font-size:16px;
                    color:#FFFFFF;'>
                    {message}
                </div>
                """,
                unsafe_allow_html=True
            )
        with right_col:
            st.write("")


def main():
    st.title("🔐 Cybersecurity Chatbot")
    st.write("Ask me anything about phishing, spam, or data leaks – using real facts.")

    # Display chat history
    for msg in st.session_state.messages:
        display_message(msg["content"], is_user=(msg["role"] == "user"))

    # Use a form so you can clear input after submit
    with st.form(key="input_form", clear_on_submit=True):
        user_input = st.text_input("You:", key="input")
        submitted = st.form_submit_button("Send")

        if submitted and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            display_message(user_input, is_user=True)

            with st.spinner("🔎 Searching knowledge base and answering..."):
                result = qa_chain({"query": user_input})
                bot_response = result['result']
                st.session_state.messages.append({"role": "bot", "content": bot_response})
                display_message(bot_response, is_user=False)



if __name__ == "__main__":
    main()
