import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS   # ✅ updated import
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Set page config FIRST
st.set_page_config(page_title="Cybersecurity Chatbot", layout="wide")

# Configure API key
GEMINI_KEY = "YOUR_GEMINI_APIKEYS"
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
    st.session_state.messages = []


def display_message(message, is_user):
    if is_user:
        # User message (light grey, right-aligned)
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


def main():
    st.title("🔐 Cybersecurity Chatbot")
    st.write("Ask me anything about phishing, spam, or data leaks – using real facts.")

    # Display chat history
    for msg in st.session_state.messages:
        display_message(msg["content"], is_user=(msg["role"] == "user"))

    # Use chat_input (like Telegram enter-to-send)
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message(user_input, is_user=True)

        # Generate bot response
        with st.spinner("🔎 Searching knowledge base and answering..."):
            result = qa_chain({"query": user_input})
            bot_response = result['result']

        # Add bot message
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        display_message(bot_response, is_user=False)


if __name__ == "__main__":
    main()
