import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Create summarization prompt
summarize_prompt = PromptTemplate(
    template="Summarize the following news article:\n\n{article}\n\nSummary:",
    input_variables=["article"]
)
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# Extract news text from URL
def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ''.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Failed to fetch news from {url}: {e}"

# Streamlit Page Config
st.set_page_config(
    page_title="AI Tech News Summarizer",
    layout="centered",
    page_icon="🧠",
)

# Custom CSS for attractive UI
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        background-attachment: fixed;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    h1, h2, h3, p, label {
        color: #00ffe7 !important;
    }
    input, .stTextInput>div>div>input {
        background-color: #111;
        color: #00ffe7;
        border: 1px solid #00ffe7;
        padding: 0.5rem;
    }
    .stButton > button {
        background-color: #ff6f61;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        border: none;
        transition: background 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #e14a3f;
        color: white;
    }
    .stSpinner {
        color: #00ffe7;
    }
    </style>
""", unsafe_allow_html=True)

# Main UI
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🧠 AI Tech News Summarizer")
st.write("Paste a news article URL and get an instant summary powered by **Gemini AI**.")

# Input
url = st.text_input("🔗 Enter News URL:")

# Action
if st.button("✨ Summarize"):
    if url:
        with st.spinner("🔍 Fetching and summarizing..."):
            article = extract_news(url)
            if article.startswith("Failed"):
                st.error(article)
            else:
                summary = summarize_chain.run(article=article)
                st.success("✅ Summary generated successfully!")
                st.markdown("### 📝 Summary:")
                st.write(summary)
    else:
        st.warning("⚠️ Please enter a valid URL.")
st.markdown("</div>", unsafe_allow_html=True)
