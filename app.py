
import os
import re
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load css
def load_css(file_name):
    with open(file_name, encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")



# Configuration
load_dotenv()

st.set_page_config(
    page_title="Cognify",
    page_icon="🤖",
    layout="wide"
)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3.8-flash"

if not API_KEY:
    st.error("GEMINI_API_KEY is missing. Check your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)


# Extract text from PDF
def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "page": number,
                "text": text
            })

    return pages


# Split document text into smaller chunks
def make_chunks(pages, chunk_size=1200, overlap=200):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            chunk_text = text[start:start + chunk_size]

            if chunk_text.strip():
                chunks.append({
                    "page": page["page"],
                    "text": chunk_text
                })

            start += chunk_size - overlap

    return chunks


# Retrieve relevant document passages
def retrieve_chunks(chunks, question, top_k=5):
    if not chunks:
        return []

    documents = [item["text"] for item in chunks]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        matrix = vectorizer.fit_transform(documents + [question])
        scores = cosine_similarity(
            matrix[-1], matrix[:-1]
        ).flatten()

        ranked = scores.argsort()[::-1]
        selected = [
            chunks[i] for i in ranked[:top_k]
            if scores[i] > 0
        ]
        return selected

    except ValueError:
        return []


# Ask Gemini
def ask_gemini(question, context=""):
    instructions = """
You are Cognify, an intelligent AI knowledge assistant.

Give accurate, clear, useful answers.
For document questions, use the supplied document context.
Do not invent facts or pretend information is in the document
when it is not. If the context is insufficient, say so.
Cite document page numbers when available.
Use headings or bullet points when helpful.
"""

    prompt = f"""
{instructions}

DOCUMENT CONTEXT:
{context if context else "No PDF context provided."}

USER QUESTION:
{question}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text or "I couldn't generate a response."


# Summarize a document

def summarize_pdf(pages):
    if not pages:
        return "No readable text found in this PDF."

    document_text = "\n\n".join(
        f"[Page {page['page']}]\n{page['text']}"
        for page in pages
    )

    # Limit the input to keep the request manageable
    document_text = document_text[:40000]

    return ask_gemini(
        "Summarize the following document clearly. Include the main "
        "topic, key points, important findings, and conclusion. "
        "Use headings and bullet points. Mention page numbers where "
        "possible. If the supplied text is incomplete, say so.\n\n"
        + document_text
    )



# Interface
st.title("Cognify")
st.markdown("### Your personal AI knowledge assistant")
st.caption(
    "Chat with Gemini, explore documents, and turn long PDFs "
    "into useful insights."
)

with st.sidebar:
    st.header("📄 Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Gemini • PDF Q&A • RAG")


# Process PDF when uploaded
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = []

if uploaded_file:
    if st.session_state.pdf_name != uploaded_file.name:
        try:
            with st.spinner("Reading your PDF..."):
                pages = extract_pdf(uploaded_file)
                st.session_state.pdf_pages = pages
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []

            if pages:
                st.sidebar.success(
                    f"Loaded {len(pages)} pages."
                )
            else:
                st.sidebar.warning(
                    "No readable text found. Scanned PDFs may need OCR."
                )

        except Exception as error:
            st.error(f"Could not read this PDF: {error}")

if "messages" not in st.session_state:
    st.session_state.messages = []


# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input(
    "Ask anything, or ask a question about your PDF..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Cognify is thinking..."):
                pages = st.session_state.pdf_pages
                lower_question = question.lower()

                summary_patterns = [
                    "summarize",
                    "summarise",
                    "summary",
                    "give me an overview",
                    "main points",
                    "key points",
                    "document overview"
                ]

                if pages and any(
                    phrase in lower_question
                    for phrase in summary_patterns
                ):
                    answer = summarize_pdf(pages)

                elif pages:
                    chunks = make_chunks(pages)
                    relevant = retrieve_chunks(chunks, question)

                    context = "\n\n".join(
                        f"[Page {item['page']}]\n{item['text']}"
                        for item in relevant
                    )

                    answer = ask_gemini(question, context)

                    if relevant:
                        cited_pages = sorted({
                            item["page"] for item in relevant
                        })
                        answer += (
                            "\n\n**Relevant PDF pages:** "
                            + ", ".join(map(str, cited_pages))
                        )
                    else:
                        
                        if pages:
                            answer = ("I couldn't find relevant information in the uploaded "
                                "PDF to answer this question reliably. Try rephrasing "
                                "your question or asking about a specific topic in the document."
                            )
                        else:
                            answer = ask_gemini(question)

                else:
                    answer = ask_gemini(question)

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        
        except Exception as error:
            error_text = str(error).lower()

            if any(word in error_text for word in [
                "429", "quota", "resource_exhausted", "rate limit"
            ]):
                st.error(
                    "Gemini's API quota or rate limit may have been reached. "
                    "Please check your API usage and try again later."
                )
            elif any(word in error_text for word in [
                "401", "403", "api key", "permission denied"
            ]):
                st.error(
                    "Gemini couldn't authorize the request. Check your API "
                    "key and model access in your environment settings."
                )
            elif any(word in error_text for word in [
                "timeout", "connection", "network"
            ]):
                st.error(
                    "The connection to Gemini failed. Check your internet "
                    "connection and try again."
                )
            else:
                st.error(
                    "Cognify couldn't complete this request. Please try "
                    "again. If the issue continues, inspect the technical details."
                )

            with st.expander("Technical details"):
                st.code(str(error))

