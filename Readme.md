# Cognify — AI Chatbot Assistant

Cognify is an AI-powered chatbot assistant built with Python, Streamlit, and the Google Gemini API. It supports general-purpose conversations and allows users to upload PDF documents, generate summaries, and ask questions about their document content.

The project demonstrates practical applications of Large Language Models (LLMs), prompt engineering, document processing, and Retrieval-Augmented Generation (RAG).

## Features

- **AI Chat:** Interact with Google's Gemini model to get responses to general questions.
- **PDF Upload:** Upload PDF documents and extract their text for analysis.
- **Document Summarization:** Generate structured summaries highlighting the main topic, key points, findings, and conclusions.
- **PDF Question Answering:** Ask questions about uploaded documents and receive answers based on relevant text passages.
- **Basic RAG Pipeline:** Retrieve relevant document chunks using TF-IDF vectorization and cosine similarity before generating answers.
- **Page References:** Include relevant PDF page numbers to help users locate supporting information.
- **Interactive UI:** Use a Streamlit interface with chat history, a sidebar, and custom CSS styling.
- **API Error Handling:** Display helpful messages for common API and request errors.

## Tech Stack

- **Programming Language:** Python
- **LLM:** Google Gemini API
- **Frontend:** Streamlit, HTML/CSS
- **PDF Processing:** pypdf
- **Text Retrieval:** scikit-learn, TF-IDF, cosine similarity
- **Environment Variables:** python-dotenv

## How It Works

### General AI Chat
1. The user enters a question in the chat interface.
2. Cognify sends the prompt to the Gemini API.
3. The generated response is displayed in the chat.

### PDF Summarization
1. The user uploads a PDF document.
2. Cognify extracts the text from its pages using pypdf.
3. The extracted content is prepared for the summarization prompt.
4. Gemini generates a structured summary.

### PDF Question Answering (Basic RAG)
1. The uploaded PDF is divided into overlapping text chunks.
2. TF-IDF converts the chunks and user question into numerical representations.
3. Cosine similarity identifies the most relevant chunks.
4. The retrieved text is passed to Gemini as context.
5. Cognify generates an answer based on the retrieved content and displays relevant page references.

## Project Structure

```text
Genai-Chatbot/
├── app.py
├── style.css
├── .env
├── .gitignore
└── README.md
```

The `.env` file stores the Gemini API key locally and should never be uploaded to GitHub.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Fowziya-Husen/Cognify-AI-Chatbot.git
cd Cognify-AI-Chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install streamlit google-genai python-dotenv pypdf scikit-learn
```

### 4. Configure the API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key from [Google AI Studio](https://aistudio.google.com/).


### 5. Run the Application

```bash
python -m streamlit run app.py
```

Cognify will open in your browser.

## Limitations

- PDF summarization processes up to the first 40,000 characters of extracted document text in the current implementation.
- Scanned PDFs may not work correctly because OCR has not been implemented.
- TF-IDF performs lexical retrieval and may miss relevant passages when wording differs. The current implementation does not use semantic embeddings or a vector database.
- Response availability depends on Gemini API access, model availability, and usage quotas.
- Generated answers should be verified against the original document, particularly for important decisions.

## Future Improvements

- Add semantic embeddings for improved document retrieval.
- Integrate a vector database for scalable document search.
- Add OCR support for scanned PDFs.
- Implement automated evaluation of retrieval relevance and answer quality.
- Improve source attribution with more precise page-level citations.
- Deploy the application so users can access it online.

## Learning Outcomes

This project explores:

- Integrating an LLM through an API
- Designing prompts for different tasks
- Extracting and processing PDF documents
- Implementing a basic RAG workflow
- Retrieving relevant text using TF-IDF and cosine similarity
- Building an interactive application with Streamlit
- Managing API credentials using environment variables

## Author

**Fowziya H**

