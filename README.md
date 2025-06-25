# chat-with-pdf-genai
# 📚 Chat with PDF using Generative AI

This is a Streamlit app that lets users upload PDF files and ask questions based on their content using a lightweight Retrieval-Augmented Generation (RAG) approach.

## 🔧 Technologies Used
- Streamlit
- PyMuPDF (PDF text extraction)
- Sentence-Transformers (for semantic embeddings)
- FAISS (vector search)
- Hugging Face Transformers (Flan-T5 tokenizer)
- Grok Mini API (via GitHub Inference)

## 🚀 Features
- Upload PDFs and ask context-aware questions
- Smart chunking for long documents
- Fast semantic search with FAISS
- Answer generation using Grok Mini
- Export answers as TXT or DOCX

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run app.py
