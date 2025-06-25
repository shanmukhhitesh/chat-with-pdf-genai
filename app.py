import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from io import BytesIO
from docx import Document

endpoint = "https://models.github.ai/inference"
model = "xai/grok-3-mini"
token = "<YOUR_TOKEN>"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(r"F:\models\flan-t5-base")

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def smart_chunk_text(text, max_tokens=256):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(tokenizer.encode(current_chunk + sentence)) < max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings


def get_top_chunks(question, chunks, index, k=5):
    q_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_embedding, k)
    return [chunks[i] for i in indices[0]]


def ask_with_grok(context, question):
    prompt = f"""Based on the following material, answer the question.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    response = client.complete(
        messages=[
            SystemMessage("You are a helpful assistant."),
            UserMessage(prompt),
        ],
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        model=model,
    )
    return response.choices[0].message.content

def export_answer(answer, export_format):
    if export_format == "TXT":
        return BytesIO(answer.encode("utf-8"))
    elif export_format == "DOCX":
        doc = Document()
        doc.add_paragraph(answer)
        byte_io = BytesIO()
        doc.save(byte_io)
        byte_io.seek(0)
        return byte_io

st.title("📚 Chat with PDF...")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question based on the PDF:")

if uploaded_file and question:
    with st.spinner("Processing..."):
        try:
            text = extract_text_from_pdf(uploaded_file)
            chunks = smart_chunk_text(text)
            index, _ = create_faiss_index(chunks)
            top_chunks = get_top_chunks(question, chunks, index)
            context = "\n".join(top_chunks)
            answer = ask_with_grok(context, question)
            st.success("Answer:")
            st.write(answer)
            export_format = st.selectbox("Choose export format:", ["TXT", "DOCX"])
            if st.button("Download Answer"):
                file_data = export_answer(answer, export_format)
                st.download_button("Download", file_data, file_name=f"answer.{export_format.lower()}")
        except Exception as e:             
            st.error(f"Error: {str(e)}")
