import streamlit as st
import pandas as pd
import numpy as np
import docx
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load Model and Data
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence_model")

@st.cache_data
def load_data():
    resume_df = pd.read_csv("resume_data.csv")
    job_df = pd.read_csv("job_data.csv")
    resume_embeddings = np.load("resume_embeddings.npy")
    job_embeddings = np.load("job_embeddings.npy")
    return resume_df, job_df, resume_embeddings, job_embeddings

model = load_model()
resume_df, job_df, resume_embeddings, job_embeddings = load_data()

nlp = spacy.load("en_core_web_sm")

SKILLS = ["python", "java", "sql", "excel", "tableau", "power bi",
          "machine learning", "nlp", "deep learning", "data analysis",
          "tensorflow", "pytorch", "communication", "leadership", "statistics"]

def extract_resume_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = str(uploaded_file.read(), 'utf-8')
    return text

def extract_skills(text):
    text_lower = text.lower()
    return [skill for skill in SKILLS if skill in text_lower]

def get_best_snippet(resume_text, job_text):
    """Return the sentence from resume most similar to job description."""
    resume_doc = [sent.text for sent in nlp(resume_text).sents if len(sent.text.strip()) > 30]
    if not resume_doc:
        return "No detailed content found in resume."
    embeddings_resume = model.encode(resume_doc)
    embedding_job = model.encode([job_text])
    sims = cosine_similarity(embedding_job, embeddings_resume)[0]
    best_index = np.argmax(sims)
    return resume_doc[best_index]

# Streamlit UI
st.set_page_config(page_title="Resume Screening App", layout="wide")
st.title("📄 Resume Screening & Matching System")
st.write("Upload your resume to get automatically matched with the most suitable job descriptions.")

uploaded_resume = st.file_uploader("📤 Upload Resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

job_description = st.text_area("🧾 Paste Job Description", height=200, placeholder="e.g. We're looking for a Junior Data Analyst...")

if uploaded_resume and job_description.strip():
    with st.spinner("Analyzing and comparing..."):
        resume_text = extract_resume_text(uploaded_resume)
        skills = extract_skills(resume_text)

        # Compute similarity
        resume_vec = model.encode([resume_text])
        job_vec = model.encode([job_description])
        similarity = cosine_similarity(resume_vec, job_vec)[0][0]

        # Get snippet
        snippet = get_best_snippet(resume_text, job_description)

        # Display Results
        st.success("✅ Analysis Complete!")
        st.markdown(f"### 🔢 Match Score: **{round(similarity * 100, 2)}%**")
        st.markdown(f"**🧠 st.write(f"🧠 Strong match for: {' '.join(job_description.split()[:20])}..."))
        st.markdown("**📌 Resume Snippet:**")
        st.info(snippet)
        st.markdown(f"**💼 Matched Skills:** {', '.join(skills) if skills else 'No skills detected'}")

else:
    st.info("Please upload your resume and paste a job description to see the results.")
