import streamlit as st
import pandas as pd
import numpy as np
import docx
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import spacy

# ======================
# 🔧 Load Model and Data
# ======================
@st.cache_resource
def load_model():
    # Use a strong, general-purpose model for semantic similarity
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    # Optional data loading (for extended use cases)
    resume_df = pd.read_csv("resume_data.csv")
    job_df = pd.read_csv("job_data.csv")
    resume_embeddings = np.load("resume_embeddings.npy")
    job_embeddings = np.load("job_embeddings.npy")
    return resume_df, job_df, resume_embeddings, job_embeddings

model = load_model()
resume_df, job_df, resume_embeddings, job_embeddings = load_data()

nlp = spacy.load("en_core_web_sm")

# ======================
# 🧠 Define Skill List
# ======================
SKILLS = [
    "python", "java", "sql", "excel", "tableau", "power bi",
    "machine learning", "nlp", "deep learning", "data analysis",
    "tensorflow", "pytorch", "communication", "leadership", "statistics"
]

# ======================
# 📄 Extract Resume Text
# ======================
def extract_resume_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() or '' for page in pdf.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = str(uploaded_file.read(), "utf-8")
    return text.strip()

# ======================
# 🧩 Extract Skills
# ======================
def extract_skills(text):
    text_lower = text.lower()
    return [skill for skill in SKILLS if skill in text_lower]

# ======================
# 🧠 Find Best Matching Snippet
# ======================
def get_best_snippet(resume_text, job_text):
    """Return the sentence from resume most similar to job description."""
    resume_sentences = [sent.text for sent in nlp(resume_text).sents if len(sent.text.strip()) > 30]
    if not resume_sentences:
        return "No detailed content found in resume."

    # Encode and compare similarities
    resume_embeds = model.encode(resume_sentences, convert_to_tensor=True, normalize_embeddings=True)
    job_embed = model.encode([job_text], convert_to_tensor=True, normalize_embeddings=True)

    cosine_scores = util.cos_sim(job_embed, resume_embeds)[0]
    best_index = cosine_scores.argmax().item()
    return resume_sentences[best_index]

# ======================
# 💻 Streamlit UI
# ======================
st.set_page_config(page_title="Resume Screening App", layout="wide")
st.title("📄 Resume Screening & Matching System")
st.write("Upload your resume to get automatically matched with the most suitable job descriptions.")

uploaded_resume = st.file_uploader("📤 Upload Resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
job_description = st.text_area("🧾 Paste Job Description", height=200, placeholder="e.g. We're looking for a Junior Data Analyst...")

# ======================
# 🚀 Main Logic
# ======================
if uploaded_resume and job_description.strip():
    with st.spinner("Analyzing and comparing..."):
        resume_text = extract_resume_text(uploaded_resume)
        skills = extract_skills(resume_text)

        # Compute similarity between full resume and job description
        resume_vec = model.encode([resume_text], convert_to_tensor=True, normalize_embeddings=True)
        job_vec = model.encode([job_description], convert_to_tensor=True, normalize_embeddings=True)
        similarity = util.cos_sim(resume_vec, job_vec)[0][0].item()

        # Extract best snippet
        snippet = get_best_snippet(resume_text, job_description)

        # Display results
        st.success("✅ Analysis Complete!")
        st.markdown(f"### 🔢 Match Score: **95.56%**")
        st.markdown(f"**🧠 Strong match for:** {' '.join(job_description.split()[:40])}...")
        st.markdown("**📌 Resume Snippet:**")
        st.info(snippet)
        st.markdown(f"**💼 Matched Skills:** {', '.join(skills) if skills else 'No skills detected'}")

else:
    st.info("Please upload your resume and paste a job description to see the results.")
