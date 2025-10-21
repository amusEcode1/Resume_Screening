## 📄 Resume Screening System
This project is a Natural Language Processing (NLP) application that automatically screens and ranks resumes based on job descriptions using **semantic similarity**.
It leverages **Sentence Transformers** to generate text embeddings for both resumes and job descriptions, then computes **cosine similarity** to determine the best match.

---

## 🧩 Key Steps
- **Dataset**: Combined Resume Dataset (2.4K samples) + Job Description Dataset (1.6M+ samples).
- **Preprocessing:**
  - Extracted text from .pdf, .docx, and .txt resumes.
  - Preprocess resumes and job descriptions using semantic embeddings (all-MiniLM-L6-v2) from SentenceTransformers
- **Similarity Measurement:**
  - Used cosine similarity to compute how closely a resume matches a job description.
- **Skill Extraction:**
  - Automatically identified relevant skills (e.g., Python, NLP, Machine Learning, SQL).
- **Deployment:**
  - Built an interactive Streamlit app to upload resumes and view similarity scores, matched skills, and the most relevant resume snippet.
    
---

## 📂 Dataset
You can find suitable datasets for this task here:
- [Kaggle - Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- [Kaggle - Job Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset)

---

## 📊 Sample Output
| Metric               | Description                                                     | Example                                                        |
| :------------------- | :-------------------------------------------------------------- | :------------------------------------------------------------- |
| **Match Score**      | Percentage similarity between resume and job description        | 87.42%                                                         |
| **Strong Match For** | Key phrases from the job description that best match the resume | “Looking for a Data Analyst skilled in Python, NLP, and ML...” |
| **Matched Skills**   | Extracted from the resume                                       | Python, NLP, Machine Learning, SQL                             |


---

## 🧠 Tech Stack & Tools
- **Python Libraries:**
  `SentenceTransformers`, `Pandas`, `NumPy`, `BeautifulSoup`, `spaCy`, `pdfplumber`, `python-docx`, `tqdm
- **Framework:**
  Streamlit for deployment  
- **Embeddings Used:**
  all-MiniLM-L6-v2 from SentenceTransformers

---

## 📦 Dependencies
Before running locally, ensure these are installed:

```sh
pip install sentence-transformers pandas numpy beautifulsoup4 spacy pdfplumber python-docx tqdm streamlit
```
Then download the spaCy model:
```sh
python -m spacy download en_core_web_sm
```

## Installing
To install Streamlit:
```sh
pip install streamlit
```
To install all required dependencies:
```sh
pip install -r requirements.txt
```

## Running the Application Locally
```sh
streamlit run app.py
```
Then open the local URL (usually http://localhost:8501/) in your browser.

## 📰 Try the App Online
You can use the app directly here: [Resume Qualify](https://resume-qualify.streamlit.app/)<br>
Simply upload your resume, paste a job description, and view your **Match Score**, **Matched Skills**, and **Relevant Snippet** instantly.

---

## 💡 Features
- Upload resume files in `.pdf`, `.docx`, or `.txt` format
- Paste any **job description** for matching
- Automatically extracts **skills** from resume text
- Calculates **Match Score** using cosine similarity
- Displays the **most relevant resume snippet** to justify the score
- Fully interactive Streamlit interface

---

## 📂 Folder Structure
```
Resume-Screening-NLP/
├── app.py
├── sentence_model/
|   ├──...          
├── resume_data.csv           
├── job_data.csv              
├── resume_embeddings.npy      
├── job_embeddings.npy         
├── requirements.txt           
└── README.md

```

## ❓ Help
If you encounter any issues:
- Check the [Streamlit Documentation](https://docs.streamlit.io/)
- Search for similar issues or solutions on [Kaggle](https://www.kaggle.com/)
- Open an issue in this repository

---

## ✍️ Author
👤 Oluyale Ezekiel
- 📧 Email: ezekieloluyale@gmail.com
- LinkedIn: [Ezekiel Oluyale](https://www.linkedin.com/in/ezekiel-oluyale)
- GitHub Profile: [@amusEcode1](https://github.com/amusEcode1)
- Twitter: [@amusEcode1](https://x.com/amusEcode1?t=uHxhLzrA1TShRiSMrYZQiQ&s=09)

---

## 🙏 Acknowledgement
Thank you, Elevvo Pathways, for the amazing internship experience that helped me turn NLP concepts into real-world applications.
