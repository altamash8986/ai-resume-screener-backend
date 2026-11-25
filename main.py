from sklearn.linear_model import LinearRegression
import tempfile
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
import numpy as np
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import io
import os.path
import re
import base64
from datetime import datetime
from typing import List

import pymupdf  
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from docx import Document
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from spacy.matcher import PhraseMatcher
import uvicorn
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "https://ai-resume-screener-frontend-six.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    ai_detector = pipeline("text-classification", model="roberta-base-openai-detector")
except Exception as e:
    print(f"Failed to load AI Plagiarism model: {e}")
    ai_detector = None

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Spacy model not found. Run: python -m spacy download en_core_web_sm")

try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Failed to load SBERT model: {e}")
    sbert_model = None

SKILL_ONTOLOGY = {
    "deep learning": ["pytorch", "tensorflow", "keras", "neural networks", "neural network", "transformers", "cnn",
                      "rnn"],
    "machine learning": ["scikit-learn", "sklearn", "supervised learning", "unsupervised learning", "xgboost",
                         "random forest"],
    "neural networks": ["pytorch", "tensorflow", "keras", "deep learning", "cnn", "rnn", "transformers"],
    "nlp": ["spacy", "hugging face", "nltk", "roberta", "bert", "llms", "nlp"],
}

GET_DEFAULT_ROLES = {
    "AI/ML Engineer": [
        "python", "numpy", "pandas", "scikit-learn", "matplotlib",
        "machine learning", "data analysis", "pytorch",
        "statistics", "git", "deep learning", "neural networks"
    ],
    "Software Engineer": [
        "python", "java", "c++", "sql", "data structures and algorithms", "git", "object oriented programming",
        "rest api", "problem solving",
    ],
    "Data Analyst": [
        "excel", "sql", "python", "pandas", "data visualization",
        "statistics", "power bi", "tableau", "data cleaning", "reporting"
    ],
    "DevOps Engineer": [
        "linux", "git", "docker", "ci/cd", "bash scripting",
        "shell scripting", "networking", "python", "github actions"
    ],
    "Frontend Developer": [
        "html", "css", "javascript", "react", "git",
        "responsive design", "bootstrap", "tailwind css", "typescript"
    ],
    "Backend Developer": [
        "python", "node.js", "sql", "rest api", "git",
        "flask", "fastapi", "mongodb", "postgresql", "authentication"
    ],
    "Full Stack Developer": [
        "html", "css", "javascript", "react", "node.js",
        "python", "sql", "rest api", "git", "mongodb", "docker"
    ],
    "Data Engineer": [
        "python", "sql", "etl", "pandas", "postgresql",
        "git", "data pipelines", "docker", "airflow"
    ],
    "Cloud Engineer": [
        "aws", "linux", "docker", "networking", "git",
        "virtual machines", "storage", "iam", "monitoring"
    ],
    "Cybersecurity Engineer": [
        "networking", "linux", "ethical hacking", "python",
        "firewalls", "cryptography", "owasp", "vulnerability assessment"
    ],
    "Mobile App Developer": [
        "java", "kotlin", "android", "flutter", "dart",
        "rest api", "firebase", "git", "ui/ux"
    ],
    "Web Developer": [
        "html", "css", "javascript", "react", "node.js",
        "rest api", "git", "responsive design", "typescript", "sql"
    ],
    "UI/UX Designer": [
        "figma", "adobe xd", "wireframing", "prototyping", "user research",
        "usability testing", "typography", "color theory", "html", "css"
    ],
    "IOS Developer": [
        "swift", "swiftui", "xcode", "objective-c", "rest api",
        "firebase", "core data", "git", "ui/ux", "app store deployment"
    ],
    "Game Developer": [
        "c++", "unity", "c#", "unreal engine", "game physics",
        "3d mathematics", "opengl", "python", "git", "ui/ux"
    ],
    "QA / Test Engineer": [
        "manual testing", "test cases", "bug tracking", "selenium",
        "pytest", "jira", "api testing", "postman", "git"
    ],
    "GenAI Engineer": [
        "python", "llm", "langchain", "openai api", "prompt engineering",
        "rag", "vector databases", "hugging face",
        "fastapi", "git", "embeddings"
    ],
    "Prompt Engineer": [
        "prompt design", "llm", "openai api", "python",
        "chatgpt", "claude", "few-shot prompting", "evaluation",
        "documentation", "langchain"
    ],
    "MLOps Engineer": [
        "python", "docker", "git", "mlflow", "fastapi",
        "ci/cd", "linux", "model deployment", "monitoring"
    ],
    "AI Product Analyst": [
        "sql", "python", "data analysis", "excel",
        "a/b testing", "data visualization", "metrics",
        "power bi", "documentation"
    ],
    "Business Analyst": [
        "sql", "excel", "data analysis", "power bi",
        "documentation", "jira", "reporting", "agile", "communication"
    ],
    "IT Support Engineer": [
        "networking", "linux", "windows", "troubleshooting",
        "hardware", "customer support", "documentation", "security"
    ]
}

CERTIFICATES = ["aws", "google", "microsoft", "amazon", "forage", "ibm", "meta", "coursera", "udemy", "freecodecamp",
                "infosys", "linkedin learning", "government", "oracle", "mckinsey", "stanford", "deeplearning.ai",
                "uc san diego"]
EXPECTED_SECTIONS = ["education", "experience", "projects", "skills", "certifications", "summary"]

HIRING_STRATEGIES = {
    "Balanced": {"skill": 30, "exp": 30, "format": 15, "jd": 15, "cert": 10},
    "Entry-Level": {"skill": 35, "exp": 5, "format": 25, "jd": 20, "cert": 15},
    "Medium-Level": {"skill": 40, "exp": 35, "jd": 10, "format": 10, "cert": 5},
    "Senior-Level": {"skill": 25, "exp": 45, "format": 5, "jd": 20, "cert": 5},
}


# get pdf file data
def extract_text_from_file(file_content: bytes, filename: str):
    text = ""
    try:
        if filename.lower().endswith(".pdf"):
            with pymupdf.open(stream=file_content, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
        elif filename.lower().endswith(".docx"):
            doc = Document(io.BytesIO(file_content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            return "Unsupported File Format"
    except Exception as e:
        return f"Error reading file: {e}"
    return text


# check is resume or not
def is_valid_resume(text: str) -> bool:
    if not text or not text.strip():
        return False
    text_lower = text.lower()
    resume_keywords = [
        "experience", "education", "skills", "summary", "objective",
        "employment", "project", "university", "college", "certification", "profile"
    ]
    match_count = sum(1 for word in resume_keywords if word in text_lower)
    return match_count >= 4


# ai authentication
def get_ai_plag(text: str) -> float:
    if not ai_detector or not text.strip():
        return 0.0
    text_snippet = text[250:2250]
    try:
        result = ai_detector(text_snippet, truncation=True, max_length=512)[0]
        label = str(result['label']).lower()
        score = result['score']

        if label in ['fake', 'chatgpt', 'label_1', '1']:
            final_score = float(round(score * 100, 1))
        else:
            final_score = float(round((1 - score) * 100, 1))
        return final_score
    except Exception as e:
        print(f"⚠️ AI Detection Error: {e}")
        return 0.0


# get the contact information from the resume
def extract_contact_info(text):
    email = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    phone = re.search(r"(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", text)
    linkedin = re.search(r"linkedin\.com/in/[\w-]+", text)
    return email.group(0) if email else "---", phone.group(0) if phone else "---", linkedin.group(
        0) if linkedin else "---"


# mathching the skills
def extract_skills_from_ner(text, skill_set):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp(skill) for skill in skill_set]
    matcher.add("SKILLS", patterns)
    doc = nlp(text)
    return list(set([doc[start:end].text.lower() for match_id, start, end in matcher(doc)]))


# for get cetification and count
def extract_certifications(text):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp(cert.lower()) for cert in CERTIFICATES]
    matcher.add("CERTS", patterns)
    doc = nlp(text)
    return list(set([doc[start:end].text.lower() for match_id, start, end in matcher(doc)]))

# experience count
def extract_experience_nlp(text):
    text_lower = text.lower()
    start_match = re.search(
        r"(?:^|\n)\s*(work\s+experience|experience|employment|professional\s+experience|internship|internships)\b",
        text_lower
    )
    if not start_match:
        return 0.0, 0.0
    start_idx = start_match.end()
    end_match = re.search(
        r"(?:^|\n)\s*(education|projects?|skills?|certifications?|achievements?|summary|technical\s+skills|awards|languages|additional\s+skills)\b",
        text_lower[start_idx:]
    )
    if end_match:
        end_idx = start_idx + end_match.start()
        exp_text_block = text[start_idx:end_idx]
    else:
        exp_text_block = text[start_idx:]

    exp_text_block = re.sub(r"[–—]", "-", exp_text_block)

    def parse_date(date_str):
        date_str = date_str.strip().lower()
        if date_str in ["present", "current", "now", "ongoing"]:
            now = datetime.now()
            return now.year, now.month
        formats = ["%b %Y", "%B %Y", "%b-%Y", "%B-%Y", "%m/%Y", "%m-%Y"]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.year, dt.month
            except:
                continue
        return None

    pattern = r"([A-Za-z]{3,9}\s*\d{4})\s*(?:-|to)\s*([A-Za-z]{3,9}\s*\d{4}|present|current|ongoing)"
    matches = re.findall(pattern, exp_text_block, re.IGNORECASE)
    if not matches:
        return 0.0, 0.0

    ranges = []
    for start, end in matches:
        start_parsed = parse_date(start)
        end_parsed = parse_date(end)
        if start_parsed and end_parsed:
            sy, sm = start_parsed
            ey, em = end_parsed
            if (ey, em) >= (sy, sm):
                ranges.append(((sy, sm), (ey, em)))

    if not ranges:
        return 0.0, 0.0

    ranges.sort()
    merged = [ranges[0]]
    for current_start, current_end in ranges[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    total_months = sum((ey - sy) * 12 + (em - sm) + 1 for (sy, sm), (ey, em) in merged)
    exp_years = round(total_months / 12, 2)
    exp_score = 100 if exp_years >= 5 else round((exp_years / 5) * 100, 2)
    return exp_years, exp_score


# smart matching of combined
def smart_match(skills_required, resume_text):
    master_skill_list = set([skill.lower() for skill in skills_required])
    for parent, children in SKILL_ONTOLOGY.items():
        master_skill_list.add(parent.lower())
        for child in children:
            master_skill_list.add(child.lower())

    found_in_resume_raw = extract_skills_from_ner(resume_text, list(master_skill_list))
    found_in_resume = set([skill.lower() for skill in found_in_resume_raw])

    matched = []
    missed = []

    for req_skill in skills_required:
        req_skill_lower = req_skill.lower()

        if req_skill_lower in found_in_resume:
            matched.append(req_skill_lower)
            continue

        is_ontology_match = False
        if req_skill_lower in SKILL_ONTOLOGY:
            for sub_skill in SKILL_ONTOLOGY[req_skill_lower]:
                if sub_skill.lower() in found_in_resume:
                    matched.append(req_skill_lower)
                    is_ontology_match = True
                    break

        if not is_ontology_match:
            missed.append(req_skill_lower)

    return list(set(matched)), list(set(missed))


def compute_similarity(jd_text, resume_text):
    if sbert_model is None:
        return 0.0

    jd_clean = re.sub(r"[^\w\s]", "", jd_text.lower())
    resume_clean = re.sub(r"[^\w\s]", "", resume_text.lower())

    if not resume_clean.strip() or not jd_clean.strip():
        return 0.0

    jd_emb = sbert_model.encode(jd_clean, convert_to_tensor=True)
    res_emb = sbert_model.encode(resume_clean, convert_to_tensor=True)

    cosine_score = util.cos_sim(jd_emb, res_emb).item()
    return round(max(cosine_score, 0) * 100, 2)


def analyze_formatting(resume_text):
    fmt_score = len([sec for sec in EXPECTED_SECTIONS if sec in resume_text.lower()]) / len(EXPECTED_SECTIONS) * 100
    return round(fmt_score, 2)


def check_authenticity(resume_text: str, matched_skills: list, jd_sim: float) -> str:
    if not is_valid_resume(resume_text):
        return "❌ Invalid Document"
    resume_lower = resume_text.lower()
    if (jd_sim > 95.0):
        return "⚠️ High Risk (JD Copy-Paste)"
    for skill in matched_skills:
        count = len(re.findall(rf'\b{re.escape(skill)}\b', resume_lower))
        if count > 15:
            return f"⚠️ High Risk (Stuffing: '{skill}')"
    ai_score = get_ai_plag(resume_text)
    if ai_score > 75.0:
        return f"🤖 AI Generated ({ai_score}%)"
    elif ai_score > 40.0:
        return f"⚠️ Mixed Content ({ai_score}% AI)"
    return "✅ Verified"

# this is for get the comparison charts
def generate_comparison_chart(name_a, name_b, scores_a, scores_b, metrics, chart_type):
    plt.figure(figsize=(12, 5))
    if chart_type == "Bar":
        x = range(len(metrics))
        plt.bar(x, scores_a, width=0.4, label=name_a)
        plt.bar([i + 0.4 for i in x], scores_b, width=0.4, label=name_b)
        plt.xticks([i + 0.2 for i in x], metrics, rotation=10)
        plt.ylim(0, 110)
        plt.legend()
    elif chart_type == "Line":
        plt.plot(metrics, scores_a, marker="o", label=name_a)
        plt.plot(metrics, scores_b, marker="o", label=name_b)
        plt.ylim(0, 110)
        plt.legend()
    plt.ylabel("Score (0-100)")
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


# for clustering
def generate_cluster_chart(report_data):
    if len(report_data) < 3:
        return ""
    names = [r["Resume"] for r in report_data]
    final_scores = [r["Final Score"] for r in report_data]
    skills = [r["skill_score"] for r in report_data]
    X = np.array(list(zip(skills, final_scores)))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    quality_scores = centers.sum(axis=1)
    sorted_idx = np.argsort(quality_scores)
    color_map = {
        sorted_idx[2]: '#10b981',
        sorted_idx[1]: '#f59e0b',
        sorted_idx[0]: '#ef4444'
    }
    plt.figure(figsize=(14, 6))
    for i in range(3):
        cluster_points = X[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    s=150, c=color_map[i],
                    label=f"Cluster {i + 1}", edgecolors='white', linewidth=2)
    for i, name in enumerate(names):
        short_name = name[:12] + "..." if len(name) > 12 else name
        plt.annotate(short_name, (skills[i], final_scores[i]),
                     xytext=(8, 8), textcoords='offset points', fontsize=9, color="white")
    plt.title("AI Candidate Clustering (K-Means)", fontsize=20, color="white", pad=15)
    plt.xlabel("Skill Match Score (%)", fontsize=12, color="white")
    plt.ylabel("Final Weighted Score (%)", fontsize=12, color="white")
    ax = plt.gca()
    ax.set_facecolor('#1e293b')
    plt.gcf().patch.set_facecolor('#0f172a')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Top Talent', markerfacecolor="#00f821", markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Average', markerfacecolor="#f39e0c", markersize=11),
        Line2D([0], [0], marker='o', color='w', label='Below Cutoff', markerfacecolor="#ff0000", markersize=10)
    ]
    plt.legend(handles=legend_elements, facecolor='#1e293b', labelcolor='white', edgecolor='#334155')
    plt.grid(True, linestyle='--', alpha=0.3, color="white")
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


np.random.seed(42)
final_scores = np.random.randint(40, 101, 1000)
experience = np.random.randint(0, 16, 1000)
salary = 25000 + final_scores * 400 + experience * 6000 + np.random.randint(-3000, 3000, 1000)
x_train = np.column_stack((final_scores, experience))
y_train = salary
model = LinearRegression()
model.fit(x_train, y_train)


def predict_salary(final_score, exp_years):
    prediction = model.predict([[final_score, exp_years]])
    return f"₹{int(prediction[0]):,} per month"



# API Endpoints are starting below
@app.get("/")
def welcome():
    return {"message": "Ai Resume Screener is Running ✅"}


@app.post("/analyze")
async def process_resumes(
        files: List[UploadFile] = File(...),
        hiring_strategy: str = Form("Balanced"),
        input_mode: str = Form("Use Job Role"),
        role: str = Form("AI/ML Engineer"),
        jd_textbox: str = Form(""),
        experience_level: str = Form("0"),
        cutoff_score: float = Form(0.0),
        report_scope: str = Form("All Resumes")
):
    try:
        experience_level = float(experience_level)
    except:
        experience_level = 0.0

    weights = HIRING_STRATEGIES.get(hiring_strategy, HIRING_STRATEGIES["Balanced"])

    if input_mode == "Use Job Role":
        skills_required = GET_DEFAULT_ROLES.get(role, [])
    else:
        all_skills = set(skill for skills in GET_DEFAULT_ROLES.values() for skill in skills)
        skills_required = extract_skills_from_ner(jd_textbox.strip().lower(), list(all_skills))

    if not skills_required:
        raise HTTPException(status_code=400, detail="No skills found/selected for screening.")

    report_data = []
    skipped_files = []

    for file in files:
        content = await file.read()
        resume_text = extract_text_from_file(content, file.filename)

        if "error reading file" in resume_text.lower() or "unsupported file" in resume_text.lower():
            skipped_files.append(file.filename)
            continue

        email, phone, linkedin = extract_contact_info(resume_text)
        processed_text = resume_text.lower()

        jd_sim = compute_similarity(" ".join(skills_required), processed_text)
        fmt_score = analyze_formatting(processed_text)

        
        matched_skills, missed_skills = smart_match(skills_required, processed_text)
        skill_score = (len(matched_skills) / len(skills_required) * 100) if skills_required else 0

        authenticity_flag = check_authenticity(processed_text, matched_skills, jd_sim)
        cert_score = min(len(extract_certifications(resume_text)) * 20, 100)

        calculated_years, experience_score_val = extract_experience_nlp(resume_text)
        total_months_int = int(round(calculated_years * 12))
        years_int = total_months_int // 12
        months_int = total_months_int % 12

        if years_int == 0:
            exp_display_str = f"{months_int} months"
        elif months_int == 0:
            exp_display_str = f"{years_int} years"
        else:
            exp_display_str = f"{years_int} years {months_int} months"

        w_skill = weights.get("skill", 0)
        w_exp = weights.get("exp", 0)
        w_cert = weights.get("cert", 0)
        w_jd = weights.get("jd", 0)
        w_fmt = weights.get("format", 0)
        total_weight = w_skill + w_exp + w_cert + w_jd + w_fmt

        if total_weight == 0:
            final_score = 0
        else:
            final_score = (
                                      w_skill * skill_score + w_exp * experience_score_val + w_cert * cert_score + w_jd * jd_sim + w_fmt * fmt_score) / total_weight

        if experience_level > 0 and total_months_int <= experience_level:
            decision = f"❌ Rejected"
            continue
        elif final_score >= cutoff_score:
            decision = "✅ Selected"
        else:
            decision = "❌ Rejected"

        report_data.append({
            "Resume": file.filename,
            "Decision": decision,
            "Authenticity": authenticity_flag,
            "Final Score": round(final_score, 2),
            "Email": email,
            "Phone": phone,
            "skill_score": round(skill_score, 2),
            "experience_years": exp_display_str,
            "experience": round(experience_score_val, 2),
            "jd_sim": round(jd_sim, 2),
            "cert_score": round(cert_score, 2),
            "Matched": ", ".join(matched_skills),
            "Missed": ", ".join(missed_skills)
        })

    if report_scope == "Selected Resumes Only":
        report_data = [r for r in report_data if "Selected" in r["Decision"]]
    report_data.sort(key=lambda x: x["Final Score"], reverse=True)

    df = pd.DataFrame(report_data)
    excel_base64 = ""
    if not df.empty:
        with io.BytesIO() as buffer:
            df.to_excel(buffer, index=False)
            excel_base64 = base64.b64encode(buffer.getvalue()).decode()

    cluster_chart_base64 = generate_cluster_chart(report_data)

    return {
        "results": report_data,
        "skipped": skipped_files,
        "excel_base64": excel_base64,
        "cluster_chart": cluster_chart_base64
    }


@app.post("/comparison")
async def compare_resumes(
        file_a: UploadFile = File(...),
        file_b: UploadFile = File(...),
        role: str = Form("AI/ML Engineer"),
        input_mode: str = Form("Use Job Role"),
        jd_textbox: str = Form(""),
        chart_type: str = Form("Bar")
):
    content_a = await file_a.read()
    content_b = await file_b.read()

    text_a = extract_text_from_file(content_a, file_a.filename)
    text_b = extract_text_from_file(content_b, file_b.filename)

    if not is_valid_resume(text_a):
        raise HTTPException(status_code=400, detail=f"'{file_a.filename}' is not a valid resume document.")
    if not is_valid_resume(text_b):
        raise HTTPException(status_code=400, detail=f"'{file_b.filename}' is not a valid resume document.")

    if input_mode == "Use Job Role":
        skills_required = GET_DEFAULT_ROLES.get(role, [])
    else:
        all_skills = set(skill for skills in GET_DEFAULT_ROLES.values() for skill in skills)
        skills_required = extract_skills_from_ner(jd_textbox.lower(), list(all_skills))

    if not skills_required:
        raise HTTPException(status_code=400, detail="No skills defined for comparison.")

    matched_a, _ = smart_match(skills_required, text_a)
    matched_b, _ = smart_match(skills_required, text_b)
    
    skill_a = len(matched_a) / len(skills_required) * 100
    skill_b = len(matched_b) / len(skills_required) * 100

    _, exp_a = extract_experience_nlp(text_a)
    _, exp_b = extract_experience_nlp(text_b)
    
    jd_sim_a = compute_similarity(" ".join(skills_required), text_a)
    jd_sim_b = compute_similarity(" ".join(skills_required), text_b)
    
    fmt_a = analyze_formatting(text_a)
    fmt_b = analyze_formatting(text_b)
    
    cert_a = min(len(extract_certifications(text_a)) * 20, 100)
    cert_b = min(len(extract_certifications(text_b)) * 20, 100)

    metrics = ["Skill", "Experience", "JD", "Format", "Cert"]
    scores_a = [skill_a, exp_a, jd_sim_a, fmt_a, cert_a]
    scores_b = [skill_b, exp_b, jd_sim_b, fmt_b, cert_b]

    chart_base64 = generate_comparison_chart(file_a.filename, file_b.filename, scores_a, scores_b, metrics, chart_type)

    return {
        "resume_a": file_a.filename,
        "resume_b": file_b.filename,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "chart_base64": chart_base64,
        "matched_a": matched_a,
        "matched_b": matched_b
    }


@app.post("/get_clusters")
async def get_clusters(results: List[dict]):
    if not results or len(results) < 3:
        raise HTTPException(status_code=400, detail="At least 3 candidates are required for clustering.")
    try:
        chart_base64 = generate_cluster_chart(results)
        return {"status": "success", "cluster_chart": chart_base64}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/salary")
async def get_salary_prediction(selected_results: List[dict]):
    if not selected_results:
        raise HTTPException(status_code=400, detail="No Selected candidate provided")
    try:
        updated_results = []
        for candidate in selected_results:
            score = candidate.get("Final Score", 0)
            exp = candidate.get("experience", 0)
            prediction = predict_salary(score, exp)
            candidate["predicted_salary"] = prediction
            updated_results.append(candidate)
        return {"status": "success", "salary_data": updated_results}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)