import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from docx import Document
from collections import Counter

RESUME_FOLDER = "Resumes/"   # folder containing .docx resumes

if not os.path.exists(RESUME_FOLDER):
    raise FileNotFoundError(f"Folder not found: {RESUME_FOLDER}")

print("Resumes found:", os.listdir(RESUME_FOLDER))

def read_docx(path):
    doc = Document(path)
    return " ".join(p.text for p in doc.paragraphs)

data = []

for i, file in enumerate(os.listdir(RESUME_FOLDER)):
    if file.lower().endswith(".docx"):
        file_path = os.path.join(RESUME_FOLDER, file)
        text = read_docx(file_path)

        data.append({
            "resume_id": i + 1,
            "resume_text": text
        })

df = pd.DataFrame(data)

print("\nDataset created successfully")
print(df.head())

def extract_education(text):
    text = text.lower()
    if "phd" in text:
        return "PhD"
    elif "master" in text:
        return "Masters"
    elif "bachelor" in text or "b.tech" in text or "b.e" in text:
        return "Bachelors"
    else:
        return "Not Specified"

df["education"] = df["resume_text"].apply(extract_education)

def extract_experience(text):
    match = re.search(r"(\d+)\+?\s+years", text.lower())
    return int(match.group(1)) if match else 0

df["experience_years"] = df["resume_text"].apply(extract_experience)

def classify_domain(text):
    text = text.lower()
    tech_terms = ["java", "python", "software", "developer", "engineer", "sql"]
    count = sum(term in text for term in tech_terms)

    if count >= 3:
        return "Tech"
    elif count == 0:
        return "Non-Tech"
    else:
        return "Mixed"

df["resume_domain"] = df["resume_text"].apply(classify_domain)

STOPWORDS = {
    "the","and","for","with","that","this","from","are","was","were",
    "has","have","had","not","but","you","your","his","her","their",
    "will","can","may","also","such","using","used","use","etc", "involved", "hibernate", "test", "spring"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

#skill extraction
def extract_skills(text, top_n=15):
    words = clean_text(text)
    freq = Counter(words)
    return ", ".join([w for w, _ in freq.most_common(top_n)])

df["skills"] = df["resume_text"].apply(extract_skills)

df = df[
    ["resume_id", "resume_text", "education", "skills", "experience_years", "resume_domain"]
]

print("\nFinal dataset preview:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nDataset Description:")
print(df.describe(include="all"))

#resume length
df["resume_length"] = df["resume_text"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8,5))
sns.histplot(df["resume_length"], bins=30)
plt.title("Resume Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Count")
plt.show()

#education
plt.figure(figsize=(6,4))
df["education"].value_counts().plot(kind="bar")
plt.title("Education Distribution")
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.show()

#skills in resumes
df["skill_count"] = df["skills"].apply(lambda x: len(x.split(", ")))

plt.figure(figsize=(8,5))
sns.histplot(df["skill_count"], bins=20)
plt.title("Extracted Skills per Resume")
plt.xlabel("Skill Count")
plt.ylabel("Count")
plt.show()

#top skills
all_skills = []
for s in df["skills"]:
    all_skills.extend(s.split(", "))

print("\nTop 20 skills across resumes:")
print(Counter(all_skills).most_common(20))

#experience
plt.figure(figsize=(8,5))
sns.histplot(df["experience_years"], bins=20)
plt.title("Experience Years Distribution")
plt.xlabel("Years of Experience")
plt.ylabel("Count")
plt.show()

#domain distribution
plt.figure(figsize=(6,4))
df["resume_domain"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", startangle=90
)
plt.title("Resume Domain Distribution")
plt.ylabel("")
plt.show()

#resume length vs experience years
plt.figure(figsize=(7,5))
sns.scatterplot(
    x="experience_years",
    y="resume_length",
    data=df
)
plt.title("Experience vs Resume Length")
plt.xlabel("Years of Experience")
plt.ylabel("Resume Length (Words)")
plt.show()

df.to_csv("resume_dataset_cleaned.csv", index=False)
print("\nCleaned dataset saved as resume_dataset_cleaned.csv")
