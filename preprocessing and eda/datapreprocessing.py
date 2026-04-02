import pandas as pd
import re

# Load cleaned resume dataset
resumes = pd.read_csv("resume_dataset_cleaned.csv")

# -----------------------------
# Basic Text Cleaning Function
# -----------------------------
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Clean main textual columns
resumes["resume_text"] = resumes["resume_text"].apply(normalize_text)
resumes["skills"] = resumes["skills"].apply(normalize_text)
resumes["education"] = resumes["education"].apply(normalize_text)

# -----------------------------
# Feature Engineering
# -----------------------------
# Combine all relevant text into ONE model feature
resumes["combined_profile"] = (
    resumes["resume_text"] + " " +
    resumes["skills"] + " " +
    resumes["education"] + " " +
    resumes["resume_domain"]
)

resumes.to_csv("resumes_preprocessed.csv", index=False)
print("Saved resumes_preprocessed.csv")

resumes.head()

# Load job dataset
jobs = pd.read_csv("job_dataset.csv")

# Normalize text columns
text_cols = ["Skills", "Keywords", "Responsibilities", "Title"]

for col in text_cols:
    jobs[col] = jobs[col].apply(normalize_text)

# Ensure experience is numeric
jobs["YearsOfExperience"] = pd.to_numeric(
    jobs["YearsOfExperience"], errors="coerce"
).fillna(0)

# -----------------------------
# Feature Engineering
# -----------------------------
# Combine text features for similarity
jobs["combined_job_profile"] = (
    jobs["Skills"] + " " +
    jobs["Keywords"] + " " +
    jobs["Responsibilities"] + " " +
    jobs["Title"]
)

jobs.to_csv("jobs_preprocessed.csv", index=False)
print("Saved jobs_preprocessed.csv")


jobs.head()

def filter_jobs_for_candidate(candidate_row, jobs_df):
    return jobs_df[
        jobs_df["YearsOfExperience"] <= candidate_row["experience_years"]
    ]

from sklearn.feature_extraction.text import TfidfVectorizer

# Combine all text for vocabulary learning
combined_corpus = pd.concat([
    resumes["combined_profile"],
    jobs["combined_job_profile"]
])

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=8000,
    ngram_range=(1,2)
)

tfidf.fit(combined_corpus)

# Transform separately
resume_vectors = tfidf.transform(resumes["combined_profile"])
job_vectors = tfidf.transform(jobs["combined_job_profile"])


from sklearn.metrics.pairwise import cosine_similarity

def recommend_jobs(candidate_index, top_n=5):
    candidate_vector = resume_vectors[candidate_index]
    
    # Filter jobs by experience
    filtered_jobs = filter_jobs_for_candidate(
        resumes.iloc[candidate_index], jobs
    )
    
    filtered_indices = filtered_jobs.index
    
    similarity_scores = cosine_similarity(
        candidate_vector,
        job_vectors[filtered_indices]
    )[0]
    
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    return filtered_jobs.iloc[top_indices][["Title", "YearsOfExperience"]]

resumes_model = resumes[
    ["resume_id", "combined_profile", "experience_years", "resume_domain"]
]

resumes_model.to_csv("resumes_model_ready.csv", index=False)
print("Saved resumes_model_ready.csv")

jobs_model = jobs[
    ["JobID", "combined_job_profile", "YearsOfExperience", "Title"]
]

jobs_model.to_csv("jobs_model_ready.csv", index=False)
print("Saved jobs_model_ready.csv")

import pickle

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Saved tfidf_vectorizer.pkl")


'''
Why separate files?

Because:

Stage	             Purpose
Raw data	    Original extracted
Preprocessed	Cleaned and normalized
Model-ready	    Only features needed for TF-IDF
Vectorizer	    Saved trained representation
'''
