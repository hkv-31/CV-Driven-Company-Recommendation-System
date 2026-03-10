import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re

resume_df = pd.read_csv("resume_dataset_cleaned.csv")
job_df = pd.read_csv("job_dataset.csv")

print(resume_df.shape)
print(job_df.shape)

resume_df.head()
job_df.head()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

resume_df["combined_text"] = (
    resume_df["skills"].fillna("") + " " +
    resume_df["education"].fillna("") + " " +
    resume_df["resume_text"].fillna("")
)
job_df["Title"] = job_df["Title"].fillna("Unknown Role")
job_df["combined_text"] = (
    job_df["Title"].fillna("") + " " +
    job_df["Skills"].fillna("") + " " +
    job_df["Keywords"].fillna("") + " " +
    job_df["Responsibilities"].fillna("")
)

resume_df["combined_text"] = resume_df["combined_text"].apply(clean_text)
job_df["combined_text"] = job_df["combined_text"].apply(clean_text)

all_text = pd.concat([
    resume_df["combined_text"],
    job_df["combined_text"]
])

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

vectorizer.fit(all_text)
resume_vectors = vectorizer.transform(resume_df["combined_text"])
job_vectors = vectorizer.transform(job_df["combined_text"])
#print(resume_vectors)
#print(job_vectors)

#Cosine Similarity Computation
similarity_matrix = cosine_similarity(resume_vectors, job_vectors)

#print(similarity_matrix.shape)

#Recommendation function
def recommend_jobs(resume_index, top_k=5):
    
    scores = similarity_matrix[resume_index]
    top_indices = np.argsort(scores)[::-1][:top_k]
    recommendations = job_df.iloc[top_indices][
        ["Title", "Skills", "YearsOfExperience"]
    ].copy()
    recommendations = recommendations.copy()
    recommendations["similarity_score"] = scores[top_indices]
    return recommendations

#Resume Upload Simulation
def recommend_from_resume_text(resume_text, top_k=5):
    
    cleaned = clean_text(resume_text)
    vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(vector, job_vectors)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = job_df.iloc[top_indices][["Title","Skills"]]
    results["score"] = scores[top_indices]
    return results

#Popularity based fallback model
popular_jobs = (
    job_df['Title']
    .value_counts()
    .head(10)
)

#eval- Precision@K, Recall@K

'''def recommend_companies(resume_index, top_k=5):

    scores = similarity_matrix[resume_index]
    job_df["similarity"] = scores
    company_scores = job_df.groupby("Company")["similarity"].max()
    top_companies = company_scores.sort_values(ascending=False).head(top_k)
    return top_companies'''

#Test
print(recommend_jobs(resume_index=0, top_k=5))
print(recommend_from_resume_text(
    "Python machine learning data analysis pandas tensorflow deep learning"
))
print(similarity_matrix.shape)