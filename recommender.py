import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

'''Backend architecture:
recommender.py
│
├── content-based filtering (TF-IDF + cosine similarity)
├── simulated collaborative filtering (SVD on interaction matrix)
├── hybrid model (weighted combination)
│
├── recommend_jobs()         → content-based recommendations
├── recommend_cf()           → CF-based recommendations
├── recommend_hybrid()       → hybrid recommendations
│
├── recommend_companies()           → content-based company recs
├── recommend_companies_hybrid()    → hybrid company recs
│
└── evaluate_model()         → Precision@K, Recall@K, HitRate

NOTE: Interaction matrix is simulated from TF-IDF similarity scores.
CF and content signals are therefore correlated. For a truly independent
hybrid, replace interaction_matrix with real application/click logs.
'''

#LOAD DATA 

jobs = pd.read_csv("tech_jobs_cleaned.csv")
resumes = pd.read_csv("tech_resumes_cleaned.csv")

#CONTENT-BASED MODEL 

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

job_vectors = vectorizer.fit_transform(jobs["combined_text"])
resume_vectors = vectorizer.transform(resumes["combined_resume"])

similarity = cosine_similarity(resume_vectors, job_vectors)

#SIMULATED INTERACTION MATRIX 

def simulate_interactions(similarity_matrix, threshold=0.3, noise=0.05, random_state=42):
    """
    Simulate binary interactions (apply/click) from content similarity scores.

    - Jobs with similarity above `threshold` are treated as relevant interactions.
    - Gaussian noise adds realism — not every relevant job gets interacted with,
      and some borderline jobs may be included.

    NOTE: This is a bootstrapping strategy for when real interaction data is
    unavailable. The simulated matrix is derived from TF-IDF similarity, so
    CF and content signals will be correlated — not truly independent.
    For production, replace this with real application/click logs.

    Args:
        similarity_matrix : np.ndarray — content similarity (n_resumes x n_jobs)
        threshold         : float — minimum similarity to count as an interaction
        noise             : float — std of Gaussian noise added to probabilities
        random_state      : int — for reproducibility

    Returns:
        np.ndarray — binary interaction matrix (n_resumes x n_jobs)
    """
    rng = np.random.default_rng(random_state)
    prob_matrix = np.clip(
        similarity_matrix + rng.normal(0, noise, similarity_matrix.shape),
        0, 1
    )
    interaction_matrix = (prob_matrix >= threshold).astype(int)
    return interaction_matrix


interaction_matrix = simulate_interactions(similarity)

#COLLABORATIVE FILTERING (SVD on interaction matrix) 

svd = TruncatedSVD(n_components=100, random_state=42)
matrix_reduced = svd.fit_transform(interaction_matrix)
reconstructed_matrix = svd.inverse_transform(matrix_reduced)

#HYBRID MODEL (NORMALIZED)
scaler_content = MinMaxScaler()
scaler_cf = MinMaxScaler()

similarity_norm = scaler_content.fit_transform(similarity)
cf_norm = scaler_cf.fit_transform(reconstructed_matrix)

hybrid_matrix = (0.6 * similarity_norm) + (0.4 * cf_norm)

#MODEL SERIALIZATION 

def save_models(path="models/"):
    """
    Serialize all fitted objects to disk so they don't need to be
    recomputed on every import. Load with load_models() at inference time.
    """
    os.makedirs(path, exist_ok=True)
    joblib.dump(vectorizer,          os.path.join(path, "vectorizer.pkl"))
    joblib.dump(svd,                 os.path.join(path, "svd.pkl"))
    joblib.dump(scaler_content,      os.path.join(path, "scaler_content.pkl"))
    joblib.dump(scaler_cf,           os.path.join(path, "scaler_cf.pkl"))
    np.save(os.path.join(path, "similarity.npy"),          similarity)
    np.save(os.path.join(path, "reconstructed_matrix.npy"), reconstructed_matrix)
    np.save(os.path.join(path, "hybrid_matrix.npy"),        hybrid_matrix)
    print(f"Models saved to '{path}'")


def load_models(path="models/"):
    """
    Load serialized models from disk. Returns a dict of all fitted objects
    and matrices. Use this instead of recomputing from scratch at inference.
    """
    return {
        "vectorizer":           joblib.load(os.path.join(path, "vectorizer.pkl")),
        "svd":                  joblib.load(os.path.join(path, "svd.pkl")),
        "scaler_content":       joblib.load(os.path.join(path, "scaler_content.pkl")),
        "scaler_cf":            joblib.load(os.path.join(path, "scaler_cf.pkl")),
        "similarity":           np.load(os.path.join(path, "similarity.npy")),
        "reconstructed_matrix": np.load(os.path.join(path, "reconstructed_matrix.npy")),
        "hybrid_matrix":        np.load(os.path.join(path, "hybrid_matrix.npy")),
    }

#INPUT VALIDATION 

def _validate_index(resume_index, matrix, top_n):
    """Shared input validation for all recommend_* functions."""
    n_resumes = matrix.shape[0]
    if not isinstance(resume_index, (int, np.integer)):
        raise TypeError(f"resume_index must be an integer, got {type(resume_index).__name__}")
    if resume_index < 0 or resume_index >= n_resumes:
        raise IndexError(f"resume_index {resume_index} out of range for {n_resumes} resumes")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

#CONTENT-BASED RECOMMENDATION

def recommend_jobs(resume_index, top_n=5):
    """
    Recommend top_n jobs for a given resume using content-based filtering
    (TF-IDF cosine similarity).

    Args:
        resume_index : int — index into the resumes dataframe
        top_n        : int — number of jobs to return

    Returns:
        pd.DataFrame — top job recommendations with similarity scores
    """
    _validate_index(resume_index, similarity, top_n)

    scores = similarity[resume_index]
    top_jobs = scores.argsort()[-top_n:][::-1]

    results = jobs.iloc[top_jobs][[
        "title",
        "company_name",
        "location",
        "formatted_experience_level",
        "company_url"
    ]].copy()

    results["similarity_score"] = scores[top_jobs]
    results = results.drop_duplicates("title")

    return results

#COLLABORATIVE FILTERING RECOMMENDATION 

def recommend_cf(resume_index, top_n=5):
    """
    Recommend top_n jobs using collaborative filtering (SVD on simulated
    interaction matrix).

    Args:
        resume_index : int — index into the resumes dataframe
        top_n        : int — number of jobs to return

    Returns:
        pd.DataFrame — top job recommendations with CF scores
    """
    _validate_index(resume_index, reconstructed_matrix, top_n)

    scores = reconstructed_matrix[resume_index]
    top_jobs = scores.argsort()[-top_n:][::-1]

    results = jobs.iloc[top_jobs][[
        "title",
        "company_name",
        "location"
    ]].copy()

    results["cf_score"] = scores[top_jobs]
    results = results.drop_duplicates("title")

    return results

#HYBRID RECOMMENDATION 

def recommend_hybrid(resume_index, top_n=5):
    """
    Recommend top_n jobs using the hybrid model (0.6 * content + 0.4 * CF),
    both normalized independently before blending.

    Args:
        resume_index : int — index into the resumes dataframe
        top_n        : int — number of jobs to return

    Returns:
        pd.DataFrame — top job recommendations with hybrid scores
    """
    _validate_index(resume_index, hybrid_matrix, top_n)

    scores = hybrid_matrix[resume_index]
    top_jobs = scores.argsort()[-top_n:][::-1]

    results = jobs.iloc[top_jobs][[
        "title",
        "company_name",
        "location",
        "company_url"
    ]].copy()

    results["hybrid_score"] = scores[top_jobs]
    results = results.drop_duplicates("title")

    return results

#COMPANY RECOMMENDATIONS

def recommend_companies(resume_index, top_n=5):
    """
    Recommend top companies using content-based scores.
    Aggregates over top 50 jobs and returns most frequent companies.

    Args:
        resume_index : int — index into the resumes dataframe
        top_n        : int — number of companies to return

    Returns:
        pd.Series — company names with job match counts
    """
    _validate_index(resume_index, similarity, top_n)

    scores = similarity[resume_index]
    top_jobs = scores.argsort()[-50:][::-1]  # Fix: reverse for true top-50

    companies = jobs.iloc[top_jobs]["company_name"]
    return companies.value_counts().head(top_n)


def recommend_companies_hybrid(resume_index, top_n=5):
    """
    Recommend top companies using hybrid scores.
    Aggregates over top 50 jobs, groups by company, and returns
    companies with the most matched job postings.

    Args:
        resume_index : int — index into the resumes dataframe
        top_n        : int — number of companies to return

    Returns:
        pd.DataFrame — companies with job_count and company_url
    """
    _validate_index(resume_index, hybrid_matrix, top_n)

    scores = hybrid_matrix[resume_index]
    top_jobs = scores.argsort()[-50:][::-1]  # Fix: reverse for true top-50

    subset = jobs.iloc[top_jobs]

    companies = (
        subset.groupby("company_name")
        .agg({
            "company_url": "first",
            "title": "count"
        })
        .rename(columns={"title": "job_count"})
        .sort_values("job_count", ascending=False)
        .head(top_n)
    )

    return companies

#EVALUATION METRICS

def precision_at_k(scores, k=5, threshold=0.3):
    """Fraction of top-k recommendations that are relevant."""
    top_k = scores.argsort()[-k:]
    relevant = scores[top_k] >= threshold
    return relevant.sum() / k


def recall_at_k(scores, k=5, threshold=0.3):
    """Fraction of all relevant items that appear in the top-k."""
    relevant_total = (scores >= threshold).sum()
    if relevant_total == 0:
        return 0
    top_k = scores.argsort()[-k:]
    relevant_recommended = (scores[top_k] >= threshold).sum()
    return relevant_recommended / relevant_total


def hit_rate(scores, k=5, threshold=0.3):
    """1 if at least one relevant item appears in the top-k, else 0."""
    top_k = scores.argsort()[-k:]
    return int((scores[top_k] >= threshold).any())


def evaluate_model(matrix, k=5, threshold=0.3):
    """
    Evaluate a recommendation matrix across all resumes.

    Args:
        matrix    : np.ndarray — similarity or hybrid score matrix
        k         : int — cutoff rank for evaluation
        threshold : float — minimum score to count as relevant.
                    NOTE: this is hardcoded at 0.3 as a reasonable baseline
                    but should ideally be calibrated on labelled data or
                    set via a percentile cutoff on your score distribution.

    Returns:
        dict — Precision@K, Recall@K, HitRate averaged over all resumes
    """
    precisions, recalls, hits = [], [], []

    for i in range(len(resumes)):
        scores = matrix[i]
        precisions.append(precision_at_k(scores, k, threshold))
        recalls.append(recall_at_k(scores, k, threshold))
        hits.append(hit_rate(scores, k, threshold))

    return {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}":    np.mean(recalls),
        "HitRate":        np.mean(hits)
    }

#MAIN (FOR TESTING ONLY) 

if __name__ == "__main__":

    print("Jobs:", jobs.shape)
    print("Resumes:", resumes.shape)
    print("Interaction matrix sparsity: "
          f"{100 * (interaction_matrix == 0).sum() / interaction_matrix.size:.1f}%")

    print("\n── CONTENT BASED ──")
    print(recommend_jobs(0))

    print("\n── COLLABORATIVE FILTERING ──")
    print(recommend_cf(0))

    print("\n── HYBRID ──")
    print(recommend_hybrid(0))

    print("\n── COMPANY RECOMMENDATIONS (HYBRID) ──")
    print(recommend_companies_hybrid(0))

    print("\n── EVALUATION ──")
    print("Content-Based: ", evaluate_model(similarity))
    print("CF:            ", evaluate_model(reconstructed_matrix))
    print("Hybrid:        ", evaluate_model(hybrid_matrix))

    # Uncomment to serialize models after first run:
    # save_models()
