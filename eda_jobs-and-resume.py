import pandas as pd

#JOBS
jobs = pd.read_csv("tech_jobs.csv")
print("Original Shape:", jobs.shape)

#DROP DUPLICATES
jobs = jobs.drop_duplicates()

#FILL MISSING VALUES
text_cols = [
    "title",
    "description",
    "skills_desc",
    "company_name",
    "company_description",
    "location"
]

for col in text_cols:
    jobs[col] = jobs[col].fillna("")

jobs["company_url"] = jobs["company_url"].replace("", "Not specified").fillna("Not specified")

#SKILL EXTRACTION
skill_keywords = [
    "python","java","c++","c#","sql","mysql","postgresql",
    "mongodb","aws","azure","gcp","docker","kubernetes",
    "spark","hadoop","tensorflow","pytorch","scikit-learn",
    "react","angular","vue","node","flask","django",
    "fastapi","spring","html","css","javascript","typescript",
    "git","linux","pandas","numpy","power bi","tableau"
]

def extract_skills(text):
    text = text.lower()
    found = [skill for skill in skill_keywords if skill in text]
    return ", ".join(found)

jobs["skills_extracted"] = (
    jobs["title"] + " " + jobs["description"]
).apply(extract_skills)

jobs["skills_extracted"] = jobs["skills_extracted"].apply(
    lambda x: ", ".join([s.strip() for s in x.split(",") if s.strip() != ""])
)

#CREATE COMBINED TEXT
jobs["combined_text"] = (
    jobs["title"] + " " +
    jobs["description"] + " " +
    jobs["skills_extracted"] + " " +
    jobs["company_description"]
).str.lower()

#Fix empty company names
jobs["company_name"] = jobs["company_name"].replace("", "Unknown Company")

print("After preprocessing:", jobs.shape)

#Save cleaned jobs
jobs.to_csv("tech_jobs_cleaned.csv", index=False)

#JOB EDA
print("\nShape:", jobs.shape)
print("\nColumns:", jobs.columns)

print("\nNull values")
print(jobs.isnull().sum())

print("\nTop job titles")
print(jobs["title"].value_counts().head(10))

print("\nTop companies")
print(jobs["company_name"].value_counts().head(10))

print("\nExperience levels")
print(jobs["formatted_experience_level"].value_counts())

print("\nTop locations")
print(jobs["location"].value_counts().head(10))

print("\nTop extracted skills")
skills = jobs["skills_extracted"].str.split(",")
skills = skills.explode()
print(skills.value_counts().head(20))

#RESUMES
resumes = pd.read_csv("tech_resumes.csv")
print("\nOriginal shape:", resumes.shape)

#DROP DUPLICATES
resumes = resumes.drop_duplicates()

#FILL NULL VALUES
text_cols = ["resume_text", "skills", "education"]

for col in text_cols:
    resumes[col] = resumes[col].fillna("")

#CREATE COMBINED RESUME TEXT
resumes["combined_resume"] = (
    resumes["resume_text"] + " " +
    resumes["skills"] + " " +
    resumes["education"]
).str.lower()

print("After preprocessing:", resumes.shape)

#Save cleaned resumes
resumes.to_csv("tech_resumes_cleaned.csv", index=False)

#RESUME EDA
print("\nShape:", resumes.shape)

print("\nDomains")
print(resumes["resume_domain"].value_counts())

print("\nExperience years")
print(resumes["experience_years"].describe())

print("\nSkill count")
print(resumes["skill_count"].describe())

print("\nTop resume skills")
skills = resumes["skills"].str.split(",")
skills = skills.explode()
print(skills.value_counts().head(20))