# RankForge - Candidate Ranking Engine v1
# By Sulthan Basha Mulla | Team Curators

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample Job Description
job_description = "Python developer with DevOps experience in Docker and CI/CD"

# Sample Candidates
candidates = {
    "Name": ["Ravi", "Priya", "Arun", "Sneha", "Kiran"],
    "Skills": [
        "Python Django Docker CI/CD GitHub Actions",
        "Java Spring Boot MySQL REST API",
        "Python FastAPI Docker Kubernetes DevOps",
        "React JavaScript Node.js MongoDB",
        "Python scikit-learn Machine Learning Docker"
    ]
}

# Create DataFrame
df = pd.DataFrame(candidates)

# Rank Candidates
vectorizer = TfidfVectorizer()
all_text = [job_description] + list(df["Skills"])
vectors = vectorizer.fit_transform(all_text)
scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
df["RankScore"] = scores
df = df.sort_values("RankScore", ascending=False)

print("🏆 RankForge - Candidate Rankings")
print("="*40)
print(df[["Name", "RankScore"]].to_string(index=False))
