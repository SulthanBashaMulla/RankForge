# RankForge - Candidate Ranking Engine v2
# By Sulthan Basha Mulla | Team Curators

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Job Description
job_description = "Python developer with DevOps experience in Docker CI/CD and Machine Learning"

# Hard Coding the Candidates
candidates = {
    "Name": ["Ravi", "Priya", "Arun", "Sneha", "Kiran",
             "Arjun", "Meena", "Rahul", "Divya", "Suresh"],
    "Location": ["Hyderabad", "Mumbai", "Nandyal", "Delhi",
                 "Bangalore", "Warangal", "Chennai", "Tirupati",
                 "Pune", "Vizag"],
    "Experience": [3, 5, 1, 4, 2, 3, 6, 2, 4, 1],
    "Skills": [
        "Python Django Docker CI/CD GitHub Actions",
        "Java Spring Boot MySQL REST API",
        "Python FastAPI Docker Kubernetes DevOps MLOps",
        "React JavaScript Node.js MongoDB",
        "Python scikit-learn Machine Learning Docker",
        "Python Docker CI/CD Jenkins DevOps",
        "Java Python Machine Learning TensorFlow",
        "Python Flask Docker GitHub Actions CI/CD",
        "JavaScript React Python REST API",
        "Python Machine Learning scikit-learn pandas"
    ]
}

# DataFrame
df = pd.DataFrame(candidates)

# Rank Candidates
vectorizer = TfidfVectorizer()
all_text = [job_description] + list(df["Skills"])
vectors = vectorizer.fit_transform(all_text)
scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
df["RankScore"] = scores.round(3)
df["Rank"] = df["RankScore"].rank(ascending=False).astype(int)
df = df.sort_values("RankScore", ascending=False)

# Saving to CSV
df.to_csv("rankforge_results.csv", index=False)

print("🏆 RankForge - Top Candidates")
print("="*50)
print(df[["Rank","Name","Location","Experience","RankScore"]].to_string(index=False))
print("="*50)
print(f"✅ Results saved to rankforge_results.csv")
print(f"📊 Total Candidates Ranked: {len(df)}")
print(f"🥇 Best Match: {df.iloc[0]['Name']} from {df.iloc[0]['Location']}")
