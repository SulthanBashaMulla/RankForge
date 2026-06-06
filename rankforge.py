# RankForge - Candidate Ranking Engine v3
# By Sulthan Basha Mulla | Team Curators
# INDIA.RUNS 2026 | Redrob AI

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Job Description
job_description = "Python developer with DevOps experience in Docker CI/CD and Machine Learning"

# Required Skills
required_skills = ["python", "docker", "ci/cd", "machine learning", "devops"]

# Hardcoded the Candidates
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

df = pd.DataFrame(candidates)

# Skill Match Analysis
def get_skill_gap(candidate_skills, required):
    candidate_lower = candidate_skills.lower()
    matched = [s for s in required if s in candidate_lower]
    missing = [s for s in required if s not in candidate_lower]
    return matched, missing

# Ranking
vectorizer = TfidfVectorizer()
all_text = [job_description] + list(df["Skills"])
vectors = vectorizer.fit_transform(all_text)
scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

# Experience boost
exp_boost = df["Experience"] * 0.01
df["RankScore"] = (scores + exp_boost).round(3)
df["Rank"] = df["RankScore"].rank(ascending=False).astype(int)
df = df.sort_values("RankScore", ascending=False)

# Skill Gap
matched_list = []
missing_list = []
for skills in df["Skills"]:
    matched, missing = get_skill_gap(skills, required_skills)
    matched_list.append(", ".join(matched) if matched else "None")
    missing_list.append(", ".join(missing) if missing else "None")

df["Matched Skills"] = matched_list
df["Missing Skills"] = missing_list

# Saving to csv file formate
df.to_csv("rankforge_v3_results.csv", index=False)

# Output formates
print("🏆 RankForge V3 - Intelligent Candidate Rankings")
print("="*60)
for i, row in df.head(3).iterrows():
    print(f"\n🥇 Rank {row['Rank']}: {row['Name']} ({row['Location']})")
    print(f"   Score     : {row['RankScore']}")
    print(f"   Experience: {row['Experience']} years")
    print(f"   ✅ Matched : {row['Matched Skills']}")
    print(f"   ❌ Missing : {row['Missing Skills']}")

print("\n" + "="*60)
print("📊 Full Rankings:")
print(df[["Rank","Name","Location","Experience","RankScore"]].to_string(index=False))
print("="*60)
print(f"✅ Saved to rankforge_v3_results.csv")
print(f"📊 Total Candidates: {len(df)}")
print(f"🥇 Best Match: {df.iloc[0]['Name']} from {df.iloc[0]['Location']}")
