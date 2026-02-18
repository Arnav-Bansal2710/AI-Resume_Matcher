from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

skills_vocab = [
    "python","java","machine learning","flask",
    "sql","nlp","scikit learn",
    "bert","transformers","cosine similarity",
    "data structures","backend"
]

def extract_skills(text):
    found = []
    text = text.lower()
    for skill in skills_vocab:
        if skill in text:
            found.append(skill)
    return found


def calculate_similarity(resume_text, job_desc):
    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        max_df=0.85,
        sublinear_tf=True
    )

    documents = [resume_text, job_desc]
    tfidf_matrix = tfidf.fit_transform(documents)

    score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    return round(score[0][0] * 100, 2)


def get_missing_keywords(resume_text, job_desc):
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(job_desc))
    missing = jd_skills - resume_skills
    return list(missing)


