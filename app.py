from flask import Flask, render_template, request
import os
from model.preprocess import extract_text_from_pdf, clean_text
from model.similarity import calculate_similarity, get_missing_keywords

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    missing_keywords = []

    if request.method == "POST":
        file = request.files["resume"]
        job_desc = request.form["job_desc"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            resume_text = extract_text_from_pdf(filepath)
            resume_text = clean_text(resume_text)
            job_desc = clean_text(job_desc)

            score = calculate_similarity(resume_text, job_desc)
            missing_keywords = get_missing_keywords(resume_text, job_desc)

    return render_template("index.html", score=score, missing_keywords=missing_keywords)


if __name__ == "__main__":
    app.run(debug=True)
