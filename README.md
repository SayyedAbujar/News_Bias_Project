# 📰 News Bias Detection Project

An end-to-end **NLP & ML based system** to detect and classify news articles as **Biased (1)** or **Un-Biased (0)**.  
This project applies **text preprocessing, feature extraction (TF-IDF / Bag of Words), and ML models** like Logistic Regression, SVM, and Random Forest.  
Deployed with **FastAPI backend** and a **Streamlit dashboard** for user-friendly predictions.

---

## 🚀 Features

- Data preprocessing (cleaning, tokenization, stemming, stopword removal).
- Feature extraction using CountVectorizer & TF-IDF.
- Machine Learning models for binary classification.
- FastAPI REST API for serving predictions.
- Streamlit dashboard for interactive testing.

---

## 🛠️ Tech Stack

- Python (Scikit-learn, Pandas, NumPy, NLTK)
- FastAPI (Backend API)
- Streamlit (Frontend UI)
- Joblib (Model serialization)
- Git & GitHub (Version control)

---

## 📂 Project Structure

News_bias_Project/
│── data/ # Dataset (processed/cleaned)
│── notebooks/ # Jupyter notebooks (EDA, model training)
│── model/ # Saved joblib models & vectorizers
│── backend/ # FastAPI app
│── frontend/ # Streamlit app
│── README.md # Project documentation
