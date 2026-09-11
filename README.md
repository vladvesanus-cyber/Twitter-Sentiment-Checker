# 🔍 Twitter Sentiment Checker

A machine learning project for text sentiment analysis (positive/negative) based on a Twitter dataset, with automated selection of the best classification model and a Flask web app for checking text sentiment in real time.

## 📁 Project Structure

```
Twitter/
├── Data/
│   └── data.csv            # source dataset (Sentiment140-like)
├── templates/
│   ├── base.html            # base template (styles, header/footer)
│   ├── home.html             # text input form
│   └── results.html          # results page
├── clean.py                  # text cleaning
├── main.py                   # model training + saving artifacts
├── app.py                     # Flask application
├── artifacts.pkl               # saved model + vectorizer (created by main.py)
└── README.md
```

## ⚙️ How It Works

**1. Data**
- Loads `Data/data.csv` in Sentiment140 format (columns: `target, id, date, flag, user, text`)
- Sentiment labels are remapped: `0` → negative, `4` → `1` (positive)
- A random sample of **30,000 rows** is used for faster training (`random_state=42`)

**2. Text Cleaning** (`clean.py`)
- Removes mentions (`@user`)
- Removes URLs (`http...`, `www...`)
- Removes hashtags (`#tag`)
- Removes extra whitespace
- Converts text to lowercase

**3. Vectorization**
- Text is converted into numerical features using **TF-IDF** (`TfidfVectorizer`)

**4. Model Training & Selection** (`main.py`)
- Data is split into train/test (80/20, `random_state=42`)
- Four classification models are compared:
  - Linear SVC
  - Multinomial Naive Bayes
  - Random Forest Classifier
  - Gradient Boosting Classifier
- For each model, **Accuracy**, **Precision**, and a full `classification_report` are printed
- The model with the highest **Precision** is selected and saved as the final model

**5. Saving Artifacts**
- The trained model and TF-IDF vectorizer are saved together in `artifacts.pkl` — this is the file `app.py` later loads

**6. Web Application** (`app.py`, Flask)
- Home page (`/`, `/home`) — a form for pasting text
- `POST /check` — cleans the input text (`clean_text`), vectorizes it with the same TF-IDF vectorizer, and classifies it with the model
- The result is shown on a separate page: **"This text has a positive sentiment"** or **"This text has a negative sentiment"**, styled with matching colors (green/red)

## 🚀 Getting Started

### Install Dependencies

The project has no `requirements.txt` — install the packages manually:

```bash
pip install flask pandas scikit-learn
```

### Train the Model

```bash
python main.py
```

The script will:
1. Load and clean `Data/data.csv`
2. Vectorize the text using TF-IDF
3. Train and compare 4 classification models on Accuracy/Precision
4. Select the best model (by Precision)
5. Save the model and vectorizer to `artifacts.pkl`

> ⚠️ Before running `main.py`, place the source dataset at `Data/data.csv` — it is not included in the repository.

### Run the Web App

```bash
python app.py
```

This starts a Flask server (by default at `http://127.0.0.1:5000`) where you can paste any text and get a sentiment prediction.

## 💡 Possible Improvements

- Add a `requirements.txt` with pinned dependency versions
- Support neutral sentiment (currently binary classification: positive/negative)
- Replace TF-IDF with embeddings (e.g. Word2Vec, BERT) for better accuracy
- Tune hyperparameters with `GridSearchCV`
- Load the model once when Flask starts rather than at module import time (move to `if __name__ == "__main__"` or lazy-load it)
- Add input length validation on the frontend

## 📄 Dataset

The project targets the format of the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset (1.6M tweets with sentiment labels).

---
*Author: vladvesanus-cyber*
