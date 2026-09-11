import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from clean import clean_data

data_full = pd.read_csv('Data/data.csv', encoding='latin-1')
data_full.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
data_full['target'] = data_full['target'].map({0: 0, 4: 1})  # Map 4 to 1 for positive sentiment
data = data_full.sample(n=30000, random_state=42)

cleaned_data = clean_data(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_data["text"])

y = cleaned_data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model ={"LinearSVC": LinearSVC(),
        "MultinomialNB": MultinomialNB(),
        "RandomForestClassifier": RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier()}
score = 0
best_model = None

for name, m in model.items():
    m.fit(X_train, y_train)
    prediction = m.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    report = classification_report(y_test, prediction)
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Classification Report:\n{report}")
    if precision > score:
        score = precision
        best_model = m

pickle.dump({"vectorizer": vectorizer,
             "model": best_model}, open("artifacts.pkl", "wb"))
