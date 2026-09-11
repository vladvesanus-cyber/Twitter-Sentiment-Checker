from flask import Flask, render_template, request
import pickle
from clean import clean_text

app = Flask(__name__)

class TextChecker:
    def __init__(self):
        self.artifacts = pickle.load(open('artifacts.pkl', 'rb'))

    def check_text(self,text):
        vectorizer = self.artifacts['vectorizer']
        model = self.artifacts['model']
        text = clean_text(text)
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)
        return prediction[0]
    
text_checker = TextChecker()

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/check', methods=['POST'])
def check():
    text = request.form.get('text', '')
    if text:
        result = text_checker.check_text(text)
        return render_template('results.html', result=result)
    else:
        return render_template('home.html', error="Please enter some text to check.")
    
if __name__=="__main__":
    app.run(debug=True)