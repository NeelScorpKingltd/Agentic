

from flask import Flask, render_template, request
from nlp_pipeline import analyze_article
from classifier import predict_category

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('article', '')
    if not text.strip():
        return render_template('index.html', error='Please provide an article to analyze.')
    result = analyze_article(text)
    
    # Classify the text using the trained Keras model
    result['category'] = predict_category(text)
    
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
