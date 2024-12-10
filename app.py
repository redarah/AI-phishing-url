from flask import Flask, render_template, request
from joblib import load
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse, urlsplit
import tldextract
import re
import nltk
import whois
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)


# Load models, vectorizers, and encoders
model_text = load(r'path to model text ')
vectorizer_text = load(r'path to vectorizer text')
label_encoder_text = load(r'path to label text ')

################################## URL  ##################################

model_url = load(r'path to model url')
vectorizer_url = load(r'path to vectorizer url')
encoder_url = load(r'path to encoder url')
svd = load(r'path to svd url')

################################## Text ##################################

def preprocess_email(email):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    words = tokenizer.tokenize(email.lower())
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

################################## URL  ##################################

def length_of_url(url):
    return 'trust' if len(url) < 54 else 'untrust'

def having_at_symbol(url):
    return 'untrust' if "@" in url else 'trust'

def double_slash_redirection(url):
    parts = urlsplit(url)
    return 'untrust' if "//" in parts.path else 'trust'

def sub_domains(url):
    domain_parts = tldextract.extract(url).subdomain.split('.')
    return 'trust' if len(domain_parts) <= 2 else 'untrust'

def domain_registered(url):
    domain_name = urlparse(url).netloc
    try:
        domain_info = whois.whois(domain_name)
        return 'trust' if domain_info.domain_name else 'untrust'
    except Exception:
        return 'untrust'

def extract_text_from_url(url):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    words = tokenizer.tokenize(url.lower())
    return ' '.join(words)

def preprocess_url(url):
    length_of_url_value = length_of_url(url)
    having_at_symbol_value = having_at_symbol(url)
    double_slash_redirection_value = double_slash_redirection(url)
    sub_domains_value = sub_domains(url)
    text_sent_value = extract_text_from_url(url)

    encoded_features = encoder_url.transform([[length_of_url_value, having_at_symbol_value, double_slash_redirection_value, sub_domains_value]])
    X_text_sent = vectorizer_url.transform([text_sent_value])
    X_text_sent_reduced = svd.transform(X_text_sent)
    
    # Ensure that the arrays being concatenated have compatible shapes
    encoded_features = encoded_features.toarray() if not isinstance(encoded_features, np.ndarray) else encoded_features
    X_text_sent_reduced = X_text_sent_reduced if not isinstance(X_text_sent_reduced, np.ndarray) else X_text_sent_reduced

    X_final = np.hstack((X_text_sent_reduced, encoded_features))

    return X_final

def predict_url(url):
    X_final = preprocess_url(url)
    prediction = model_url.predict(X_final)
    return prediction[0]

################################## Merge ##################################

def classify_email(email):
    # Extract URLs
    urls = re.findall(r'(https?://[^\s]+)', email)
    urls = [re.sub(r'^https?://', '', url) for url in urls]  # Remove http:// or https://

    # Classify the email text
    email_text = re.sub(r'(https?://[^\s]+)', '', email)  # Remove URLs from email text
    text_vectorized = vectorizer_text.transform([preprocess_email(email_text)])
    predicted_class_index = model_text.predict(text_vectorized)[0]
    result_text = label_encoder_text.classes_[predicted_class_index]

    # Classify each URL
    url_results = [predict_url(url) for url in urls]
    
    if result_text and url_results:
        # Determine the final verdict
        if result_text == "Safe Email" and all(result == "good" for result in url_results):
            verdict = "Legitimate Email"
        elif result_text == "Safe Email" and any(result == "bad" for result in url_results):
            verdict = "Suspicious Email"
        elif result_text == "Phishing Email" and all(result == "good" for result in url_results):
            verdict = "Suspicious Email"
        else:
            verdict = "Malicious Email"
         
    elif not url_results:
        verdict = result_text
   
    return verdict

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    user_input = ""
    if request.method == 'POST':
        input_type = request.form['input_type']
        user_input = request.form['user_input']
        
        if input_type == 'email':
            result = classify_email(user_input)
        elif input_type == 'url':
            result = predict_url(user_input)
        elif input_type == 'both':
            result = classify_email(user_input)

    return render_template('index.html', result=result, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
