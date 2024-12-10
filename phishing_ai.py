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

################################## Text ##################################

def preprocess_email(email):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    words = tokenizer.tokenize(email.lower())
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

# Load models, vectorizers, and encoders
model_text = load(r'C:\Users\owner\Desktop\phishing\presentation\text\text_model.joblib')
vectorizer_text = load(r'C:\Users\owner\Desktop\phishing\presentation\text\vectorizer_text.joblib')
label_encoder_text = load(r'C:\Users\owner\Desktop\phishing\presentation\text\label_encoder_text.joblib')

################################## URL  ##################################

model_url = load(r'C:\Users\owner\Desktop\phishing\presentation\url\url_model.joblib')
vectorizer_url = load(r'C:\Users\owner\Desktop\phishing\presentation\url\vectorizer_url.joblib')
encoder_url = load(r'C:\Users\owner\Desktop\phishing\presentation\url\encoder_url.joblib')
svd = load(r'C:\Users\owner\Desktop\phishing\presentation\url\svd_model.joblib')

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
    
    if result_text and url_results :
        # Determine the final verdict
        if result_text == "Safe Email" and all(result == "good" for result in url_results):
            print(result_text,url_results)
            verdict = "Legitimate Email"
        elif result_text == "Safe Email" and any(result == "bad" for result in url_results):
            print(result_text,url_results)
            verdict = "Suspicious Email"
        elif result_text == "Phishing Email" and all(result == "good" for result in url_results):
            print(result_text,url_results)
            verdict = "Suspicious Email"
        else:
            print(result_text,url_results)
            verdict = "Malicious Email"
         
    elif not  url_results:
        print("Only email")
        verdict = result_text
   
    
    
    return verdict

# Example email
email_text = """
Dear Hassan,

We regret to inform you that there has been a security breach detected on your account. Our system detected unauthorized access attempts from multiple locations, indicating a potential compromise of your account security.

To safeguard your account and prevent any further unauthorized access, we urge you to take immediate action by following the steps below:

Click on the following link to verify your account
Enter your username and password to confirm your identity.
Update your account information to secure it against future attacks.
Please note that failure to take action within the next 24 hours may result in permanent suspension of your account. We value your security and privacy and are committed to resolving this issue promptly.

If you have any questions or concerns, please do not hesitate to contact our support team at support@update-yourdetails.com.

Thank you for your cooperation in maintaining the security of our platform.

Sincerely,
Francis
It/support Global Sec Inc
"""

# Classify the email
verdict = classify_email(email_text)
print(f"Verdict for the email: {verdict}")
