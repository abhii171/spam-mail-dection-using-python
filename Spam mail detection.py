import nltk
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics


nltk.download('stopwords')
from nltk.corpus import stopwords


def preprocess_text(text):
    
    text = text.lower()
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    text = re.sub(r'\d+', '', text)
    
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


data = [
    ("Win money now!!!", "spam"),
    ("Hi, how are you?", "ham"),
    ("Limited time offer!!!", "spam"),
    ("Meeting at 10am tomorrow", "ham"),
    ("Claim your free prize", "spam"),
    ("Are you available for a call?", "ham")
]


texts, labels = zip(*data)


texts = [preprocess_text(text) for text in texts]


X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

model = Pipeline([
    ('vect', CountVectorizer()), 
    ('clf', MultinomialNB()),]  )   


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


def predict_spam(email):
    processed_email = preprocess_text(email)
    prediction = model.predict([processed_email])
    return prediction[0]


user_email = input("Please enter the email message to classify: ")
classification = predict_spam(user_email)
print(f'The email is classified as: {classification}')
