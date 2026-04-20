import pandas as pd
import re
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = text.lower()
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    
    return " ".join(cleaned_words)

try:
    df = pd.read_csv('day1.csv')
    print("Done Load")
except FileNotFoundError:
    print("Fail not found ")


df_small = df.head(5000).copy()

print("\n Clear data ")

if 'review' in df_small.columns:
    df_small['cleaned_review'] = df_small['review'].apply(clean_text)
    
   
    print("\n After Clear ")
    print(df_small[['review', 'cleaned_review']].head())
    
else:
    print(" Column review not found")

print("df['review'].head(1000)")



def preprocess_text(text):
    text = text.lower()
    
    tokens = word_tokenize(text)
    
    cleaned_tokens = [
        stemmer.stem(w) 
        for w in tokens 
        if w not in stop_words and w.isalpha()
    ]
    
    return " ".join(cleaned_tokens)


df_small['processed_text'] = df_small['review'].apply(preprocess_text)

print(df_small[['review', 'processed_text']].head())



tfidf = TfidfVectorizer(max_features=1000)

X = tfidf.fit_transform(df_small['processed_text'])

print("\n--- Feature Extraction ---")
print(f" (Rows, Columns): {X.shape}")
print(" DONE , confirm text to num ")



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df_small['processed_text']) 
y = df_small['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Result ")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"(Accuracy): {accuracy * 100:.2f}%")

print("\nreport fav :")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('(Predicted)')
plt.ylabel('(Actual)')
plt.title('Confusion Matrix ')
plt.savefig('my_plot.png') 
print("my_plot.png")


metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

values = [0.85, 0.84, 0.86, 0.85] 

plt.figure(figsize=(10, 5))
plt.bar(metrics, values, color=['skyblue', 'orange', 'green', 'red'])
plt.ylim(0, 1)
plt.title('Performance Metrics Visualization')
plt.savefig('my_plot.png') 
print("my_plot.png")



from gensim.models import Word2Vec
import numpy as np

sentences = [text.split() for text in df_small['processed_text']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def get_sentence_vec(tokens):
    vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

X_emb = np.array([get_sentence_vec(s) for s in sentences])

X_train_emb, X_test_emb, y_train, y_test = train_test_split(X_emb, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf_emb = RandomForestClassifier()
clf_emb.fit(X_train_emb, y_train)

acc_tfidf = accuracy_score(y_test, y_pred) 
acc_emb = accuracy_score(y_test, clf_emb.predict(X_test_emb))

print("\n--- Feature Comparison ---")
print(f"TF-IDF Accuracy: {acc_tfidf:.4f}")
print(f"Word2Vec Embeddings Accuracy: {acc_emb:.4f}")

