import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("spam.csv", encoding="latin-1")
print("Dataset Columns:")
print(data.columns)
print("\nFirst 5 rows:")
print(data.head())

data = data[['Label', 'EmailText']]
data.columns = ['label', 'message']

data['label'] = data['label'].str.lower()


data['label'] = data['label'].map({
    'spam': 1,
    'ham': 0
})

data.dropna(inplace=True)
print("\nLabel Distribution After Cleaning:")
print(data['label'].value_counts())

X = data['message']
y = data['label']

vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print("\nNaive Bayes Model Trained Successfully!")
















