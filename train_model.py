import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
with open('spam.csv', encoding='ISO-8859-1') as f:
    df = pd.read_csv(f)

# 2. Keep only the necessary columns
df = df[['label', 'text']]

# Convert labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 4. Build the model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])

# 5. Train the model
model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save the model
with open('spam_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as spam_classifier.pkl")
