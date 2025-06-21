import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Charger les données
df = pd.read_csv("real_data.csv")

# print("Nombre de lignes après nettoyage :", len(df))


X = df['Cleaned_Text']
y = df['Sentiment']

# Définir les ratios
ratios = [0.2, 0.25, 0.3]
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": LinearSVC()
}

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5)

for ratio in ratios:
    print(f"\n📊 Ratio entraînement/test : {int((1 - ratio) * 100)}:{int(ratio * 100)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("Nombre de lignes après nettoyage :", len(X_train),len(X_test) )

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        print(f"\n🔎 Modèle : {name}")
        print(classification_report(y_test, y_pred))
