from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# les données
df = pd.read_csv("real_data.csv")

# Colonnes binaires des catégories
Y = df[['Food', 'Service', 'Value', 'Ambiance']]
X = df['Cleaned_Text']

ratios = [0.2, 0.25, 0.3]
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": LinearSVC()
}

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

for ratio in ratios:
    print(f"\n Ratio entraînement/test : {int((1 - ratio) * 100)}:{int(ratio * 100)}")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, random_state=42)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    from sklearn.metrics import multilabel_confusion_matrix

    for name, base_model in models.items():
        model = OneVsRestClassifier(base_model)
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n Modèle : {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=Y.columns, zero_division=0))

        #  Matrice de confusion pour chaque étiquette (Food, Service, etc.)
        mcm = multilabel_confusion_matrix(y_test, y_pred)

        # for idx, label in enumerate(Y.columns):
        #     cm = mcm[idx]
        #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Négatif", "Positif"])
        #     # plt.figure(figsize=(5, 4))
        #     disp.plot(cmap='Blues', values_format='d')
        #     plt.title(f"Matrice de confusion - {name} - {label}")
        #     plt.tight_layout()
        #     plt.show()
