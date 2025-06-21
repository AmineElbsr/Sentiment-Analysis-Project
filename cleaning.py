import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Charger un échantillon ou tout le fichier
df = pd.read_csv("yelp Restaurant Reviews.csv")  # ajuste si besoin

# Supprimer les lignes où Review Text ou Rating sont vides
df = df.dropna(subset=['Review Text', 'Rating'])

# Supprimer les doublons sur le texte des avis
df = df.drop_duplicates(subset='Review Text')

# Réinitialiser l’index
df = df.reset_index(drop=True)


# Fonction de nettoyage
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Mettre en minuscules
    text = re.sub(r'<.*?>', '', text)  # Supprimer les balises HTML
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Supprimer les liens
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Supprimer la ponctuation et chiffres
    tokens = word_tokenize(text)  # Tokenisation
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Supprimer les stopwords
    return " ".join(tokens)

# Appliquer à la colonne Review Text
df['Cleaned_Text'] = df['Review Text'].apply(clean_text)

# Afficher le résultat
print(df[['Review Text', 'Cleaned_Text']].head())

# Afficher un résumé
# print("Nombre de lignes après nettoyage :", len(df))
# print(df.head())

# Sauvegarder toutes les colonnes, y compris la colonne Cleaned_Text
df.to_csv("yelp_reviews_cleaned.csv", index=False)

# Sauvegarder uniquement les colonnes utiles
df[['Cleaned_Text', 'Rating']].to_csv("cleaned_reviews_simple.csv", index=False)

