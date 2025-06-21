import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('stopwords')

# le fichier
df = pd.read_csv("cleaned2.csv")  


# Remove non-numeric values like 'Like'
df = df[pd.to_numeric(df['Rating'], errors='coerce').notnull()]

# Convert to float
df['Rating'] = df['Rating'].astype(float)

# Keep only ratings 1, 2, 4, or 5
df = df[df['Rating'].isin([1.0, 2.0, 4.0, 5.0])]

# Create sentiment label
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

print(df[['Cleaned_Text', 'Rating', 'Sentiment']].head())
# Sauvegarder toutes les colonnes, y compris la colonne Cleaned_Text
df.to_csv("data3.csv", index=False)


# Créer la colonne label de sentiment

# # Supprimer les lignes où Review Text ou Rating sont vides
# df = df.dropna(subset=['Review', 'Rating'])

# # Supprimer les doublons sur le texte des avis
# df = df.drop_duplicates(subset='Review')

# # Réinitialiser l’index
# df = df.reset_index(drop=True)


# # Fonction de nettoyage
# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()  # Mettre en minuscules
#     text = re.sub(r'<.*?>', '', text)  # Supprimer les balises HTML
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Supprimer les liens
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Supprimer la ponctuation et chiffres
#     tokens = word_tokenize(text)  # Tokenisation
#     tokens = [word for word in tokens if word not in stopwords.words('english')]  # Supprimer les stopwords
#     return " ".join(tokens)

# # Appliquer à la colonne Review Text
# df['Cleaned_Text'] = df['Review'].apply(clean_text)

# # Afficher le résultat
# print(df[['Review', 'Cleaned_Text']].head())

# # Afficher un résumé
# # print("Nombre de lignes après nettoyage :", len(df))
# # print(df.head())



# Sauvegarder uniquement les colonnes utiles
# df[['Cleaned_Text', 'Rating']].to_csv("cleaned2.csv", index=False)

