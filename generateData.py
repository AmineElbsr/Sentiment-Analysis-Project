import pandas as pd

#  fichier CSV 
df = pd.read_csv("data3.csv")  

#  Définir les mots-clés pour chaque aspect 
# Liste d'exemples d'aliments
food_names = [
    'ice cream', 'cookies', 'pizza', 'burger', 'pasta', 'steak', 'chicken', 'sushi',
    'salad', 'noodles', 'fries', 'sandwich', 'cheese', 'cake', 'dessert', 'donut'
]

# Mise à jour des mots-clés pour Food
aspect_keywords = {
    'Food': ['food', 'tasty', 'delicious', 'steak', 'dish', 'meal', 'bland', 'cold', 'hot', 'menu'] + food_names,
    'Service': ['waiter', 'waitress', 'staff', 'service', 'friendly', 'rude', 'slow', 'quick', 'attentive'],
    'Value': ['price', 'expensive', 'cheap', 'value', 'worth', 'overpriced', 'cost'],
    'Ambiance': ['ambiance', 'atmosphere', 'music', 'lighting', 'decor', 'vibe', 'environment']
}


#  Fonction de détection des aspects 
def tag_aspects(text):
    text = str(text).lower()
    return {
        aspect: int(any(keyword in text for keyword in keywords))
        for aspect, keywords in aspect_keywords.items()
    }

#  Appliquer la détection à chaque ligne 
aspects_df = df['Cleaned_Text'].apply(tag_aspects).apply(pd.Series)

#  Fusionner avec le dataframe original 
df_annotated = pd.concat([df, aspects_df], axis=1)

#  Sauvegarder le fichier annoté 
df_annotated.to_csv("yelp_aspect_annotated.csv", index=False)

print("Fichier annoté sauvegardé : yelp_aspect_annotated.csv")
