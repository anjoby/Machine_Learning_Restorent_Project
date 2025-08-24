import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

csv_file = "D:\ML INTERN\Dataset .csv"

restaurants = pd.read_csv(csv_file)

restaurants['Cuisines'] = restaurants['Cuisines'].fillna('Unknown')


combined_info = []
for i in range(len(restaurants)):
    cuisine = str(restaurants.loc[i, 'Cuisines'])
    city = str(restaurants.loc[i, 'City'])
    currency = str(restaurants.loc[i, 'Currency'])

    combo = cuisine + " " + city + " " + currency
    combined_info.append(combo)

restaurants['combined_info'] = combined_info

text_vectorizer = CountVectorizer(stop_words='english')
restaurant_vectors = text_vectorizer.fit_transform(restaurants['combined_info'])

user_fav_cuisine = "Japanese"
user_location = "Makati City"
user_money_type = "Botswana Pula(P)"

user_query = user_fav_cuisine + " " + user_location + " " + user_money_type
user_vector = text_vectorizer.transform([user_query])

similarities = cosine_similarity(user_vector, restaurant_vectors).flatten()

top_indexes = similarities.argsort()[-5:][::-1]

top_matches = restaurants.iloc[top_indexes]

final_columns = ['Restaurant Name', 'Cuisines', 'City', 'Average Cost for two', 'Aggregate rating']
recommended = top_matches[final_columns].reset_index(drop=True)


print("\nTop 5 Recommended Restaurants Based on What You Like:\n")
print(recommended.to_string(index=True))  # keeps the row numbers just like your screenshot
