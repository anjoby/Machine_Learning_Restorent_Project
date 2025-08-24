# 🛠️ First, import the basic tools (or "libraries") we'll need
import pandas as pd           # for reading and handling data
import matplotlib.pyplot as plt   # for making charts
import seaborn as sns         # prettier charts
import folium                 # for interactive maps
from folium.plugins import MarkerCluster  # to group map markers

# ------------------------------
# STEP 1: Load the restaurant data
# ------------------------------

# 📄 This is the file path to the data file on my computer
my_file = r"D:\ML INTERN\Dataset .csv"

# Let's read the CSV file using pandas
try:
    df = pd.read_csv(my_file)
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ File not found. Please check the path!")

# Quick look at the data - just to understand what it looks like
print("\nData Preview:")
print(df.head())

print("\nColumns in our dataset:")
print(df.columns)

# ------------------------------
# STEP 2: Plotting restaurants on a map
# ------------------------------

# 🧭 We'll center the map around the average latitude & longitude
average_lat = df["Latitude"].mean()
average_lon = df["Longitude"].mean()

# Create a map starting from this center point
my_map = folium.Map(location=[average_lat, average_lon], zoom_start=12)

# Group markers together so map doesn't get messy
marker_cluster = MarkerCluster().add_to(my_map)

# Add each restaurant as a blue marker
for i, row in df.iterrows():
    lat = row["Latitude"]
    lon = row["Longitude"]
    name = row["Restaurant Name"]

    folium.Marker(
        location=[lat, lon],
        popup=name,
        icon=folium.Icon(color="blue", icon="cutlery", prefix='fa')
    ).add_to(marker_cluster)

# Save the map to an HTML file so it can be opened in a browser
my_map.save("restaurant_map.html")
print("🗺️ Map saved to 'restaurant_map.html'")

# ------------------------------
# STEP 3: Where are the most restaurants?
# ------------------------------

# Count number of restaurants in each city
top_cities = df["City"].value_counts().head(10)

# Count number of restaurants in each locality
top_localities = df["Locality"].value_counts().head(10)

# 📊 Plotting both in one figure
plt.figure(figsize=(16, 6))

# Bar chart for cities
plt.subplot(1, 2, 1)
sns.barplot(x=top_cities.values, y=top_cities.index, palette="Blues_d")
plt.title("Top 10 Cities with Most Restaurants")
plt.xlabel("Number of Restaurants")
plt.ylabel("City")

# Bar chart for localities
plt.subplot(1, 2, 2)
sns.barplot(x=top_localities.values, y=top_localities.index, palette="Greens_d")
plt.title("Top 10 Localities with Most Restaurants")
plt.xlabel("Number of Restaurants")
plt.ylabel("Locality")

plt.tight_layout()
plt.show()

# ------------------------------
# STEP 4: Average ratings and cost per city
# ------------------------------

# Group data by city and calculate average rating, cost, and number of restaurants
city_stats = df.groupby("City").agg({
    "Aggregate rating": "mean",
    "Average Cost for two": "mean",
    "Restaurant ID": "count"    # we'll just count how many IDs per city
}).rename(columns={"Restaurant ID": "Restaurant Count"})

# Only show top 10 cities with the most restaurants
city_stats_sorted = city_stats.sort_values(by="Restaurant Count", ascending=False).head(10)

# Plot: Average Rating
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=city_stats_sorted["Aggregate rating"], y=city_stats_sorted.index, palette="coolwarm")
plt.title("Average Rating by City")
plt.xlabel("Rating")
plt.ylabel("City")

# Plot: Average Cost for Two
plt.subplot(1, 2, 2)
sns.barplot(x=city_stats_sorted["Average Cost for two"], y=city_stats_sorted.index, palette="PuBuGn_d")
plt.title("Average Cost for Two by City")
plt.xlabel("Cost")
plt.ylabel("City")

plt.tight_layout()
plt.show()

# ------------------------------
# STEP 5: Most popular cuisines in each city
# ------------------------------

# First, fill in missing values in the Cuisines column
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# Turn the comma-separated cuisines into a list
df["Cuisine List"] = df["Cuisines"].str.split(", ")

# Turn that list into separate rows (this is called "exploding")
df_expanded = df.explode("Cuisine List")

# Pick top 5 cities again (just to be safe)
top_cities_list = df["City"].value_counts().head(5).index

# For each city, show top 5 cuisines
for one_city in top_cities_list:
    city_data = df_expanded[df_expanded["City"] == one_city]
    top_cuisines = city_data["Cuisine List"].value_counts().head(5)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette="pastel")
    plt.title(f"Top 5 Cuisines in {one_city}")
    plt.xlabel("Number of Restaurants")
    plt.ylabel("Cuisine")
    plt.tight_layout()
    plt.show()
