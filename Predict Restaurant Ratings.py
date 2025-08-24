import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

file_path = r"D:\ML INTERN\Dataset .csv"
data = pd.read_csv(file_path)

columns_to_remove = [
    'Restaurant ID', 'Restaurant Name', 'Address', 'Locality',
    'Locality Verbose', 'Rating color', 'Rating text'
]
data.drop(columns=columns_to_remove, inplace=True)


data.fillna("Unknown", inplace=True)

label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

X = data.drop('Aggregate rating', axis=1)
y = data['Aggregate rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", round(mse, 2))
print("R-squared Score:", round(r2, 2))

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
important_features = feature_importance.sort_values(ascending=False).head(5)

print("\nTop 5 Important Features:")
print(important_features)
