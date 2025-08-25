import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv(r"C:\Users\anjoj\Downloads\Dataset.csv")

data = data.dropna(subset=['Cuisines'])

data = data.fillna(method='ffill')

label_encoders = {}
for col in data.columns:
    if data[col].dtype == 'object' and col != 'Cuisines':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

cuisine_encoder = LabelEncoder()
data['Cuisines'] = cuisine_encoder.fit_transform(data['Cuisines'])

counts = data['Cuisines'].value_counts()
data = data[data['Cuisines'].isin(counts[counts >= 2].index)]

X = data.drop('Cuisines', axis=1)
y = data['Cuisines']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

report = classification_report(
    y_test, y_pred,
    labels=np.unique(np.concatenate((y_test, y_pred))),
    target_names=cuisine_encoder.inverse_transform(np.unique(np.concatenate((y_test, y_pred)))),
    output_dict=True
)
pd.DataFrame(report).transpose().to_csv('classification_report.csv')

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()


importances = model.feature_importances_
cols = X.columns
idxs = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[idxs])
plt.xticks(range(len(importances)), cols[idxs], rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("All set. Files saved:")
print(" - classification_report.csv")
print(" - confusion_matrix.png")
print(" - feature_importance.png")

