# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset('titanic')

# Step 1: Raw data performance (with minimal preprocessing)
raw_df = df.copy()
raw_df = raw_df[['survived', 'pclass', 'sex', 'age', 'fare']]
raw_df.dropna(inplace=True)
raw_df['sex'] = raw_df['sex'].map({'male': 0, 'female': 1})

X_raw = raw_df.drop('survived', axis=1)
y_raw = raw_df['survived']
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

model_raw = RandomForestClassifier(random_state=42)
model_raw.fit(X_train_raw, y_train_raw)
y_pred_raw = model_raw.predict(X_test_raw)

print("🔹 Raw Data Model Accuracy:", accuracy_score(y_test_raw, y_pred_raw))

# Step 2: Improved data quality
df_clean = df.copy()
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].median())
df_clean = df_clean[['survived', 'pclass', 'sex', 'age', 'fare']]
df_clean.dropna(inplace=True)
df_clean['sex'] = df_clean['sex'].map({'male': 0, 'female': 1})

X = df_clean.drop('survived', axis=1)
y = df_clean['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_clean = RandomForestClassifier(random_state=42)
model_clean.fit(X_train, y_train)
y_pred_clean = model_clean.predict(X_test)

print("\n🔹 Clean Data Model Accuracy:", accuracy_score(y_test, y_pred_clean))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred_clean))

# Optional: Accuracy comparison plot
acc_raw = accuracy_score(y_test_raw, y_pred_raw)
acc_clean = accuracy_score(y_test, y_pred_clean)

plt.bar(['Raw Data', 'Cleaned Data'], [acc_raw, acc_clean], color=['red', 'green'])
plt.title('Accuracy Before vs After Data Cleaning')
plt.ylabel('Accuracy')
plt.show()
