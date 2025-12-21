# -------------------------
# LOAN DEFAULT PREDICTION PROJECT
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1️⃣ Load dataset
data = pd.read_csv("loan.csv")

print("Columns:")
print(data.columns)
print("\nFirst 5 rows:")
print(data.head())

# 2️⃣ Identify column types
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
categorical_cols = data.select_dtypes(include='object').columns.tolist()

# Remove ID and Target
if 'LoanID' in categorical_cols:
    categorical_cols.remove('LoanID')
if 'Default' in numeric_cols:
    numeric_cols.remove('Default')

# 3️⃣ Handle missing values
for col in numeric_cols:
    data[col].fillna(data[col].median(), inplace=True)

for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# 4️⃣ Encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# 5️⃣ Split features and target
X = data.drop(['LoanID', 'Default'], axis=1)
y = data['Default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7️⃣ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Evaluate model
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

