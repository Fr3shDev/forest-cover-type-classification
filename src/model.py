import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/covtype.csv')
print(data.head())
print('Shape:', data.shape)
print(data.dtypes)
print('IS NOT NULL SUM', data.isna().sum())

X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']

# 80% data for training, 20% for testing
# stratify=y ensures classe proportions remain balanced in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

#Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed metrics
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

importances = rf.feature_importances_
features = X.columns

# Sort features by importance
sortec_indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(10), importances[sortec_indices][:10])
plt.xticks(range(10), features[sortec_indices][:10], rotation=45, ha='right')
plt.title('Top 10 Feature Importances - Random Forest')
plt.show()