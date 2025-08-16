import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

data = pd.read_csv('data/covtype.csv')
print(data.head())
print('Shape:', data.shape)
print(data.dtypes)
print('IS NOT NULL SUM', data.isna().sum())

X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']

# For XGBoost compatibility, convert the labels from 1-7 to 0-6
y_adjusted = y-1

# 80% data for training, 20% for testing
# stratify=y ensures classe proportions remain balanced in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X, y_adjusted, test_size=0.2, random_state=42, stratify=y_adjusted)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

#Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed metrics
print("Random Forest Classification Report:")
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
sorted_indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(10), importances[sorted_indices][:10])
plt.xticks(range(10), features[sorted_indices][:10], rotation=45, ha='right')
plt.title('Top 10 Feature Importances - Random Forest')
plt.show()

xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_clf.fit(X_train_xgb, y_train_xgb)

y_pred_xgb = xgb_clf.predict(X_test_xgb)

print("XGBoost Accuracy:", accuracy_score(y_test_xgb, y_pred_xgb))
print(classification_report(y_test_xgb, y_pred_xgb))

# XGBoost Confusion Matrix
cm_xgb = confusion_matrix(y_test_xgb, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
plt.show()

# XGBoost Feature Importance
importances_xgb = xgb_clf.feature_importances_
sorted_indices_xgb = importances_xgb.argsort()[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(10), importances_xgb[sorted_indices_xgb][:10])
plt.xticks(range(10), features[sorted_indices_xgb][:10], rotation=45, ha='right')
plt.title('Top 10 Feature Importances - XGBoost')
plt.tight_layout()
plt.show()

# Compare models
print(f"\nModel Comparison:")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"XGBoost Accuracy: {accuracy_score(y_test_xgb, y_pred_xgb):.4f}")