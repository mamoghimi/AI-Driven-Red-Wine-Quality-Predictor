import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the dataset
file_path = r'D:\Important Documents\Python file\Udemy Practic/winequality-red.csv'
data = pd.read_csv(file_path)

# Separate features and the target variable
X = data.drop('quality', axis=1)
y = data['quality']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plotting distribution of the target variable 'quality'
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=data, palette='viridis')
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Count')  
plt.grid(axis='y')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Wine Features')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_clf.fit(X_train, y_train)

# Predictions on the test set
y_pred = rf_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Initialize the PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)

# Create polynomial features from scaled data
X_poly = poly.fit_transform(X_scaled)

# Split the polynomial features into training and testing sets
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=40)

# Retrain the model with polynomial features
rf_clf_poly = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf_poly.fit(X_train_poly, y_train)

# Predictions on the test set with polynomial features
y_pred_poly = rf_clf_poly.predict(X_test_poly)

# Evaluate the model with polynomial features
accuracy_poly = accuracy_score(y_test, y_pred_poly)
class_report_poly = classification_report(y_test, y_pred_poly)

# Print accuracy and classification report for the model without polynomial features
print("Model without Polynomial Features:")
print("----------------------------------")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(class_report)

# Space to separate the outputs
print("\n" + "="*50 + "\n")

# Print accuracy and classification report for the model with polynomial features
print("Model with Polynomial Features:")
print("-------------------------------")
print(f"Accuracy with Polynomial Features: {accuracy_poly:.4f}")
print("Classification Report with Polynomial Features:")
print(class_report_poly)