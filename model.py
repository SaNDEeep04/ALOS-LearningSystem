import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Step 1: Load the dataset
data = pd.read_csv("D:\\Final Year Project\\Datasets\\student-por.csv")
print("Dataset loaded successfully!")

# Step 2: Preprocess the data
# Define the columns to be used
columns_to_use = ['G1', 'G2', 'studytime', 'failures', 'health', 'age', 'absences', 'sex']
target_column = 'G3'

# Handle categorical data
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])  # Encoding 'sex' column to numeric values

# Normalize numerical columns
numerical_columns = ['G1', 'G2', 'studytime', 'failures', 'health', 'age', 'absences','sex']
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 3: Define target and features
X = data[columns_to_use]  # Features
y = data[target_column]  # Target

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and test sets!")

# Step 5: Model Selection and Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("Model training completed!")

# Step 6: Save the Model, Scaler, and Label Encoder
joblib.dump(model, "student_performance_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model, scaler, and label encoder saved!")
