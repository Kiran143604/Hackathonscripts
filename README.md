# Hackathonscripts
Python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Simulate dataset based on provided criteria
np.random.seed(42)
n_samples = 1000

data = {
    'auditor_location_distance': np.random.randint(1, 50, n_samples),  # km from store
    'auditor_availability': np.random.choice([0, 1], n_samples),  # 0=busy, 1=available
    'auditor_skill_level': np.random.randint(1, 5, n_samples),  # 1-4 scale
    'store_risk_score': np.random.randint(1, 10, n_samples),  # 1-10 risk
    'audit_priority': np.random.randint(1, 5, n_samples),  # 1-4 priority
    'traffic_condition': np.random.choice([0, 1], n_samples),  # 0=normal, 1=heavy
    'weather_condition': np.random.choice([0, 1], n_samples),  # 0=clear, 1=bad
    'audit_deadline_hours': np.random.randint(1, 24, n_samples),  # hours left
}

# Target: reassignment needed (1=yes, 0=no)
target = []
for i in range(n_samples):
    score = 0
    if data['auditor_availability'][i] == 0:
        score += 2
    if data['auditor_location_distance'][i] > 30:
        score += 1
    if data['store_risk_score'][i] > 7:
        score += 1
    if data['audit_priority'][i] > 3:
        score += 1
    if data['traffic_condition'][i] == 1 or data['weather_condition'][i] == 1:
        score += 1
    if data['audit_deadline_hours'][i] < 4:
        score += 1
    target.append(1 if score >= 3 else 0)

df = pd.DataFrame(data)
df['reassignment_needed'] = target

# 2. Split data
X = df.drop('reassignment_needed', axis=1)
y = df['reassignment_needed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Training Complete")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:", conf_matrix)
print("Classification Report:", report)

# 5. Example prediction
example = pd.DataFrame({
    'auditor_location_distance': [35],
    'auditor_availability': [0],
    'auditor_skill_level': [2],
    'store_risk_score': [9],
    'audit_priority': [4],
    'traffic_condition': [1],
    'weather_condition': [0],
    'audit_deadline_hours': [3]
})
prediction = model.predict(example)[0]
print("Example Input:", example)
print(f"Predicted Reassignment Needed: {'Yes' if prediction == 1 else 'No'}")
# Take user input for prediction
print("\nEnter details for prediction:")
auditor_location_distance = int(input("Auditor location distance (km): "))
auditor_availability = int(input("Auditor availability (0=busy, 1=available): "))
auditor_skill_level = int(input("Auditor skill level (1-4): "))
store_risk_score = int(input("Store risk score (1-10): "))
audit_priority = int(input("Audit priority (1-4): "))
traffic_condition = int(input("Traffic condition (0=normal, 1=heavy): "))
weather_condition = int(input("Weather condition (0=clear, 1=bad): "))
audit_deadline_hours = int(input("Audit deadline hours remaining: "))

# Create DataFrame for prediction
user_input = pd.DataFrame({
    'auditor_location_distance': [auditor_location_distance],
    'auditor_availability': [auditor_availability],
    'auditor_skill_level': [auditor_skill_level],
    'store_risk_score': [store_risk_score],
    'audit_priority': [audit_priority],
    'traffic_condition': [traffic_condition],
    'weather_condition': [weather_condition],
    'audit_deadline_hours': [audit_deadline_hours]
})

# Predict
prediction = model.predict(user_input)[0]
print("\nPrediction Result:")
print(f"Reassignment Needed: {'Yes' if prediction == 1 else 'No'}")
