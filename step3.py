import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming your training data is stored in a DataFrame called df
# Make sure to preprocess your data appropriately before training the model
df = pd.read_csv('C:/Users/Barry/Documents/Uni/Projects/Object Tracking/Pose Estimation/Gunestimation/XGboost/level2/dataset/dataset.csv')

# Define features (X) and target variable (y)
X = df.drop(['label', 'image_name'], axis=1)  # Assuming 'label' is the column containing the target variable
y = df['label'].map({'gun': 0, 'nongun': 1})  # Convert labels to 0 and 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost classifier
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model
model.save_model("C:/Users/Barry/Documents/Uni/Projects/Object Tracking/Pose Estimation/Gunestimation/XGboost/level3/model_weights.xgb")