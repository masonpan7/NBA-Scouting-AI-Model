from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
import pickle

import numpy as np
import pandas as pd

# Load the dataset
from Data_Preprocessing import df_grouped

# Create a copy of the data table to work with in this file
df_model = df_grouped.copy()

# Shift the 'breakout' column to create the 'breakout_next_season' column
df_model['breakout_next_season'] = df_model.groupby('Player')['breakout'].shift(-1)

# Drop rows where 'breakout_next_season' is NaN
df_model = df_model.dropna(subset=['breakout_next_season'])

# Convert 'breakout_next_season' to integer
df_model['breakout_next_season'] = df_model['breakout_next_season'].astype(int)

# Split the data into features and target
X = df_model.drop(columns=['breakout_next_season', 'Player', 'Pos', 'PER', 'BLK', 'BLK/G'])
y = df_model['breakout_next_season']

features = X.columns

columns = [col for col in df_model.columns if col != 'breakout_next_season'] + ['breakout_next_season']
df_model = df_model[columns]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)  # Convert back to DataFrame
X_test = pd.DataFrame(scaler.transform(X_test), columns=features)

weighted_features = ['MP/G_change', 'USG%_change', 'PTS/G_change', 'WS/48_change', 'G', 'BPM']

for feature in weighted_features:
    X_train[feature] *= 2  # Scale up high-importance features
    X_test[feature] *= 2

# Train Random Forest Model
model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.01,
    scale_pos_weight=10,
    min_child_weight=4,
    reg_alpha=0.005,
    random_state=100
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# Predict on test data
df_model['predicted_breakout'] = model.predict_proba(scaler.transform(X))[:,1]

shifted_probs = ((df_model['predicted_breakout'] - 0.615) / (1 - 0.615)) * 0.5 + 0.5

# Clip the probabilities to ensure they stay within [0, 1] range
shifted_probs = np.clip(shifted_probs, 0, 1)

# Add the shifted probabilities to the dataframe
df_model['shifted_predicted_breakout'] = shifted_probs

print(df_model.head(20))

print(len(df_model[df_model['predicted_breakout'] == 1]))
# Filter rows where breakout_next_season is 1 and predicted_breakout is less than 0.4
filtered_rows = df_model[(df_model['breakout_next_season'] == 1) & (df_model['shifted_predicted_breakout'] < 0.4)]
filtered_rows1 = df_model[(df_model['breakout_next_season'] == 1) & (df_model['shifted_predicted_breakout'] > 0.4)]

filtered_row2 = df_model[(df_model['breakout_next_season'] == 0) & (df_model['shifted_predicted_breakout'] > 0.6)]
filtered_rows3 = df_model[(df_model['breakout_next_season'] == 0) & (df_model['shifted_predicted_breakout'] < 0.6)]

# Print the number of such rows
print(f"Number of rows where breakout_next_season is 1 but predicted_breakout < 0.4: {len(filtered_rows)}")
print(f"Number of rows where breakout_next_season is 1 but predicted_breakout > 0.4: {len(filtered_rows1)}")

# Print the number of such rows
print(f"Number of rows where breakout_next_season is 0 but predicted_breakout > 0.6: {len(filtered_row2)}")
print(f"Number of rows where breakout_next_season is 0 but predicted_breakout < 0.6: {len(filtered_rows3)}")

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Find the threshold that maximizes the F1-score
f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)  # Avoid division by zero
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"Optimal Threshold: {best_threshold:.3f}")

y_pred_adjusted = (y_probs >= 0.615).astype(int)

# Evaluate new performance
new_accuracy = accuracy_score(y_test, y_pred_adjusted)
new_precision = precision_score(y_test, y_pred_adjusted)
new_recall = recall_score(y_test, y_pred_adjusted)
new_f1 = f1_score(y_test, y_pred_adjusted)

print(f"New Accuracy: {new_accuracy:.4f}")
print(f"New Precision: {new_precision:.4f}")
print(f"New Recall: {new_recall:.4f}")
print(f"New F1 Score: {new_f1:.4f}")


def tune_hyperparameters(X_train, y_train):
    # Define the model
    model = XGBClassifier(random_state=100)
    
    # Set up the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 250, 300],               # Number of trees
        'max_depth': [3, 4, 5, 6],                           # Maximum depth of trees
        'learning_rate': [0.01, 0.02, 0.05, 0.1],            # Step size shrinkage
        'scale_pos_weight': [1, 10, 20, 30],                  # To handle class imbalance
        'min_child_weight': [1, 3, 4, 5],                     # Minimum sum of instance weight
    }
    
    # Create custom scorer for F1 Score
    f1_scorer = make_scorer(f1_score)

    # Perform Grid Search for F1 Score
    grid_search_f1 = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=f1_scorer, n_jobs=-1)
    grid_search_f1.fit(X_train, y_train)

    # Get the best parameters and score for F1
    best_f1_params = grid_search_f1.best_params_
    best_f1_score = grid_search_f1.best_score_

    # Print out the result
    print("Best parameters for F1 Score:", best_f1_params)
    print("Best F1 Score:", best_f1_score)
    
    # Return best parameters
    return best_f1_params

# Example usage (assuming X_train, y_train are your data):
# best_f1_params = tune_hyperparameters(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, file)
