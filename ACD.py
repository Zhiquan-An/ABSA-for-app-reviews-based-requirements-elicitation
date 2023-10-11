import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import ast

# Load and preprocess the data
file_path = 'Apps-Reviews-ABSA-main/Social_Networking_text_representation.csv'
df_1 = pd.read_csv(file_path)
df_1['fine_tuned_bert'] = [np.array(ast.literal_eval(x)) for x in df_1['fine_tuned_bert']]
#df_1['tfidf'] = [np.array(ast.literal_eval(x)) for x in df_1['tfidf']]
#df_1['w2v'] = [np.array(ast.literal_eval(x)) for x in df_1['w2v']]
#df_1['bert'] = [np.array(ast.literal_eval(x)) for x in df_1['bert']]
X = np.array(df_1["fine_tuned_bert"].tolist())
y = df_1['category_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base models
xgb_model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', enable_categorical=True)
rf_model = RandomForestClassifier()
svm_model = SVC()
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
mlp_model = MLPClassifier(max_iter=1000)

# Define parameter grids for each base model
param_grid_xgb = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20]
}

param_grid_svm = {
    'C': [1, 5, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

param_grid_logistic = {
    'C': [0.1, 1, 10],
    'max_iter':[500,1000,5000,10000]
}

param_grid_mlp = {
    'hidden_layer_sizes': [(100,), (100, 50), (200,)],
    'max_iter': [1000, 1500]
}

# Perform grid search for each base model
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, scoring='f1_micro')
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='f1_micro')
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, scoring='f1_micro')
grid_search_logistic = GridSearchCV(estimator=logistic_model, param_grid=param_grid_logistic, cv=5, scoring='f1_micro')
grid_search_mlp = GridSearchCV(estimator=mlp_model, param_grid=param_grid_mlp, cv=5, scoring='f1_micro')

# Fit grid search models for each base model
grid_search_xgb.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_svm.fit(X_train, y_train)
grid_search_logistic.fit(X_train, y_train)
grid_search_mlp.fit(X_train, y_train)

# Get best parameter values for each base model
best_params_xgb = grid_search_xgb.best_params_
best_params_rf = grid_search_rf.best_params_
best_params_svm = grid_search_svm.best_params_
best_params_logistic = grid_search_logistic.best_params_
best_params_mlp = grid_search_mlp.best_params_

print("Best Parameters - XGBoost:", best_params_xgb)
print("Best Parameters - RandomForest:", best_params_rf)
print("Best Parameters - SVM:", best_params_svm)
print("Best Parameters - LogisticRegression:", best_params_logistic)
print("Best Parameters - MLP:", best_params_mlp)

# Create base models with best parameters
xgb_model_best = xgb.XGBClassifier(**best_params_xgb, objective='multi:softmax', eval_metric='mlogloss', enable_categorical=True)
rf_model_best = RandomForestClassifier(**best_params_rf)
svm_model_best = SVC(**best_params_svm)
logistic_model_best = LogisticRegression(**best_params_logistic)
mlp_model_best = MLPClassifier(**best_params_mlp)

# Create the stacking model with base models using best parameters
stacking_model = StackingClassifier(estimators=[
    ('xgboost', xgb_model_best),
    ('random_forest', rf_model_best),
    ('svm', svm_model_best),
    ('logistic', logistic_model_best),
    ('mlp', mlp_model_best)
], final_estimator=LogisticRegression(logistic_model_best))

# Fit the stacking model to the training data
stacking_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = stacking_model.predict(X_test)

# Calculate the F1 score on the test data
f1_value = f1_score(y_test, y_pred, average='micro')
print("F1 Score:", f1_value)
