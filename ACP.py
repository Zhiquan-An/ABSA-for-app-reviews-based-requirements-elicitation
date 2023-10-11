
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import ast
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score


data_path = "./dataset/Games_text_representation.csv"
data = pd.read_csv(data_path, encoding='ISO-8859-1')
data['bert'] = data['bert'].apply(ast.literal_eval)


bert_cols = [f'bert_{i}' for i in range(len(data['bert'][0]))]
bert_df = pd.DataFrame(data['bert'].tolist(), columns=bert_cols)
X = bert_df.values
y = data['sentiment_id'].astype(int).values


base_models = [
    ('LR', LogisticRegression(max_iter=1000, C=1, penalty='l1', solver='liblinear')),
    ('MLP', MLPClassifier(hidden_layer_sizes=(50, 50), random_state=42, max_iter=1000)),
    ('SVM', SVC(probability=True, kernel='linear', random_state=42)),
    ('XGB', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=300, random_state=42, learning_rate=0.1, max_depth=10)),
    ('RF', RandomForestClassifier(n_estimators=300, random_state=42))
]

kf = KFold(n_splits=10, shuffle=True, random_state=42)
meta_X = np.zeros((len(X), 1)) 

model_f1_scores = {name: [] for name, _ in base_models}

smote = SMOTE(random_state=42)


for i, (name, model) in enumerate(base_models):
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

 
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]  

        f1 = f1_score(y_val, y_pred, average='macro')
        model_f1_scores[name].append(f1)

   
        if i == 0:
            meta_X[val_index, 0] = f1 * y_pred_proba
        else:
            meta_X[val_index, 0] += f1 * y_pred_proba

    avg_f1 = np.mean(model_f1_scores[name])
    print(f"{name} Model Average F1 Score across 10 Folds: {avg_f1:.4f}")
    print("-" * 40)


meta_X /= len(base_models)

accuracy_scores_meta = []  

for train_index, val_index in kf.split(meta_X):
    X_train_meta, X_val_meta = meta_X[train_index], meta_X[val_index]
    y_train_meta, y_val_meta = y[train_index], y[val_index]


    meta_model = RandomForestClassifier()
    meta_model.fit(X_train_meta, y_train_meta)
    y_pred_meta = meta_model.predict(X_val_meta)

    accuracy_meta = accuracy_score(y_val_meta, y_pred_meta)  
    accuracy_scores_meta.append(accuracy_meta)


print(f"Meta Model Average Accuracy: {np.mean(accuracy_scores_meta):.4f}")
print(f"Best Meta Model Accuracy: {max(accuracy_scores_meta):.4f}")
