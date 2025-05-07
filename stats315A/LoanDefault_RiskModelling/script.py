# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, f1_score

train_file_path = "loan-train.csv"
test_file_path = "loan-testx.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
numerical_features = train_df.select_dtypes(exclude=['object']).columns.tolist()

numerical_features.remove("default")

for df in [train_df, test_df]:
    df.fillna(df.median(numeric_only=True), inplace=True)
    df[categorical_features] = df[categorical_features].fillna('Unknown')

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

bootstrap_samples = 3000
bootstrap_indices = np.random.choice(train_df.index, size=bootstrap_samples, replace=True)
train_bootstrap = train_df.iloc[bootstrap_indices]

X = train_bootstrap.drop(columns=['default'])
y = train_bootstrap['default']

X_train, X_test_split, y_train, y_test_split = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_split_processed = preprocessor.transform(X_test_split)
X_test_final_processed = preprocessor.transform(test_df)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)
adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

stacking_model = StackingClassifier(
    estimators=[
        ('log_reg', log_reg),
        ('rf', rf),
        ('dt', dt),
        ('svm', svm),
        ('adaboost', adaboost)
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    passthrough=True
)

log_reg.fit(X_train_processed, y_train)
rf.fit(X_train_processed, y_train)
dt.fit(X_train_processed, y_train)
svm.fit(X_train_processed, y_train)
adaboost.fit(X_train_processed, y_train)
xgb_model.fit(X_train_processed, y_train)
stacking_model.fit(X_train_processed, y_train)

stacking_probs = stacking_model.predict_proba(X_test_split_processed)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test_split_processed)[:, 1]


final_probs = (stacking_probs + xgb_probs) / 2

auc_score = roc_auc_score(y_test_split, final_probs)
log_loss_score = log_loss(y_test_split, final_probs)
f1 = f1_score(y_test_split, (final_probs > 0.5).astype(int))


fpr, tpr, _ = roc_curve(y_test_split, final_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Stacking Model (AUC = {auc_score:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Model (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stacking Model with Bootstrapping")
plt.legend()
plt.show()

stacking_test_probs = stacking_model.predict_proba(X_test_final_processed)[:, 1]
xgb_test_probs = xgb_model.predict_proba(X_test_final_processed)[:, 1]


final_test_probs = (stacking_test_probs + xgb_test_probs) / 2

final_output_file_path = "risk_scores_bootstrap.txt"
pd.DataFrame(final_test_probs).to_csv(final_output_file_path, index=False, header=False)


final_output_file_path

