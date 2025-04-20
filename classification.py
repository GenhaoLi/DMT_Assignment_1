#%%
import pandas as pd
#%%
df = pd.read_csv('cleaned.csv')
df.columns = df.columns.str.strip()
#%% md
# ## Target: What programme are you in?
# 
# Predict which program the student is in, based on the 4 courses and use of chatgpt
# 
# because the corr in the heatmap is high (not now though), and in reality courses that student take can indeed characterize the program they are in
# 
#%%
target_col = 'What programme are you in?'
feature_cols = ['Have you taken a course on machine learning?',
                'Have you taken a course on information retrieval?',
                'Have you taken a course on statistics?',
                'Have you taken a course on databases?',
                'I have used ChatGPT to help me with some of my study assignments']

# take a look at the target column and the features
def check_value_counts(df: pd.DataFrame, cols: list[str]):
    for col in cols:
        print(df[col].value_counts(), '\n')
check_value_counts(df, [*feature_cols, target_col])
#%% md
# Encoding the features and the target variable
#%%
for col in feature_cols:
    df[col] = df[col].map({'yes': 1, '1': 1,
                           'no': 0, '0': 0,
                           'unknown': 0.5, 'unkown': 0.5})
check_value_counts(df, feature_cols)
#%%
major_classes = ['computer science', 'artificial intelligence']
df[target_col] = df[target_col].apply(lambda x: x if x in major_classes else 'other')
df[target_col].value_counts()

#%%

from sklearn.preprocessing import LabelEncoder
le_program = LabelEncoder()
encoded_target_col = 'encoded_program'
df[encoded_target_col] = le_program.fit_transform(df[target_col])
check_value_counts(df, [encoded_target_col])
#%% md
# ## Train-test split
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[encoded_target_col], test_size=0.2, random_state=123)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#%%
set(y_train), set(y_test)
#%% md
# ## Apply two classification algorithms
# ### Random Forest
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=1,
)
rf = RandomForestClassifier(
    class_weight='balanced',
    max_depth=5,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=30,
    random_state=1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_rf_labels = le_program.inverse_transform(y_pred_rf)
pd.Series(y_pred_rf_labels).value_counts()
#%%
def print_metrics(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score (weighted):", f1_score(y_test, y_pred, average='weighted', zero_division=0))  # set `zero_division` to 0 to handle case when there are no true positives
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
#%%
print_metrics(y_test, y_pred_rf)
#%% md
# Optimize the hyperparameters, using `GridSearchCV`, could take 2 minutes
#%%
from sklearn.model_selection import GridSearchCV

rf_param_grid = {
    'n_estimators': list(range(10, 300, 20)),
    'max_depth': [None] + list(range(5, 30, 5)),
    'min_samples_split': list(range(1, 10, 2)),
    'min_samples_leaf': list(range(1, 5)),
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
}

# These were the best params in one of our searches, but it is somehow not found after we expand the param grid
# ANSWER: OVERFITTING
rf_param_best = {'class_weight': 'balanced', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 30} # Acc: 0.6122448979591837, F1: 0.6140872154332858
rf_param_grid = {key: [value] for key, value in rf_param_best.items()}
for k, v in rf_param_best.items():
    if v not in rf_param_grid[k]:
        print(f"param {k} not in grid search")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=1, class_weight='balanced'),
    rf_param_grid,
    scoring='f1_weighted',
    cv=10,
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)

print("best rf params:", rf_grid.best_params_)
y_pred_rf_optimized = rf_grid.predict(X_test)
print_metrics(y_test, y_pred_rf_optimized)
##%% md
#%% md
# ##%%
#%%
from sklearn.metrics import classification_report, accuracy_score, f1_score

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
# transform back to the original labels
y_pred_knn_labels = le_program.inverse_transform(y_pred_knn)
pd.Series(y_pred_knn_labels).value_counts()
##%%
#%%
##%% md
#%% md
# ##%%
#%%
    'n_neighbors': [3],
    # 'n_neighbors': list(range(1, 20, 2)),
    # 'weights': ['uniform', 'distance'],
    # 'p': [1, 2],
    'leaf_size': [30],
    # 'leaf_size': list(range(10, 50, 2)),
    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    # 'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_param_grid,
    scoring='f1_weighted',
    cv=10,
    n_jobs=-1,
    verbose=1
)
knn_grid.fit(X_train, y_train)

print("best knn params:", knn_grid.best_params_)
print("best knn score:", knn_grid.best_score_)
y_pred_knn_optimized = knn_grid.predict(X_test)
print_metrics(y_test, y_pred_knn_optimized)
##%% md
#%% md
# Metrics:
# - Accuracy: overall correctness
# - F1 Score (weighted): accounts for class imbalance, since class `other` is much less than the other two classes
# - `classification_report` shows precision, recall, F1 and support for each class
# ##%% md
#%% md
# ##%%
#%%
##%% md
#%% md
# ##%%
#%%
##%% md
#%% md
# ##%%
# results = {
#%%
    "Best Params": [rf_grid.best_params_, knn_grid.best_params_],
    "Accuracy": [accuracy_score(y_test, y_pred_rf_optimized), accuracy_score(y_test, y_pred_knn_optimized)],
    "F1 (weighted)": [f1_score(y_test, y_pred_rf_optimized, average='weighted'), f1_score(y_test, y_pred_knn_optimized, average='weighted')],
}

pd.DataFrame(results)
