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
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_rf_labels = le_program.inverse_transform(y_pred_rf)
pd.Series(y_pred_rf_labels).value_counts()
#%% md
# We adjusted the hyperparameters a bit to avoid the case where no `other` class is predicted, which could also cause insufficient data for `classification_report`
# 
#%%
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1 Score (macro):", f1_score(y_test, y_pred_rf, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
#%% md
# ### K-Nearest Neighbors
#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
# transform back to the original labels
y_pred_knn_labels = le_program.inverse_transform(y_pred_knn)
pd.Series(y_pred_knn_labels).value_counts()
#%% md
# ## Performance
# Metrics:
# - Accuracy: overall correctness
# - F1 Score (weighted): accounts for class imbalance, since class `other` is much less than the other two classes
# - `classification_report` shows precision, recall, F1 and support for each class
#%%
def print_metrics(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score (macro):", f1_score(y_test, y_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
#%% md
# ### Random Forest Performance
#%%
print_metrics(y_test, y_pred_rf)
#%% md
# ### K-Nearest Neighbors Performance
#%%
print_metrics(y_test, y_pred_knn)
#%% md
# TODO: optimization of the hyperparameters, using `GridSearchCV` or manually?