# import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# read dataset 
df = pd.read_csv("data/cs-training.csv")

# drop fisrt col if it's "ID"
if df.columns[0] != "SeriousDlqin2yrs":
    df = df.drop(columns=[df.columns[0]])

print("data:", df.shape)

# y = target, X = input feature
y = df["SeriousDlqin2yrs"]
X = df.drop(columns=["SeriousDlqin2yrs"])

# fill na to median 
X = X.fillna(X.median())

# train, test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# train model 
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# extract 
y_prob = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_prob)
print("Baseline ROC-AUC:", round(auc,4))