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

# calculate PD(Probaility of Default (부도 확률) (2년 내 심각한 연체가 날 확률)) 
y_prob = model.predict_proba(X_test)[:,1]
print(y_prob)

# calculate AUC  (0.7~ : good)(연체 고객을 정상 고객 보다 더 높은 확률로 잘 구분 하는지)
auc = roc_auc_score(y_test, y_prob)
print("Baseline ROC-AUC:", round(auc,4))

#PD : 연체 확률
LGD = 0.4 #연체 시 손실률: 40% 가정
EAD = 10_000_000 #평균 대출 금액 


eval_df = pd.DataFrame({
    "y_true": y_test.values,
    "PD": y_prob
})

# Expected Loss (EL = PD * LGD * EAD)
eval_df["EL"] = eval_df["PD"] * LGD * EAD

print("평균 PD:", round(eval_df["PD"].mean(),4))
print("평균 Expected Loss:", int(eval_df["EL"].mean()))
print("총 Expected Loss:", int(eval_df["EL"].sum()))


# cut off 설정: 대출 승인 기준을 어떻게 고정하면 얼마나 줄어들까?

cutoffs = [0.05, 0.07, 0.1, 0.15, 0.2]

results = []

for c in cutoffs:
    approved = eval_df["PD"] < c
    approval_rate = approved.mean()
    total_loss = eval_df.loc[approved, "EL"].sum()
    bad_rate = eval_df.loc[approved, "y_true"].mean()

    results.append([c, approval_rate, bad_rate, total_loss])

result_df = pd.DataFrame(results, 
                         columns=["Cutoff_PD", "Approval_Rate", "Observed_Bad_Rate", "Total_Expected_Loss"])

print(result_df)

