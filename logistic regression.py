import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix,classification_report, roc_curve,auc,roc_auc_score

df=pd.read_csv(r"archive\Social_Network_Ads.csv")

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#SIMPLE MODEL
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
cr= classification_report(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("Simple LR")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)
print(f"ROC AUC:{roc:.4f}")

#ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

#AUC
roc_auc = auc(fpr, tpr)

#plot ROC
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#LR W/ L1
model = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
cr= classification_report(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("LR with L1")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)
print(f"ROC AUC:{roc:.4f}")

#ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

#AUC
roc_auc = auc(fpr, tpr)

#plot ROC
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


#LR W/ L2
model = LogisticRegressionCV(cv=5, penalty='l2', solver='liblinear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
cr= classification_report(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("LR with L2")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)
print(f"ROC AUC:{roc:.4f}")

#ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

#AUC
roc_auc = auc(fpr, tpr)

#plot ROC
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

