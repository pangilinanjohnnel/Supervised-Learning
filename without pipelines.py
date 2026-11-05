import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics

df=pd.read_csv(r"archive\movie_review_train.csv")

X=df["text"]
y=df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive-Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_pred = nb.predict(X_test_vec)

print("Naive-Bayes")
print(metrics.confusion_matrix(y_test,nb_pred))
print(metrics.classification_report(y_test,nb_pred))

# LinearSVC
svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_pred = svm.predict(X_test_vec)

print("LinearSVC")
print(metrics.confusion_matrix(y_test,svm_pred))
print(metrics.classification_report(y_test,svm_pred))

#Basic review
myreview = "This movie is absolute cinema" #positive review
myreview2 = "This movie is absolutely terrible" #negative review
myreview3 = "..." #no comment
myreview4 = "It left me speechless" #can be taken either way
myreview5 = "Recommended 10 out of 10 the Good, the Bad and the Ugly " #negative words in title
myreview6 = "Give this director an Oscar for best way to ruin a franchise" #sarcasm

vectorized_text = vectorizer.transform([myreview, myreview2, myreview3, myreview4, myreview5, myreview6])

np = nb.predict(vectorized_text)
sp = svm.predict(vectorized_text)

#result comparison 0 for neg and 1 for pos
print(f"NB w/o pipeline: {np}")
print(f"SVM w/o pipeline: {sp}")
