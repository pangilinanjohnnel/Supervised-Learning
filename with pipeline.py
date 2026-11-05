import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics

df=pd.read_csv(r"archive\movie_review_train.csv")

X=df["text"]
y=df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Naive-Bayes
nb = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('nb', MultinomialNB())])
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

print("Naive-Bayes")
print(metrics.confusion_matrix(y_test,nb_pred))
print(metrics.classification_report(y_test,nb_pred))

# LinearSVC
svm = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('svc', LinearSVC())])
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

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

np = nb.predict([myreview, myreview2, myreview3, myreview4, myreview5, myreview6])
sp = svm.predict([myreview, myreview2, myreview3, myreview4, myreview5, myreview6])

#result comparison 0 for neg and 1 for pos
print(f"Pipeline NB: {np}")
print(f"Pipeline SVM: {sp}")
