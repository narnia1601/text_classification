import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

df = pd.read_csv('./moviereviews2.tsv', sep='\t')
df = df.dropna()

blanks = []

for i,lb,rv in df.itertuples():
    if rv.isspace():
        blanks.append(i)

df.drop(blanks)

X = df['review']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=42)

clf = Pipeline([('tfidf',TfidfVectorizer()), ('clf',LinearSVC())])

clf.fit(X_train,y_train)

pred = clf.predict(X_test)

print(clf.predict(['This movie is good']))
print(accuracy_score(y_test,pred))