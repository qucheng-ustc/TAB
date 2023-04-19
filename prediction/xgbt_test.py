from xgboost import XGBClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logisitic')
bst.fit(X_train, y_train)
preds = bst.predict(X_test)

print(y_test)
print(preds)
print(accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
