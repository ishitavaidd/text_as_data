import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

DATA_DIR = Path("data")

with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]
X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

vectorizer = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.9)
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

clf = LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))

nonzero = np.count_nonzero(clf.coef_[0])
print(f"Non-zero coefficients: {nonzero} / {len(clf.coef_[0])}")

features = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]
top_pos = np.argsort(coefs)[-15:][::-1]
top_neg = np.argsort(coefs)[:15]

print("\nTop 15 positive words (CORE):")
for i in top_pos:
    print(f"  {features[i]:20s} {coefs[i]:.4f}")

print("\nTop 15 negative words (NEG):")
for i in top_neg:
    print(f"  {features[i]:20s} {coefs[i]:.4f}")