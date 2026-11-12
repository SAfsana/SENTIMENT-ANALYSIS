import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, f1_score)

# ---- EDIT THESE to match your dataset ----
CSV = "zomato_reviews.csv"
TEXT_COL = "Review"
RATING_COL = "Rating"
# ------------------------------------------

df = pd.read_csv(CSV)
df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
df = df.dropna(subset=[TEXT_COL, RATING_COL])

def label(r):
    if r <= 2.0: return "Negative"
    if r <= 3.5: return "Neutral"
    return "Positive"

def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

df["label"] = df[RATING_COL].apply(label)
df["clean"] = df[TEXT_COL].apply(clean)
print(df["label"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
print(f"train/test: {len(X_train)} / {len(X_test)}")

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Multinomial NB": MultinomialNB(),
}
classes = ["Negative", "Neutral", "Positive"]
results, preds = [], {}
for name, m in models.items():
    m.fit(Xtr, y_train)
    p = m.predict(Xte)
    preds[name] = p
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, p, average="macro")
    results.append([name, accuracy_score(y_test, p), pr, rc, f1])
    print(f"{name}: per-class F1 =", dict(zip(classes, f1_score(y_test, p, labels=classes, average=None).round(2))))

res = pd.DataFrame(results, columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1"])
print((res.set_index("Algorithm") * 100).round(2))

best = res.sort_values("F1", ascending=False).iloc[0]["Algorithm"]
print("\nBest model:", best)
print(pd.DataFrame(confusion_matrix(y_test, preds[best], labels=classes),
                   index=classes, columns=classes))

# ---- Final output: predict new comments ----
best_model = models[best]
def predict(texts):
    return best_model.predict(tfidf.transform([clean(t) for t in texts]))

print(predict(["Food arrived hot and the rider was very polite",
               "It was okay, nothing special",
               "Cold food and very late delivery"]))
