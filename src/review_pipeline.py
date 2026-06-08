import re
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from snownlp import SnowNLP
except Exception:
    SnowNLP = None


def clean_review(text: str) -> str:
    text = re.sub(r"http\S+", "", str(text))
    return re.sub(r"\s+", " ", text).strip()


def sentiment_score(text: str) -> float:
    if SnowNLP is None:
        positive = sum(w in text for w in ["甜", "新鲜", "好吃", "满意"])
        negative = sum(w in text for w in ["坏", "贵", "烂", "慢"])
        return (positive + 1) / (positive + negative + 2)
    return float(SnowNLP(text).sentiments)


def top_tfidf_terms(df: pd.DataFrame, top_k=30) -> pd.DataFrame:
    docs = [" ".join(jieba.lcut(clean_review(t))) for t in df["review"]]
    vec = TfidfVectorizer(max_features=500)
    mat = vec.fit_transform(docs)
    scores = mat.mean(axis=0).A1
    return pd.DataFrame({"term": vec.get_feature_names_out(), "tfidf": scores}).sort_values("tfidf", ascending=False).head(top_k)
