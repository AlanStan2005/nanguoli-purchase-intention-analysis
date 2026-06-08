import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["quality_score", "price_acceptance", "brand_awareness", "channel_trust", "sentiment"]


def segment_consumers(df: pd.DataFrame, k=4) -> pd.DataFrame:
    x = StandardScaler().fit_transform(df[FEATURE_COLS])
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x)
    return df.assign(segment=labels)


def purchase_intention_model(df: pd.DataFrame):
    x = df[FEATURE_COLS]
    y = df["purchase_intent"]
    model = LogisticRegression(max_iter=1000)
    model.fit(x, y)
    prob = model.predict_proba(x)[:, 1]
    return model, {"auc": roc_auc_score(y, prob)}
