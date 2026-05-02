import os, re, time, requests, gc
import xml.etree.ElementTree as ET
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ── Config ─────────────────────────────────────────
RSS_CACHE_TTL = 600
SAMPLE_PER_CLASS = 3000
MAX_TFIDF_FEATURES = 2000

# ── Lebanese TV Sources ────────────────────────────
TV_SOURCES = [
    {"name": "MTV Lebanon", "name_ar": "MTV لبنان", "color": "#e63946", "rss": "https://www.mtv.com.lb/rss"},
    {"name": "Al Jadeed TV", "name_ar": "الجديد", "color": "#f4a261", "rss": "https://www.aljadeed.tv/rss/arabic/news"},
    {"name": "LBCI", "name_ar": "LBCI", "color": "#2a9d8f", "rss": "https://www.lbci.com/rss"},
    {"name": "NNA Lebanon", "name_ar": "الوكالة الوطنية", "color": "#457b9d", "rss": "https://www.nna-leb.gov.lb/ar/rss"},
    {"name": "OTV Lebanon", "name_ar": "OTV", "color": "#6a0572", "rss": "https://www.otv.com.lb/rss"},
    {"name": "Annahar", "name_ar": "النهار", "color": "#d4a017", "rss": "https://www.annahar.com/rss"},
    {"name": "Al Manar", "name_ar": "المنار", "color": "#264653", "rss": "https://www.almanar.com.lb/rss"},
]

# ── RSS Cache ──────────────────────────────────────
_rss_cache = {}

def fetch_rss(source):
    cached = _rss_cache.get(source["name"])
    if cached and (time.time() - cached["fetched_at"]) < RSS_CACHE_TTL:
        return cached["articles"]

    articles = []
    try:
        resp = requests.get(source["rss"], timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.content)

        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = (item.findtext("description") or "").strip()

            if title:
                articles.append({
                    "title": title,
                    "link": link,
                    "description": desc
                })

    except Exception as e:
        print("RSS error:", e)

    _rss_cache[source["name"]] = {
        "articles": articles,
        "fetched_at": time.time()
    }

    return articles


# ── ML GLOBAL STATE ────────────────────────────────
model = None
tfidf = None
records = []


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\u0600-\u06FF ]', '', text)
    return text.strip()


# ── TRAIN MODEL ───────────────────────────────────
def train():
    global model, tfidf, records

    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        fake_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "emineyetm/fake-news-detection-datasets",
            "News _dataset/Fake.csv"
        )
        fake_df["label"] = 1

        true_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "emineyetm/fake-news-detection-datasets",
            "News _dataset/True.csv"
        )
        true_df["label"] = 0

    except Exception as e:
        return False, str(e)

    fake_df = fake_df.sample(n=min(SAMPLE_PER_CLASS, len(fake_df)))
    true_df = true_df.sample(n=min(SAMPLE_PER_CLASS, len(true_df)))

    df = pd.concat([fake_df, true_df]).sample(frac=1)

    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)

    X_raw = df["text"]
    Y = df["label"]

    tfidf = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES)
    X = tfidf.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Build records
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    records = []
    for i in range(len(preds)):
        records.append({
            "id": i + 1,
            "title": str(df.iloc[i]["title"])[:120],
            "text": str(df.iloc[i]["text"])[:400],
            "actual": int(y_test.iloc[i]),
            "predicted": int(preds[i]),
            "fake_prob": round(probs[i][1] * 100, 1),
            "real_prob": round(probs[i][0] * 100, 1),
            "correct": bool(preds[i] == y_test.iloc[i])
        })

    acc = round(sum(r["correct"] for r in records) / len(records) * 100, 2)
    return True, {"total": len(records), "accuracy": acc}


# ── 🔥 NEW TV MATCH FUNCTION ──────────────────────
def match_sources(user_text):
    global tfidf

    if not user_text.strip() or tfidf is None:
        return []

    user_vec = tfidf.transform([clean_text(user_text)])
    results = []

    for source in TV_SOURCES:
        articles = fetch_rss(source)

        best_score = 0
        best_article = None

        for art in articles:
            combined = (art["title"] + " " + art["description"]).strip()
            if not combined:
                continue

            art_vec = tfidf.transform([clean_text(combined)])
            score = cosine_similarity(user_vec, art_vec)[0][0]

            if score > best_score:
                best_score = score
                best_article = art

        if best_score >= 0.20 and best_article:
            results.append({
                "source": source["name"],
                "source_ar": source["name_ar"],
                "color": source["color"],
                "title": best_article["title"],
                "link": best_article["link"],
                "score": round(best_score * 100, 1),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:3]


# ── ROUTES ───────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train_route():
    ok, result = train()
    return jsonify(result if ok else {"error": result})


@app.route("/predict_ml", methods=["POST"])
def predict_ml():
    if model is None or tfidf is None:
        return jsonify({"error": "Model not trained"})

    text = request.json.get("text", "")
    vec = tfidf.transform([clean_text(text)])

    pred = int(model.predict(vec)[0])
    probs = model.predict_proba(vec)[0]

    return jsonify({
        "verdict": "FAKE" if pred else "REAL",
        "fake_prob": round(probs[1] * 100, 1),
        "real_prob": round(probs[0] * 100, 1)
    })


@app.route("/predict_tv", methods=["POST"])
def predict_tv():
    text = request.json.get("text", "")
    matches = match_sources(text)

    if not matches:
        return jsonify({
            "tv_verdict": "NOT FOUND"
        })

    return jsonify({
        "tv_verdict": "FOUND",
        "best_source": matches[0]["source"],
        "best_link": matches[0]["link"],
        "tv_matches": matches
    })


@app.route("/sources")
def sources():
    return jsonify(TV_SOURCES)


# ── RUN ──────────────────────────────────────────
if __name__ == "__main__":
    print("Running FakeLens on http://localhost:5000")
    app.run(debug=True)
