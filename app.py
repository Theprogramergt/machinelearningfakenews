import os, re, time, requests
import xml.etree.ElementTree as ET
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "dataset")
FAKE_CSV      = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_CSV      = os.path.join(DATASET_DIR, "True.csv")
RSS_CACHE_TTL = 600

# ── Lebanese / Arabic TV News Sources ─────────────────────────────────────────
TV_SOURCES = [
    {"name": "MTV Lebanon",  "name_ar": "MTV لبنان",        "color": "#e63946", "rss": "https://www.mtv.com.lb/rss"},
    {"name": "Al Jadeed TV", "name_ar": "الجديد",            "color": "#f4a261", "rss": "https://www.aljadeed.tv/rss/arabic/news"},
    {"name": "LBCI",         "name_ar": "LBCI",              "color": "#2a9d8f", "rss": "https://www.lbci.com/rss"},
    {"name": "NNA Lebanon",  "name_ar": "الوكالة الوطنية",   "color": "#457b9d", "rss": "https://www.nna-leb.gov.lb/ar/rss"},
    {"name": "OTV Lebanon",  "name_ar": "OTV",               "color": "#6a0572", "rss": "https://www.otv.com.lb/rss"},
    {"name": "Annahar",      "name_ar": "النهار",            "color": "#d4a017", "rss": "https://www.annahar.com/rss"},
    {"name": "Al Manar",     "name_ar": "المنار",            "color": "#264653", "rss": "https://www.almanar.com.lb/rss"},
]

# ── RSS Cache ─────────────────────────────────────────────────────────────────
_rss_cache = {}

def fetch_rss(source):
    cached = _rss_cache.get(source["name"])
    if cached and (time.time() - cached["fetched_at"]) < RSS_CACHE_TTL:
        return cached["articles"]
    articles = []
    try:
        resp = requests.get(source["rss"], timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            _rss_cache[source["name"]] = {"articles": [], "fetched_at": time.time()}
            return []
        root = ET.fromstring(resp.content)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            desc  = (item.findtext("description") or "").strip()
            if title:
                articles.append({"title": title, "link": link, "description": desc})
        if not articles:
            for entry in root.findall("atom:entry", ns):
                title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
                link_el = entry.find("atom:link", ns)
                link = link_el.get("href", "") if link_el is not None else ""
                if title:
                    articles.append({"title": title, "link": link, "description": ""})
    except Exception as e:
        print(f"  RSS error [{source['name']}]: {e}")
    _rss_cache[source["name"]] = {"articles": articles, "fetched_at": time.time()}
    return articles

def extract_keywords(text, min_len=3):
    words = re.findall(r'[\u0600-\u06FF]{' + str(min_len) + r',}|[a-zA-Z]{' + str(min_len) + r',}', text)
    stop_ar = {"من","إلى","على","في","عن","مع","هذا","هذه","التي","الذي",
                "كان","كانت","وقد","وقال","قال","لقد","أن","إن","ما","لا",
                "كما","ذلك","بعد","قبل","حيث","أو","ثم","أيضا","بين","حول"}
    return [w for w in words if w not in stop_ar]

def match_sources(user_text):
    keywords = extract_keywords(user_text, min_len=3)
    if len(keywords) < 2:
        return []
    kw_set = set(keywords[:20])
    matches = []
    for source in TV_SOURCES:
        articles = fetch_rss(source)
        best_score, best_article = 0, None
        for art in articles:
            combined  = (art["title"] + " " + art["description"]).strip()
            art_words = set(extract_keywords(combined, min_len=3))
            if not art_words:
                continue
            overlap = len(kw_set & art_words)
            score   = overlap / max(len(kw_set), 1)
            if score > best_score:
                best_score, best_article = score, art
        if best_score >= 0.25 and best_article:
            matches.append({
                "source":    source["name"],
                "source_ar": source["name_ar"],
                "color":     source["color"],
                "title":     best_article["title"],
                "link":      best_article["link"],
                "score":     round(best_score * 100, 1),
            })
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches

# ── Global ML state ───────────────────────────────────────────────────────────
model = None; tfidf = None; records = []

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'^[A-Z][A-Z,\s]+\([^)]+\)\s*[-–]\s*', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text.strip()

def download_dataset():
    """Download dataset from Kaggle if not already present."""
    if os.path.exists(FAKE_CSV) and os.path.exists(TRUE_CSV):
        return True, "Dataset already exists"
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        os.makedirs(DATASET_DIR, exist_ok=True)
        print("Downloading Fake.csv from Kaggle...")
        fake_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "emineyetm/fake-news-detection-datasets",
            "News _dataset/Fake.csv"
        )
        fake_df.to_csv(FAKE_CSV, index=False)
        print("Downloading True.csv from Kaggle...")
        true_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "emineyetm/fake-news-detection-datasets",
            "News _dataset/True.csv"
        )
        true_df.to_csv(TRUE_CSV, index=False)
        print("Dataset downloaded successfully!")
        return True, "Downloaded successfully"
    except Exception as e:
        return False, f"Failed to download dataset: {str(e)}"

def train():
    global model, tfidf, records

    # Download dataset from Kaggle if not present
    ok, msg = download_dataset()
    if not ok:
        return False, msg

    fake_df = pd.read_csv(FAKE_CSV); fake_df["label"] = 1
    true_df = pd.read_csv(TRUE_CSV); true_df["label"] = 0
    df = pd.concat([fake_df, true_df], ignore_index=True).sample(frac=1, random_state=42)
    df["text"] = (df.get("title", "") + " " + df.get("text", "")).apply(clean_text)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    X_raw = df["text"]; Y = df["label"]
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression(C=1.0, solver="liblinear")
    model.fit(X_train, y_train)
    train_idx = y_train.index.tolist()
    probs = model.predict_proba(X_train)
    preds = model.predict(X_train)
    records = []
    for i, idx in enumerate(train_idx):
        row = df.iloc[idx]
        records.append({
            "id": i+1, "title": str(row.get("title",""))[:120] or "—",
            "text": str(row.get("text",""))[:600],
            "actual": int(row["label"]), "predicted": int(preds[i]),
            "fake_prob": round(float(probs[i][1])*100,1),
            "real_prob": round(float(probs[i][0])*100,1),
            "correct": bool(preds[i] == row["label"]),
        })
    acc = round(sum(r["correct"] for r in records) / len(records) * 100, 2)
    return True, {"total": len(records), "accuracy": acc}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/train", methods=["POST"])
def train_route():
    ok, result = train()
    return (jsonify(result) if ok else jsonify({"error": result}), 200 if ok else 500)

@app.route("/articles")
def articles():
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 25))
    filter_ = request.args.get("filter", "all")
    filtered = records
    if filter_ == "fake":  filtered = [r for r in records if r["predicted"] == 1]
    elif filter_ == "real": filtered = [r for r in records if r["predicted"] == 0]
    elif filter_ == "wrong": filtered = [r for r in records if not r["correct"]]
    total = len(filtered); start = (page-1)*per_page
    return jsonify({"items": filtered[start:start+per_page], "total": total,
                    "page": page, "pages": (total+per_page-1)//per_page})

@app.route("/article/<int:article_id>")
def article_detail(article_id):
    for r in records:
        if r["id"] == article_id: return jsonify(r)
    return jsonify({"error": "Not found"}), 404

@app.route("/predict_ml", methods=["POST"])
def predict_ml():
    if model is None or tfidf is None:
        return jsonify({"error": "Model not trained yet. Click Train Model first."}), 400
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])
    pred    = int(model.predict(vec)[0])
    probs   = model.predict_proba(vec)[0]
    return jsonify({
        "verdict":   "FAKE" if pred == 1 else "REAL",
        "fake_prob": round(float(probs[1]) * 100, 1),
        "real_prob": round(float(probs[0]) * 100, 1),
    })

@app.route("/predict_tv", methods=["POST"])
def predict_tv():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    tv_matches = match_sources(text)
    return jsonify({
        "tv_verdict": "CONFIRMED" if tv_matches else "NOT FOUND",
        "tv_matches": tv_matches,
    })

@app.route("/sources")
def sources():
    return jsonify([{"name": s["name"], "name_ar": s["name_ar"],
                     "color": s["color"]} for s in TV_SOURCES])

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Fake News Browser  +  TV Source Check")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Sources : {len(TV_SOURCES)} Lebanese TV channels")
    print("  Open    : http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
