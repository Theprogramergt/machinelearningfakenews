import os, re, time, requests, gc
import xml.etree.ElementTree as ET
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RSS_CACHE_TTL    = 120        # ← 2 min refresh (was 10 min) = more "live"
SAMPLE_PER_CLASS = 3000
MAX_TFIDF_FEATURES = 2000

# ── Lebanese / Arabic TV News Sources ─────────────────────────────────────────
TV_SOURCES = [
    {"name": "MTV Lebanon",  "name_ar": "MTV لبنان",       "color": "#e63946", "rss": "https://www.mtv.com.lb/rss"},
    {"name": "Al Jadeed TV", "name_ar": "الجديد",           "color": "#f4a261", "rss": "https://www.aljadeed.tv/rss/arabic/news"},
    {"name": "LBCI",         "name_ar": "LBCI",             "color": "#2a9d8f", "rss": "https://www.lbci.com/rss"},
    {"name": "NNA Lebanon",  "name_ar": "الوكالة الوطنية",  "color": "#457b9d", "rss": "https://www.nna-leb.gov.lb/ar/rss"},
    {"name": "OTV Lebanon",  "name_ar": "OTV",              "color": "#6a0572", "rss": "https://www.otv.com.lb/rss"},
    {"name": "Annahar",      "name_ar": "النهار",           "color": "#d4a017", "rss": "https://www.annahar.com/rss"},
    {"name": "Al Manar",     "name_ar": "المنار",           "color": "#264653", "rss": "https://www.almanar.com.lb/rss"},
]

# ── Arabic stop words (expanded) ──────────────────────────────────────────────
STOP_AR = {
    "من","إلى","على","في","عن","مع","هذا","هذه","التي","الذي","الذين",
    "كان","كانت","وقد","وقال","قال","لقد","أن","إن","ما","لا","كل",
    "كما","ذلك","بعد","قبل","حيث","أو","ثم","أيضا","بين","حول","وفي",
    "وعلى","ومن","وإلى","وأن","وكان","التي","وهو","وهي","وقال","وأضاف",
    "اليوم","أمس","الآن","خلال","حتى","منذ","عند","لدى","إذ","إذا",
    "هذا","هذه","ذلك","تلك","هناك","هنا","كانت","يكون","تكون","فإن",
    "لكن","غير","بشكل","نحو","نفس","ضمن","عبر","لأن","بما","أكثر",
}

# ── RSS Cache ─────────────────────────────────────────────────────────────────
_rss_cache = {}

def fetch_rss(source):
    cached = _rss_cache.get(source["name"])
    if cached and (time.time() - cached["fetched_at"]) < RSS_CACHE_TTL:
        return cached["articles"]
    articles = []
    try:
        resp = requests.get(source["rss"], timeout=8,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"})
        if resp.status_code != 200:
            print(f"  RSS {source['name']} → HTTP {resp.status_code}")
            _rss_cache[source["name"]] = {"articles": [], "fetched_at": time.time()}
            return []
        root = ET.fromstring(resp.content)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}

        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            desc  = re.sub(r'<[^>]+>', '', item.findtext("description") or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()
            if title:
                articles.append({"title": title, "link": link,
                                  "description": desc, "pubDate": pub})

        if not articles:
            for entry in root.findall("atom:entry", ns):
                title   = (entry.findtext("atom:title", namespaces=ns) or "").strip()
                link_el = entry.find("atom:link", ns)
                link    = link_el.get("href", "") if link_el is not None else ""
                desc    = (entry.findtext("atom:summary", namespaces=ns) or "").strip()
                if title:
                    articles.append({"title": title, "link": link,
                                     "description": desc, "pubDate": ""})

    except Exception as e:
        print(f"  RSS error [{source['name']}]: {e}")

    _rss_cache[source["name"]] = {"articles": articles, "fetched_at": time.time()}
    print(f"  RSS [{source['name']}] fetched {len(articles)} articles")
    return articles


# ── Smarter keyword extraction ────────────────────────────────────────────────
def extract_keywords(text, min_len=3):
    """Extract Arabic + Latin words, remove stop words, deduplicate root-like stems."""
    words = re.findall(
        r'[\u0600-\u06FF]{' + str(min_len) + r',}|[a-zA-Z]{' + str(min_len) + r',}',
        text
    )
    filtered = [w for w in words if w not in STOP_AR]
    return filtered


def arabic_stem(word):
    """
    Very lightweight Arabic stemmer:
    strips common prefixes (وال ال وب وك وف بال) and suffixes (ين ون ات ة ها هم تي ني).
    Good enough for headline matching without any external library.
    """
    prefixes = ["وال", "بال", "كال", "فال", "ال", "وب", "وك", "وف", "وم", "وس"]
    suffixes = ["ين", "ون", "ات", "ية", "ها", "هم", "هن", "تي", "ني", "كم", "ة"]
    w = word
    for p in prefixes:
        if w.startswith(p) and len(w) - len(p) >= 3:
            w = w[len(p):]
            break
    for s in suffixes:
        if w.endswith(s) and len(w) - len(s) >= 3:
            w = w[:-len(s)]
            break
    return w


def score_match(query_words, article_text):
    """
    Dual scoring:
      - exact overlap  (high weight)
      - stemmed overlap (medium weight)
    Returns 0.0–1.0
    """
    if not query_words or not article_text:
        return 0.0

    art_words  = extract_keywords(article_text, min_len=3)
    if not art_words:
        return 0.0

    q_set      = set(query_words)
    a_set      = set(art_words)

    # exact
    exact      = len(q_set & a_set)

    # stemmed
    q_stems    = {arabic_stem(w) for w in q_set}
    a_stems    = {arabic_stem(w) for w in a_set}
    stemmed    = len(q_stems & a_stems)

    # weighted score normalised to query length
    raw = (exact * 1.0 + stemmed * 0.6) / max(len(q_set), 1)
    return min(raw, 1.0)          # cap at 1.0


def match_sources(user_text):
    keywords = extract_keywords(user_text, min_len=3)
    if len(keywords) < 2:
        return []

    # use ALL keywords (no 20-word cap)
    matches = []
    for source in TV_SOURCES:
        articles = fetch_rss(source)
        best_score, best_article = 0.0, None

        for art in articles:
            # score against title + description together
            combined = art["title"] + " " + art.get("description", "")
            s = score_match(keywords, combined)
            if s > best_score:
                best_score, best_article = s, art

        # ↓ lower threshold: 0.15 instead of 0.25
        if best_score >= 0.15 and best_article:
            matches.append({
                "source":    source["name"],
                "source_ar": source["name_ar"],
                "color":     source["color"],
                "title":     best_article["title"],
                "link":      best_article["link"],
                "pubDate":   best_article.get("pubDate", ""),
                "score":     round(best_score * 100, 1),
            })

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches


# ── Global ML state ───────────────────────────────────────────────────────────
model   = None
tfidf   = None
records = []

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'^[A-Z][A-Z,\s]+\([^)]+\)\s*[-–]\s*', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text.strip()

def train():
    global model, tfidf, records
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        print("Downloading Fake.csv …")
        fake_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "emineyetm/fake-news-detection-datasets",
            "News _dataset/Fake.csv"
        )
        fake_df["label"] = 1
        print("Downloading True.csv …")
        true_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "emineyetm/fake-news-detection-datasets",
            "News _dataset/True.csv"
        )
        true_df["label"] = 0
    except Exception as e:
        return False, f"Failed to download dataset: {str(e)}"

    fake_df = fake_df.sample(n=min(SAMPLE_PER_CLASS, len(fake_df)), random_state=42)
    true_df = true_df.sample(n=min(SAMPLE_PER_CLASS, len(true_df)), random_state=42)
    df = pd.concat([fake_df, true_df], ignore_index=True).sample(frac=1, random_state=42)
    del fake_df, true_df; gc.collect()

    df["text"] = (
        df.get("title", pd.Series(dtype=str)).fillna("") + " " +
        df.get("text",  pd.Series(dtype=str)).fillna("")
    ).apply(clean_text)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    X_raw = df["text"]; Y = df["label"]
    tfidf = TfidfVectorizer(stop_words="english", max_features=MAX_TFIDF_FEATURES)
    X     = tfidf.fit_transform(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=200)
    model.fit(X_train, y_train)
    del X, X_train; gc.collect()

    probs    = model.predict_proba(X_test)
    preds    = model.predict(X_test)
    test_idx = y_test.index.tolist()
    records  = []
    for i, idx in enumerate(test_idx):
        row = df.iloc[idx]
        records.append({
            "id":        i + 1,
            "title":     str(row.get("title", ""))[:120] or "—",
            "text":      str(row.get("text",  ""))[:400],
            "actual":    int(row["label"]),
            "predicted": int(preds[i]),
            "fake_prob": round(float(probs[i][1]) * 100, 1),
            "real_prob": round(float(probs[i][0]) * 100, 1),
            "correct":   bool(preds[i] == row["label"]),
        })
    del X_test, probs, preds; gc.collect()
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
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 25))
    filter_  = request.args.get("filter", "all")
    filtered = records
    if filter_ == "fake":   filtered = [r for r in records if r["predicted"] == 1]
    elif filter_ == "real":  filtered = [r for r in records if r["predicted"] == 0]
    elif filter_ == "wrong": filtered = [r for r in records if not r["correct"]]
    total = len(filtered); start = (page - 1) * per_page
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
    data    = request.get_json()
    text    = data.get("text", "").strip()
    if not text: return jsonify({"error": "No text provided."}), 400
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
    if not text: return jsonify({"error": "No text provided."}), 400
    tv_matches = match_sources(text)
    return jsonify({
        "tv_verdict": "CONFIRMED" if tv_matches else "NOT FOUND",
        "tv_matches": tv_matches,
    })

@app.route("/sources")
def sources():
    return jsonify([{"name": s["name"], "name_ar": s["name_ar"],
                     "color": s["color"]} for s in TV_SOURCES])

# ── Debug route: see raw RSS for a source ─────────────────────────────────────
@app.route("/debug_rss/<source_name>")
def debug_rss(source_name):
    for s in TV_SOURCES:
        if s["name"].lower().replace(" ", "") == source_name.lower().replace(" ", ""):
            arts = fetch_rss(s)
            return jsonify({"source": s["name"], "count": len(arts), "articles": arts[:10]})
    return jsonify({"error": "Source not found"}), 404

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Fake News Browser  +  TV Source Check")
    print(f"  Sources : {len(TV_SOURCES)} Lebanese TV channels")
    print(f"  Sample  : {SAMPLE_PER_CLASS} articles per class")
    print("  Open    : http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
