import os, re, time, requests, gc
import xml.etree.ElementTree as ET
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RSS_CACHE_TTL = 300          # 5 min cache (fresher results)
SAMPLE_PER_CLASS = 3000
MAX_TFIDF_FEATURES = 2000

# ── Lebanese / Arabic TV News Sources ─────────────────────────────────────────
TV_SOURCES = [
    {"name": "MTV Lebanon",  "name_ar": "MTV لبنان",        "color": "#e63946",
     "rss": ["https://www.mtv.com.lb/rss",
             "https://www.mtv.com.lb/rss/latest"]},

    {"name": "Al Jadeed TV", "name_ar": "الجديد",            "color": "#f4a261",
     "rss": ["https://www.aljadeed.tv/rss/arabic/news",
             "https://www.aljadeed.tv/rss/arabic"]},

    {"name": "LBCI",         "name_ar": "LBCI",              "color": "#2a9d8f",
     "rss": ["https://www.lbci.com/rss",
             "https://www.lbci.com/rss/news"]},

    {"name": "NNA Lebanon",  "name_ar": "الوكالة الوطنية",   "color": "#457b9d",
     "rss": ["https://www.nna-leb.gov.lb/ar/rss",
             "https://nna-leb.gov.lb/ar/rss"]},

    {"name": "OTV Lebanon",  "name_ar": "OTV",               "color": "#6a0572",
     "rss": ["https://www.otv.com.lb/rss"]},

    {"name": "Annahar",      "name_ar": "النهار",            "color": "#d4a017",
     "rss": ["https://www.annahar.com/rss",
             "https://www.annahar.com/rss/latest"]},

    {"name": "Al Manar",     "name_ar": "المنار",            "color": "#264653",
     "rss": ["https://www.almanar.com.lb/rss",
             "https://www.almanar.com.lb/rss/news"]},
]

# Arabic stop words – expanded
STOP_AR = {
    "من","إلى","على","في","عن","مع","هذا","هذه","التي","الذي",
    "كان","كانت","وقد","وقال","قال","لقد","أن","إن","ما","لا",
    "كما","ذلك","بعد","قبل","حيث","أو","ثم","أيضا","بين","حول",
    "وفي","وقال","وأن","هو","هي","هم","نحن","أنت","يكون","تكون",
    "وهو","وهي","وهم","الى","عند","منذ","حتى","لكن","لكي","لان",
    "لأن","وعلى","للـ","للا","كل","قد","لم","لن","ليس","ليست",
    "عبر","خلال","بشكل","بسبب","نتيجة","وفقا","طبقا","حسب","وفق",
    "يوم","اليوم","الذين","اللذين","اللواتي","اللتان","ذين","هذين",
    "وذلك","كذلك","ولكن","كانوا","كنا","كنت","تكن","يكن","يكونوا",
}

# ── RSS Cache ─────────────────────────────────────────────────────────────────
_rss_cache = {}

def fetch_rss_url(url):
    """Fetch a single RSS URL, return list of article dicts."""
    articles = []
    try:
        resp = requests.get(url, timeout=8,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; FakeLens/1.0)"})
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}

        # ── RSS <item> ──────────────────────────────────────────────────────
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            desc  = re.sub(r'<[^>]+>', '', item.findtext("description") or "").strip()
            pubdate = (item.findtext("pubDate") or "").strip()
            if title:
                articles.append({"title": title, "link": link,
                                  "description": desc, "pubDate": pubdate})

        # ── Atom <entry> ────────────────────────────────────────────────────
        if not articles:
            for entry in root.findall("atom:entry", ns):
                title   = (entry.findtext("atom:title",   namespaces=ns) or "").strip()
                summary = (entry.findtext("atom:summary", namespaces=ns) or "").strip()
                link_el = entry.find("atom:link", ns)
                link    = link_el.get("href", "") if link_el is not None else ""
                if title:
                    articles.append({"title": title, "link": link,
                                     "description": summary, "pubDate": ""})
    except Exception as e:
        print(f"  RSS fetch error [{url}]: {e}")
    return articles

def fetch_rss(source):
    """Try all RSS URLs for a source, merge results, cache them."""
    cached = _rss_cache.get(source["name"])
    if cached and (time.time() - cached["fetched_at"]) < RSS_CACHE_TTL:
        return cached["articles"]

    all_articles = []
    seen_links   = set()
    rss_urls     = source["rss"] if isinstance(source["rss"], list) else [source["rss"]]

    for url in rss_urls:
        for art in fetch_rss_url(url):
            if art["link"] not in seen_links:
                seen_links.add(art["link"])
                all_articles.append(art)
        if all_articles:          # stop after first successful URL
            break

    _rss_cache[source["name"]] = {"articles": all_articles, "fetched_at": time.time()}
    return all_articles

# ── Text helpers ──────────────────────────────────────────────────────────────

def normalize_ar(text):
    """Normalize Arabic text: strip diacritics, unify alef variants, etc."""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)   # diacritics
    text = re.sub(r'[أإآاٱ]', 'ا', text)                # alef variants
    text = re.sub(r'[ىي]', 'ي', text)                   # ya variants
    text = re.sub(r'ة', 'ه', text)                      # ta marbuta
    return text

def extract_keywords(text, min_len=3):
    """Return meaningful Arabic and Latin words from text."""
    norm = normalize_ar(text)
    words = re.findall(
        r'[\u0600-\u06FF]{' + str(min_len) + r',}|[a-zA-Z]{' + str(min_len) + r',}',
        norm
    )
    return [w for w in words if w not in STOP_AR]

def tokenize(text):
    """Normalize + tokenize into a set of significant tokens."""
    return set(extract_keywords(text, min_len=3))

def similarity_score(query_tokens, article_tokens):
    """
    Jaccard-style overlap biased toward query coverage.
    Returns 0-1.
    """
    if not query_tokens or not article_tokens:
        return 0.0
    overlap = len(query_tokens & article_tokens)
    # How much of the query is covered?
    query_coverage  = overlap / len(query_tokens)
    # Jaccard to avoid rewarding very short article token sets
    jaccard         = overlap / len(query_tokens | article_tokens)
    # Weighted blend
    return 0.7 * query_coverage + 0.3 * jaccard

def match_sources(user_text, min_score=0.15):
    """
    For each TV source, find the best-matching article.
    Lower threshold (0.15) so partial matches are caught.
    """
    user_tokens = tokenize(user_text)
    if len(user_tokens) < 2:
        return []

    matches = []
    for source in TV_SOURCES:
        articles = fetch_rss(source)
        if not articles:
            continue

        best_score, best_article = 0.0, None
        for art in articles:
            combined     = art["title"] + " " + art["description"]
            art_tokens   = tokenize(combined)
            if not art_tokens:
                continue
            score = similarity_score(user_tokens, art_tokens)
            if score > best_score:
                best_score, best_article = score, art

        if best_score >= min_score and best_article:
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

        print("Downloading Fake.csv from Kaggle…")
        fake_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "emineyetm/fake-news-detection-datasets",
            "News _dataset/Fake.csv"
        )
        fake_df["label"] = 1

        print("Downloading True.csv from Kaggle…")
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
    del fake_df, true_df
    gc.collect()

    df["text"] = (
        df.get("title", pd.Series(dtype=str)).fillna("") + " " +
        df.get("text",  pd.Series(dtype=str)).fillna("")
    ).apply(clean_text)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    X_raw = df["text"]
    Y     = df["label"]

    tfidf = TfidfVectorizer(stop_words="english", max_features=MAX_TFIDF_FEATURES)
    X     = tfidf.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=200)
    model.fit(X_train, y_train)

    del X, X_train
    gc.collect()

    probs    = model.predict_proba(X_test)
    preds    = model.predict(X_test)
    test_idx = y_test.index.tolist()

    records = []
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

    del X_test, probs, preds
    gc.collect()

    acc = round(sum(r["correct"] for r in records) / len(records) * 100, 2)
    return True, {"total": len(records), "accuracy": acc}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

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
    elif filter_ == "real": filtered = [r for r in records if r["predicted"] == 0]
    elif filter_ == "wrong":filtered = [r for r in records if not r["correct"]]
    total = len(filtered)
    start = (page - 1) * per_page
    return jsonify({
        "items": filtered[start:start + per_page],
        "total": total,
        "page":  page,
        "pages": (total + per_page - 1) // per_page,
    })

@app.route("/article/<int:article_id>")
def article_detail(article_id):
    for r in records:
        if r["id"] == article_id:
            return jsonify(r)
    return jsonify({"error": "Not found"}), 404

@app.route("/predict_ml", methods=["POST"])
def predict_ml():
    if model is None or tfidf is None:
        return jsonify({"error": "Model not trained yet. Click Train Model first."}), 400
    data    = request.get_json()
    text    = data.get("text", "").strip()
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
        "tv_verdict":  "CONFIRMED" if tv_matches else "NOT FOUND",
        "tv_matches":  tv_matches,
        "sources_checked": len(TV_SOURCES),
    })

@app.route("/sources")
def sources():
    return jsonify([{
        "name": s["name"], "name_ar": s["name_ar"], "color": s["color"]
    } for s in TV_SOURCES])

# ── Debug route: see what a source returned (useful for tuning) ───────────────
@app.route("/debug/rss/<source_name>")
def debug_rss(source_name):
    for s in TV_SOURCES:
        if s["name"].lower().replace(" ", "") == source_name.lower().replace(" ", ""):
            arts = fetch_rss(s)
            return jsonify({"source": s["name"], "count": len(arts), "sample": arts[:10]})
    return jsonify({"error": "source not found"}), 404

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  FakeLens – Fake News Browser + TV Source Check")
    print(f"  Sources : {len(TV_SOURCES)} Lebanese TV channels")
    print(f"  Sample  : {SAMPLE_PER_CLASS} articles per class")
    print("  Open    : http://localhost:5000")
    print("  Debug   : /debug/rss/<SourceName>")
    print("=" * 55 + "\n")
    app.run(debug=True, port=5000)
