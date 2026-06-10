
import io
import re
import json
import zipfile
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageStat, ExifTags

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    import networkx as nx
except Exception:
    nx = None

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    torch = None
    CLIPModel = None
    CLIPProcessor = None


APP_TITLE = "LifeLens V8.6 – Family Analytics + Safe AI Review"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}
SUPPORTED_EXT = IMAGE_EXT | VIDEO_EXT

FREE_IMAGE_LIMIT = 500
PREMIUM_UNLOCK_CODE = "LIFELENS-PREMIUM-DEMO"

TOPIC_TO_CATEGORY = {
    "vonat": "Járművek",
    "traktor": "Járművek",
    "autó": "Járművek",
    "jármű": "Járművek",
    "bicikli": "Sport / mozgás",
    "foci": "Sport / mozgás",
    "túrázás": "Utazás / természet",
    "hegy": "Utazás / természet",
    "nyaralás": "Utazás / természet",
    "strand": "Utazás / természet",
    "állatkert": "Állatok",
    "kutya": "Állatok",
    "rajz": "Kreatív",
    "gitár": "Kreatív",
    "zene": "Kreatív",
    "sakk": "Tanulás / játék",
    "társasjáték": "Tanulás / játék",
    "karácsony": "Ünnepek",
    "születésnap": "Ünnepek",
    "játszótér": "Játék",
    "plüss": "Játék",
    "fagyi": "Családi pillanatok",
}



def infer_topics_from_path(path: Path):
    """Saját képeknél csak fájlnév + mappanév alapján címkézünk.
    Fontos: dátum alapján már nem teszünk rá automatikusan nyaralás/karácsony taget,
    mert az túl sok téves találatot adott.
    """
    text = str(path).lower()
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ö": "o", "ő": "o",
        "ú": "u", "ü": "u", "ű": "u"
    }
    norm = text
    for a, b in replacements.items():
        norm = norm.replace(a, b)

    keyword_map = {
        "vonat": ["vonat", "train", "railway"],
        "traktor": ["traktor", "tractor"],
        "autó": ["auto", "autó", "car", "kocsi"],
        "bicikli": ["bicikli", "bike", "bicycle", "bringa"],
        "foci": ["foci", "football", "soccer", "meccs"],
        "túrázás": ["tura", "túra", "turazas", "kirandulas", "kirándulás", "hiking"],
        "hegy": ["hegy", "mountain"],
        "nyaralás": ["nyaralas", "nyaralás", "balaton", "holiday", "vacation", "strand", "tenger"],
        "strand": ["strand", "beach"],
        "állatkert": ["allatkert", "állatkert", "zoo"],
        "kutya": ["kutya", "dog"],
        "rajz": ["rajz", "drawing", "festes", "festés"],
        "gitár": ["gitar", "gitár", "guitar", "zene", "music"],
        "sakk": ["sakk", "chess"],
        "karácsony": ["karacsony", "karácsony", "christmas", "xmas"],
        "születésnap": ["szulinap", "szülinap", "birthday", "torta", "cake"],
        "játszótér": ["jatszoter", "játszótér", "playground"],
        "plüss": ["pluss", "plüss", "plush"],
        "fagyi": ["fagyi", "icecream", "ice_cream", "ice cream"],
    }

    topics = []
    for topic, words in keyword_map.items():
        if any(w.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ö","o").replace("ő","o").replace("ú","u").replace("ü","u").replace("ű","u") in norm for w in words):
            topics.append(topic)

    return sorted(set(topics))



AI_TOPIC_PROMPTS = {
    "vonat": ["toy train", "child playing with train", "railway train"],
    "traktor": ["tractor", "toy tractor", "child with tractor"],
    "autó": ["car", "toy car", "child with car"],
    "bicikli": ["child riding bicycle", "bicycle", "balance bike"],
    "foci": ["soccer ball", "child playing football", "football match"],
    "túrázás": ["family hiking", "mountain hiking", "forest trail"],
    "hegy": ["mountain landscape", "family in mountains"],
    "nyaralás": ["family vacation", "holiday travel", "family trip"],
    "strand": ["beach", "family at beach", "child at beach"],
    "állatkert": ["zoo animals", "family at zoo", "child watching animals"],
    "kutya": ["dog", "child with dog"],
    "rajz": ["child drawing", "drawing with pencils", "painting activity"],
    "gitár": ["child playing guitar", "guitar lesson", "family music"],
    "sakk": ["chess board", "child playing chess", "people playing chess"],
    "karácsony": ["christmas tree", "christmas gifts", "family christmas"],
    "születésnap": ["birthday cake", "birthday party", "child birthday"],
    "játszótér": ["playground", "child on playground", "swing and slide"],
    "plüss": ["stuffed animal", "plush toy", "child with stuffed toy"],
    "fagyi": ["ice cream", "child eating ice cream"],
}


@st.cache_resource(show_spinner=False)
def load_clip_model():
    if CLIPModel is None or CLIPProcessor is None or torch is None:
        return None, None, None
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor, device


def ai_detect_topics(img: Image.Image, threshold: float = 0.18):
    model, processor, device = load_clip_model()
    if model is None:
        return [], ""

    labels = []
    prompts = []
    for topic, topic_prompts in AI_TOPIC_PROMPTS.items():
        for prompt in topic_prompts:
            labels.append(topic)
            prompts.append(prompt)

    inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu().numpy()

    best = {}
    for label, prob in zip(labels, probs):
        best[label] = max(best.get(label, 0), float(prob))

    selected = [k for k, v in best.items() if v >= threshold]
    selected = sorted(selected, key=lambda k: best[k], reverse=True)
    reason = ", ".join([f"{k}:{best[k]:.2f}" for k in selected[:6]])
    return selected[:8], reason


st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

st.markdown("""
<style>
.insight-card {
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 12px;
    background: #f8fafc;
    min-height: 140px;
}
.locked-card {
    border: 1px solid #f59e0b;
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 12px;
    background: #fffbeb;
}
.small-muted { color:#64748b; font-size:0.9rem; }
.big-kpi { font-size: 2.1rem; font-weight: 800; margin: 0; }
</style>
""", unsafe_allow_html=True)

st.title("📊 LifeLens V8.6 – Family Analytics")
st.caption("Beépített demo · kattintható insightok · Korszakmotor · Family DNA · Nostalgia Score · Dashboardból album")

st.info(
    "A V8.6 már automatikusan betölti a Demo_Family_Alpha_V2.zip fájlt, ha az app.py mellett van a GitHub repo-ban. "
    "A dashboard insightjai kattintható gombokkal képgalériára/drill-down nézetre visznek."
)


def safe_name(text: str) -> str:
    text = str(text or "album").strip()
    text = text.replace("ő", "o").replace("ű", "u").replace("Ő", "O").replace("Ű", "U")
    text = re.sub(r"[^\w\- .áéíóöúüÁÉÍÓÖÚÜ]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120] or "album"


def is_premium() -> bool:
    return bool(st.session_state.get("premium_unlocked", False))


def premium_badge():
    if is_premium():
        st.success("✅ Premium mód aktív")
    else:
        st.warning(f"🔒 Free mód · max {FREE_IMAGE_LIMIT} média")


def locked_feature_box(feature_name: str):
    st.markdown(
        f"""
        <div class="locked-card">
        <b>🔒 {feature_name}</b><br>
        Ez Premium funkció. A dashboard preview látható, de az export / teljes elemzés zárolt.
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_drilldown(kind: str, title: str, topic: str = "", year=None, category: str = "", year_month: str = ""):
    st.session_state["drilldown"] = {
        "kind": kind,
        "title": title,
        "topic": topic,
        "year": year,
        "category": category,
        "year_month": year_month,
    }


def get_drilldown_df(df: pd.DataFrame) -> pd.DataFrame:
    dd = st.session_state.get("drilldown")
    if not dd:
        return pd.DataFrame()
    out = df.copy()
    if dd.get("topic"):
        topic = dd["topic"]
        out = out[out["topics"].fillna("").apply(lambda x: topic in [v.strip() for v in str(x).split(",")])]
    if dd.get("year") is not None:
        out = out[out["year"] == int(dd["year"])]
    if dd.get("category"):
        out = out[out["category"] == dd["category"]]
    if dd.get("year_month"):
        out = out[out["year_month"] == dd["year_month"]]
    return out


def render_drilldown_panel(df: pd.DataFrame):
    dd = st.session_state.get("drilldown")
    if not dd:
        return
    sub = get_drilldown_df(df)
    st.markdown("---")
    st.subheader("🔎 Kiválasztott insight képei")
    st.write(f"**{dd.get('title','')}** · találat: **{len(sub)}**")
    if is_premium() and not sub.empty:
        st.download_button(
            "📦 Album ZIP ebből a kiválasztásból",
            create_album_zip(sub, dd.get("title", "album")),
            file_name=f"{safe_name(dd.get('title', 'album'))}.zip",
            mime="application/zip",
            key="drilldown_zip",
        )
    elif not is_premium():
        locked_feature_box("Kiválasztott insight album export")
    render_gallery(sub, max_items=48)


def get_bundled_demo_zip():
    candidates = [
        Path(__file__).resolve().parent / "Demo_Family_Alpha_V2.zip",
        Path.cwd() / "Demo_Family_Alpha_V2.zip",
        Path("/mnt/data/Demo_Family_Alpha_V2.zip"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def split_values(series):
    vals = []
    for item in series.fillna(""):
        vals += [x.strip() for x in str(item).split(",") if x.strip()]
    return vals


def explode_values(df, col):
    out = []
    if df.empty or col not in df.columns:
        return pd.DataFrame()
    for _, r in df.iterrows():
        for v in [x.strip() for x in str(r.get(col, "")).split(",") if x.strip()]:
            row = r.to_dict()
            row["value"] = v
            out.append(row)
    return pd.DataFrame(out)


def get_exif_datetime(img):
    try:
        exif = img.getexif()
        if not exif:
            return None
        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
            if key in tag_map:
                try:
                    return datetime.strptime(str(tag_map[key]), "%Y:%m:%d %H:%M:%S")
                except Exception:
                    pass
    except Exception:
        return None
    return None


def quality_score(img: Image.Image) -> float:
    try:
        small = ImageOps.grayscale(img).resize((256, 256))
        arr = np.asarray(small, dtype=np.float32)
        sharp = np.diff(arr, axis=1).var() + np.diff(arr, axis=0).var()
        stat = ImageStat.Stat(small.resize((128, 128)))
        mean = stat.mean[0]
        std = stat.stddev[0]
        exposure = max(0, 1 - abs(mean - 128) / 128) * 60 + min(std / 64, 1) * 40
        return round(float(sharp * 0.7 + exposure * 0.3), 2)
    except Exception:
        return 0.0


def load_demo_zip(uploaded_zip) -> pd.DataFrame:
    temp_root = Path(tempfile.mkdtemp(prefix="lifelens_demo_"))
    with zipfile.ZipFile(uploaded_zip) as z:
        z.extractall(temp_root)

    meta_files = list(temp_root.rglob("demo_family_metadata.csv"))
    if not meta_files:
        st.error("A ZIP-ben nem találom a demo_family_metadata.csv fájlt.")
        return pd.DataFrame()

    meta_path = meta_files[0]
    df = pd.read_csv(meta_path)
    base = meta_path.parent

    rows = []
    for i, r in df.iterrows():
        p = base / str(r["filename"])
        if not p.exists():
            continue
        rows.append({
            "media_id": f"demo_{i}",
            "media_type": "image",
            "path": str(p),
            "filename": Path(str(r["filename"])).name,
            "folder": str(p.parent),
            "preview_path": str(p),
            "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
            "md5": hashlib.md5(str(p).encode()).hexdigest(),
            "date": pd.to_datetime(r["date"], errors="coerce"),
            "year": int(r["year"]),
            "month": int(r["month"]),
            "year_month": f"{int(r['year'])}-{int(r['month']):02d}",
            "width": None,
            "height": None,
            "duration_sec": None,
            "quality_score": 75 + (i % 20),
            "persons": str(r.get("persons", "")),
            "topics": str(r.get("topics", "")),
            "main_topic": str(r.get("main_topic", "")),
            "category": str(r.get("category", TOPIC_TO_CATEGORY.get(str(r.get("main_topic", "")), "Egyéb"))),
            "events": "",
            "era_title": str(r.get("era_title", "")),
            "family_name": str(r.get("family_name", "Demo Family Alpha")),
            "demo": True,
        })
    return pd.DataFrame(rows).sort_values("date")


def scan_zip(uploaded_file, limit: int, use_ai: bool = False, ai_threshold: float = 0.35, max_ai_topics: int = 2, excluded_ai_topics=None, auto_apply_ai: bool = False) -> pd.DataFrame:
    """Saját ZIP indexelése.
    Ha a ZIP tartalmaz demo_family_metadata.csv vagy lifelens_metadata.csv fájlt, akkor metadata alapján töltünk.
    Egyébként csak fájlnév/mappanév alapján címkézünk, hogy ne hozzon mindent minden szűrésnél.
    """
    temp_root = Path(tempfile.mkdtemp(prefix="lifelens_zip_"))
    if excluded_ai_topics is None:
        excluded_ai_topics = []
    with zipfile.ZipFile(uploaded_file) as z:
        z.extractall(temp_root)

    # Metadata támogatás saját csomaghoz is
    meta_candidates = list(temp_root.rglob("lifelens_metadata.csv")) + list(temp_root.rglob("demo_family_metadata.csv"))
    if meta_candidates:
        meta_path = meta_candidates[0]
        base = meta_path.parent
        meta = pd.read_csv(meta_path)
        rows = []
        for i, r in meta.iterrows():
            rel = str(r.get("filename", ""))
            p = base / rel
            if not p.exists():
                # fallback: filename basename keresése
                matches = list(temp_root.rglob(Path(rel).name))
                p = matches[0] if matches else p
            if not p.exists():
                continue
            dt = pd.to_datetime(r.get("date", None), errors="coerce")
            if pd.isna(dt):
                try:
                    dt = pd.to_datetime(datetime.fromtimestamp(p.stat().st_mtime))
                except Exception:
                    dt = pd.NaT
            topics = str(r.get("topics", ""))
            main_topic = str(r.get("main_topic", "") or (topics.split(",")[0].strip() if topics else ""))
            category = str(r.get("category", TOPIC_TO_CATEGORY.get(main_topic, "Egyéb")))
            rows.append({
                "media_id": f"user_meta_{i}",
                "media_type": "image",
                "path": str(p),
                "filename": p.name,
                "folder": str(p.parent),
                "preview_path": str(p),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
                "md5": hashlib.md5(str(p).encode()).hexdigest(),
                "date": dt,
                "year": int(r.get("year", dt.year if not pd.isna(dt) else 0)),
                "month": int(r.get("month", dt.month if not pd.isna(dt) else 0)),
                "year_month": f"{int(r.get('year', dt.year if not pd.isna(dt) else 0))}-{int(r.get('month', dt.month if not pd.isna(dt) else 0)):02d}",
                "quality_score": 75 + (i % 20),
                "persons": str(r.get("persons", "")),
                "topics": topics,
                "main_topic": main_topic,
                "category": category,
                "events": "",
                "ai_reason": ai_reason if "ai_reason" in locals() else "",
                "ai_analyzed": bool(use_ai),
                "demo": False,
            })
        return pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()

    files = []
    for p in temp_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            files.append(p)
    files = files[:limit]

    rows = []
    for i, p in enumerate(files):
        try:
            dt = datetime.fromtimestamp(p.stat().st_mtime)
            q = 0
            width = height = None
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                exif_dt = get_exif_datetime(img)
                if exif_dt:
                    dt = exif_dt
                width, height = img.size
                q = quality_score(img)

            topics = infer_topics_from_path(p)
            ai_reason = ""
            if use_ai:
                if CLIPModel is None or torch is None:
                    pass
                else:
                    try:
                        with Image.open(p) as ai_img:
                            ai_img = ImageOps.exif_transpose(ai_img).convert("RGB")
                            ai_topics, ai_reason = ai_detect_topics(ai_img, threshold=ai_threshold)
                            ai_topics = [t for t in ai_topics if t not in excluded_ai_topics][:max_ai_topics]
                            # V8.6: az AI alapból csak javaslat, nem automatikus címke.
                            # Automatikusan csak akkor kerül a topics mezőbe, ha a felhasználó külön bekapcsolja.
                            if auto_apply_ai:
                                topics = sorted(set(topics) | set(ai_topics))
                    except Exception as _exc:
                        ai_reason = f"AI hiba: {_exc}"
            main_topic = topics[0] if topics else ""
            rows.append({
                "media_id": f"user_{i}",
                "media_type": "image",
                "path": str(p),
                "filename": p.name,
                "folder": str(p.parent),
                "preview_path": str(p),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
                "md5": hashlib.md5(str(p).encode()).hexdigest(),
                "date": pd.to_datetime(dt),
                "year": dt.year,
                "month": dt.month,
                "year_month": f"{dt.year}-{dt.month:02d}",
                "quality_score": q,
                "persons": "",
                "topics": ", ".join(topics),
                "ai_suggestions": ", ".join(ai_topics) if "ai_topics" in locals() else "",
                "ai_reason": ai_reason if "ai_reason" in locals() else "",
                "ai_analyzed": bool(use_ai),
                "main_topic": main_topic,
                "category": TOPIC_TO_CATEGORY.get(main_topic, "Egyéb") if main_topic else "Nincs címke",
                "events": "",
                "demo": False,
            })
        except Exception:
            pass
    return pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()


def build_year_topic_table(df: pd.DataFrame) -> pd.DataFrame:
    ex = explode_values(df, "topics")
    if ex.empty:
        return pd.DataFrame()

    total_by_year = df.groupby("year").size().rename("year_total").reset_index()
    g = ex.groupby(["year", "value"]).agg(
        count=("path", "count"),
        months=("month", "nunique"),
        avg_quality=("quality_score", "mean"),
        persons=("persons", lambda s: len(set(split_values(s)))),
    ).reset_index().rename(columns={"value": "topic"})

    g = g.merge(total_by_year, on="year", how="left")
    g["share_pct"] = (g["count"] / g["year_total"] * 100).round(2)
    g = g.sort_values(["topic", "year"])
    g["prev_share_pct"] = g.groupby("topic")["share_pct"].shift(1).fillna(0)
    g["growth_pp"] = (g["share_pct"] - g["prev_share_pct"]).round(2)

    max_count = max(g["count"].max(), 1)
    max_share = max(g["share_pct"].max(), 1)
    max_growth = max(g["growth_pp"].max(), 1)

    g["era_score"] = (
        0.45 * (g["growth_pp"].clip(lower=0) / max_growth * 100) +
        0.25 * (g["share_pct"] / max_share * 100) +
        0.20 * (g["count"] / max_count * 100) +
        0.10 * (g["months"] / 12 * 100)
    ).round(1)

    return g.sort_values("era_score", ascending=False)


def build_period_cards(df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    yt = build_year_topic_table(df)
    if yt.empty:
        return pd.DataFrame()

    cards = yt[yt["era_score"] >= 20].copy()
    cards["title"] = cards["topic"].str.capitalize() + " korszak"
    cards["period"] = cards["year"].astype(int).astype(str)
    cards["subtitle"] = (
        cards["count"].astype(int).astype(str)
        + " kép · "
        + cards["share_pct"].astype(str)
        + "% arány · +"
        + cards["growth_pp"].clip(lower=0).astype(str)
        + " pp növekedés"
    )
    return cards.sort_values("era_score", ascending=False).head(top_n)


def build_family_dna(df: pd.DataFrame) -> pd.DataFrame:
    ex = explode_values(df, "topics")
    if ex.empty:
        return pd.DataFrame()

    ex["category"] = ex["value"].map(TOPIC_TO_CATEGORY).fillna("Egyéb")
    total = len(df)

    g = ex.groupby("category").agg(
        count=("path", "count"),
        years=("year", "nunique"),
        months=("year_month", "nunique"),
    ).reset_index()

    g["share_pct"] = (g["count"] / max(total, 1) * 100).round(2)

    by_year = ex.groupby(["category", "year"]).size().reset_index(name="year_count")
    by_year_total = df.groupby("year").size().reset_index(name="total")
    by_year = by_year.merge(by_year_total, on="year", how="left")
    by_year["year_share"] = by_year["year_count"] / by_year["total"] * 100

    trend_rows = []
    for cat, part in by_year.groupby("category"):
        part = part.sort_values("year")
        trend = float(part["year_share"].iloc[-1] - part["year_share"].iloc[0]) if len(part) >= 2 else 0.0
        trend_rows.append({"category": cat, "trend_pp": trend})
    trend_df = pd.DataFrame(trend_rows)

    g = g.merge(trend_df, on="category", how="left").fillna({"trend_pp": 0})

    max_count = max(g["count"].max(), 1)
    max_share = max(g["share_pct"].max(), 1)
    max_months = max(g["months"].max(), 1)
    max_trend = max(g["trend_pp"].clip(lower=0).max(), 1)

    g["dna_score"] = (
        0.40 * (g["share_pct"] / max_share * 100) +
        0.30 * (g["count"] / max_count * 100) +
        0.20 * (g["months"] / max_months * 100) +
        0.10 * (g["trend_pp"].clip(lower=0) / max_trend * 100)
    ).round(1)

    return g.sort_values("dna_score", ascending=False)


def build_hidden_memories(df: pd.DataFrame, period_cards: pd.DataFrame, top_n=20) -> pd.DataFrame:
    ex = explode_values(df, "topics")
    if ex.empty:
        return pd.DataFrame()

    g = ex.groupby(["year", "year_month", "value"]).agg(
        count=("path", "count"),
        avg_quality=("quality_score", "mean"),
        persons=("persons", lambda s: len(set(split_values(s)))),
    ).reset_index().rename(columns={"value": "topic"})

    strength_map = {}
    if not period_cards.empty:
        for _, r in period_cards.iterrows():
            strength_map[(int(r["year"]), r["topic"])] = float(r["era_score"])

    g["era_strength"] = g.apply(lambda r: strength_map.get((int(r["year"]), r["topic"]), 0), axis=1)
    now_year = pd.Timestamp.now().year
    g["age_years"] = now_year - g["year"].fillna(now_year)

    max_count = max(g["count"].max(), 1)
    max_age = max(g["age_years"].max(), 1)
    max_era = max(g["era_strength"].max(), 1)
    g["rarity_score"] = (1 - (g["count"] / max_count)).clip(0, 1) * 100

    g["nostalgia_score"] = (
        0.40 * (g["age_years"] / max_age * 100) +
        0.30 * (g["count"] / max_count * 100) +
        0.20 * (g["era_strength"] / max_era * 100) +
        0.10 * g["rarity_score"]
    ).round(1)

    return g.sort_values("nostalgia_score", ascending=False).head(top_n)


def family_evolution_score(df: pd.DataFrame) -> float:
    yt = build_year_topic_table(df)
    if yt.empty:
        return 0.0
    return round(float(yt["growth_pp"].clip(lower=0).mean() * 6), 1)


def family_diversity_score(df: pd.DataFrame) -> int:
    return len(set(split_values(df.get("topics", pd.Series(dtype=str)))))


def compute_relationships(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        people = [x.strip() for x in str(r.get("persons", "")).split(",") if x.strip()]
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                rows.append({
                    "person_a": people[i],
                    "person_b": people[j],
                    "year": r.get("year"),
                    "path": r.get("path"),
                })
    rel = pd.DataFrame(rows)
    if rel.empty:
        return rel
    return rel.groupby(["person_a", "person_b"]).size().reset_index(name="count").sort_values("count", ascending=False)


def create_album_zip(rows_df: pd.DataFrame, name: str) -> bytes:
    mem = io.BytesIO()
    used = set()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for _, r in rows_df.iterrows():
            p = Path(str(r.get("path", "")))
            if not p.exists():
                continue
            arc = safe_name(r.get("filename", p.name))
            if arc in used:
                arc = f"{Path(arc).stem}_{hashlib.md5(str(p).encode()).hexdigest()[:6]}{p.suffix}"
            used.add(arc)
            z.write(p, arc)
        z.writestr("lifelens_album_index.csv", rows_df.to_csv(index=False).encode("utf-8-sig"))
    mem.seek(0)
    return mem.read()


def render_gallery(rows_df: pd.DataFrame, max_items=24):
    if rows_df is None or rows_df.empty:
        st.info("Nincs megjeleníthető kép.")
        return
    cols = st.columns(4)
    for i, (_, row) in enumerate(rows_df.head(max_items).iterrows()):
        p = Path(str(row.get("preview_path") or row.get("path") or ""))
        with cols[i % 4]:
            if p.exists():
                st.image(str(p), caption=f"{row.get('filename','')} · {row.get('topics','')}", use_container_width=True)
            else:
                st.write(row.get("filename", ""))


def filter_by_topic_year(df, topic, year=None):
    mask = df["topics"].fillna("").apply(lambda x: topic in [v.strip() for v in str(x).split(",")])
    out = df[mask]
    if year is not None:
        out = out[out["year"] == year]
    return out


if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()

with st.sidebar:
    st.header("0. Licenc / mód")
    premium_badge()

    unlock_code = st.text_input("Premium feloldó kód", type="password", help="Demo kód: LIFELENS-PREMIUM-DEMO")
    if st.button("Premium feloldása", use_container_width=True):
        if unlock_code.strip() == PREMIUM_UNLOCK_CODE:
            st.session_state["premium_unlocked"] = True
            st.success("Premium mód feloldva.")
            st.rerun()
        else:
            st.error("Hibás feloldó kód.")

    st.divider()
    st.header("1. Adatforrás")

    bundled_demo = get_bundled_demo_zip()
    if bundled_demo:
        st.success("✅ Beépített demo csomag megtalálva")
        if st.button("Demo Family automatikus betöltése", use_container_width=True):
            st.session_state["df"] = load_demo_zip(bundled_demo)
            st.success("Demo Family Alpha betöltve.")
    else:
        st.warning("Nem találom a Demo_Family_Alpha_V2.zip fájlt az app mellett.")
        st.caption("GitHub repo-ba tedd fel az app.py mellé Demo_Family_Alpha_V2.zip néven.")
        demo_zip = st.file_uploader("Demo Family Alpha V2 ZIP kézi feltöltése", type=["zip"], key="demo_zip")
        if demo_zip is not None and st.button("Demo Family betöltése", use_container_width=True):
            st.session_state["df"] = load_demo_zip(demo_zip)
            st.success("Demo Family Alpha betöltve.")

    st.caption("Saját adatokkal csak teszt jelleggel:")
    st.caption("V8.6: az AI alapból csak javaslatot ad. Így a téves AI címkék nem rontják el a szűrést.")
    use_ai_for_zip = st.checkbox("AI képfelismerés futtatása saját ZIP-re", value=False)
    ai_threshold = st.slider(
        "AI szigorúság / küszöb",
        0.10, 0.70, 0.35, 0.01,
        help="Magasabb érték = kevesebb, de biztosabb javaslat. Javaslat: 0.35–0.50."
    )
    max_ai_topics = st.slider("Max. AI javaslat képenként", 1, 5, 2)
    excluded_ai_topics = st.multiselect(
        "Kizárt AI kategóriák",
        sorted(AI_TOPIC_PROMPTS.keys()) if "AI_TOPIC_PROMPTS" in globals() else [],
        default=[],
        help="Ha például nincs gitár/fagyi a csomagban, zárd ki ezeket."
    )
    auto_apply_ai = st.checkbox(
        "Kísérleti: AI javaslatok automatikus címkévé alakítása",
        value=False,
        help="Ha kikapcsolva marad, az AI csak javaslatot ad az AI ellenőrzés fülön, és nem befolyásolja a szűrőket."
    )
    if use_ai_for_zip and (CLIPModel is None or torch is None):
        st.warning("AI-hoz kell a requirements_ai.txt vagy: torch + transformers.")
    user_zip = st.file_uploader("Saját képek ZIP-ben", type=["zip"], key="user_zip")
    if user_zip is not None and st.button("Saját ZIP indexelése", use_container_width=True):
        limit = FREE_IMAGE_LIMIT if not is_premium() else 5000
        st.session_state["df"] = scan_zip(
            user_zip,
            limit,
            use_ai=use_ai_for_zip,
            ai_threshold=ai_threshold,
            max_ai_topics=max_ai_topics,
            excluded_ai_topics=excluded_ai_topics,
            auto_apply_ai=auto_apply_ai,
        )
        st.success("Saját ZIP index kész.")

df = st.session_state.get("df", pd.DataFrame())

if df.empty:
    st.markdown("## Kezdés")
    st.write("Bal oldalon kattints a **Demo Family automatikus betöltése** gombra.")
    st.write("Ha a gomb nem látszik, tedd fel a `Demo_Family_Alpha_V2.zip` fájlt az app.py mellé a GitHub repo-ba.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["month"] = pd.to_numeric(df["month"], errors="coerce")
df["year_month"] = df["year_month"].fillna("Dátum nélkül")

all_years = sorted([int(y) for y in df["year"].dropna().unique()])
all_topics = sorted(set(split_values(df["topics"])))
all_people = sorted(set(split_values(df["persons"])))
all_categories = sorted(set(df["category"].fillna("Egyéb")))

with st.sidebar:
    st.divider()
    st.header("2. Dashboard slicerek")
    sel_years = st.multiselect("Év", all_years, default=all_years)
    sel_topics = st.multiselect("Téma / korszak", all_topics)
    sel_people = st.multiselect("Személy", all_people)
    sel_categories = st.multiselect("Family DNA kategória", all_categories)

fdf = df.copy()
if sel_years:
    fdf = fdf[fdf["year"].isin(sel_years)]
if sel_topics:
    fdf = fdf[fdf["topics"].fillna("").apply(lambda x: all(t in [v.strip() for v in str(x).split(",")] for t in sel_topics))]
if sel_people:
    fdf = fdf[fdf["persons"].fillna("").apply(lambda x: all(p in [v.strip() for v in str(x).split(",")] for p in sel_people))]
if sel_categories:
    fdf = fdf[fdf["category"].isin(sel_categories)]

if "demo" in fdf.columns and not bool(fdf["demo"].fillna(False).any()):
    unlabeled = int((fdf["topics"].fillna("").str.strip() == "").sum())
    if unlabeled:
        st.warning(
            f"Saját képek: {unlabeled} képhez nincs címke. "
            "Kapcsold be az AI képfelismerést, vagy adj metadata CSV-t a pontosabb szűréshez."
        )

if "ai_suggestions" in fdf.columns and fdf["ai_suggestions"].fillna("").str.strip().ne("").any() and not fdf["topics"].fillna("").str.contains(",").any():
    st.info("AI javaslatok készültek, de nem lettek automatikusan címkévé alakítva. Nézd meg az 🤖 AI ellenőrzés fület.")

tabs = st.tabs([
    "📊 V8.1 Dashboard",
    "🧭 Korszakmotor",
    "🧬 Family DNA",
    "🕰️ Nostalgia / Hidden Memories",
    "🔗 Kapcsolati háló",
    "🖼️ Képek + album",
    "📦 Export",
    "🤖 AI ellenőrzés"
])

period_cards = build_period_cards(fdf, top_n=12)
family_dna = build_family_dna(fdf)
nostalgia = build_hidden_memories(fdf, period_cards, top_n=20)
relationships = compute_relationships(fdf)

with tabs[0]:
    st.subheader("Family Analytics Dashboard")

    strongest_era = period_cards.iloc[0] if not period_cards.empty else None
    top_dna = family_dna.iloc[0] if not family_dna.empty else None
    top_nostalgia = nostalgia.iloc[0] if not nostalgia.empty else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Média", len(fdf))
    c2.metric("Korszakjelölt", len(period_cards))
    c3.metric("Family DNA főtéma", top_dna["category"] if top_dna is not None else "-")
    c4.metric("Diverzitás", family_diversity_score(fdf))
    c5.metric("Evolution", family_evolution_score(fdf))

    st.markdown("### Top insight kártyák")
    cols = st.columns(3)

    with cols[0]:
        with st.container(border=True):
            st.caption("legerősebb korszak")
            if strongest_era is not None:
                st.markdown(f"### {strongest_era['title']}")
                st.metric("Era score", f"{strongest_era['era_score']}/100")
                st.write(strongest_era["subtitle"])
                if st.button("Képek megnyitása", key="open_top_era", use_container_width=True):
                    set_drilldown("era", strongest_era["title"], topic=strongest_era["topic"], year=int(strongest_era["year"]))
                    st.rerun()
            else:
                st.write("Nincs elég adat.")

    with cols[1]:
        with st.container(border=True):
            st.caption("Family DNA")
            if top_dna is not None:
                st.markdown(f"### {top_dna['category']}")
                st.metric("DNA score", f"{top_dna['dna_score']}/100")
                st.write(f"{int(top_dna['count'])} találat · {top_dna['share_pct']}% arány")
                if st.button("Kategória képei", key="open_top_dna", use_container_width=True):
                    set_drilldown("dna", f"Family DNA · {top_dna['category']}", category=top_dna["category"])
                    st.rerun()
            else:
                st.write("Nincs elég adat.")

    with cols[2]:
        with st.container(border=True):
            st.caption("Nostalgia Score")
            if top_nostalgia is not None:
                title = f"{top_nostalgia['year_month']} · {top_nostalgia['topic']}"
                st.markdown(f"### {title}")
                st.metric("Nostalgia", f"{top_nostalgia['nostalgia_score']}/100")
                st.write(f"{int(top_nostalgia['count'])} kép")
                if st.button("Hidden képek", key="open_top_nostalgia", use_container_width=True):
                    set_drilldown("nostalgia", title, topic=top_nostalgia["topic"], year_month=top_nostalgia["year_month"])
                    st.rerun()
            else:
                st.write("Nincs elég adat.")

    left, right = st.columns(2)

    with left:
        st.markdown("### Korszakmotor")
        if px is not None and not period_cards.empty:
            chart = period_cards.copy()
            chart["label"] = chart["title"] + " · " + chart["period"].astype(str)
            fig = px.bar(
                chart.sort_values("era_score"),
                x="era_score",
                y="label",
                orientation="h",
                labels={"era_score": "Era score", "label": ""},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Kattintás helyett stabil drill-down: válassz az alábbi gyorsgombok közül.")

        if not period_cards.empty:
            quick_cols = st.columns(min(3, len(period_cards)))
            for i, (_, r) in enumerate(period_cards.head(3).iterrows()):
                with quick_cols[i % len(quick_cols)]:
                    if st.button(f"{r['topic']} {int(r['year'])}", key=f"quick_era_{i}", use_container_width=True):
                        set_drilldown("era", f"{r['topic']} {int(r['year'])}", topic=r["topic"], year=int(r["year"]))
                        st.rerun()

    with right:
        st.markdown("### Family DNA")
        if px is not None and not family_dna.empty:
            fig = px.pie(
                family_dna.head(8),
                names="category",
                values="dna_score",
                hole=0.45,
            )
            st.plotly_chart(fig, use_container_width=True)
            if not family_dna.empty:
                dna_cols = st.columns(min(3, len(family_dna)))
                for i, (_, r) in enumerate(family_dna.head(3).iterrows()):
                    with dna_cols[i % len(dna_cols)]:
                        if st.button(str(r["category"]), key=f"quick_dna_{i}", use_container_width=True):
                            set_drilldown("dna", f"Family DNA · {r['category']}", category=r["category"])
                            st.rerun()

    render_drilldown_panel(fdf)

with tabs[1]:
    st.subheader("Korszakmotor V1")
    yt = build_year_topic_table(fdf)
    if yt.empty:
        st.info("Nincs elég témaadat.")
    else:
        st.dataframe(yt[["topic", "year", "count", "share_pct", "growth_pp", "months", "era_score"]].head(50), use_container_width=True)

        if px is not None:
            top_topics = yt.groupby("topic")["era_score"].max().sort_values(ascending=False).head(6).index.tolist()
            trend = yt[yt["topic"].isin(top_topics)]
            fig = px.line(trend, x="year", y="share_pct", color="topic", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        if not period_cards.empty:
            labels = [f"{r.topic} · {int(r.year)} · score {r.era_score}" for r in period_cards.itertuples()]
            choice = st.selectbox("Válassz korszakot", labels)
            if choice:
                topic = choice.split(" · ")[0]
                year = int(choice.split(" · ")[1])
                sub = filter_by_topic_year(fdf, topic, year)
                st.metric("Képek ebben a korszakban", len(sub))
                if is_premium():
                    st.download_button("📦 Album ZIP ebből", create_album_zip(sub, f"{topic}_{year}"), file_name=f"{safe_name(topic)}_{year}.zip", mime="application/zip")
                else:
                    locked_feature_box("Korszak album export")
                render_gallery(sub, max_items=32)

with tabs[2]:
    st.subheader("Family DNA V1")
    st.write("Súlyozott score: 40% arány + 30% darabszám + 20% tartósság + 10% növekedési trend.")
    if family_dna.empty:
        st.info("Nincs Family DNA adat.")
    else:
        st.dataframe(family_dna[["category", "count", "share_pct", "years", "months", "trend_pp", "dna_score"]], use_container_width=True)
        if px is not None:
            fig = px.bar(family_dna.sort_values("dna_score"), x="dna_score", y="category", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Kategória drill-down")
        cat_choice = st.selectbox("Válassz Family DNA kategóriát", family_dna["category"].tolist(), key="dna_category_select")
        if cat_choice:
            sub = fdf[fdf["category"].eq(cat_choice)]
            st.metric("Talált kép", len(sub))
            if st.button("Kiválasztott kategória rögzítése", key="set_dna_drill", use_container_width=True):
                set_drilldown("dna", f"Family DNA · {cat_choice}", category=cat_choice)
                st.rerun()
            render_gallery(sub, max_items=32)

with tabs[3]:
    st.subheader("Nostalgia / Hidden Memories")
    if nostalgia.empty:
        st.info("Nincs nostalgia találat.")
    else:
        st.dataframe(nostalgia[["year_month", "topic", "count", "age_years", "era_strength", "nostalgia_score"]], use_container_width=True)
        if px is not None:
            fig = px.bar(nostalgia.sort_values("nostalgia_score"), x="nostalgia_score", y="year_month", color="topic", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

        labels = [f"{r.year_month} · {r.topic} · score {r.nostalgia_score}" for r in nostalgia.itertuples()]
        choice = st.selectbox("Válassz hidden memory csomagot", labels)
        if choice:
            ym = choice.split(" · ")[0]
            topic = choice.split(" · ")[1]
            sub = filter_by_topic_year(fdf[fdf["year_month"].eq(ym)], topic)
            st.metric("Talált kép", len(sub))
            render_gallery(sub, max_items=32)

with tabs[4]:
    st.subheader("Kapcsolati háló")
    if relationships.empty:
        st.info("Nincs kapcsolatadat.")
    else:
        st.dataframe(relationships, use_container_width=True)
        if go is not None and nx is not None:
            G = nx.Graph()
            for _, r in relationships.iterrows():
                G.add_edge(r["person_a"], r["person_b"], weight=int(r["count"]))
            pos = nx.spring_layout(G, seed=7)
            fig = go.Figure()
            max_w = max([G[u][v].get("weight", 1) for u, v in G.edges()] or [1])
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                weight = G[u][v].get("weight", 1)
                width = 1.5 + 8 * weight / max_w
                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=width),
                    hoverinfo="text",
                    text=[f"{u} – {v}: {weight} közös kép"],
                    showlegend=False,
                ))

            node_x, node_y, text = [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                text.append(node)
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=text, textposition="top center", marker=dict(size=32), hoverinfo="text", showlegend=False))
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=520)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Kapcsolat drill-down")
        rel_labels = [f"{r.person_a} + {r.person_b} · {int(r.count)} kép" for r in relationships.itertuples()]
        rel_choice = st.selectbox("Válassz kapcsolatot", rel_labels, key="rel_select")
        if rel_choice:
            pair = rel_choice.split(" · ")[0]
            a, b = [x.strip() for x in pair.split("+")]
            sub = fdf[fdf["persons"].fillna("").apply(lambda x: a in [v.strip() for v in str(x).split(",")] and b in [v.strip() for v in str(x).split(",")])]
            st.metric("Közös képek", len(sub))
            render_gallery(sub, max_items=32)

with tabs[5]:
    st.subheader("Képek + album")
    st.write(f"Jelenlegi szűrés: **{len(fdf)}** kép.")
    render_gallery(fdf.sort_values("date"), max_items=60)
    st.download_button(
        "📦 Album ZIP a jelenlegi szűrésből",
        create_album_zip(fdf, "filtered_album"),
        file_name="filtered_album.zip",
        mime="application/zip",
        key="gallery_filtered_zip",
        use_container_width=True,
    )

with tabs[6]:
    st.subheader("Export")

    st.markdown("### Szűrt képek / album ZIP")
    st.write(f"A jelenlegi slicerek alapján exportálandó képek száma: **{len(fdf)}**")

    if len(fdf) == 0:
        st.info("Nincs exportálható kép a jelenlegi szűrésben.")
    else:
        st.download_button(
            "📦 Szűrt képek ZIP export",
            data=create_album_zip(fdf, "filtered_album"),
            file_name="lifelens_filtered_album.zip",
            mime="application/zip",
            key="export_filtered_zip_always",
            use_container_width=True,
        )

    st.markdown("### Adat export")
    st.download_button(
        "⬇ Index CSV",
        fdf.to_csv(index=False).encode("utf-8-sig"),
        file_name="lifelens_v8_6_index.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if not period_cards.empty:
        st.download_button(
            "⬇ Korszak score CSV",
            period_cards.to_csv(index=False).encode("utf-8-sig"),
            file_name="lifelens_v8_6_period_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if not family_dna.empty:
        st.download_button(
            "⬇ Family DNA CSV",
            family_dna.to_csv(index=False).encode("utf-8-sig"),
            file_name="lifelens_v8_6_family_dna.csv",
            mime="text/csv",
            use_container_width=True,
        )

with tabs[7]:
    st.subheader("AI ellenőrzés / címkék finomhangolása")
    st.write("Itt látszik, melyik kép milyen AI címkét kapott. Ha sok a téves címke, emeld az AI küszöböt, vagy zárj ki kategóriákat.")
    cols_to_show = [c for c in ["filename", "topics", "ai_suggestions", "ai_reason", "ai_analyzed", "category"] if c in fdf.columns]
    if cols_to_show:
        st.dataframe(fdf[cols_to_show].head(500), use_container_width=True)
        st.download_button(
            "⬇ AI javaslatok CSV",
            fdf[cols_to_show].to_csv(index=False).encode("utf-8-sig"),
            file_name="lifelens_ai_suggestions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Nincs AI címkeinformáció.")
