import io
import os
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
    import cv2
except Exception:
    cv2 = None

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


APP_TITLE = "LifeLens V7.1 – Family Analytics + Deep Analysis"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}
SUPPORTED_EXT = IMAGE_EXT | VIDEO_EXT

PERSON_HINTS = {
    "Lenke": ["lenke", "lencsi"],
    "Zente": ["zente"],
    "Apa": ["apa", "apu", "gabor", "gábor"],
    "Anya": ["anya", "anyu"],
    "Nagyi": ["nagyi", "mama", "nagymama"],
}

TOPIC_PROMPTS = {
    "sakk": ["a child playing chess", "chess board", "people playing chess"],
    "traktor": ["tractor", "child with tractor", "farm vehicle", "toy tractor"],
    "vonat": ["train", "toy train", "railway"],
    "bicikli": ["child riding a bicycle", "bicycle", "balance bike"],
    "foci": ["football soccer", "soccer ball", "child playing football"],
    "fagyi": ["ice cream", "child eating ice cream"],
    "rajzolás": ["child drawing", "colored pencils", "drawing on paper"],
    "karácsony": ["christmas tree", "christmas family", "christmas gifts"],
    "születésnap": ["birthday cake", "birthday party", "child birthday"],
    "nyaralás": ["family vacation", "beach holiday", "travel family"],
    "játszótér": ["playground", "child on playground", "swing slide"],
    "kirándulás": ["family hiking", "mountain landscape", "outdoor trip"],
    "kutya": ["dog", "child with dog"],
    "autó": ["car", "child near car"],
    "iskola": ["school classroom", "child in school", "first day of school"],
    "óvoda": ["kindergarten", "preschool child", "children in kindergarten"],
    "strand": ["beach", "child on beach", "family at beach"],
    "hegy": ["mountain landscape", "child in front of mountain", "family hiking in mountains"],
    "torta": ["cake", "birthday cake", "child with cake"],
    "plüss": ["stuffed animal", "child with stuffed toy", "plush toy"],
    "jármű": ["vehicle", "toy vehicle", "child with vehicle"],
}

EVENT_KEYWORDS = {
    "karácsony": ["karacsony", "karácsony", "christmas", "xmas", "december"],
    "születésnap": ["szulinap", "szülinap", "birthday", "torta", "party", "buli"],
    "nyaralás": ["balaton", "nyaralas", "nyaralás", "holiday", "vacation", "strand", "hotel", "wellness"],
    "iskola/óvoda": ["ovi", "ovoda", "óvoda", "iskola", "bolcsi", "bölcsi", "ballagas", "farsang"],
    "sport": ["foci", "football", "edzes", "edzés", "meccs", "torna", "uszas", "úszás"],
}

ACTIVE_TOPICS = {"bicikli", "foci", "nyaralás", "játszótér", "kirándulás", "traktor", "strand", "hegy"}
MOOD_POSITIVE_TOPICS = {"karácsony", "születésnap", "nyaralás", "játszótér", "fagyi", "torta", "strand"}

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

st.markdown("""
<style>
.insight-card {
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 12px;
    background: #f8fafc;
}
.deep-box {
    border: 1px solid #7c3aed;
    background: #f5f3ff;
    color: #2e1065;
    padding: 16px;
    border-radius: 18px;
    margin-bottom: 12px;
}
.small-muted { color:#64748b; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 LifeLens V7.1 – Family Analytics")
st.caption("Helyi családi képelemző dashboard · gyors index + opcionális Deep Analysis")

st.markdown("""
<div class="deep-box">
<b>V7.1 újdonság:</b> az app először gyors indexet készít. 
Utána külön elindítható a <b>Deep Analysis</b>, amely helyi CLIP AI-val mélyebben elemzi a képtartalmat.
</div>
""", unsafe_allow_html=True)


def normalize_text(text: str) -> str:
    import unicodedata
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("\\", " ").replace("/", " ").replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def safe_name(text: str) -> str:
    text = str(text or "album").strip()
    text = text.replace("ő", "o").replace("ű", "u").replace("Ő", "O").replace("Ű", "U")
    text = re.sub(r"[^\w\- .áéíóöúüÁÉÍÓÖÚÜ]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120] or "album"


def md5_file(path: Path, chunk_size=1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


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


def fallback_datetime(path: Path):
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
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


def iter_media_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p


@st.cache_resource(show_spinner=False)
def load_clip():
    if CLIPModel is None or CLIPProcessor is None or torch is None:
        return None, None, None
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor, device


def clip_analyze_image(img: Image.Image, threshold: float = 0.20):
    model, processor, device = load_clip()
    if model is None:
        return {}, [], ""
    labels = []
    prompts = []
    for topic, ps in TOPIC_PROMPTS.items():
        for p in ps:
            labels.append(topic)
            prompts.append(p)

    inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu().numpy()

    best_by_topic = {}
    for label, prob in zip(labels, probs):
        best_by_topic[label] = max(best_by_topic.get(label, 0), float(prob))

    selected = [k for k, v in best_by_topic.items() if v >= threshold]
    selected = sorted(selected, key=lambda k: best_by_topic[k], reverse=True)
    reason = ", ".join([f"{k}:{best_by_topic[k]:.2f}" for k in selected[:7]])
    return best_by_topic, selected[:10], reason


def infer_filename_tags(path: Path, dt):
    text = normalize_text(f"{path.name} {path.parent}")
    topics = set()
    events = set()
    persons = set()

    for person, hints in PERSON_HINTS.items():
        if any(normalize_text(h) in text for h in hints):
            persons.add(person)

    for topic in TOPIC_PROMPTS.keys():
        if normalize_text(topic) in text:
            topics.add(topic)

    for event, words in EVENT_KEYWORDS.items():
        if any(normalize_text(w) in text for w in words):
            events.add(event)
            if event in TOPIC_PROMPTS:
                topics.add(event)

    if dt:
        if dt.month == 12:
            events.add("karácsony")
            topics.add("karácsony")
        if dt.month in [6, 7, 8]:
            topics.add("nyaralás")

    return sorted(persons), sorted(topics), sorted(events)


def get_video_preview(path: Path, preview_dir: Path):
    if cv2 is None:
        return "", None, None, None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return "", None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = round(frames / fps, 1) if fps else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames * 0.35) if frames else 0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return "", width, height, duration
    preview_dir.mkdir(parents=True, exist_ok=True)
    out = preview_dir / (hashlib.md5(str(path).encode()).hexdigest() + ".jpg")
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img.save(out, quality=85)
    return str(out), width, height, duration


def scan_media(root: Path, limit: int):
    files = list(iter_media_files(root))[:limit]
    rows = []
    preview_dir = Path(st.session_state.work_dir) / "previews"
    progress = st.progress(0, text="Gyors index készül...")

    for i, path in enumerate(files):
        media_type = "video" if path.suffix.lower() in VIDEO_EXT else "image"
        try:
            dt = fallback_datetime(path)
            preview_path, width, height, duration = "", None, None, None
            q = 0

            if media_type == "image":
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    exif_dt = get_exif_datetime(img)
                    if exif_dt:
                        dt = exif_dt
                    width, height = img.size
                    q = quality_score(img)
                    preview_path = str(path)
            else:
                preview_path, width, height, duration = get_video_preview(path, preview_dir)
                if preview_path:
                    try:
                        with Image.open(preview_path) as img:
                            q = quality_score(img.convert("RGB"))
                    except Exception:
                        q = 0

            persons, topics, events = infer_filename_tags(path, dt)
            active_score = sum(1 for t in topics if t in ACTIVE_TOPICS)
            mood_score = sum(1 for t in topics if t in MOOD_POSITIVE_TOPICS)

            rows.append({
                "media_type": media_type,
                "path": str(path),
                "filename": path.name,
                "folder": str(path.parent),
                "preview_path": preview_path,
                "size_mb": round(path.stat().st_size / (1024*1024), 2),
                "md5": md5_file(path),
                "date": dt,
                "year": dt.year if dt else None,
                "month": dt.month if dt else None,
                "year_month": dt.strftime("%Y-%m") if dt else "Dátum nélkül",
                "width": width,
                "height": height,
                "duration_sec": duration,
                "quality_score": q,
                "persons": ", ".join(persons),
                "topics": ", ".join(topics),
                "events": ", ".join(sorted(set(events))),
                "ai_topics": "",
                "ai_reason": "",
                "deep_analyzed": False,
                "active_flag": active_score > 0,
                "positive_visual_flag": mood_score > 0,
                "person_count": len(persons),
            })
        except Exception as exc:
            rows.append({"media_type": media_type, "path": str(path), "filename": path.name, "error": str(exc)})
        progress.progress((i + 1) / max(1, len(files)), text=f"Gyors index: {i+1}/{len(files)}")
    progress.empty()
    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df = df.sort_values("date", na_position="last")
    return df


def run_deep_analysis(df: pd.DataFrame, threshold: float, max_items: int):
    if df.empty:
        return df
    if CLIPModel is None or torch is None:
        st.error("A Deep Analysis-hez telepíteni kell: torch + transformers. Futtasd az INSTALL_AI_IMAGE_RECOGNITION.bat fájlt.")
        return df

    work = df.copy()
    candidates = work[~work.get("deep_analyzed", False).fillna(False)].head(max_items).copy()
    progress = st.progress(0, text="Deep Analysis indul...")

    for idx_i, (idx, row) in enumerate(candidates.iterrows()):
        p = Path(str(row.get("preview_path") or row.get("path") or ""))
        try:
            if p.exists():
                with Image.open(p) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    _, ai_topics, ai_reason = clip_analyze_image(img, threshold)
                    old_topics = [x.strip() for x in str(work.at[idx, "topics"]).split(",") if x.strip()]
                    merged = sorted(set(old_topics) | set(ai_topics))
                    work.at[idx, "topics"] = ", ".join(merged)
                    work.at[idx, "ai_topics"] = ", ".join(ai_topics)
                    work.at[idx, "ai_reason"] = ai_reason
                    work.at[idx, "deep_analyzed"] = True
                    work.at[idx, "active_flag"] = any(t in ACTIVE_TOPICS for t in merged)
                    work.at[idx, "positive_visual_flag"] = any(t in MOOD_POSITIVE_TOPICS for t in merged)
        except Exception as exc:
            work.at[idx, "ai_reason"] = f"Deep error: {exc}"
            work.at[idx, "deep_analyzed"] = True
        progress.progress((idx_i + 1) / max(1, len(candidates)), text=f"Deep Analysis: {idx_i+1}/{len(candidates)}")
    progress.empty()
    return work


def split_values(series):
    vals = []
    for item in series.fillna(""):
        vals += [x.strip() for x in str(item).split(",") if x.strip()]
    return vals


def explode_values(df, col):
    out = []
    for _, r in df.iterrows():
        for v in [x.strip() for x in str(r.get(col, "")).split(",") if x.strip()]:
            row = r.to_dict()
            row["value"] = v
            out.append(row)
    return pd.DataFrame(out)


def compute_topic_year_scores(df):
    if df.empty or "topics" not in df.columns:
        return pd.DataFrame()
    total_by_year = df.groupby("year").size().rename("year_total")
    ex = explode_values(df, "topics")
    if ex.empty:
        return pd.DataFrame()
    g = ex.groupby(["year", "value"]).agg(
        count=("path", "count"),
        videos=("media_type", lambda s: int((s == "video").sum())),
        avg_quality=("quality_score", "mean"),
        months=("month", "nunique"),
        persons=("persons", lambda s: len(set(split_values(s))))
    ).reset_index()
    g = g.merge(total_by_year, on="year", how="left")
    g["share_pct"] = (g["count"] / g["year_total"] * 100).round(1)
    max_count = max(g["count"].max(), 1)
    g["period_score"] = (
        0.40 * (g["share_pct"] / max(g["share_pct"].max(), 1) * 100) +
        0.30 * (g["count"] / max_count * 100) +
        0.20 * (g["months"] / 12 * 100) +
        0.10 * (g["persons"] / max(g["persons"].max(), 1) * 100)
    ).round(1)
    g = g.rename(columns={"value": "topic"})
    return g.sort_values("period_score", ascending=False)


def compute_relationships(df):
    rows = []
    for _, r in df.iterrows():
        people = [x.strip() for x in str(r.get("persons", "")).split(",") if x.strip()]
        for i in range(len(people)):
            for j in range(i+1, len(people)):
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


def compute_family_scores(df):
    if df.empty:
        return pd.DataFrame()
    out = df.groupby("year").agg(
        total=("path", "count"),
        active=("active_flag", "sum"),
        positive=("positive_visual_flag", "sum"),
        together=("person_count", lambda s: int((s >= 2).sum())),
        four_plus=("person_count", lambda s: int((s >= 4).sum())),
    ).reset_index()
    out["activity_index"] = (out["active"] / out["total"] * 100).round(1)
    out["visual_positive_index"] = (out["positive"] / out["total"] * 100).round(1)
    out["togetherness_score"] = ((out["together"] / out["total"] * 70) + (out["four_plus"] / out["total"] * 30)).round(1)
    return out


def hidden_memories(df, min_age_years=3, top_n=50):
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    now = pd.Timestamp.now()
    tmp = df.copy()
    tmp["date_dt"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp["age_years"] = ((now - tmp["date_dt"]).dt.days / 365.25).fillna(0)
    tmp["topic_count"] = tmp["topics"].fillna("").apply(lambda x: len([v for v in str(x).split(",") if v.strip()]))
    tmp["person_bonus"] = tmp["person_count"].fillna(0) * 8
    tmp["hidden_score"] = (
        tmp["age_years"].clip(0, 10) * 6 +
        tmp["quality_score"].fillna(0).rank(pct=True) * 35 +
        tmp["topic_count"] * 6 +
        tmp["person_bonus"]
    ).round(1)
    return tmp[tmp["age_years"] >= min_age_years].sort_values("hidden_score", ascending=False).head(top_n)


def create_album_zip(rows_df, name):
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


def render_gallery(rows_df, max_items=24):
    if rows_df is None or rows_df.empty:
        st.info("Nincs megjeleníthető kép/videó.")
        return
    cols = st.columns(4)
    for i, (_, row) in enumerate(rows_df.head(max_items).iterrows()):
        p = Path(str(row.get("preview_path") or row.get("path") or ""))
        with cols[i % 4]:
            if p.exists():
                st.image(str(p), caption=f"{row.get('filename','')} · {row.get('topics','')}", use_container_width=True)
            else:
                st.write(row.get("filename", ""))


if "work_dir" not in st.session_state:
    st.session_state.work_dir = tempfile.mkdtemp(prefix="lifelens_v7_1_")

with st.sidebar:
    st.header("1. Forrás")
    folder = st.text_input("Helyi képmappa útvonala", value="")
    st.caption("Példa: C:\\Users\\Gabor\\Pictures vagy külső HDD mappája")

    st.header("2. Gyors index")
    limit = st.number_input("Max. fájl első teszthez", min_value=100, max_value=200000, value=3000, step=100)
    if st.button("Gyors index indítása", use_container_width=True):
        p = Path(folder)
        if not p.exists() or not p.is_dir():
            st.error("Nem találom ezt a mappát.")
        else:
            df = scan_media(p, int(limit))
            st.session_state["df"] = df
            save_path = p / "lifelens_v7_1_index.csv"
            try:
                df.to_csv(save_path, index=False, encoding="utf-8-sig")
                st.success(f"Index mentve: {save_path}")
            except Exception as exc:
                st.warning(f"Index készült, de nem tudtam menteni: {exc}")

    uploaded_index = st.file_uploader("Korábbi lifelens_v7_1_index.csv betöltése", type=["csv"])
    if uploaded_index:
        st.session_state["df"] = pd.read_csv(uploaded_index)
        st.success("Index betöltve.")

    st.header("3. Deep Analysis")
    deep_threshold = st.slider("AI találati küszöb", 0.05, 0.45, 0.20, 0.01)
    deep_limit = st.number_input("Deep Analysis max. elem / futás", min_value=50, max_value=50000, value=500, step=50)
    st.caption("Javaslat: először 500-1000 képpel teszteld. Nagy archívumnál hosszú lehet.")
    if CLIPModel is None or torch is None:
        st.warning("AI csomag nincs telepítve. Futtasd: INSTALL_AI_IMAGE_RECOGNITION.bat")

df = st.session_state.get("df", pd.DataFrame())

if not df.empty:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
    if "deep_analyzed" not in df.columns:
        df["deep_analyzed"] = False

if df.empty:
    st.markdown("## Első lépés")
    st.write("Adj meg egy helyi képmappát bal oldalon, majd indítsd a gyors indexet.")
    st.write("Utána külön indíthatod a Deep Analysis-t, ami AI-val megnézi a képek tartalmát.")
    st.stop()

st.sidebar.header("4. Slicerek")
all_people = sorted(set(split_values(df.get("persons", pd.Series(dtype=str)))))
all_topics = sorted(set(split_values(df.get("topics", pd.Series(dtype=str)))))
years = sorted([int(y) for y in df["year"].dropna().unique()]) if "year" in df.columns else []

sel_years = st.sidebar.multiselect("Év", years, default=years)
sel_people = st.sidebar.multiselect("Személy", all_people)
sel_topics = st.sidebar.multiselect("Téma / korszak", all_topics)
media_filter = st.sidebar.multiselect("Média", ["image", "video"], default=["image", "video"])

if st.sidebar.button("🔥 Deep Analysis indítása", use_container_width=True):
    df2 = run_deep_analysis(df, float(deep_threshold), int(deep_limit))
    st.session_state["df"] = df2
    if folder:
        try:
            p = Path(folder)
            if p.exists():
                df2.to_csv(p / "lifelens_v7_1_index.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass
    st.rerun()

fdf = df.copy()
if sel_years:
    fdf = fdf[fdf["year"].isin(sel_years)]
if sel_people:
    mask = fdf["persons"].fillna("").apply(lambda x: all(p in [v.strip() for v in str(x).split(",")] for p in sel_people))
    fdf = fdf[mask]
if sel_topics:
    mask = fdf["topics"].fillna("").apply(lambda x: all(t in [v.strip() for v in str(x).split(",")] for t in sel_topics))
    fdf = fdf[mask]
if media_filter:
    fdf = fdf[fdf["media_type"].isin(media_filter)]

tabs = st.tabs([
    "📊 Family Dashboard",
    "🔥 Deep Analysis állapot",
    "🧭 Korszak Explorer",
    "🔗 Kapcsolati háló",
    "📈 Trendek",
    "🕵️ Elfelejtett emlékek",
    "🖼️ Képgaléria / Album",
    "📦 Export"
])

with tabs[0]:
    st.subheader("Family Insights Dashboard")
    topic_scores = compute_topic_year_scores(fdf)
    family_scores = compute_family_scores(fdf)
    rel = compute_relationships(fdf)
    hidden = hidden_memories(fdf, min_age_years=3, top_n=30)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Média", len(fdf))
    c2.metric("Deep elemzett", int(fdf.get("deep_analyzed", pd.Series(False)).fillna(False).sum()))
    c3.metric("Témák", len(all_topics))
    c4.metric("Korszak-jelölt", len(topic_scores[topic_scores["period_score"] >= 55]) if not topic_scores.empty else 0)
    c5.metric("Elfelejtett emlék", len(hidden))

    st.markdown("### Top insight kártyák")
    cards = []
    if not topic_scores.empty:
        top = topic_scores.iloc[0]
        cards.append(("legerősebb korszak", f"{top['topic']} – {int(top['year'])}", f"{int(top['count'])} kép/videó · {top['share_pct']}% arány"))
    if not rel.empty:
        top = rel.iloc[0]
        cards.append(("legerősebb kapcsolat", f"{top['person_a']} + {top['person_b']}", f"{int(top['count'])} közös kép/videó"))
    if not family_scores.empty:
        top = family_scores.sort_values("activity_index", ascending=False).iloc[0]
        cards.append(("legaktívabb év", f"{int(top['year'])}", f"Aktivitási index: {top['activity_index']}"))
    if not hidden.empty:
        top = hidden.iloc[0]
        cards.append(("rejtett emlék", str(top.get("filename","")), f"Hidden score: {top.get('hidden_score','')}"))
    if all_topics:
        top_topic = pd.Series(split_values(fdf.get("topics", pd.Series(dtype=str)))).value_counts().head(1)
        if not top_topic.empty:
            cards.append(("Family DNA főtéma", top_topic.index[0], f"{int(top_topic.iloc[0])} találat"))

    cols = st.columns(2)
    for i, (label, title, desc) in enumerate(cards):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="insight-card">
            <div class="small-muted">{label}</div>
            <h3>{title}</h3>
            <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    if px is not None and not topic_scores.empty:
        st.markdown("### Top korszak-jelöltek")
        top_chart = topic_scores.head(12).copy()
        top_chart["label"] = top_chart["topic"] + " · " + top_chart["year"].astype(int).astype(str)
        fig = px.bar(top_chart.sort_values("period_score"), x="period_score", y="label", orientation="h",
                     labels={"period_score": "Korszak score", "label": ""})
        st.plotly_chart(fig, use_container_width=True)

    if px is not None and not family_scores.empty:
        st.markdown("### Családi indexek évenként")
        fig = px.line(family_scores, x="year", y=["activity_index", "togetherness_score", "visual_positive_index"],
                      markers=True, labels={"value": "Index", "year": "Év", "variable": "Mutató"})
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Deep Analysis állapot")
    total = len(df)
    done = int(df.get("deep_analyzed", pd.Series(False)).fillna(False).sum())
    st.metric("Elemzett arány", f"{done}/{total}")
    if total:
        st.progress(done / total)
    st.write("A Deep Analysis a képek tényleges tartalma alapján bővíti a témákat, ezért jobb korszakokat és Family DNA-t ad.")
    if "ai_reason" in df.columns:
        st.dataframe(df[df.get("deep_analyzed", False).fillna(False)][["filename", "topics", "ai_topics", "ai_reason"]].head(200), use_container_width=True)

with tabs[2]:
    st.subheader("Korszak Explorer")
    topic_scores = compute_topic_year_scores(fdf)
    if topic_scores.empty:
        st.info("Nincs elég témaadat.")
    else:
        st.dataframe(topic_scores[["topic", "year", "count", "share_pct", "months", "period_score"]], use_container_width=True)
        labels = [f"{r.topic} · {int(r.year)}" for r in topic_scores.itertuples()]
        choice = st.selectbox("Válassz korszakot drill-downhoz", labels)
        if choice:
            topic, year = choice.split(" · ")
            year = int(year)
            sub = fdf[(fdf["year"] == year) & (fdf["topics"].fillna("").str.contains(re.escape(topic), case=False, regex=True))]
            st.metric("Talált média", len(sub))
            st.download_button("📦 Album ZIP ebből", create_album_zip(sub, choice), file_name=f"{safe_name(choice)}.zip", mime="application/zip")
            st.caption("Tipp: bal oldalon tegyél rá plusz személy-szűrőt, pl. Zente.")
            render_gallery(sub, 32)

with tabs[3]:
    st.subheader("Kapcsolati háló")
    rel = compute_relationships(fdf)
    if rel.empty:
        st.info("A hálóhoz személycímkék kellenek. Első körben fájlnév/mappa alapján működik, később jöhet arcfelismerés-tanítás.")
    else:
        st.dataframe(rel, use_container_width=True)
        if go is not None and nx is not None:
            G = nx.Graph()
            for _, r in rel.iterrows():
                G.add_edge(r["person_a"], r["person_b"], weight=int(r["count"]))
            pos = nx.spring_layout(G, seed=7)
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
            node_x, node_y, text = [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x); node_y.append(y); text.append(node)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=2), hoverinfo="none"))
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=text, textposition="top center",
                                     marker=dict(size=28), hoverinfo="text"))
            fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=10,b=10), height=500)
            st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Érdeklődési trendek")
    ex = explode_values(fdf, "topics")
    if ex.empty:
        st.info("Nincs témaadat.")
    else:
        top_topics = pd.Series(ex["value"]).value_counts().head(8).index.tolist()
        chosen = st.multiselect("Témák", sorted(set(ex["value"])), default=top_topics[:5])
        trend = ex[ex["value"].isin(chosen)].groupby(["year", "value"]).size().reset_index(name="count")
        if px is not None and not trend.empty:
            fig = px.line(trend, x="year", y="count", color="value", markers=True, labels={"count": "Kép/videó", "year": "Év", "value": "Téma"})
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(trend, use_container_width=True)

with tabs[5]:
    st.subheader("Elfelejtett emlékek")
    min_age = st.slider("Minimum életkor évben", 1, 10, 3)
    hidden = hidden_memories(fdf, min_age_years=min_age, top_n=100)
    if hidden.empty:
        st.info("Nincs találat.")
    else:
        st.dataframe(hidden[["filename", "date", "topics", "persons", "quality_score", "hidden_score"]], use_container_width=True)
        st.download_button("📦 Hidden Memories album ZIP", create_album_zip(hidden, "hidden_memories"), file_name="hidden_memories.zip", mime="application/zip")
        render_gallery(hidden, 40)

with tabs[6]:
    st.subheader("Képgaléria / Album")
    st.write(f"A jelenlegi slicerek alapján: **{len(fdf)}** média.")
    sort_by = st.selectbox("Rendezés", ["date", "quality_score", "filename"], index=0)
    asc = st.checkbox("Növekvő", value=True)
    gallery = fdf.sort_values(sort_by, ascending=asc, na_position="last") if sort_by in fdf.columns else fdf
    st.download_button("📦 Album ZIP a jelenlegi szűrésből", create_album_zip(gallery, "filtered_album"), file_name="filtered_album.zip", mime="application/zip")
    render_gallery(gallery, 60)

with tabs[7]:
    st.subheader("Export")
    st.download_button("⬇ Index CSV letöltése", fdf.to_csv(index=False).encode("utf-8-sig"), file_name="lifelens_v7_1_filtered_index.csv", mime="text/csv")
    full_scores = compute_topic_year_scores(df)
    if not full_scores.empty:
        st.download_button("⬇ Korszak score CSV", full_scores.to_csv(index=False).encode("utf-8-sig"), file_name="lifelens_v7_1_period_scores.csv", mime="text/csv")
    st.caption("Későbbi V7.2: személytanítás / arcfelismerés, PDF dashboard export, HTML Family Wrapped.")
