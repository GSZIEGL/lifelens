
import io, re, zipfile, hashlib, tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageStat, ExifTags

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

APP_TITLE = "LifeLens V8 – Family Analytics"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
SUPPORTED_EXT = IMAGE_EXT | {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}
FREE_IMAGE_LIMIT = 500
PREMIUM_UNLOCK_CODE = "LIFELENS-PREMIUM-DEMO"

TOPIC_TO_CATEGORY = {
    "vonat": "Járművek", "traktor": "Járművek", "autó": "Járművek", "jármű": "Járművek",
    "bicikli": "Sport / mozgás", "foci": "Sport / mozgás", "túrázás": "Utazás / természet",
    "hegy": "Utazás / természet", "nyaralás": "Utazás / természet", "strand": "Utazás / természet",
    "állatkert": "Állatok", "kutya": "Állatok", "rajz": "Kreatív", "gitár": "Kreatív", "zene": "Kreatív",
    "sakk": "Tanulás / játék", "társasjáték": "Tanulás / játék", "karácsony": "Ünnepek",
    "születésnap": "Ünnepek", "játszótér": "Játék", "plüss": "Játék", "fagyi": "Családi pillanatok",
}

PERSON_HINTS = {
    "Lili": ["lili", "lenke", "kislany", "kislány"],
    "Marci": ["marci", "zente", "kisfiu", "kisfiú"],
    "Anya": ["anya", "anyu"],
    "Apa": ["apa", "apu"],
}

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.markdown("""
<style>
.insight-card{border:1px solid #e2e8f0;border-radius:18px;padding:16px;margin-bottom:12px;background:#f8fafc;min-height:140px}
.locked-card{border:1px solid #f59e0b;border-radius:18px;padding:16px;margin-bottom:12px;background:#fffbeb}
.small-muted{color:#64748b;font-size:.9rem}.big-kpi{font-size:2.1rem;font-weight:800;margin:0}
</style>
""", unsafe_allow_html=True)
st.title("📊 LifeLens V8 – Family Analytics")
st.caption("Korszakmotor · Family DNA · Nostalgia Score · Demo család · Dashboardból album")
st.info("V8: beépített demo család Cloudban is kipróbálható saját képek feltöltése nélkül.")

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

def is_premium() -> bool:
    return bool(st.session_state.get("premium_unlocked", False))

def premium_badge():
    if is_premium(): st.success("✅ Premium mód aktív")
    else: st.warning(f"🔒 Free mód · max {FREE_IMAGE_LIMIT} média")

def locked_feature_box(feature_name: str):
    st.markdown(f"""<div class="locked-card"><b>🔒 {feature_name}</b><br>Premium funkció. A preview látható, de az export / teljes elemzés zárolt.</div>""", unsafe_allow_html=True)

def split_values(series):
    vals = []
    for item in series.fillna(""):
        vals += [x.strip() for x in str(item).split(",") if x.strip()]
    return vals

def explode_values(df, col):
    out = []
    if df.empty or col not in df.columns: return pd.DataFrame()
    for _, r in df.iterrows():
        for v in [x.strip() for x in str(r.get(col, "")).split(",") if x.strip()]:
            row = r.to_dict(); row["value"] = v; out.append(row)
    return pd.DataFrame(out)

def make_demo_image(path: Path, title: str, subtitle: str, year: int, topics: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = abs(hash(str(path))) % (2**32)
    rng = np.random.default_rng(seed)
    bg = tuple(int(x) for x in rng.integers(150, 235, size=3))
    img = Image.new("RGB", (900, 650), bg)
    draw = ImageDraw.Draw(img)
    for _ in range(8):
        x0, y0 = int(rng.integers(0, 850)), int(rng.integers(0, 600))
        x1, y1 = x0 + int(rng.integers(40, 130)), y0 + int(rng.integers(40, 130))
        color = tuple(int(x) for x in rng.integers(80, 180, size=3))
        draw.ellipse([x0, y0, x1, y1], outline=color, width=4)
    draw.rectangle([30, 420, 870, 620], fill=(255, 255, 255))
    draw.text((55, 445), title, fill=(15, 23, 42))
    draw.text((55, 485), subtitle, fill=(51, 65, 85))
    draw.text((55, 525), f"{year} · {topics}", fill=(100, 116, 139))
    draw.text((55, 565), "DEMO FAMILY – synthetic placeholder image", fill=(148, 163, 184))
    img.save(path, quality=90)

@st.cache_data(show_spinner=False)
def build_demo_family() -> pd.DataFrame:
    root = Path(tempfile.gettempdir()) / "lifelens_v8_demo_family"
    root.mkdir(parents=True, exist_ok=True)
    plan = [
        (2019, "vonat", "Vonatos korszak", 70),
        (2020, "traktor", "Traktoros korszak", 95),
        (2021, "állatkert", "Állatkertes év", 80),
        (2022, "rajz", "Rajzos-kreatív korszak", 90),
        (2023, "sakk", "Sakk korszak", 115),
        (2024, "gitár", "Gitáros-zenei korszak", 105),
        (2025, "túrázás", "Kirándulós év", 85),
    ]
    rows, idx = [], 0
    people_pool = [["Lili"], ["Marci"], ["Lili","Marci"], ["Apa","Lili"], ["Anya","Marci"], ["Apa","Anya","Lili","Marci"]]
    for year, main_topic, era_name, count in plan:
        for i in range(count):
            month, day = (i % 12) + 1, (i % 25) + 1
            extra = []
            if month in [6,7,8]: extra.append("nyaralás")
            if month == 12: extra.append("karácsony")
            if i % 17 == 0: extra.append("születésnap")
            if i % 13 == 0: extra.append("játszótér")
            if i % 19 == 0: extra.append("fagyi")
            topics = sorted(set([main_topic] + extra))
            category = TOPIC_TO_CATEGORY.get(main_topic, "Egyéb")
            persons = people_pool[i % len(people_pool)]
            date = pd.Timestamp(year=year, month=month, day=day)
            filename = f"{year}_{month:02d}_{safe_name(main_topic)}_{i:03d}.jpg"
            img_path = root / str(year) / safe_name(main_topic) / filename
            if not img_path.exists():
                make_demo_image(img_path, era_name, "Kovács család – fiktív demo", year, ", ".join(topics))
            rows.append({
                "media_id": f"demo_{idx}", "media_type": "image", "path": str(img_path), "filename": filename,
                "folder": str(img_path.parent), "preview_path": str(img_path), "size_mb": round(img_path.stat().st_size/(1024*1024),3),
                "md5": hashlib.md5(str(img_path).encode()).hexdigest(), "date": date, "year": year, "month": month,
                "year_month": f"{year}-{month:02d}", "width": 900, "height": 650, "duration_sec": None,
                "quality_score": 65 + (i % 25), "persons": ", ".join(persons), "topics": ", ".join(topics),
                "main_topic": main_topic, "category": category,
                "events": ", ".join([t for t in topics if t in ["karácsony","születésnap","nyaralás"]]),
                "deep_analyzed": True, "demo": True,
            })
            idx += 1
    return pd.DataFrame(rows).sort_values("date")

def get_exif_datetime(img):
    try:
        exif = img.getexif()
        if not exif: return None
        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
            if key in tag_map:
                try: return datetime.strptime(str(tag_map[key]), "%Y:%m:%d %H:%M:%S")
                except Exception: pass
    except Exception: return None
    return None

def fallback_datetime(path: Path):
    try: return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception: return None

def quality_score(img: Image.Image) -> float:
    try:
        small = ImageOps.grayscale(img).resize((256,256))
        arr = np.asarray(small, dtype=np.float32)
        sharp = np.diff(arr, axis=1).var() + np.diff(arr, axis=0).var()
        stat = ImageStat.Stat(small.resize((128,128)))
        mean, std = stat.mean[0], stat.stddev[0]
        exposure = max(0, 1 - abs(mean-128)/128)*60 + min(std/64, 1)*40
        return round(float(sharp*.7 + exposure*.3), 2)
    except Exception: return 0.0

def infer_filename_tags(path: Path, dt):
    text = normalize_text(f"{path.name} {path.parent}")
    topics, persons = set(), set()
    for person, hints in PERSON_HINTS.items():
        if any(normalize_text(h) in text for h in hints): persons.add(person)
    for topic in TOPIC_TO_CATEGORY:
        if normalize_text(topic) in text: topics.add(topic)
    if dt:
        if dt.month == 12: topics.add("karácsony")
        if dt.month in [6,7,8]: topics.add("nyaralás")
    return sorted(persons), sorted(topics)

def iter_media_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p

def md5_file(path: Path, chunk_size=1024*1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size): h.update(chunk)
    return h.hexdigest()

def scan_zip(uploaded_file, limit: int):
    temp_root = Path(tempfile.mkdtemp(prefix="lifelens_zip_"))
    with zipfile.ZipFile(uploaded_file) as z: z.extractall(temp_root)
    return scan_folder(temp_root, limit)

def scan_folder(root: Path, limit: int):
    files = list(iter_media_files(root))[:limit]
    rows = []
    progress = st.progress(0, text="Indexelés...")
    for i, path in enumerate(files):
        media_type = "video" if path.suffix.lower() not in IMAGE_EXT else "image"
        try:
            dt = fallback_datetime(path)
            preview_path = str(path) if media_type == "image" else ""
            width = height = None; q = 0
            if media_type == "image":
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    exif_dt = get_exif_datetime(img)
                    if exif_dt: dt = exif_dt
                    width, height = img.size
                    q = quality_score(img)
            persons, topics = infer_filename_tags(path, dt)
            category = TOPIC_TO_CATEGORY.get(topics[0], "Egyéb") if topics else "Egyéb"
            rows.append({
                "media_id": f"user_{i}", "media_type": media_type, "path": str(path), "filename": path.name,
                "folder": str(path.parent), "preview_path": preview_path, "size_mb": round(path.stat().st_size/(1024*1024),2),
                "md5": md5_file(path), "date": dt, "year": dt.year if dt else None, "month": dt.month if dt else None,
                "year_month": dt.strftime("%Y-%m") if dt else "Dátum nélkül", "width": width, "height": height, "duration_sec": None,
                "quality_score": q, "persons": ", ".join(persons), "topics": ", ".join(topics),
                "main_topic": topics[0] if topics else "", "category": category, "events": "", "deep_analyzed": False, "demo": False,
            })
        except Exception as exc:
            rows.append({"media_id": f"user_{i}", "media_type": media_type, "path": str(path), "filename": path.name, "error": str(exc)})
        progress.progress((i+1)/max(1,len(files)), text=f"Indexelés: {i+1}/{len(files)}")
    progress.empty()
    return pd.DataFrame(rows).sort_values("date", na_position="last") if rows else pd.DataFrame()

def build_year_topic_table(df):
    ex = explode_values(df, "topics")
    if ex.empty: return pd.DataFrame()
    total_by_year = df.groupby("year").size().rename("year_total").reset_index()
    g = ex.groupby(["year","value"]).agg(count=("path","count"), months=("month","nunique"), avg_quality=("quality_score","mean"), persons=("persons", lambda s: len(set(split_values(s))))).reset_index().rename(columns={"value":"topic"})
    g = g.merge(total_by_year, on="year", how="left")
    g["share_pct"] = (g["count"] / g["year_total"] * 100).round(2)
    g = g.sort_values(["topic","year"])
    g["prev_share_pct"] = g.groupby("topic")["share_pct"].shift(1).fillna(0)
    g["growth_pp"] = (g["share_pct"] - g["prev_share_pct"]).round(2)
    max_count, max_share, max_growth = max(g["count"].max(),1), max(g["share_pct"].max(),1), max(g["growth_pp"].max(),1)
    g["era_score"] = (0.45*(g["growth_pp"].clip(lower=0)/max_growth*100) + 0.25*(g["share_pct"]/max_share*100) + 0.20*(g["count"]/max_count*100) + 0.10*(g["months"]/12*100)).round(1)
    return g.sort_values("era_score", ascending=False)

def build_period_cards(df, top_n=10):
    yt = build_year_topic_table(df)
    if yt.empty: return pd.DataFrame()
    cards = yt[yt["era_score"] >= 20].copy()
    cards["title"] = cards["topic"].str.capitalize() + " korszak"
    cards["period"] = cards["year"].astype(int).astype(str)
    cards["subtitle"] = cards["count"].astype(int).astype(str)+" kép/videó · "+cards["share_pct"].astype(str)+"% arány · +"+cards["growth_pp"].clip(lower=0).astype(str)+" pp növekedés"
    return cards.sort_values("era_score", ascending=False).head(top_n)

def build_family_dna(df):
    ex = explode_values(df, "topics")
    if ex.empty: return pd.DataFrame()
    ex["category"] = ex["value"].map(TOPIC_TO_CATEGORY).fillna("Egyéb")
    total = len(df)
    g = ex.groupby("category").agg(count=("path","count"), years=("year","nunique"), months=("year_month","nunique")).reset_index()
    g["share_pct"] = (g["count"]/max(total,1)*100).round(2)
    by_year = ex.groupby(["category","year"]).size().reset_index(name="year_count")
    by_year_total = df.groupby("year").size().reset_index(name="total")
    by_year = by_year.merge(by_year_total, on="year", how="left")
    by_year["year_share"] = by_year["year_count"]/by_year["total"]*100
    trends = []
    for cat, part in by_year.groupby("category"):
        part = part.sort_values("year")
        trends.append({"category": cat, "trend_pp": float(part["year_share"].iloc[-1] - part["year_share"].iloc[0]) if len(part)>=2 else 0})
    g = g.merge(pd.DataFrame(trends), on="category", how="left").fillna({"trend_pp":0})
    max_count, max_share, max_months, max_trend = max(g["count"].max(),1), max(g["share_pct"].max(),1), max(g["months"].max(),1), max(g["trend_pp"].clip(lower=0).max(),1)
    g["dna_score"] = (0.40*(g["share_pct"]/max_share*100)+0.30*(g["count"]/max_count*100)+0.20*(g["months"]/max_months*100)+0.10*(g["trend_pp"].clip(lower=0)/max_trend*100)).round(1)
    return g.sort_values("dna_score", ascending=False)

def build_hidden_memories(df, period_cards, top_n=20):
    ex = explode_values(df, "topics")
    if ex.empty: return pd.DataFrame()
    g = ex.groupby(["year","year_month","value"]).agg(count=("path","count"), avg_quality=("quality_score","mean"), persons=("persons", lambda s: len(set(split_values(s))))).reset_index().rename(columns={"value":"topic"})
    strength = {}
    if not period_cards.empty:
        for _, r in period_cards.iterrows(): strength[(int(r["year"]), r["topic"])] = float(r["era_score"])
    now_year = pd.Timestamp.now().year
    g["era_strength"] = g.apply(lambda r: strength.get((int(r["year"]), r["topic"]), 0), axis=1)
    g["age_years"] = now_year - g["year"].fillna(now_year)
    max_count, max_age, max_era = max(g["count"].max(),1), max(g["age_years"].max(),1), max(g["era_strength"].max(),1)
    g["rarity_score"] = (1-(g["count"]/max_count)).clip(0,1)*100
    g["nostalgia_score"] = (0.40*(g["age_years"]/max_age*100)+0.30*(g["count"]/max_count*100)+0.20*(g["era_strength"]/max_era*100)+0.10*g["rarity_score"]).round(1)
    return g.sort_values("nostalgia_score", ascending=False).head(top_n)

def family_evolution_score(df):
    yt = build_year_topic_table(df)
    if yt.empty: return 0.0
    return round(float(yt["growth_pp"].clip(lower=0).mean()*6), 1)

def family_diversity_score(df):
    return len(set(split_values(df.get("topics", pd.Series(dtype=str)))))

def compute_relationships(df):
    rows = []
    for _, r in df.iterrows():
        people = [x.strip() for x in str(r.get("persons","")).split(",") if x.strip()]
        for i in range(len(people)):
            for j in range(i+1, len(people)):
                rows.append({"person_a": people[i], "person_b": people[j], "year": r.get("year"), "path": r.get("path")})
    rel = pd.DataFrame(rows)
    if rel.empty: return rel
    return rel.groupby(["person_a","person_b"]).size().reset_index(name="count").sort_values("count", ascending=False)

def create_album_zip(rows_df, name):
    mem = io.BytesIO(); used = set()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for _, r in rows_df.iterrows():
            p = Path(str(r.get("path","")))
            if not p.exists(): continue
            arc = safe_name(r.get("filename", p.name))
            if arc in used: arc = f"{Path(arc).stem}_{hashlib.md5(str(p).encode()).hexdigest()[:6]}{p.suffix}"
            used.add(arc); z.write(p, arc)
        z.writestr("lifelens_album_index.csv", rows_df.to_csv(index=False).encode("utf-8-sig"))
    mem.seek(0); return mem.read()

def render_gallery(rows_df, max_items=24):
    if rows_df is None or rows_df.empty:
        st.info("Nincs megjeleníthető kép."); return
    cols = st.columns(4)
    for i, (_, row) in enumerate(rows_df.head(max_items).iterrows()):
        p = Path(str(row.get("preview_path") or row.get("path") or ""))
        with cols[i % 4]:
            if p.exists() and p.suffix.lower() in IMAGE_EXT:
                st.image(str(p), caption=f"{row.get('filename','')} · {row.get('topics','')}", use_container_width=True)
            else: st.write(row.get("filename",""))

def filter_by_topic_year(df, topic, year=None):
    mask = df["topics"].fillna("").apply(lambda x: topic in [v.strip() for v in str(x).split(",")])
    out = df[mask]
    if year is not None: out = out[out["year"] == year]
    return out

if "df" not in st.session_state: st.session_state["df"] = pd.DataFrame()
with st.sidebar:
    st.header("0. Licenc / mód"); premium_badge()
    unlock_code = st.text_input("Premium feloldó kód", type="password", help="Demo kód: LIFELENS-PREMIUM-DEMO")
    if st.button("Premium feloldása", use_container_width=True):
        if unlock_code.strip() == PREMIUM_UNLOCK_CODE:
            st.session_state["premium_unlocked"] = True; st.success("Premium mód feloldva."); st.rerun()
        else: st.error("Hibás feloldó kód.")
    st.divider()
    st.header("1. Adatforrás")
    source = st.radio("Mit szeretnél elemezni?", ["Demo család", "Saját ZIP feltöltése", "Helyi mappaútvonal (csak localhost)"], index=0)
    if source == "Demo család":
        if st.button("Demo család betöltése", use_container_width=True):
            st.session_state["df"] = build_demo_family(); st.success("Demo család betöltve.")
    elif source == "Saját ZIP feltöltése":
        uploaded = st.file_uploader("Képek ZIP-ben", type=["zip"])
        if uploaded is not None and st.button("ZIP indexelése", use_container_width=True):
            st.session_state["df"] = scan_zip(uploaded, FREE_IMAGE_LIMIT if not is_premium() else 5000); st.success("ZIP index kész.")
    else:
        folder = st.text_input("Helyi mappaútvonal", value="", help="Csak saját gépen, localhoston működik.")
        if folder and st.button("Helyi mappa indexelése", use_container_width=True):
            p = Path(folder)
            if not p.exists() or not p.is_dir(): st.error("Nem találom ezt a mappát.")
            else: st.session_state["df"] = scan_folder(p, FREE_IMAGE_LIMIT if not is_premium() else 5000)
    st.divider()
    st.header("2. Dashboard slicerek")

df = st.session_state.get("df", pd.DataFrame())
if df.empty:
    st.markdown("## Kezdés")
    st.write("Bal oldalon válaszd a **Demo család** opciót, majd kattints a betöltésre. Ez Cloudban is működik saját képek nélkül.")
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
    sel_years = st.multiselect("Év", all_years, default=all_years)
    sel_topics = st.multiselect("Téma / korszak", all_topics)
    sel_people = st.multiselect("Személy", all_people)
    sel_categories = st.multiselect("Family DNA kategória", all_categories)

fdf = df.copy()
if sel_years: fdf = fdf[fdf["year"].isin(sel_years)]
if sel_topics: fdf = fdf[fdf["topics"].fillna("").apply(lambda x: all(t in [v.strip() for v in str(x).split(",")] for t in sel_topics))]
if sel_people: fdf = fdf[fdf["persons"].fillna("").apply(lambda x: all(p in [v.strip() for v in str(x).split(",")] for p in sel_people))]
if sel_categories: fdf = fdf[fdf["category"].isin(sel_categories)]

period_cards = build_period_cards(fdf, top_n=12)
family_dna = build_family_dna(fdf)
nostalgia = build_hidden_memories(fdf, period_cards, top_n=20)
relationships = compute_relationships(fdf)

tabs = st.tabs(["📊 V8 Dashboard","🧭 Korszakmotor","🧬 Family DNA","🕰️ Nostalgia / Hidden Memories","🔗 Kapcsolati háló","🖼️ Képek + album","📦 Export"])

with tabs[0]:
    st.subheader("Family Analytics Dashboard")
    if not is_premium(): st.caption(f"Free preview: max {FREE_IMAGE_LIMIT} média. Premium: korlátlan elemzés + album export.")
    strongest_era = period_cards.iloc[0] if not period_cards.empty else None
    top_dna = family_dna.iloc[0] if not family_dna.empty else None
    top_nostalgia = nostalgia.iloc[0] if not nostalgia.empty else None
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Média", len(fdf)); c2.metric("Korszakjelölt", len(period_cards)); c3.metric("Family DNA főtéma", top_dna["category"] if top_dna is not None else "-"); c4.metric("Diverzitás", family_diversity_score(fdf)); c5.metric("Evolution", family_evolution_score(fdf))
    st.markdown("### Top insight kártyák")
    cols = st.columns(3)
    with cols[0]:
        if strongest_era is not None:
            st.markdown(f"""<div class="insight-card"><div class="small-muted">legerősebb korszak</div><h3>{strongest_era["title"]}</h3><p class="big-kpi">{strongest_era["era_score"]}/100</p><p>{strongest_era["subtitle"]}</p></div>""", unsafe_allow_html=True)
    with cols[1]:
        if top_dna is not None:
            st.markdown(f"""<div class="insight-card"><div class="small-muted">Family DNA</div><h3>{top_dna["category"]}</h3><p class="big-kpi">{top_dna["dna_score"]}/100</p><p>{int(top_dna["count"])} találat · {top_dna["share_pct"]}% arány</p></div>""", unsafe_allow_html=True)
    with cols[2]:
        if top_nostalgia is not None:
            st.markdown(f"""<div class="insight-card"><div class="small-muted">Nostalgia Score</div><h3>{top_nostalgia["year_month"]} · {top_nostalgia["topic"]}</h3><p class="big-kpi">{top_nostalgia["nostalgia_score"]}/100</p><p>{int(top_nostalgia["count"])} kép/videó</p></div>""", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("### Korszakmotor")
        if px is not None and not period_cards.empty:
            chart = period_cards.copy(); chart["label"] = chart["title"] + " · " + chart["period"].astype(str)
            fig = px.bar(chart.sort_values("era_score"), x="era_score", y="label", orientation="h", labels={"era_score":"Era score","label":""})
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Family DNA")
        if px is not None and not family_dna.empty:
            fig = px.pie(family_dna.head(8), names="category", values="dna_score", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Korszakmotor V1")
    st.write("A korszakmotor főleg a pozitív változást nézi: mi lett hirtelen jellemzőbb egy adott évben.")
    yt = build_year_topic_table(fdf)
    if yt.empty: st.info("Nincs elég témaadat.")
    else:
        st.dataframe(yt[["topic","year","count","share_pct","growth_pp","months","era_score"]].head(60), use_container_width=True)
        if px is not None:
            top_topics = yt.groupby("topic")["era_score"].max().sort_values(ascending=False).head(6).index.tolist()
            fig = px.line(yt[yt["topic"].isin(top_topics)], x="year", y="share_pct", color="topic", markers=True, labels={"share_pct":"Arány az adott év képein belül (%)","year":"Év","topic":"Téma"})
            st.plotly_chart(fig, use_container_width=True)
        if not period_cards.empty:
            labels = [f"{r.topic} · {int(r.year)} · score {r.era_score}" for r in period_cards.itertuples()]
            choice = st.selectbox("Válassz korszakot", labels)
            topic, year = choice.split(" · ")[0], int(choice.split(" · ")[1])
            sub = filter_by_topic_year(fdf, topic, year)
            st.metric("Képek/videók ebben a korszakban", len(sub))
            if is_premium(): st.download_button("📦 Album ZIP ebből", create_album_zip(sub, f"{topic}_{year}"), file_name=f"{safe_name(topic)}_{year}_album.zip", mime="application/zip")
            else: locked_feature_box("Korszak album export")
            render_gallery(sub, max_items=32)

with tabs[2]:
    st.subheader("Family DNA V1")
    st.write("Súlyozott score: 40% arány + 30% darabszám + 20% tartósság + 10% növekedési trend.")
    if family_dna.empty: st.info("Nincs Family DNA adat.")
    else:
        st.dataframe(family_dna[["category","count","share_pct","years","months","trend_pp","dna_score"]], use_container_width=True)
        if px is not None:
            fig = px.bar(family_dna.sort_values("dna_score"), x="dna_score", y="category", orientation="h", labels={"dna_score":"Family DNA score","category":""})
            st.plotly_chart(fig, use_container_width=True)
        category_choice = st.selectbox("Kategória drill-down", family_dna["category"].tolist())
        sub = fdf[fdf["category"].eq(category_choice)]
        st.metric("Talált média", len(sub)); render_gallery(sub, max_items=32)

with tabs[3]:
    st.subheader("Nostalgia / Hidden Memories")
    st.write("Score: 40% kor + 30% képszám + 20% korszak-erősség + 10% ritkaság.")
    if nostalgia.empty: st.info("Nincs nostalgia találat.")
    else:
        st.dataframe(nostalgia[["year_month","topic","count","age_years","era_strength","nostalgia_score"]], use_container_width=True)
        if px is not None:
            fig = px.bar(nostalgia.sort_values("nostalgia_score"), x="nostalgia_score", y="year_month", color="topic", orientation="h", labels={"nostalgia_score":"Nostalgia score","year_month":"Időszak"})
            st.plotly_chart(fig, use_container_width=True)
        labels = [f"{r.year_month} · {r.topic} · score {r.nostalgia_score}" for r in nostalgia.itertuples()]
        choice = st.selectbox("Válassz hidden memory csomagot", labels)
        ym, topic = choice.split(" · ")[0], choice.split(" · ")[1]
        sub = filter_by_topic_year(fdf[fdf["year_month"].eq(ym)], topic)
        st.metric("Talált média", len(sub))
        if is_premium(): st.download_button("📦 Hidden Memory album ZIP", create_album_zip(sub, f"{ym}_{topic}"), file_name=f"{safe_name(ym)}_{safe_name(topic)}_hidden_memory.zip", mime="application/zip")
        else: locked_feature_box("Hidden Memory album export")
        render_gallery(sub, max_items=32)

with tabs[4]:
    st.subheader("Kapcsolati háló")
    if relationships.empty: st.info("Nincs kapcsolatadat.")
    else:
        st.dataframe(relationships, use_container_width=True)
        if go is not None and nx is not None:
            G = nx.Graph()
            for _, r in relationships.iterrows(): G.add_edge(r["person_a"], r["person_b"], weight=int(r["count"]))
            pos = nx.spring_layout(G, seed=7)
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0,y0 = pos[edge[0]]; x1,y1 = pos[edge[1]]
                edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
            node_x, node_y, text = [], [], []
            for node in G.nodes():
                x,y = pos[node]; node_x.append(x); node_y.append(y); text.append(node)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=3), hoverinfo="none"))
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=text, textposition="top center", marker=dict(size=32), hoverinfo="text"))
            fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=10,b=10), height=520)
            st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("Képek + album")
    st.write(f"Jelenlegi szűrés: **{len(fdf)}** média.")
    sort_by = st.selectbox("Rendezés", ["date","quality_score","filename"], index=0)
    ascending = st.checkbox("Növekvő", value=True)
    gallery = fdf.sort_values(sort_by, ascending=ascending, na_position="last") if sort_by in fdf.columns else fdf
    if is_premium(): st.download_button("📦 Album ZIP a jelenlegi szűrésből", create_album_zip(gallery, "filtered_album"), file_name="filtered_album.zip", mime="application/zip")
    else: locked_feature_box("Szűrt album export")
    render_gallery(gallery, max_items=60)

with tabs[6]:
    st.subheader("Export")
    st.download_button("⬇ Index CSV", fdf.to_csv(index=False).encode("utf-8-sig"), file_name="lifelens_v8_index.csv", mime="text/csv")
    if not period_cards.empty: st.download_button("⬇ Korszak score CSV", period_cards.to_csv(index=False).encode("utf-8-sig"), file_name="lifelens_v8_period_scores.csv", mime="text/csv")
    if not family_dna.empty: st.download_button("⬇ Family DNA CSV", family_dna.to_csv(index=False).encode("utf-8-sig"), file_name="lifelens_v8_family_dna.csv", mime="text/csv")
    st.caption("Következő fejlesztés: látványosabb demo pack, majd személytanítás.")
