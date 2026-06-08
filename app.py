
import io
import os
import re
import zipfile
import hashlib
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageStat, ExifTags, ImageOps

try:
    import cv2
except Exception:
    cv2 = None


APP_TITLE = "LifeLens AI Private V5 – Privát fotó és videó rendező"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}
SUPPORTED_EXT = IMAGE_EXT | VIDEO_EXT



def get_download_file_bytes(filename: str):
    """Letölthető csomag beolvasása az app mappájából vagy az aktuális munkamappából."""
    candidates = [
        Path(__file__).resolve().parent / filename,
        Path.cwd() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p.read_bytes()
    return None


def render_desktop_download_button(location: str = "main"):
    """Desktop / helyi privát csomag letöltő gomb."""
    desktop_bytes = (
        get_download_file_bytes("LifeLens_Private_V5_Desktop_Package.zip")
        or get_download_file_bytes("LifeLens_AI_Private_Desktop_Builder.zip")
        or get_download_file_bytes("LifeLens_Private_V4_Desktop_Package.zip")
    )
    if desktop_bytes:
        st.download_button(
            "⬇ LifeLens Desktop / helyi privát verzió letöltése",
            data=desktop_bytes,
            file_name="LifeLens_Private_Desktop_Package.zip",
            mime="application/zip",
            use_container_width=True,
            key=f"download_desktop_{location}",
        )
    else:
        st.warning(
            "A desktop letöltőcsomag nincs az app mappájában. "
            "Tedd a LifeLens_Private_V5_Desktop_Package.zip fájlt ugyanabba a mappába, ahol az app fut."
        )


def get_local_builder_zip_bytes():
    """Ha a telepítő/indító csomag ugyanabban a mappában van, letöltésre kínálja."""
    candidates = [
        Path(__file__).resolve().parent / "LifeLens_Private_V5_Desktop_Package.zip",
        Path.cwd() / "LifeLens_Private_V5_Desktop_Package.zip",
        Path(__file__).resolve().parent / "LifeLens_AI_Private_Desktop_Builder.zip",
        Path.cwd() / "LifeLens_AI_Private_Desktop_Builder.zip",
        Path(__file__).resolve().parent / "LifeLens_Private_V4_Desktop_Package.zip",
        Path.cwd() / "LifeLens_Private_V4_Desktop_Package.zip",
    ]
    for p in candidates:
        if p.exists():
            return p.read_bytes()
    return None



st.set_page_config(page_title=APP_TITLE, page_icon="🔒", layout="wide")

st.markdown(
    """
    <style>
    .privacy-box {
        padding: 1rem;
        border-radius: 16px;
        background: #ecfdf5;
        border: 1px solid #10b981;
        color: #064e3b;
        margin-bottom: 1rem;
    }
    .warn-box {
        padding: 1rem;
        border-radius: 16px;
        background: #fff7ed;
        border: 1px solid #f97316;
        color: #7c2d12;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔒 LifeLens AI Private V5")
st.caption("Privát fotó- és videórendező · helyi futtatásra optimalizálva")

st.markdown(
    """
    <div class="privacy-box">
    <b>Privát mód:</b> ezt az appot helyben futtasd a saját gépeden.
    A képeket és videókat ez a kód nem küldi külső szerverre, nem használja tanításra, és nem osztja meg senkivel.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warn-box">
    <b>Fontos:</b> ha ezt Streamlit Cloudra vagy más szerverre teszed, akkor a ZIP/fájl feltöltés technikailag arra a szerverre kerül.
    Privát családi képeknél első körben csak lokális futtatást javaslok: <code>streamlit run lifelens_private_app.py</code>.
    </div>
    """,
    unsafe_allow_html=True,
)



st.markdown("## 🚀 Hogyan szeretnéd használni?")

col_online, col_local = st.columns(2)

with col_online:
    st.markdown(
        """
        ### 🌐 Online próba
        Kisebb, nem érzékeny tesztanyaggal kipróbálható.

        - ZIP feltöltés
        - gyors demó
        - a helyi `C:\\` meghajtó nem látható

        ⚠️ Privát családi képekhez nem ezt ajánljuk.
        """
    )

with col_local:
    st.markdown(
        """
        ### 💻 Helyi privát verzió
        Ez az ajánlott mód családi képekhez.

        - saját gépen fut
        - látja a helyi mappákat, külső HDD-t, NAS-t
        - a képek/videók nem kerülnek felhőbe
        - nagy archívumhoz ideális
        """
    )
    render_desktop_download_button("landing")

st.link_button("🐍 Python letöltése, ha a helyi indító kéri", "https://www.python.org/downloads/")
st.divider()


def safe_name(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("ő", "o").replace("ű", "u").replace("Ő", "O").replace("Ű", "U")
    text = re.sub(r"[^\w\- .áéíóöúüÁÉÍÓÖÚÜ]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120] if text else "ismeretlen"


def md5_file(path: Path, chunk_size=1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def average_hash(img: Image.Image, hash_size: int = 8) -> str:
    try:
        g = ImageOps.grayscale(img).resize((hash_size, hash_size))
        arr = np.asarray(g, dtype=np.float32)
        avg = arr.mean()
        bits = arr > avg
        return "".join("1" if b else "0" for b in bits.flatten())
    except Exception:
        return ""


def get_exif_datetime(img: Image.Image):
    try:
        exif = img.getexif()
        if not exif:
            return None
        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
            if key in tag_map:
                raw = str(tag_map[key])
                try:
                    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
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


def classify_basic_tags(path: Path, dt: datetime | None, media_type: str, w=None, h=None) -> list[str]:
    tags = []
    name = path.name.lower()

    if media_type == "video":
        tags.append("videó")
    else:
        tags.append("kép")

    if "screenshot" in name or "képernyőkép" in name or "kepernyokep" in name:
        tags.append("screenshot")

    if w and h:
        ratio = max(w, h) / max(1, min(w, h))
        if ratio > 1.9 and (w > 1200 or h > 1200):
            tags.append("panoráma")
        if w < 900 or h < 900:
            tags.append("kis felbontás")

    if dt:
        if dt.month == 12:
            tags.append("december")
            tags.append("karácsony/időszak")
        if dt.month in [6, 7, 8]:
            tags.append("nyár")
        if dt.month in [1, 2]:
            tags.append("tél")

    for keyword in [
        "balaton", "nyaralas", "nyaralás", "karacsony", "karácsony", "szulinap", "szülinap",
        "foci", "football", "traktor", "auto", "autó", "kutya", "macska", "ovi", "bölcsi", "bolcsi"
    ]:
        if keyword in name:
            tags.append(keyword)

    return sorted(set(tags))


def sharpness_score(img: Image.Image) -> float:
    try:
        g = np.asarray(ImageOps.grayscale(img).resize((256, 256)), dtype=np.float32)
        dx = np.diff(g, axis=1)
        dy = np.diff(g, axis=0)
        return float(dx.var() + dy.var())
    except Exception:
        return 0.0


def exposure_score(img: Image.Image) -> float:
    try:
        small = ImageOps.grayscale(img).resize((128, 128))
        stat = ImageStat.Stat(small)
        mean = stat.mean[0]
        std = stat.stddev[0]
        middle = 1 - abs(mean - 128) / 128
        contrast = min(std / 64, 1)
        return float(max(0, middle) * 60 + contrast * 40)
    except Exception:
        return 0.0


def quality_score(img: Image.Image) -> float:
    return round(sharpness_score(img) * 0.7 + exposure_score(img) * 0.3, 2)


def extract_zip(uploaded_file, dest: Path):
    with zipfile.ZipFile(uploaded_file) as z:
        z.extractall(dest)


def iter_media_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p


def get_video_metadata(path: Path, preview_dir: Path):
    if cv2 is None:
        return {
            "duration_sec": None,
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
            "preview_path": "",
            "video_quality_score": 0,
            "phash": "",
        }

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "duration_sec": None,
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
            "preview_path": "",
            "video_quality_score": 0,
            "phash": "",
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps else None

    # középső képkocka preview
    target_frame = int(frame_count * 0.35) if frame_count else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()

    preview_path = ""
    q = 0
    phash = ""

    if ok and frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        q = quality_score(img)
        phash = average_hash(img)
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / (hashlib.md5(str(path).encode()).hexdigest() + ".jpg")
        img.save(preview_path, quality=85)
        preview_path = str(preview_path)

    return {
        "duration_sec": round(duration, 1) if duration else None,
        "fps": round(fps, 1) if fps else None,
        "frame_count": int(frame_count) if frame_count else None,
        "width": width,
        "height": height,
        "preview_path": preview_path,
        "video_quality_score": q,
        "phash": phash,
    }


def scan_media(root: Path, limit: int | None = None) -> pd.DataFrame:
    rows = []
    files = list(iter_media_files(root))
    if limit:
        files = files[:limit]

    preview_dir = Path(st.session_state.work_dir) / "video_previews"
    progress = st.progress(0, text="Média indexelése...")

    for i, path in enumerate(files):
        media_type = "video" if path.suffix.lower() in VIDEO_EXT else "image"
        try:
            dt = fallback_datetime(path)

            if media_type == "image":
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    exif_dt = get_exif_datetime(img)
                    if exif_dt is not None:
                        dt = exif_dt

                    w, h = img.size
                    phash = average_hash(img)
                    q = quality_score(img)
                    tags = classify_basic_tags(path, dt, media_type, w, h)

                    rows.append({
                        "media_type": "image",
                        "path": str(path),
                        "filename": path.name,
                        "folder": str(path.parent),
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                        "md5": md5_file(path),
                        "phash": phash,
                        "date": dt,
                        "year": dt.year if dt else None,
                        "month": dt.month if dt else None,
                        "width": w,
                        "height": h,
                        "duration_sec": None,
                        "fps": None,
                        "preview_path": str(path),
                        "quality_score": q,
                        "tags": ", ".join(tags),
                        "is_screenshot": "screenshot" in tags,
                    })
            else:
                meta = get_video_metadata(path, preview_dir)
                tags = classify_basic_tags(path, dt, media_type, meta.get("width"), meta.get("height"))

                rows.append({
                    "media_type": "video",
                    "path": str(path),
                    "filename": path.name,
                    "folder": str(path.parent),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                    "md5": md5_file(path),
                    "phash": meta.get("phash", ""),
                    "date": dt,
                    "year": dt.year if dt else None,
                    "month": dt.month if dt else None,
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "duration_sec": meta.get("duration_sec"),
                    "fps": meta.get("fps"),
                    "preview_path": meta.get("preview_path", ""),
                    "quality_score": meta.get("video_quality_score", 0),
                    "tags": ", ".join(tags),
                    "is_screenshot": False,
                })

        except Exception as exc:
            rows.append({
                "media_type": media_type,
                "path": str(path),
                "filename": path.name,
                "folder": str(path.parent),
                "error": str(exc),
            })

        progress.progress((i + 1) / max(len(files), 1), text=f"Indexelés: {i+1}/{len(files)}")
    progress.empty()

    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df = df.sort_values("date", ascending=True, na_position="last")
    return df


def duplicate_summary(df: pd.DataFrame):
    if df.empty or "md5" not in df.columns:
        return pd.DataFrame()
    dup = df[df.duplicated("md5", keep=False)].copy()
    if dup.empty:
        return pd.DataFrame()
    dup["duplicate_group"] = dup.groupby("md5").ngroup() + 1
    return dup.sort_values(["duplicate_group", "filename"])


def build_album(df: pd.DataFrame, mode: str, top_n: int = 100) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "quality_score" not in work.columns:
        work["quality_score"] = 0

    if mode == "Top képek és videók":
        return work.sort_values("quality_score", ascending=False).head(top_n)

    if mode == "Top fotók":
        return work[work["media_type"].eq("image")].sort_values("quality_score", ascending=False).head(top_n)

    if mode == "Top videók":
        return work[work["media_type"].eq("video")].sort_values("quality_score", ascending=False).head(top_n)

    if mode == "Családi / esemény válogatás":
        work = work[~work.get("is_screenshot", False).fillna(False)]
        return work.sort_values("quality_score", ascending=False).head(top_n)

    if mode == "Karácsony":
        mask = work["tags"].fillna("").str.contains("karácsony|karacsony|december", case=False, regex=True)
        return work[mask].sort_values("quality_score", ascending=False).head(top_n)

    if mode == "Nyár / nyaralás":
        mask = work["tags"].fillna("").str.contains("nyár|nyar|nyaralás|nyaralas|balaton", case=False, regex=True)
        return work[mask].sort_values("quality_score", ascending=False).head(top_n)

    return work.head(top_n)


def folder_for_row(row, structure: str):
    year = int(row.get("year", 0) or 0)
    month = int(row.get("month", 0) or 0)
    tags = str(row.get("tags", "")).lower()
    media_type = row.get("media_type", "image")

    media_folder = "Videók" if media_type == "video" else "Képek"

    if structure == "Év / hónap":
        if year and month:
            return Path(str(year)) / f"{month:02d}" / media_folder
        return Path("Dátum_nélkül") / media_folder

    if structure == "Téma szerint":
        if "karácsony" in tags or "karacsony" in tags or "december" in tags:
            return Path("Ünnepek") / "Karácsony" / media_folder
        if "nyár" in tags or "nyar" in tags or "balaton" in tags:
            return Path("Nyaralás") / media_folder
        if "screenshot" in tags:
            return Path("Screenshotok")
        if "traktor" in tags or "auto" in tags or "autó" in tags:
            return Path("Járművek") / media_folder
        if "foci" in tags or "football" in tags:
            return Path("Sport") / media_folder
        return Path("Egyéb") / media_folder

    if structure == "Év / téma":
        base = Path(str(year)) if year else Path("Dátum_nélkül")
        return base / folder_for_row(row, "Téma szerint")

    return Path("Rendezett") / media_folder


def create_organized_zip(df: pd.DataFrame, structure: str, include_mode: str, top_n: int = 500) -> bytes:
    if df.empty:
        return b""

    work = df.copy()
    if include_mode == "Csak top média":
        work = work.sort_values("quality_score", ascending=False).head(top_n)
    elif include_mode == "Duplikátumok nélkül":
        work = work.drop_duplicates("md5", keep="first")
    elif include_mode == "Screenshotok nélkül":
        work = work[~work.get("is_screenshot", False).fillna(False)]
    elif include_mode == "Csak képek":
        work = work[work["media_type"].eq("image")]
    elif include_mode == "Csak videók":
        work = work[work["media_type"].eq("video")]

    mem = io.BytesIO()
    used_names = set()

    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for _, row in work.iterrows():
            src = Path(row["path"])
            if not src.exists():
                continue
            folder = folder_for_row(row, structure)
            filename = safe_name(row.get("filename", src.name))
            arc = str(folder / filename)

            if arc in used_names:
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                arc = str(folder / f"{stem}_{hashlib.md5(str(src).encode()).hexdigest()[:6]}{suffix}")
            used_names.add(arc)
            z.write(src, arc)

        index_csv = work.drop(columns=["md5", "phash"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        z.writestr("LifeLens_index.csv", index_csv)

    mem.seek(0)
    return mem.read()


def create_html_album(df: pd.DataFrame, title: str, max_items: int = 80) -> bytes:
    work = df.sort_values("quality_score", ascending=False).head(max_items).copy()
    rows = []
    for _, row in work.iterrows():
        src = Path(row["path"])
        preview = row.get("preview_path", "")
        preview_html = f'<img src="{preview}" style="width:100%;border-radius:12px;max-height:180px;object-fit:cover;">' if preview and Path(str(preview)).exists() else f"<div class='thumb'>{row.get('filename','')}</div>"
        rows.append(f"""
        <div class="card">
            {preview_html}
            <p><b>{row.get('media_type','')}</b> · {row.get('filename','')}</p>
            <p><b>Dátum:</b> {row.get('date','')}</p>
            <p><b>Pontszám:</b> {row.get('quality_score','')}</p>
            <p><b>Címkék:</b> {row.get('tags','')}</p>
            <p><small>{src}</small></p>
        </div>
        """)

    html = f"""
    <!doctype html>
    <html lang="hu">
    <head>
      <meta charset="utf-8">
      <title>{title}</title>
      <style>
        body {{ font-family: Arial, sans-serif; background:#f8fafc; color:#0f172a; margin:40px; }}
        .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(260px,1fr)); gap:16px; }}
        .card {{ background:white; border:1px solid #e2e8f0; border-radius:16px; padding:16px; }}
        .thumb {{ height:120px; border-radius:12px; background:#e2e8f0; display:flex; align-items:center; justify-content:center; text-align:center; padding:8px; }}
      </style>
    </head>
    <body>
      <h1>{title}</h1>
      <p>LifeLens AI Private – helyben generált album index</p>
      <div class="grid">
        {''.join(rows)}
      </div>
    </body>
    </html>
    """
    return html.encode("utf-8")


# Sidebar

st.sidebar.markdown("---")
st.sidebar.subheader("Helyi futtatási segítség")
st.sidebar.caption("Privát családi képekhez használd a 💻 Helyi privát futtatás fület.")


st.sidebar.markdown("---")
st.sidebar.subheader("Desktop verzió")
if st.sidebar.button("Miért jobb helyben futtatni?", use_container_width=True):
    st.sidebar.info("Privát képeknél a helyi verzió ajánlott: a képek nem kerülnek felhőbe, és működik a C:\ / D:\ mappaútvonal.")

st.sidebar.header("1. Forrás kiválasztása")

source_mode = st.sidebar.radio(
    "Honnan jöjjenek a képek/videók?",
    ["ZIP feltöltés", "Helyi mappa útvonala"],
    help="Privát képekhez helyi futtatást javaslok. Streamlit Cloud esetén a ZIP a szerverre kerülne.",
)

work_root = None

if "work_dir" not in st.session_state:
    st.session_state.work_dir = tempfile.mkdtemp(prefix="lifelens_")

base_tmp = Path(st.session_state.work_dir)

if source_mode == "ZIP feltöltés":
    uploaded = st.sidebar.file_uploader("Képek és videók ZIP fájlban", type=["zip"])
    if uploaded:
        extract_dir = base_tmp / "uploaded_zip"
        if st.sidebar.button("ZIP kibontása és beolvasása", use_container_width=True):
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            extract_zip(uploaded, extract_dir)
            st.session_state["photo_root"] = str(extract_dir)
            st.success("ZIP kibontva.")
else:
    folder_text = st.sidebar.text_input("Helyi mappa útvonala", value="")
    st.sidebar.caption("Példa Windows: C:\\Users\\Gabor\\Pictures")
    if st.sidebar.button("Mappa beállítása", use_container_width=True):
        p = Path(folder_text)
        if p.exists() and p.is_dir():
            st.session_state["photo_root"] = str(p)
            st.success("Mappa beállítva.")
        else:
            st.error("Ez a mappa nem található.")

if st.session_state.get("photo_root"):
    work_root = Path(st.session_state["photo_root"])

st.sidebar.header("2. Elemzés")
limit_scan = st.sidebar.number_input("Max. médiafájl teszthez", min_value=100, max_value=100000, value=3000, step=100)
scan_btn = st.sidebar.button("Indexelés indítása", use_container_width=True, disabled=work_root is None)

if scan_btn and work_root:
    df = scan_media(work_root, int(limit_scan))
    st.session_state["photo_index"] = df
    st.success("Indexelés kész.")

df = st.session_state.get("photo_index", pd.DataFrame())

if work_root is None:
    st.info("Első lépés: válassz ZIP-et vagy helyi mappát a bal oldalon.")
    st.stop()

if df.empty:
    st.info("A forrás beállítva. Indítsd el az indexelést a bal oldalon.")
    st.stop()


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Áttekintés",
    "🔎 Keresés",
    "🎞️ Videók",
    "🧹 Nagytakarítás",
    "🎯 Albumok",
    "📦 Rendezett ZIP",
    "📖 Fotókönyv / idővonal",
    "🧭 Források",
    "💻 Helyi privát futtatás",
])

with tab1:
    st.subheader("Fotó- és videóarchívum áttekintés")

    total = len(df)
    img_count = int(df["media_type"].eq("image").sum()) if "media_type" in df.columns else 0
    vid_count = int(df["media_type"].eq("video").sum()) if "media_type" in df.columns else 0
    years = df["year"].nunique() if "year" in df.columns else 0
    total_size = df["size_mb"].sum() if "size_mb" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Összes média", f"{total:,}".replace(",", " "))
    c2.metric("Képek", img_count)
    c3.metric("Videók", vid_count)
    c4.metric("Évek", years)
    c5.metric("Méret", f"{total_size:.1f} MB")

    if "year" in df.columns:
        st.markdown("### Évek szerinti eloszlás")
        year_df = df.groupby(["year", "media_type"]).size().reset_index(name="Darab").dropna()
        pivot = year_df.pivot(index="year", columns="media_type", values="Darab").fillna(0)
        st.bar_chart(pivot)

    show_cols = ["media_type", "filename", "date", "year", "month", "size_mb", "duration_sec", "quality_score", "tags", "folder"]
    st.dataframe(df[[c for c in show_cols if c in df.columns]].head(300), use_container_width=True)


with tab2:
    st.subheader("Keresés képekben és videókban")
    q = st.text_input("Mit keresel?", placeholder="pl. Balaton, karácsony, traktor, 2024, videó, screenshot")
    result = df.copy()

    if q:
        ql = q.lower().strip()
        mask = (
            result["filename"].fillna("").str.lower().str.contains(ql, regex=False)
            | result["folder"].fillna("").str.lower().str.contains(ql, regex=False)
            | result["tags"].fillna("").str.lower().str.contains(ql, regex=False)
            | result["media_type"].fillna("").str.lower().str.contains(ql, regex=False)
            | result["year"].fillna("").astype(str).str.contains(ql, regex=False)
        )
        result = result[mask]

    st.write(f"Találatok: **{len(result)}**")
    st.dataframe(result[["media_type", "filename", "date", "duration_sec", "quality_score", "tags", "path"]].head(500), use_container_width=True)

    if not result.empty:
        st.markdown("### Gyors előnézet")
        cols = st.columns(4)
        for i, (_, row) in enumerate(result.head(12).iterrows()):
            p = Path(str(row.get("preview_path") or row.get("path")))
            if p.exists():
                with cols[i % 4]:
                    st.image(str(p), caption=f"{row['media_type']} · {row['filename']}", use_container_width=True)


with tab3:
    st.subheader("Videók")
    videos = df[df["media_type"].eq("video")].copy()
    if videos.empty:
        st.info("Nem találtam videót.")
    else:
        total_duration = videos["duration_sec"].fillna(0).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Videók száma", len(videos))
        c2.metric("Összes hossz", f"{total_duration/60:.1f} perc")
        c3.metric("Videók mérete", f"{videos['size_mb'].sum():.1f} MB")

        st.dataframe(videos[["filename", "date", "duration_sec", "width", "height", "fps", "quality_score", "tags", "path"]].head(500), use_container_width=True)

        st.markdown("### Preview képkockák")
        cols = st.columns(4)
        for i, (_, row) in enumerate(videos.sort_values("quality_score", ascending=False).head(12).iterrows()):
            p = Path(str(row.get("preview_path", "")))
            if p.exists():
                with cols[i % 4]:
                    st.image(str(p), caption=f"{row['filename']} · {row.get('duration_sec')} mp", use_container_width=True)


with tab4:
    st.subheader("Nagytakarítás")
    dup = duplicate_summary(df)
    dup_count = len(dup)
    dup_size = dup["size_mb"].sum() if not dup.empty and "size_mb" in dup.columns else 0
    screenshots = int(df.get("is_screenshot", pd.Series(dtype=bool)).fillna(False).sum()) if "is_screenshot" in df.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Pontos duplikátum sorok", dup_count)
    c2.metric("Duplikátumok mérete", f"{dup_size:.1f} MB")
    c3.metric("Screenshotok", screenshots)

    if dup.empty:
        st.success("Nem találtam pontos MD5 duplikátumot.")
    else:
        st.dataframe(dup[["duplicate_group", "media_type", "filename", "size_mb", "path"]].head(1000), use_container_width=True)

    st.markdown("### Gyenge minőség gyanús média")
    lowq = df.sort_values("quality_score", ascending=True).head(100)
    st.dataframe(lowq[["media_type", "filename", "quality_score", "size_mb", "path"]], use_container_width=True)


with tab5:
    st.subheader("Albumok")
    album_mode = st.selectbox("Album típusa", ["Top képek és videók", "Top fotók", "Top videók", "Családi / esemény válogatás", "Karácsony", "Nyár / nyaralás"])
    top_n = st.slider("Médiafájlok száma", 10, 500, 80, step=10)
    album = build_album(df, album_mode, top_n)

    st.write(f"Album elemek: **{len(album)}**")
    st.dataframe(album[["media_type", "filename", "date", "duration_sec", "quality_score", "tags", "path"]].head(500), use_container_width=True)

    if not album.empty:
        cols = st.columns(4)
        for i, (_, row) in enumerate(album.head(16).iterrows()):
            p = Path(str(row.get("preview_path") or row.get("path")))
            if p.exists():
                with cols[i % 4]:
                    st.image(str(p), caption=f"{row['media_type']} · {row['filename']}", use_container_width=True)

        html_bytes = create_html_album(album, f"LifeLens album – {album_mode}", max_items=top_n)
        st.download_button(
            "HTML album index letöltése",
            data=html_bytes,
            file_name=f"LifeLens_album_{safe_name(album_mode)}.html",
            mime="text/html",
            use_container_width=True,
        )


with tab6:
    st.subheader("Rendezett ZIP export")
    st.caption("Az eredeti fájlokat nem törli és nem mozgatja. Új ZIP-et készít másolatokkal.")

    structure = st.selectbox("Mappastruktúra", ["Év / hónap", "Téma szerint", "Év / téma"])
    include_mode = st.selectbox("Mit tegyen bele?", ["Minden média", "Duplikátumok nélkül", "Screenshotok nélkül", "Csak top média", "Csak képek", "Csak videók"])
    export_top_n = st.slider("Top média száma, ha azt választod", 50, 2000, 500, step=50)

    if st.button("Rendezett ZIP elkészítése", use_container_width=True):
        with st.spinner("ZIP készítése..."):
            zip_bytes = create_organized_zip(df, structure, include_mode, export_top_n)
            st.session_state["organized_zip"] = zip_bytes
        st.success("ZIP elkészült.")

    if st.session_state.get("organized_zip"):
        st.download_button(
            "Rendezett ZIP letöltése",
            data=st.session_state["organized_zip"],
            file_name=f"LifeLens_Rendezett_{safe_name(structure)}.zip",
            mime="application/zip",
            use_container_width=True,
        )



with tab7:
    st.subheader("📖 Fotókönyv / családi idővonal")
    st.markdown(
        """
        Ez a következő nagy WOW funkció előkészített helye.

        **Tervezett exportok:**
        - éves családi fotókönyv,
        - gyerek-idővonal,
        - nyaralás album,
        - karácsonyi album,
        - top 100 családi kép,
        - HTML mini album,
        - később PDF fotókönyv.

        A jelenlegi MVP-ben már készíthető HTML album és rendezett ZIP. 
        A PDF fotókönyv a következő fejlesztési körben jön.
        """
    )

    if not df.empty:
        st.markdown("### Gyors idővonal előnézet")
        if "year" in df.columns and "media_type" in df.columns:
            timeline = (
                df.groupby(["year", "media_type"])
                .size()
                .reset_index(name="Darab")
                .dropna()
                .sort_values("year")
            )
            if not timeline.empty:
                st.dataframe(timeline, use_container_width=True)
                pivot = timeline.pivot(index="year", columns="media_type", values="Darab").fillna(0)
                st.bar_chart(pivot)

        st.markdown("### Legaktívabb napok")
        tmp = df.copy()
        tmp["day"] = pd.to_datetime(tmp["date"], errors="coerce").dt.date
        day_df = tmp.dropna(subset=["day"]).groupby("day").size().reset_index(name="Médiafájl").sort_values("Médiafájl", ascending=False).head(20)
        st.dataframe(day_df, use_container_width=True)



with tab8:
    st.subheader("Források és későbbi csatlakozók")
    st.markdown(
        """
        **Google Drive**
        - OAuth belépés
        - Drive mappák listázása
        - képek/videók indexelése
        - export új Drive mappába

        **OneDrive**
        - Microsoft OAuth
        - képek/videók listázása
        - rendezett export OneDrive-ba

        **Telefon**
        - első körben telefon backup mappa
        - később mobilapp

        **Google Photos**
        - lehetséges, de API-korlátok miatt későbbi kör

        **Helyi desktop verzió**
        - ez lenne a legprivátabb
        - nem kell feltölteni semmit
        - közvetlenül tud mappát/HDD-t/NAS-t olvasni
        """
    )


with tab9:
    st.subheader("💻 Helyi privát futtatás")
    st.markdown(
        """
        Ez a rész azoknak szól, akik **nem szeretnék feltölteni** a családi képeiket/videóikat online felületre.

        **A legbiztonságosabb működés:**
        - az app a saját gépeden fut,
        - helyi mappát vagy külső meghajtót olvas,
        - a képek nem mennek fel szerverre,
        - az eredeti képeket nem törli és nem mozgatja.
        """
    )

    st.info(
        "Online Streamlit Cloudon a helyi C:\\ vagy D:\\ útvonal nem működik, mert a felhős szerver nem látja a saját géped meghajtóit. "
        "Helyi futtatásnál viszont működik."
    )

    st.markdown("### 1. Python telepítése, ha még nincs")
    st.markdown(
        """
        Töltsd le a hivatalos Python telepítőt a Python oldaláról.  
        Telepítéskor fontos: **Add Python to PATH** legyen bepipálva.
        """
    )
    st.link_button("Python letöltése – hivatalos oldal", "https://www.python.org/downloads/")

    st.markdown("### 2. Dupla kattintós indító / telepítő csomag")
    st.markdown(
        """
        A csomagban van:
        - `RUN_SOURCE_LOCAL.bat` – gyors helyi futtatás,
        - `BUILD_EXE_WINDOWS.bat` – Windows EXE készítés,
        - `BUILD_INSTALLER_WINDOWS.bat` – telepítő készítés Inno Setup-pal,
        - adatvédelmi szöveg és útmutató.
        """
    )

    render_desktop_download_button("local_tab")

    st.markdown("### 3. Használat egyszerűen")
    st.code(
        """1. ZIP kicsomagolása például: C:\\LifeLens
2. Dupla katt: RUN_SOURCE_LOCAL.bat
3. Megnyílik: http://localhost:8501
4. Helyi mappa útvonala működni fog, pl. C:\\Users\\T470\\Desktop\\Foto_App""",
        language="text",
    )

    st.markdown("### Miért nem lehet egy online appból Python-telepítést indítani?")
    st.markdown(
        """
        A böngésző biztonsági okból nem engedi, hogy egy webapp programokat telepítsen a gépedre vagy közvetlenül olvassa a mappáidat.  
        Ez jó dolog: ettől biztonságosabb a géped. Ezért kell a helyi indítócsomag vagy később egy rendes Windows telepítő.
        """
    )

import io
import os
import re
import zipfile
import hashlib
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageStat, ExifTags, ImageOps

try:
    import cv2
except Exception:
    cv2 = None


APP_TITLE = "LifeLens AI Private – Fotó és videó rendező MVP"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}
SUPPORTED_EXT = IMAGE_EXT | VIDEO_EXT


def get_local_builder_zip_bytes():
    """Ha a telepítő/indító csomag ugyanabban a mappában van, letöltésre kínálja."""
    candidates = [
        Path(__file__).resolve().parent / "LifeLens_AI_Private_Desktop_Builder.zip",
        Path.cwd() / "LifeLens_AI_Private_Desktop_Builder.zip",
    ]
    for p in candidates:
        if p.exists():
            return p.read_bytes()
    return None



st.set_page_config(page_title=APP_TITLE, page_icon="🔒", layout="wide")

st.markdown(
    """
    <style>
    .privacy-box {
        padding: 1rem;
        border-radius: 16px;
        background: #ecfdf5;
        border: 1px solid #10b981;
        color: #064e3b;
        margin-bottom: 1rem;
    }
    .warn-box {
        padding: 1rem;
        border-radius: 16px;
        background: #fff7ed;
        border: 1px solid #f97316;
        color: #7c2d12;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔒 LifeLens AI Private")
st.caption("Privát, helyben futó fotó- és videórendező MVP")

st.markdown(
    """
    <div class="privacy-box">
    <b>Privát mód:</b> ezt az appot helyben futtasd a saját gépeden.
    A képeket és videókat ez a kód nem küldi külső szerverre, nem használja tanításra, és nem osztja meg senkivel.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warn-box">
    <b>Fontos:</b> ha ezt Streamlit Cloudra vagy más szerverre teszed, akkor a ZIP/fájl feltöltés technikailag arra a szerverre kerül.
    Privát családi képeknél első körben csak lokális futtatást javaslok: <code>streamlit run lifelens_private_app.py</code>.
    </div>
    """,
    unsafe_allow_html=True,
)


def safe_name(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("ő", "o").replace("ű", "u").replace("Ő", "O").replace("Ű", "U")
    text = re.sub(r"[^\w\- .áéíóöúüÁÉÍÓÖÚÜ]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120] if text else "ismeretlen"


def md5_file(path: Path, chunk_size=1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def average_hash(img: Image.Image, hash_size: int = 8) -> str:
    try:
        g = ImageOps.grayscale(img).resize((hash_size, hash_size))
        arr = np.asarray(g, dtype=np.float32)
        avg = arr.mean()
        bits = arr > avg
        return "".join("1" if b else "0" for b in bits.flatten())
    except Exception:
        return ""


def get_exif_datetime(img: Image.Image):
    try:
        exif = img.getexif()
        if not exif:
            return None
        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
            if key in tag_map:
                raw = str(tag_map[key])
                try:
                    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
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


def classify_basic_tags(path: Path, dt: datetime | None, media_type: str, w=None, h=None) -> list[str]:
    tags = []
    name = path.name.lower()

    if media_type == "video":
        tags.append("videó")
    else:
        tags.append("kép")

    if "screenshot" in name or "képernyőkép" in name or "kepernyokep" in name:
        tags.append("screenshot")

    if w and h:
        ratio = max(w, h) / max(1, min(w, h))
        if ratio > 1.9 and (w > 1200 or h > 1200):
            tags.append("panoráma")
        if w < 900 or h < 900:
            tags.append("kis felbontás")

    if dt:
        if dt.month == 12:
            tags.append("december")
            tags.append("karácsony/időszak")
        if dt.month in [6, 7, 8]:
            tags.append("nyár")
        if dt.month in [1, 2]:
            tags.append("tél")

    for keyword in [
        "balaton", "nyaralas", "nyaralás", "karacsony", "karácsony", "szulinap", "szülinap",
        "foci", "football", "traktor", "auto", "autó", "kutya", "macska", "ovi", "bölcsi", "bolcsi"
    ]:
        if keyword in name:
            tags.append(keyword)

    return sorted(set(tags))


def sharpness_score(img: Image.Image) -> float:
    try:
        g = np.asarray(ImageOps.grayscale(img).resize((256, 256)), dtype=np.float32)
        dx = np.diff(g, axis=1)
        dy = np.diff(g, axis=0)
        return float(dx.var() + dy.var())
    except Exception:
        return 0.0


def exposure_score(img: Image.Image) -> float:
    try:
        small = ImageOps.grayscale(img).resize((128, 128))
        stat = ImageStat.Stat(small)
        mean = stat.mean[0]
        std = stat.stddev[0]
        middle = 1 - abs(mean - 128) / 128
        contrast = min(std / 64, 1)
        return float(max(0, middle) * 60 + contrast * 40)
    except Exception:
        return 0.0


def quality_score(img: Image.Image) -> float:
    return round(sharpness_score(img) * 0.7 + exposure_score(img) * 0.3, 2)


def extract_zip(uploaded_file, dest: Path):
    with zipfile.ZipFile(uploaded_file) as z:
        z.extractall(dest)


def iter_media_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p


def get_video_metadata(path: Path, preview_dir: Path):
    if cv2 is None:
        return {
            "duration_sec": None,
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
            "preview_path": "",
            "video_quality_score": 0,
            "phash": "",
        }

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "duration_sec": None,
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
            "preview_path": "",
            "video_quality_score": 0,
            "phash": "",
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps else None

    # középső képkocka preview
    target_frame = int(frame_count * 0.35) if frame_count else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()

    preview_path = ""
    q = 0
    phash = ""

    if ok and frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        q = quality_score(img)
        phash = average_hash(img)
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / (hashlib.md5(str(path).encode()).hexdigest() + ".jpg")
        img.save(preview_path, quality=85)
        preview_path = str(preview_path)

    return {
        "duration_sec": round(duration, 1) if duration else None,
        "fps": round(fps, 1) if fps else None,
        "frame_count": int(frame_count) if frame_count else None,
        "width": width,
        "height": height,
        "preview_path": preview_path,
        "video_quality_score": q,
        "phash": phash,
    }


def scan_media(root: Path, limit: int | None = None) -> pd.DataFrame:
    rows = []
    files = list(iter_media_files(root))
    if limit:
        files = files[:limit]

    preview_dir = Path(st.session_state.work_dir) / "video_previews"
    progress = st.progress(0, text="Média indexelése...")

    for i, path in enumerate(files):
        media_type = "video" if path.suffix.lower() in VIDEO_EXT else "image"
        try:
            dt = fallback_datetime(path)

            if media_type == "image":
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    exif_dt = get_exif_datetime(img)
                    if exif_dt is not None:
                        dt = exif_dt

                    w, h = img.size
                    phash = average_hash(img)
                    q = quality_score(img)
                    tags = classify_basic_tags(path, dt, media_type, w, h)

                    rows.append({
                        "media_type": "image",
                        "path": str(path),
                        "filename": path.name,
                        "folder": str(path.parent),
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                        "md5": md5_file(path),
                        "phash": phash,
                        "date": dt,
                        "year": dt.year if dt else None,
                        "month": dt.month if dt else None,
                        "width": w,
                        "height": h,
                        "duration_sec": None,
                        "fps": None,
                        "preview_path": str(path),
                        "quality_score": q,
                        "tags": ", ".join(tags),
                        "is_screenshot": "screenshot" in tags,
                    })
            else:
                meta = get_video_metadata(path, preview_dir)
                tags = classify_basic_tags(path, dt, media_type, meta.get("width"), meta.get("height"))

                rows.append({
                    "media_type": "video",
                    "path": str(path),
                    "filename": path.name,
                    "folder": str(path.parent),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                    "md5": md5_file(path),
                    "phash": meta.get("phash", ""),
                    "date": dt,
                    "year": dt.year if dt else None,
                    "month": dt.month if dt else None,
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "duration_sec": meta.get("duration_sec"),
                    "fps": meta.get("fps"),
                    "preview_path": meta.get("preview_path", ""),
                    "quality_score": meta.get("video_quality_score", 0),
                    "tags": ", ".join(tags),
                    "is_screenshot": False,
                })

        except Exception as exc:
            rows.append({
                "media_type": media_type,
                "path": str(path),
                "filename": path.name,
                "folder": str(path.parent),
                "error": str(exc),
            })

        progress.progress((i + 1) / max(len(files), 1), text=f"Indexelés: {i+1}/{len(files)}")
    progress.empty()

    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df = df.sort_values("date", ascending=True, na_position="last")
    return df


def duplicate_summary(df: pd.DataFrame):
    if df.empty or "md5" not in df.columns:
        return pd.DataFrame()
    dup = df[df.duplicated("md5", keep=False)].copy()
    if dup.empty:
        return pd.DataFrame()
    dup["duplicate_group"] = dup.groupby("md5").ngroup() + 1
    return dup.sort_values(["duplicate_group", "filename"])


def build_album(df: pd.DataFrame, mode: str, top_n: int = 100) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "quality_score" not in work.columns:
        work["quality_score"] = 0

    if mode == "Top képek és videók":
        return work.sort_values("quality_score", ascending=False).head(top_n)

    if mode == "Top fotók":
        return work[work["media_type"].eq("image")].sort_values("quality_score", ascending=False).head(top_n)

    if mode == "Top videók":
        return work[work["media_type"].eq("video")].sort_values("quality_score", ascending=False).head(top_n)

    if mode == "Családi / esemény válogatás":
        work = work[~work.get("is_screenshot", False).fillna(False)]
        return work.sort_values("quality_score", ascending=False).head(top_n)

    if mode == "Karácsony":
        mask = work["tags"].fillna("").str.contains("karácsony|karacsony|december", case=False, regex=True)
        return work[mask].sort_values("quality_score", ascending=False).head(top_n)

    if mode == "Nyár / nyaralás":
        mask = work["tags"].fillna("").str.contains("nyár|nyar|nyaralás|nyaralas|balaton", case=False, regex=True)
        return work[mask].sort_values("quality_score", ascending=False).head(top_n)

    return work.head(top_n)


def folder_for_row(row, structure: str):
    year = int(row.get("year", 0) or 0)
    month = int(row.get("month", 0) or 0)
    tags = str(row.get("tags", "")).lower()
    media_type = row.get("media_type", "image")

    media_folder = "Videók" if media_type == "video" else "Képek"

    if structure == "Év / hónap":
        if year and month:
            return Path(str(year)) / f"{month:02d}" / media_folder
        return Path("Dátum_nélkül") / media_folder

    if structure == "Téma szerint":
        if "karácsony" in tags or "karacsony" in tags or "december" in tags:
            return Path("Ünnepek") / "Karácsony" / media_folder
        if "nyár" in tags or "nyar" in tags or "balaton" in tags:
            return Path("Nyaralás") / media_folder
        if "screenshot" in tags:
            return Path("Screenshotok")
        if "traktor" in tags or "auto" in tags or "autó" in tags:
            return Path("Járművek") / media_folder
        if "foci" in tags or "football" in tags:
            return Path("Sport") / media_folder
        return Path("Egyéb") / media_folder

    if structure == "Év / téma":
        base = Path(str(year)) if year else Path("Dátum_nélkül")
        return base / folder_for_row(row, "Téma szerint")

    return Path("Rendezett") / media_folder


def create_organized_zip(df: pd.DataFrame, structure: str, include_mode: str, top_n: int = 500) -> bytes:
    if df.empty:
        return b""

    work = df.copy()
    if include_mode == "Csak top média":
        work = work.sort_values("quality_score", ascending=False).head(top_n)
    elif include_mode == "Duplikátumok nélkül":
        work = work.drop_duplicates("md5", keep="first")
    elif include_mode == "Screenshotok nélkül":
        work = work[~work.get("is_screenshot", False).fillna(False)]
    elif include_mode == "Csak képek":
        work = work[work["media_type"].eq("image")]
    elif include_mode == "Csak videók":
        work = work[work["media_type"].eq("video")]

    mem = io.BytesIO()
    used_names = set()

    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for _, row in work.iterrows():
            src = Path(row["path"])
            if not src.exists():
                continue
            folder = folder_for_row(row, structure)
            filename = safe_name(row.get("filename", src.name))
            arc = str(folder / filename)

            if arc in used_names:
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                arc = str(folder / f"{stem}_{hashlib.md5(str(src).encode()).hexdigest()[:6]}{suffix}")
            used_names.add(arc)
            z.write(src, arc)

        index_csv = work.drop(columns=["md5", "phash"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        z.writestr("LifeLens_index.csv", index_csv)

    mem.seek(0)
    return mem.read()


def create_html_album(df: pd.DataFrame, title: str, max_items: int = 80) -> bytes:
    work = df.sort_values("quality_score", ascending=False).head(max_items).copy()
    rows = []
    for _, row in work.iterrows():
        src = Path(row["path"])
        preview = row.get("preview_path", "")
        preview_html = f'<img src="{preview}" style="width:100%;border-radius:12px;max-height:180px;object-fit:cover;">' if preview and Path(str(preview)).exists() else f"<div class='thumb'>{row.get('filename','')}</div>"
        rows.append(f"""
        <div class="card">
            {preview_html}
            <p><b>{row.get('media_type','')}</b> · {row.get('filename','')}</p>
            <p><b>Dátum:</b> {row.get('date','')}</p>
            <p><b>Pontszám:</b> {row.get('quality_score','')}</p>
            <p><b>Címkék:</b> {row.get('tags','')}</p>
            <p><small>{src}</small></p>
        </div>
        """)

    html = f"""
    <!doctype html>
    <html lang="hu">
    <head>
      <meta charset="utf-8">
      <title>{title}</title>
      <style>
        body {{ font-family: Arial, sans-serif; background:#f8fafc; color:#0f172a; margin:40px; }}
        .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(260px,1fr)); gap:16px; }}
        .card {{ background:white; border:1px solid #e2e8f0; border-radius:16px; padding:16px; }}
        .thumb {{ height:120px; border-radius:12px; background:#e2e8f0; display:flex; align-items:center; justify-content:center; text-align:center; padding:8px; }}
      </style>
    </head>
    <body>
      <h1>{title}</h1>
      <p>LifeLens AI Private – helyben generált album index</p>
      <div class="grid">
        {''.join(rows)}
      </div>
    </body>
    </html>
    """
    return html.encode("utf-8")


# Sidebar

st.sidebar.markdown("---")
st.sidebar.subheader("Helyi futtatási segítség")
st.sidebar.caption("Privát családi képekhez használd a 💻 Helyi privát futtatás fület.")

st.sidebar.header("1. Forrás kiválasztása")

source_mode = st.sidebar.radio(
    "Honnan jöjjenek a képek/videók?",
    ["ZIP feltöltés", "Helyi mappa útvonala"],
    help="Privát képekhez helyi futtatást javaslok. Streamlit Cloud esetén a ZIP a szerverre kerülne.",
)

work_root = None

if "work_dir" not in st.session_state:
    st.session_state.work_dir = tempfile.mkdtemp(prefix="lifelens_")

base_tmp = Path(st.session_state.work_dir)

if source_mode == "ZIP feltöltés":
    uploaded = st.sidebar.file_uploader("Képek és videók ZIP fájlban", type=["zip"])
    if uploaded:
        extract_dir = base_tmp / "uploaded_zip"
        if st.sidebar.button("ZIP kibontása és beolvasása", use_container_width=True):
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            extract_zip(uploaded, extract_dir)
            st.session_state["photo_root"] = str(extract_dir)
            st.success("ZIP kibontva.")
else:
    folder_text = st.sidebar.text_input("Helyi mappa útvonala", value="")
    st.sidebar.caption("Példa Windows: C:\\Users\\Gabor\\Pictures")
    if st.sidebar.button("Mappa beállítása", use_container_width=True):
        p = Path(folder_text)
        if p.exists() and p.is_dir():
            st.session_state["photo_root"] = str(p)
            st.success("Mappa beállítva.")
        else:
            st.error("Ez a mappa nem található.")

if st.session_state.get("photo_root"):
    work_root = Path(st.session_state["photo_root"])

st.sidebar.header("2. Elemzés")
limit_scan = st.sidebar.number_input("Max. médiafájl teszthez", min_value=100, max_value=100000, value=3000, step=100)
scan_btn = st.sidebar.button("Indexelés indítása", use_container_width=True, disabled=work_root is None)

if scan_btn and work_root:
    df = scan_media(work_root, int(limit_scan))
    st.session_state["photo_index"] = df
    st.success("Indexelés kész.")

df = st.session_state.get("photo_index", pd.DataFrame())

if work_root is None:
    st.info("Első lépés: válassz ZIP-et vagy helyi mappát a bal oldalon.")
    st.stop()

if df.empty:
    st.info("A forrás beállítva. Indítsd el az indexelést a bal oldalon.")
    st.stop()


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Áttekintés",
    "🔎 Keresés",
    "🎞️ Videók",
    "🧹 Nagytakarítás",
    "🎯 Albumok",
    "📦 Rendezett ZIP",
    "🧭 Jövőbeli források",
    "💻 Helyi privát futtatás",
])

with tab1:
    st.subheader("Fotó- és videóarchívum áttekintés")

    total = len(df)
    img_count = int(df["media_type"].eq("image").sum()) if "media_type" in df.columns else 0
    vid_count = int(df["media_type"].eq("video").sum()) if "media_type" in df.columns else 0
    years = df["year"].nunique() if "year" in df.columns else 0
    total_size = df["size_mb"].sum() if "size_mb" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Összes média", f"{total:,}".replace(",", " "))
    c2.metric("Képek", img_count)
    c3.metric("Videók", vid_count)
    c4.metric("Évek", years)
    c5.metric("Méret", f"{total_size:.1f} MB")

    if "year" in df.columns:
        st.markdown("### Évek szerinti eloszlás")
        year_df = df.groupby(["year", "media_type"]).size().reset_index(name="Darab").dropna()
        pivot = year_df.pivot(index="year", columns="media_type", values="Darab").fillna(0)
        st.bar_chart(pivot)

    show_cols = ["media_type", "filename", "date", "year", "month", "size_mb", "duration_sec", "quality_score", "tags", "folder"]
    st.dataframe(df[[c for c in show_cols if c in df.columns]].head(300), use_container_width=True)


with tab2:
    st.subheader("Keresés képekben és videókban")
    q = st.text_input("Mit keresel?", placeholder="pl. Balaton, karácsony, traktor, 2024, videó, screenshot")
    result = df.copy()

    if q:
        ql = q.lower().strip()
        mask = (
            result["filename"].fillna("").str.lower().str.contains(ql, regex=False)
            | result["folder"].fillna("").str.lower().str.contains(ql, regex=False)
            | result["tags"].fillna("").str.lower().str.contains(ql, regex=False)
            | result["media_type"].fillna("").str.lower().str.contains(ql, regex=False)
            | result["year"].fillna("").astype(str).str.contains(ql, regex=False)
        )
        result = result[mask]

    st.write(f"Találatok: **{len(result)}**")
    st.dataframe(result[["media_type", "filename", "date", "duration_sec", "quality_score", "tags", "path"]].head(500), use_container_width=True)

    if not result.empty:
        st.markdown("### Gyors előnézet")
        cols = st.columns(4)
        for i, (_, row) in enumerate(result.head(12).iterrows()):
            p = Path(str(row.get("preview_path") or row.get("path")))
            if p.exists():
                with cols[i % 4]:
                    st.image(str(p), caption=f"{row['media_type']} · {row['filename']}", use_container_width=True)


with tab3:
    st.subheader("Videók")
    videos = df[df["media_type"].eq("video")].copy()
    if videos.empty:
        st.info("Nem találtam videót.")
    else:
        total_duration = videos["duration_sec"].fillna(0).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Videók száma", len(videos))
        c2.metric("Összes hossz", f"{total_duration/60:.1f} perc")
        c3.metric("Videók mérete", f"{videos['size_mb'].sum():.1f} MB")

        st.dataframe(videos[["filename", "date", "duration_sec", "width", "height", "fps", "quality_score", "tags", "path"]].head(500), use_container_width=True)

        st.markdown("### Preview képkockák")
        cols = st.columns(4)
        for i, (_, row) in enumerate(videos.sort_values("quality_score", ascending=False).head(12).iterrows()):
            p = Path(str(row.get("preview_path", "")))
            if p.exists():
                with cols[i % 4]:
                    st.image(str(p), caption=f"{row['filename']} · {row.get('duration_sec')} mp", use_container_width=True)


with tab4:
    st.subheader("Nagytakarítás")
    dup = duplicate_summary(df)
    dup_count = len(dup)
    dup_size = dup["size_mb"].sum() if not dup.empty and "size_mb" in dup.columns else 0
    screenshots = int(df.get("is_screenshot", pd.Series(dtype=bool)).fillna(False).sum()) if "is_screenshot" in df.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Pontos duplikátum sorok", dup_count)
    c2.metric("Duplikátumok mérete", f"{dup_size:.1f} MB")
    c3.metric("Screenshotok", screenshots)

    if dup.empty:
        st.success("Nem találtam pontos MD5 duplikátumot.")
    else:
        st.dataframe(dup[["duplicate_group", "media_type", "filename", "size_mb", "path"]].head(1000), use_container_width=True)

    st.markdown("### Gyenge minőség gyanús média")
    lowq = df.sort_values("quality_score", ascending=True).head(100)
    st.dataframe(lowq[["media_type", "filename", "quality_score", "size_mb", "path"]], use_container_width=True)


with tab5:
    st.subheader("Albumok")
    album_mode = st.selectbox("Album típusa", ["Top képek és videók", "Top fotók", "Top videók", "Családi / esemény válogatás", "Karácsony", "Nyár / nyaralás"])
    top_n = st.slider("Médiafájlok száma", 10, 500, 80, step=10)
    album = build_album(df, album_mode, top_n)

    st.write(f"Album elemek: **{len(album)}**")
    st.dataframe(album[["media_type", "filename", "date", "duration_sec", "quality_score", "tags", "path"]].head(500), use_container_width=True)

    if not album.empty:
        cols = st.columns(4)
        for i, (_, row) in enumerate(album.head(16).iterrows()):
            p = Path(str(row.get("preview_path") or row.get("path")))
            if p.exists():
                with cols[i % 4]:
                    st.image(str(p), caption=f"{row['media_type']} · {row['filename']}", use_container_width=True)

        html_bytes = create_html_album(album, f"LifeLens album – {album_mode}", max_items=top_n)
        st.download_button(
            "HTML album index letöltése",
            data=html_bytes,
            file_name=f"LifeLens_album_{safe_name(album_mode)}.html",
            mime="text/html",
            use_container_width=True,
        )


with tab6:
    st.subheader("Rendezett ZIP export")
    st.caption("Az eredeti fájlokat nem törli és nem mozgatja. Új ZIP-et készít másolatokkal.")

    structure = st.selectbox("Mappastruktúra", ["Év / hónap", "Téma szerint", "Év / téma"])
    include_mode = st.selectbox("Mit tegyen bele?", ["Minden média", "Duplikátumok nélkül", "Screenshotok nélkül", "Csak top média", "Csak képek", "Csak videók"])
    export_top_n = st.slider("Top média száma, ha azt választod", 50, 2000, 500, step=50)

    if st.button("Rendezett ZIP elkészítése", use_container_width=True):
        with st.spinner("ZIP készítése..."):
            zip_bytes = create_organized_zip(df, structure, include_mode, export_top_n)
            st.session_state["organized_zip"] = zip_bytes
        st.success("ZIP elkészült.")

    if st.session_state.get("organized_zip"):
        st.download_button(
            "Rendezett ZIP letöltése",
            data=st.session_state["organized_zip"],
            file_name=f"LifeLens_Rendezett_{safe_name(structure)}.zip",
            mime="application/zip",
            use_container_width=True,
        )


with tab7:
    st.subheader("Következő csatlakozók")
    st.markdown(
        """
        **Google Drive**
        - OAuth belépés
        - Drive mappák listázása
        - képek/videók indexelése
        - export új Drive mappába

        **OneDrive**
        - Microsoft OAuth
        - képek/videók listázása
        - rendezett export OneDrive-ba

        **Telefon**
        - első körben telefon backup mappa
        - később mobilapp

        **Google Photos**
        - lehetséges, de API-korlátok miatt későbbi kör

        **Helyi desktop verzió**
        - ez lenne a legprivátabb
        - nem kell feltölteni semmit
        - közvetlenül tud mappát/HDD-t/NAS-t olvasni
        """
    )


with tab8:
    st.subheader("💻 Helyi privát futtatás")
    st.markdown(
        """
        Ez a rész azoknak szól, akik **nem szeretnék feltölteni** a családi képeiket/videóikat online felületre.

        **A legbiztonságosabb működés:**
        - az app a saját gépeden fut,
        - helyi mappát vagy külső meghajtót olvas,
        - a képek nem mennek fel szerverre,
        - az eredeti képeket nem törli és nem mozgatja.
        """
    )

    st.info(
        "Online Streamlit Cloudon a helyi C:\\ vagy D:\\ útvonal nem működik, mert a felhős szerver nem látja a saját géped meghajtóit. "
        "Helyi futtatásnál viszont működik."
    )

    st.markdown("### 1. Python telepítése, ha még nincs")
    st.markdown(
        """
        Töltsd le a hivatalos Python telepítőt a Python oldaláról.  
        Telepítéskor fontos: **Add Python to PATH** legyen bepipálva.
        """
    )
    st.link_button("Python letöltése – hivatalos oldal", "https://www.python.org/downloads/")

    st.markdown("### 2. Dupla kattintós indító / telepítő csomag")
    st.markdown(
        """
        A csomagban van:
        - `RUN_SOURCE_LOCAL.bat` – gyors helyi futtatás,
        - `BUILD_EXE_WINDOWS.bat` – Windows EXE készítés,
        - `BUILD_INSTALLER_WINDOWS.bat` – telepítő készítés Inno Setup-pal,
        - adatvédelmi szöveg és útmutató.
        """
    )

    builder_bytes = get_local_builder_zip_bytes()
    if builder_bytes:
        st.download_button(
            "LifeLens helyi indító / desktop builder csomag letöltése",
            data=builder_bytes,
            file_name="LifeLens_AI_Private_Desktop_Builder.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.warning(
            "A desktop builder ZIP nincs az app mappájában. "
            "Tedd a LifeLens_AI_Private_Desktop_Builder.zip fájlt ugyanabba a mappába, ahol az app fut."
        )

    st.markdown("### 3. Használat egyszerűen")
    st.code(
        """1. ZIP kicsomagolása például: C:\\LifeLens
2. Dupla katt: RUN_SOURCE_LOCAL.bat
3. Megnyílik: http://localhost:8501
4. Helyi mappa útvonala működni fog, pl. C:\\Users\\T470\\Desktop\\Foto_App""",
        language="text",
    )

    st.markdown("### Miért nem lehet egy online appból Python-telepítést indítani?")
    st.markdown(
        """
        A böngésző biztonsági okból nem engedi, hogy egy webapp programokat telepítsen a gépedre vagy közvetlenül olvassa a mappáidat.  
        Ez jó dolog: ettől biztonságosabb a géped. Ezért kell a helyi indítócsomag vagy később egy rendes Windows telepítő.
        """
    )
