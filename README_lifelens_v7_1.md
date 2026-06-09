# LifeLens V7.1 – Family Analytics + Deep Analysis

Új irány: nem sima fotórendező, hanem családi adatelemző dashboard.

## V7.1 újdonság

- Gyors index külön
- Opcionális Deep Analysis külön
- Deep Analysis helyi CLIP AI-val elemzi a képek tényleges tartalmát
- A dashboard már AI nélkül is működik
- AI után jobb lesz:
  - korszakfelismerés
  - Family DNA
  - érdeklődési trend
  - hidden memories

## Fő funkciók

- Helyi képmappa beolvasása
- Képek és videók indexelése
- Videó preview képkocka
- Slicerek: év, személy, téma, média
- Family Dashboard
- Korszak Explorer
- Kapcsolati háló
- Érdeklődési trendek
- Hidden Memories
- Képgaléria
- Album ZIP export
- Index CSV mentés/betöltés

## Indítás

1. Csomagold ki a ZIP-et.
2. Kattints duplán:

START_LIFELENS_V7_1_LOCAL.bat

Vagy kézzel:

```bash
pip install -r requirements.txt
streamlit run lifelens_v7_1_family_analytics.py
```

## Deep Analysis / AI képfelismerés

Ha AI-t is szeretnél:

1. Futtasd:

INSTALL_AI_IMAGE_RECOGNITION.bat

2. Az appban indítsd a Deep Analysis-t.

Első AI futáskor a CLIP modell letöltődik a gépre, utána helyben fut.

## Privát használat

Privát családi képeknél helyi futtatás javasolt.  
Az app nem tölti fel a képeket külső szerverre.
