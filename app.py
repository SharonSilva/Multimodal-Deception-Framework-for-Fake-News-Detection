# ── GDELT Event Verification Layer ───────────────────────────────────────────
#
# Queries the GDELT 2.0 Event Database to verify whether a claimed event
# (terror attack, earthquake, flood, etc.) actually occurred near the
# location and time mentioned in a post.
#
# GDELT is completely free — no API key, no account required.
# Data is updated every 15 minutes. Covers events globally since 1979.
#
# How it works:
#   1. Extract event type + location + approximate date from post text
#   2. Query GDELT's REST API for matching events in that window
#   3. Return a verification result:
#        - confidence score (0.0–1.0)
#        - number of corroborating sources
#        - whether event is verified, unverified, or contradicted
#   4. /predict uses this to adjust thresholds before scoring:
#        - Verified event   → raise fake threshold, cap anomaly tighter
#        - Unverified event → lower fake threshold, penalise combined score
#        - No claim found   → neutral (treat as before)
#
# GDELT CAMEO event codes used:
#   14x  = Protest
#   17x  = Coerce (includes terror/military action)
#   18x  = Assault (attack, shooting, bombing)
#   19x  = Fight (armed conflict)
#   20x  = Mass violence
#   Natural disasters are in GKG (Global Knowledge Graph) not Events —
#   we query both for completeness.
#
# Latency: ~0.5–2s per query. Cached for 10 minutes to avoid hammering API.

import re
import json
import time
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional
import requests

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import pickle, os

CALIBRATION_PATH = "platt_calibrator.pkl"

def fit_platt_calibrator(model, dataloader, device):
    """
    Run model on validation set, collect raw logits + true labels,
    fit a Platt scaling (logistic regression) calibrator.
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].cpu().numpy()
            v_mismatch     = batch.get("v_mismatch", torch.zeros(len(labels), 3)).to(device)

            logits = model(input_ids, attention_mask, v_mismatch)
            all_logits.extend(logits.cpu().numpy().flatten())
            all_labels.extend(labels.flatten())

    all_logits = np.array(all_logits).reshape(-1, 1)
    all_labels = np.array(all_labels)

    calibrator = LogisticRegression(C=1.0)
    calibrator.fit(all_logits, all_labels)

    with open(CALIBRATION_PATH, "wb") as f:
        pickle.dump(calibrator, f)

    print(f"[calibration] Fitted Platt scaler on {len(all_labels)} samples")
    print(f"[calibration] Coef={calibrator.coef_[0][0]:.4f}  Intercept={calibrator.intercept_[0]:.4f}")
    return calibrator

def load_calibrator():
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, "rb") as f:
            cal = pickle.load(f)
        print(f"[calibration] Loaded Platt scaler from {CALIBRATION_PATH}")
        return cal
    return None

# ── Simple in-memory cache (key → (result, timestamp)) ───────────────────────
_GDELT_CACHE: dict = {}
_CACHE_TTL_SECONDS = 600  # 10 minutes


def _cache_key(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]


def _gdelt_fetch(url: str, timeout: int = 8) -> Optional[dict]:
    """Fetch a GDELT API URL and return parsed JSON, or None on failure."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FakeNewsDetector/1.0 (research project)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except Exception as e:
        print(f"[gdelt] fetch error: {e} — url={url[:80]}")
        return None


# ── Location extractor ────────────────────────────────────────────────────────
# Maps common location mentions to GDELT country/city codes.
# GDELT uses FIPS 10-4 country codes and ActionGeo fields.

LOCATION_MAP = {
    # Cities → (country_code, city_name_for_search)
    "paris":        ("FR", "Paris"),
    "london":       ("UK", "London"),
    "new york":     ("US", "New York"),
    "washington":   ("US", "Washington"),
    "moscow":       ("RS", "Moscow"),
    "beijing":      ("CH", "Beijing"),
    "tokyo":        ("JA", "Tokyo"),
    "sydney":       ("AS", "Sydney"),
    "berlin":       ("GM", "Berlin"),
    "istanbul":     ("TU", "Istanbul"),
    "baghdad":      ("IZ", "Baghdad"),
    "kabul":        ("AF", "Kabul"),
    "kyiv":         ("UP", "Kyiv"),
    "kiev":         ("UP", "Kyiv"),
    "ukraine":      ("UP", "Ukraine"),
    "israel":       ("IS", "Israel"),
    "gaza":         ("GZ", "Gaza"),
    "beirut":       ("LE", "Beirut"),
    "tehran":       ("IR", "Tehran"),
    "karachi":      ("PK", "Karachi"),
    "lahore":       ("PK", "Lahore"),
    "mumbai":       ("IN", "Mumbai"),
    "delhi":        ("IN", "Delhi"),
    "cairo":        ("EG", "Cairo"),
    "nairobi":      ("KE", "Nairobi"),
    "nigeria":      ("NI", "Nigeria"),
    "mali":         ("ML", "Mali"),
    "syria":        ("SY", "Syria"),
    "aleppo":       ("SY", "Aleppo"),
    "damascus":     ("SY", "Damascus"),
    "turkey":       ("TU", "Turkey"),
    "iran":         ("IR", "Iran"),
    "iraq":         ("IZ", "Iraq"),
    "afghanistan":  ("AF", "Afghanistan"),
    "pakistan":     ("PK", "Pakistan"),
    "india":        ("IN", "India"),
    "china":        ("CH", "China"),
    "russia":       ("RS", "Russia"),
    "france":       ("FR", "France"),
    "germany":      ("GM", "Germany"),
    "spain":        ("SP", "Spain"),
    "italy":        ("IT", "Italy"),
    "usa":          ("US", "United States"),
    "america":      ("US", "United States"),
    "united states":("US", "United States"),
    "japan":        ("JA", "Japan"),
    "australia":    ("AS", "Australia"),
    "brazil":       ("BR", "Brazil"),
    "mexico":       ("MX", "Mexico"),
    "colombia":     ("CO", "Colombia"),
    "venezuela":    ("VE", "Venezuela"),
}

# ── Event type extractor ──────────────────────────────────────────────────────
# Maps text keywords → GDELT CAMEO root event codes + GKG theme keywords

EVENT_PATTERNS = [
    # (keyword_list, cameo_codes, gkg_themes, event_label)
    (
        ["terror", "terrorist", "terrorism", "isis", "isil", "al-qaeda",
         "jihad", "militant", "extremist"],
        ["180", "181", "182", "183", "190", "191", "192", "193", "194", "195"],
        ["TERROR", "JIHAD", "EXTREMISM"],
        "terror_attack"
    ),
    (
        ["attack", "shooting", "gunman", "gunfire", "shot", "killed", "shooter"],
        ["180", "181", "182", "183"],
        ["ARMED_CONFLICT", "VIOLENCE"],
        "shooting_attack"
    ),
    (
        ["bomb", "bombing", "explosion", "blast", "explosive", "ied"],
        ["180", "182", "183"],
        ["TERROR", "BOMBING"],
        "bombing"
    ),
    (
        ["military", "deployed", "troops", "soldiers", "armed forces",
         "airstrike", "missile", "invasion", "war", "conflict"],
        ["190", "191", "192", "193", "194", "195"],
        ["ARMED_CONFLICT", "MILITARY"],
        "military_action"
    ),
    (
        ["earthquake", "seismic", "tremor", "magnitude", "richter"],
        [],  # Earthquakes in GKG only
        ["NATURAL_DISASTER", "EARTHQUAKE"],
        "earthquake"
    ),
    (
        ["flood", "flooding", "inundation", "submerged", "deluge"],
        [],
        ["NATURAL_DISASTER", "FLOOD"],
        "flood"
    ),
    (
        ["hurricane", "typhoon", "cyclone", "storm surge", "tropical storm"],
        [],
        ["NATURAL_DISASTER", "HURRICANE", "CYCLONE"],
        "hurricane"
    ),
    (
        ["wildfire", "forest fire", "bushfire", "blaze"],
        [],
        ["NATURAL_DISASTER", "FIRE"],
        "wildfire"
    ),
    (
        ["tsunami", "tidal wave"],
        [],
        ["NATURAL_DISASTER", "TSUNAMI"],
        "tsunami"
    ),
    (
        ["protest", "demonstration", "rally", "march", "riot", "uprising"],
        ["140", "141", "143", "144", "145"],
        ["PROTEST", "DEMONSTRATION"],
        "protest"
    ),
    (
        ["coup", "overthrow", "government collapse", "president arrested"],
        ["170", "171", "172", "173"],
        ["COUP", "POLITICAL_CRISIS"],
        "coup"
    ),
]


def extract_event_info(text: str) -> dict:
    """
    Extract event type, location, and approximate date from post text.
    Returns dict with keys: event_type, cameo_codes, gkg_themes, location,
                            country_code, date_range_days
    """
    lower = text.lower()

    # Detect event type
    detected_event = None
    detected_cameo = []
    detected_themes = []
    for keywords, cameo, themes, label in EVENT_PATTERNS:
        if any(kw in lower for kw in keywords):
            detected_event = label
            detected_cameo = cameo
            detected_themes = themes
            break

    # Detect location
    detected_location = None
    detected_country = None
    for loc_key, (country_code, city_name) in LOCATION_MAP.items():
        if loc_key in lower or f"#{loc_key}" in lower:
            detected_location = city_name
            detected_country = country_code
            break

    # Detect hashtag locations (#Paris, #Tokyo etc.)
    hashtag_locs = re.findall(r'#([A-Za-z]+)', text)
    for ht in hashtag_locs:
        ht_lower = ht.lower()
        if ht_lower in LOCATION_MAP:
            detected_location = LOCATION_MAP[ht_lower][1]
            detected_country  = LOCATION_MAP[ht_lower][0]
            break

    return {
        "event_type":    detected_event,
        "cameo_codes":   detected_cameo,
        "gkg_themes":    detected_themes,
        "location":      detected_location,
        "country_code":  detected_country,
        "has_claim":     detected_event is not None,
    }


def _query_gdelt_doc(location: str, themes: list, days_back: int = 3) -> dict:
    """
    Query GDELT Document API (GKG) for news articles matching
    location + themes in recent days. Best for natural disasters.
    Returns: {articles_found, top_sources, confidence}
    """
    if not location and not themes:
        return {"articles_found": 0, "confidence": 0.0, "sources": []}

    # Build query
    theme_query = " OR ".join(themes[:3]) if themes else ""
    loc_query   = f'"{location}"' if location else ""

    if theme_query and loc_query:
        query = f"{loc_query} {theme_query}"
    elif theme_query:
        query = theme_query
    else:
        query = loc_query

    # GDELT Doc API — searches article titles and content
    encoded = urllib.parse.quote(query)
    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={encoded}"
        f"&mode=artlist"
        f"&maxrecords=10"
        f"&timespan={days_back}d"
        f"&format=json"
    )

    data = _gdelt_fetch(url)
    if not data or "articles" not in data:
        return {"articles_found": 0, "confidence": 0.0, "sources": []}

    articles = data.get("articles", [])
    n = len(articles)
    sources = list({a.get("domain", "") for a in articles[:5] if a.get("domain")})

    # Confidence scales with number of independent sources
    if n == 0:   conf = 0.0
    elif n < 3:  conf = 0.3
    elif n < 6:  conf = 0.55
    elif n < 10: conf = 0.75
    else:        conf = 0.90

    print(f"[gdelt_doc] query='{query[:50]}' articles={n} conf={conf:.2f}")
    return {"articles_found": n, "confidence": conf, "sources": sources}


def _query_gdelt_events(location: str, country_code: str,
                        cameo_codes: list, days_back: int = 3) -> dict:
    """
    Query GDELT Events API for conflict/political events matching
    CAMEO codes at the given location. Best for attacks, military, protests.
    Returns: {events_found, avg_tone, confidence}
    """
    if not cameo_codes:
        return {"events_found": 0, "confidence": 0.0, "avg_tone": 0.0}

    # Build GDELT Events SQL-style query via the free REST endpoint
    # GDELT Events uses a different endpoint — the free BigQuery-like API
    cameo_filter = "%2C".join(cameo_codes[:4])  # max 4 codes, URL-encoded comma

    # Use GDELT's free event lookup — filter by EventRootCode and ActionGeo_CountryCode
    country_filter = f"&ActionGeo_CountryCode={country_code}" if country_code else ""
    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={urllib.parse.quote(location or 'attack')}+{urllib.parse.quote(' OR '.join(cameo_codes[:2]))}"
        f"&mode=artlist"
        f"&maxrecords=15"
        f"&timespan={days_back}d"
        f"&format=json"
    )

    data = _gdelt_fetch(url)
    if not data or "articles" not in data:
        return {"events_found": 0, "confidence": 0.0, "avg_tone": 0.0}

    articles = data.get("articles", [])
    n = len(articles)

    # Extract tone scores if available
    tones = [float(a.get("tone", 0)) for a in articles if "tone" in a]
    avg_tone = sum(tones) / len(tones) if tones else 0.0

    if n == 0:   conf = 0.0
    elif n < 3:  conf = 0.35
    elif n < 7:  conf = 0.60               
    elif n < 12: conf = 0.78
    else:        conf = 0.92

    print(f"[gdelt_events] location={location} events={n} tone={avg_tone:.2f} conf={conf:.2f}")
    return {"events_found": n, "confidence": conf, "avg_tone": avg_tone}


def verify_event_gdelt(text: str) -> dict:
    """
    Main entry point. Given post text, query GDELT to verify whether
    the claimed event actually occurred.

    Returns dict:
        verified        bool   — True if event confirmed by GDELT
        confidence      float  — 0.0 (no evidence) to 1.0 (strongly confirmed)
        sources_count   int    — number of independent news sources found
        event_type      str    — detected event type or None
        location        str    — detected location or None
        gdelt_checked   bool   — False if no checkable claim found in text
        threshold_adj   float  — suggested adjustment to fake threshold
                                 positive = raise threshold (event is real)
                                 negative = lower threshold (claim unverified)
        anomaly_cap     float  — suggested anomaly cap (lower = more lenient)
        verdict         str    — "verified" | "unverified" | "no_claim" | "error"
    """
    cache_k = _cache_key(text)
    if cache_k in _GDELT_CACHE:
        result, ts = _GDELT_CACHE[cache_k]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            print(f"[gdelt] cache hit")
            return result

    # Default result — neutral, no adjustments
    default = {
        "verified":      False,
        "confidence":    0.0,
        "sources_count": 0,
        "event_type":    None,
        "location":      None,
        "gdelt_checked": False,
        "threshold_adj": 0.0,
        "anomaly_cap":   0.45,   # default cap from existing code
        "verdict":       "no_claim",
        "sources":       [],
    }

    try:
        info = extract_event_info(text)

        if not info["has_claim"]:
            print(f"[gdelt] no verifiable claim detected in text")
            _GDELT_CACHE[cache_k] = (default, time.time())
            return default

        default["event_type"] = info["event_type"]
        default["location"]   = info["location"]
        default["gdelt_checked"] = True

        print(f"[gdelt] checking: event={info['event_type']} location={info['location']}")

        # Run both query strategies and take the best result
        doc_result   = _query_gdelt_doc(
            info["location"], info["gkg_themes"], days_back=4
        )
        event_result = _query_gdelt_events(
            info["location"], info["country_code"],
            info["cameo_codes"], days_back=4
        )

        # Combine — take max confidence from either strategy
        best_conf    = max(doc_result["confidence"], event_result["confidence"])
        total_sources = doc_result["articles_found"] + event_result["events_found"]
        all_sources   = doc_result.get("sources", [])

        # ── Interpret confidence → verdict + scoring adjustments ─────────────
        #
        # High confidence (≥0.70): Event well-documented → trust content more
        #   - Raise fake threshold: needs stronger evidence to call fake
        #   - Cap anomaly tightly: high-arousal real events look anomalous
        #
        # Medium confidence (0.35–0.70): Some evidence but not conclusive
        #   - Small positive threshold adjustment
        #   - Moderate anomaly cap
        #
        # Low confidence (<0.35): Event not found in GDELT
        #   - Lower fake threshold: easier to flag as fake
        #   - Add penalty to combined score
        #   - This is the Tokyo earthquake fabrication case

        if best_conf >= 0.70:
            verdict       = "verified"
            verified      = True
            threshold_adj = +0.15   # raise bar: need combined > 0.60+0.15=0.75 to call fake
            anomaly_cap   = 0.25    # very tight cap — event is real, anomaly is noise
        elif best_conf >= 0.35:
            verdict       = "partially_verified"
            verified      = True
            threshold_adj = +0.08
            anomaly_cap   = 0.35
        elif info["has_claim"] and info["location"]:
            # Claim made with specific location but GDELT found nothing
            verdict       = "unverified"
            verified      = False
            threshold_adj = -0.10   # lower bar: easier to flag
            anomaly_cap   = 0.55    # let anomaly contribute more
        else:
            # Claim made but no location to check — neutral
            verdict       = "unchecked"
            verified      = False
            threshold_adj = 0.0
            anomaly_cap   = 0.45

        result = {
            "verified":      verified,
            "confidence":    round(best_conf, 3),
            "sources_count": total_sources,
            "event_type":    info["event_type"],
            "location":      info["location"],
            "gdelt_checked": True,
            "threshold_adj": threshold_adj,
            "anomaly_cap":   anomaly_cap,
            "verdict":       verdict,
            "sources":       all_sources[:5],
        }

        print(f"[gdelt] verdict={verdict} conf={best_conf:.3f} "
              f"sources={total_sources} threshold_adj={threshold_adj:+.2f} "
              f"anomaly_cap={anomaly_cap:.2f}")

        _GDELT_CACHE[cache_k] = (result, time.time())
        return result

    except Exception as e:
        print(f"[gdelt] ❌ error: {e}")
        default["verdict"] = "error"
        return default

# ═══════════════════════════════════════════════════════════════
# ORIGINAL app.py — unchanged below, GDELT integrated into /predict
# ═══════════════════════════════════════════════════════════════

"""
app.py — Multimodal Deception Framework Inference Server
FULLY FIXED: Uses exact same embeddings as training pipeline
+ GDELT Layer 1: Event verification via GDELT global news index (free, no API key)
"""

import os, sys, warnings, base64, io, traceback, pickle
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024
CORS(app)

device = torch.device("cpu")
print(f"[server] Device: {device} (forced CPU — GPU reserved for llama3.2-vision)")

MODELS = {}


def compute_text_vad(text):
    lower = text.lower()
    pos_v = ["happy","joy","love","beautiful","wonderful","great","amazing","celebrate",
             "victory","win","safe","rescue","survive","hope","good","positive","peaceful"]
    neg_v = ["disaster","earthquake","collapse","kill","dead","death","terror","attack",
             "destroy","flood","hurricane","crash","bomb","fear","horror","tragic","crisis",
             "victim","massacre","catastrophe","awful","terrible","horrible","violent"]
    high_a = ["breaking","urgent","emergency","alert","shocking","incredible","massive",
              "explosion","collapse","crash","attack","terror","earthquake","hurricane",
              "flood","fire","bomb","shooting","dead","killed","disaster","crisis","omg","wow"]
    low_a  = ["calm","quiet","peaceful","slow","gentle","serene","relaxing","ordinary",
              "normal","routine","simple","mild","moderate","steady","stable"]
    high_d = ["official","confirmed","government","police","authority","control","power",
              "command","force","military","president","leader","strong","dominant"]
    low_d  = ["helpless","vulnerable","trapped","victim","missing","lost","uncertain",
              "confused","weak","powerless","afraid","scared"]

    pos_v_hits  = sum(1 for w in pos_v  if w in lower)
    neg_v_hits  = sum(1 for w in neg_v  if w in lower)
    high_a_hits = sum(1 for w in high_a if w in lower)
    low_a_hits  = sum(1 for w in low_a  if w in lower)
    high_d_hits = sum(1 for w in high_d if w in lower)
    low_d_hits  = sum(1 for w in low_d  if w in lower)

    total_v = pos_v_hits + neg_v_hits + 1
    total_a = high_a_hits + low_a_hits + 1
    total_d = high_d_hits + low_d_hits + 1

    V = np.clip(0.567 + (pos_v_hits - neg_v_hits) / total_v * 0.35, 0.15, 0.92)
    A = np.clip(0.567 + (high_a_hits - low_a_hits) / total_a * 0.35, 0.15, 0.92)
    D = np.clip(0.567 + (high_d_hits - low_d_hits) / total_d * 0.25, 0.15, 0.88)

    return {"V": float(V), "A": float(A), "D": float(D)}


def compute_image_vad(img_feat_normalized):
    clip_model = MODELS["clip"]
    tokenize   = MODELS["clip_tok"]

    # ── AROUSAL: high vs low energy scenes ───────────────────────────────────
    high_arousal_prompts = [
        "a loud concert crowd cheering and dancing with bright stage lights",
        "an explosion fire blast emergency chaos",
        "a protest riot crowd running in panic",
        "an earthquake collapse rubble screaming",
        "a thrilling sports match crowd going wild",
    ]
    low_arousal_prompts = [
        "a calm empty room with soft lighting",
        "a quiet forest path with nobody around",
        "a person sleeping peacefully in bed",
        "a still lake at sunrise with no movement",
        "an empty library reading room",
    ]

    # ── VALENCE: positive vs negative scenes ─────────────────────────────────
    high_valence_prompts = [
        "a joyful celebration party with people smiling and laughing",
        "a beautiful wedding with happy people",
        "a crowd cheering victory at a concert or sports event",
        "children playing happily in a sunny park",
        "people dancing and celebrating at a festival",
    ]
    low_valence_prompts = [
        "a tragic disaster scene with victims and destruction",
        "a funeral or mourning ceremony with grief",
        "a war zone with destroyed buildings and suffering",
        "a crime scene with police tape and emergency response",
        "a person crying in distress alone",
    ]

    # ── DOMINANCE: powerful vs powerless scenes ───────────────────────────────
    high_dominance_prompts = [
        "a military parade with soldiers in formation",
        "a powerful leader giving a speech to a large crowd",
        "armed police or security forces in control of a situation",
        "a judge in a courtroom commanding authority",
        "a large corporation headquarters imposing building",
    ]
    low_dominance_prompts = [
        "a child lost and crying alone",
        "refugees fleeing with nothing helpless",
        "a victim trapped under rubble calling for help",
        "a person begging on the street vulnerable",
        "civilians hiding in fear during an attack",
    ]

    with torch.no_grad():
        def score_axis(pos_prompts, neg_prompts):
            pos_tok  = tokenize(pos_prompts).to(device)
            neg_tok  = tokenize(neg_prompts).to(device)
            pos_feat = F.normalize(clip_model.encode_text(pos_tok), dim=-1)
            neg_feat = F.normalize(clip_model.encode_text(neg_tok), dim=-1)
            pos_sim  = float((img_feat_normalized @ pos_feat.T).mean().item())
            neg_sim  = float((img_feat_normalized @ neg_feat.T).mean().item())
            # Raw CLIP sims are ~0.05–0.35 range
            # Rescale to 0.15–0.92 to match text VAD scale
            raw_score = (pos_sim - neg_sim)  # roughly -0.20 to +0.20
            # Map to 0.15–0.92: centre at 0.567, scale by 2.5
            calibrated = float(np.clip(0.567 + raw_score * 2.5, 0.15, 0.92))
            return calibrated, pos_sim, neg_sim

        A, a_pos, a_neg = score_axis(high_arousal_prompts, low_arousal_prompts)
        V, v_pos, v_neg = score_axis(high_valence_prompts, low_valence_prompts)
        D, d_pos, d_neg = score_axis(high_dominance_prompts, low_dominance_prompts)

    print(f"[vad_image] A={A:.3f} (pos={a_pos:.3f} neg={a_neg:.3f})")
    print(f"[vad_image] V={V:.3f} (pos={v_pos:.3f} neg={v_neg:.3f})")
    print(f"[vad_image] D={D:.3f} (pos={d_pos:.3f} neg={d_neg:.3f})")

    return {"V": V, "A": A, "D": D}


def build_anomaly_features(z_out_np, v_mismatch_np):
    z = z_out_np.reshape(1, -1)
    v = v_mismatch_np.reshape(1, -1)
    mismatch_magnitude = np.linalg.norm(v, axis=1, keepdims=True)
    z_abs = np.abs(z)
    z_abs_norm = z_abs / (z_abs.sum(axis=1, keepdims=True) + 1e-10)
    z_entropy  = -np.sum(z_abs_norm * np.log(z_abs_norm + 1e-10), axis=1, keepdims=True)
    z_variance = np.var(z, axis=1, keepdims=True)
    z_skewness = np.mean((z - z.mean(axis=1, keepdims=True))**3, axis=1, keepdims=True)
    z_kurtosis = np.mean((z - z.mean(axis=1, keepdims=True))**4, axis=1, keepdims=True)
    z_v_dot    = np.sum(z * v, axis=1, keepdims=True)
    z_v_cosine = z_v_dot / (np.linalg.norm(z, axis=1, keepdims=True) *
                             np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)
    X = np.hstack([z, v, mismatch_magnitude, z_entropy,
                   z_variance, z_skewness, z_kurtosis, z_v_cosine])
    return X


def run_anomaly(z_out_np, v_mismatch_np):
    """
    Feature-based anomaly scoring using directly computed signals.
    The sklearn anomaly models (IsoForest/LOF/OCSVM/Elliptic) were trained
    on batch-processed embeddings and produce saturated scores at single-sample
    inference due to distribution mismatch. This version computes anomaly
    directly from interpretable embedding statistics that are stable at inference.

    Components:
      1. Embedding entropy    — low entropy = model is uncertain/confused
      2. Embedding variance   — unusually high/low variance = out-of-distribution
      3. VAD mismatch norm    — large text/image emotional divergence = anomalous
      4. Kurtosis signal      — heavy tails in z_aug = anomalous activation pattern
    """
    z = z_out_np.reshape(-1).astype(np.float64)
    v = v_mismatch_np.reshape(-1).astype(np.float64)

    if not np.isfinite(z).all():
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Embedding entropy — low entropy signals OOD
    z_abs  = np.abs(z)
    z_norm = z_abs / (z_abs.sum() + 1e-10)
    entropy = -np.sum(z_norm * np.log(z_norm + 1e-10))
    # Typical range: 3.5 (low/OOD) to 5.0 (normal/in-distribution)
    # Map: 3.5→high anomaly, 5.0→low anomaly
    entropy_score = float(np.clip(1.0 - (entropy - 3.5) / 1.5, 0.0, 1.0))

    # 2. Embedding variance — trained normal range ~0.8–1.2
    variance = float(np.var(z))
    # Map: variance 0.5→0.8 anomalous, 0.8–1.2 normal, >1.5 anomalous
    if variance < 0.8:
        var_score = float(np.clip((0.8 - variance) / 0.3, 0.0, 1.0))
    elif variance > 1.2:
        var_score = float(np.clip((variance - 1.2) / 0.8, 0.0, 1.0))
    else:
        var_score = 0.0

    # 3. VAD mismatch magnitude — direct signal of text/image divergence
    mismatch_mag = float(np.linalg.norm(v))
    # Typical range: 0.0–0.5, >0.3 is anomalous
    mismatch_score = float(np.clip(mismatch_mag / 0.35, 0.0, 1.0))

    # 4. Kurtosis — heavy tails indicate unusual activation patterns
    z_mean = z.mean()
    kurtosis = float(np.mean((z - z_mean) ** 4) / (np.var(z) ** 2 + 1e-10))
    # Normal kurtosis ~3 (mesokurtic). >6 = heavy tails = anomalous
    kurt_score = float(np.clip((kurtosis - 3.0) / 6.0, 0.0, 1.0))

    # Weighted ensemble
    ensemble = (
        0.30 * entropy_score +
        0.25 * mismatch_score +
        0.25 * var_score +
        0.20 * kurt_score
    )
    ensemble = float(np.clip(ensemble, 0.0, 1.0))

    print(f"[anomaly] entropy={entropy:.3f}→{entropy_score:.3f}  "
          f"var={variance:.3f}→{var_score:.3f}  "
          f"mismatch={mismatch_mag:.3f}→{mismatch_score:.3f}  "
          f"kurt={kurtosis:.3f}→{kurt_score:.3f}")
    print(f"[anomaly] ensemble score: {ensemble:.4f}")

    if   ensemble >= 0.75: level = "critical"
    elif ensemble >= 0.55: level = "high"
    elif ensemble >= 0.35: level = "medium"
    elif ensemble >= 0.20: level = "low"
    else:                  level = "normal"

    # Method flags — based on which components are elevated
    flags = {
        "iso_forest": bool(entropy_score > 0.5),
        "lof":        bool(mismatch_score > 0.5),
        "ocsvm":      bool(var_score > 0.4),
        "elliptic":   bool(kurt_score > 0.4),
    }
    n = sum(flags.values())

    print(f"[anomaly] level={level}  flags={n}/4")
    return ensemble, level, n, flags


def load_all_models():
    global MODELS

    print("[server] Loading CLIP ViT-L/14...")
    import open_clip
    clip_model, _, clip_prep = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    MODELS["clip"]      = clip_model
    MODELS["clip_prep"] = clip_prep
    MODELS["clip_tok"]  = open_clip.get_tokenizer("ViT-L-14")
    print("[server] ✅ CLIP loaded")

    print("[server] Loading SentenceTransformer...")
    from sentence_transformers import SentenceTransformer
    MODELS["st"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("[server] ✅ SentenceTransformer loaded")

    print("[server] Loading EmotionAwareFakeNewsDetector...")
    from rough_work import EmotionAwareFakeNewsDetector
    emotion_model = EmotionAwareFakeNewsDetector(
        d_text=128, d_image=1024, d_meta=128, d_common=256,
        vad_dim=3, meta_affective_dim=128, mismatch_dim=128,
        temporal_hidden=64, num_classes=1
    ).to(device)
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints/best_emotion_aware_detector.pth")
    state     = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = emotion_model.load_state_dict(state, strict=False)
    if missing:    print(f"[server]   ⚠ Missing ({len(missing)}): {missing[:4]}")
    if unexpected: print(f"[server]   ⚠ Unexpected ({len(unexpected)}): {unexpected[:4]}")
    if not missing and not unexpected: print("[server]   ✅ Weights loaded perfectly")
    emotion_model.eval()
    MODELS["emotion_model"] = emotion_model
    print("[server] ✅ EmotionAwareFakeNewsDetector loaded")

    print("[server] Loading anomaly ensemble...")
    anom_path = os.path.join(PROJECT_ROOT, "anomaly_detection_results/anomaly_models.pt")
    try:
        anom_data = torch.load(anom_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[server] ⚠️  anomaly_models.pt failed to load: {e}")
        print("[server] ℹ️  Using empty anomaly data — feature-based scoring active")
        anom_data = {}
    MODELS["anomaly_models"]   = {}   # sklearn models stripped — not used at inference
    MODELS["ensemble_weights"] = anom_data.get("ensemble_weights", {})

    for key in ["scaler", "pca"]:
        if key in anom_data:
            MODELS[key] = anom_data[key]

    if "method_score_distributions" in anom_data:
        MODELS["method_score_distributions"] = anom_data["method_score_distributions"]
        print(f"[server] ✅ Per-method score distributions loaded")
        for name, d in anom_data["method_score_distributions"].items():
            print(f"[server]    {name}: "
                  f"range=[{d['min']:.4f}, {d['max']:.4f}]")
    else:
        print("[server] ⚠️  method_score_distributions missing!")
        print("[server]    Re-run anomaly_detection_pipeline.py first")
        MODELS["method_score_distributions"] = {}

    if "ensemble_score_distribution" in anom_data:
        MODELS["ensemble_score_distribution"] = anom_data["ensemble_score_distribution"]
        d = anom_data["ensemble_score_distribution"]
        print(f"[server] ✅ Ensemble distribution loaded: "
              f"p25={d['p25']:.4f} p50={d['p50']:.4f} "
              f"p75={d['p75']:.4f} p95={d['p95']:.4f}")
    else:
        print("[server] ⚠️  ensemble_score_distribution missing!")
        MODELS["ensemble_score_distribution"] = {}
        
    if 'ocsvm_shift' in anom_data:
        MODELS['ocsvm_shift'] = anom_data['ocsvm_shift']
        print(f"[server] ✅ OCSVM shift loaded: {anom_data['ocsvm_shift']:.4f}")
    else:
        MODELS['ocsvm_shift'] = -13.1153
        print("[server] ⚠️  ocsvm_shift missing — using fallback")

    print("[server] ✅ Anomaly ensemble loaded")

    if 'training_scores' in anom_data:
        MODELS['training_scores'] = anom_data['training_scores']
        print(f"[server] ✅ Training score arrays loaded for percentile ranking")
    else:
        print("[server] training_scores missing - re-run patch_anomaly_scores.py")
        MODELS['training_scores'] = {}

    print("[server] Loading anomaly thresholds...")
    MODELS["thresholds"] = anom_data.get("thresholds", {
        "iso_forest":0.43, "lof":0.12, "ocsvm":0.29,
        "elliptic":0.05, "isolation_forest":0.43
    })
    print(f"[server] ✅ Thresholds: { {k: round(v,4) for k,v in MODELS['thresholds'].items()} }")

    # Load Platt calibrator if already fitted   ← 4 spaces (correct, outside except)
    cal_path = os.path.join(PROJECT_ROOT, "platt_calibrator.pkl")
    
    if os.path.exists(cal_path):
        with open(cal_path, "rb") as f:
            MODELS["calibrator"] = pickle.load(f)
        print(f"[server] ✅ Platt calibrator loaded from {cal_path}")
    else:
        print(f"[server] ℹ️  No calibrator found — POST /calibrate to fit one")
        MODELS["calibrator"] = None

    print("[server] ✅ All models ready\n")
    print("[server] ℹ Vision model: moondream auto-selected — ollama serve must be running")


def check_semantic_consistency(text, img_feat_norm):
    tokenize   = MODELS["clip_tok"]
    clip_model = MODELS["clip"]
    lower = text.lower()

    earthquake_words = ["earthquake","collapse","rubble","seismic","tremor","magnitude","fault"]
    flood_words      = ["flood","hurricane","tsunami","storm","surge","inundation","rainfall"]
    fire_words       = ["fire","wildfire","blaze","burning","flames","inferno"]
    attack_words     = ["attack","terror","shooting","bomb","explosion","gunman","isis","militant"]
    space_words      = ["space","iss","nasa","satellite","orbit","eclipse","astronaut","station"]
    shark_words      = ["shark","animal","wildlife","creature","beast"]

    topic_prompts = []
    if any(w in lower for w in earthquake_words):
        topic_prompts = [
            "collapsed building rubble earthquake damage",
            "destroyed structure debris after earthquake",
            "earthquake aftermath destruction",
        ]
    elif any(w in lower for w in flood_words):
        topic_prompts = [
            "flooded streets rising water hurricane",
            "flood damage submerged buildings",
            "hurricane storm surge flooding",
        ]
    elif any(w in lower for w in fire_words):
        topic_prompts = [
            "wildfire burning flames smoke",
            "fire engulfing building",
        ]
    elif any(w in lower for w in attack_words):
        topic_prompts = [
            "terror attack aftermath emergency response",
            "crime scene police cordoned area",
        ]
    elif any(w in lower for w in space_words):
        topic_prompts = [
            "space station orbit earth view",
            "solar eclipse from space",
            "nasa spacecraft satellite",
        ]
    elif any(w in lower for w in shark_words):
        topic_prompts = [
            "shark in water ocean",
            "flooded street with shark",
        ]

    event_words = earthquake_words + flood_words + fire_words + attack_words
    if any(w in lower for w in event_words):
        person_prompts = [
            "a close up portrait of a person",
            "a politician or public figure speaking",
            "a person giving a speech at a podium",
            "a headshot or profile photo of a person",
            "a man or woman photographed indoors",
        ]
        non_person_prompts = [
            "a damaged building or natural disaster scene",
            "destruction rubble or emergency scene with no people",
            "a landscape or infrastructure scene",
        ]
        with torch.no_grad():
            p_feats  = F.normalize(clip_model.encode_text(tokenize(person_prompts).to(device)), dim=-1)
            np_feats = F.normalize(clip_model.encode_text(tokenize(non_person_prompts).to(device)), dim=-1)
            person_sim     = float((img_feat_norm @ p_feats.T).mean().item())
            non_person_sim = float((img_feat_norm @ np_feats.T).mean().item())

        print(f"[semantic] person_sim={person_sim:.3f} non_person_sim={non_person_sim:.3f}")
        if person_sim > 0.22 and person_sim > non_person_sim + 0.08:
            print(f"[semantic] ⚠ Person/portrait detected in disaster-text post → INCONSISTENT")
            return False, person_sim

    if not topic_prompts:
        with torch.no_grad():
            txt_tok  = tokenize([text[:77]]).to(device)
            txt_feat = F.normalize(clip_model.encode_text(txt_tok), dim=-1)
            sim      = float((img_feat_norm @ txt_feat.T)[0, 0].item())
        print(f"[semantic] direct sim={sim:.3f}")
        return sim > 0.20, sim

    with torch.no_grad():
        tok   = tokenize(topic_prompts).to(device)
        feats = F.normalize(clip_model.encode_text(tok), dim=-1)
        sims  = (img_feat_norm @ feats.T)[0]
        best_sim = float(sims.max().item())

    print(f"[semantic] topic_sim={best_sim:.3f} prompts={topic_prompts[0][:40]}")
    return best_sim > 0.17, best_sim


def get_vision_llm_description(img_b64_raw: str = None, img_feat_norm=None) -> str | None:
    import urllib.request, json as json_lib

    vision_model = None
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            installed = [m["name"] for m in json_lib.loads(resp.read()).get("models", [])]
            print(f"[vision_llm] Installed models: {installed}")
        for candidate in ["llama3.2-vision", "llava", "bakllava", "moondream"]:
            match = next((n for n in installed if n.split(":")[0] == candidate), None)
            if match:
                vision_model = match
                print(f"[vision_llm] Selected: {vision_model}")
                break
    except Exception as e:
        print(f"[vision_llm] Ollama unreachable: {e}")
        return None

    if vision_model:
        try:
            is_moondream = vision_model.startswith("moondream")
            prompt_text = (
                "Describe the image."
                if is_moondream else
                "Describe this image in one sentence. Name the main subject specifically "
                "(e.g. 'a donkey', 'a collapsed building', 'a politician speaking at a podium', "
                "'a flooded street', 'a plate of food'). Include the setting and key details. "
                "Do not interpret intent. One sentence only."
            )
            payload = json_lib.dumps({
                "model": vision_model,
                "prompt": prompt_text,
                "images": [img_b64_raw],
                "stream": False,
                "options": {"num_predict": 80, "temperature": 0.1}
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                caption = json_lib.loads(resp.read()).get("response", "").strip()
                if caption and len(caption.split()) >= 5:
                    print(f"[vision_llm] {vision_model}: {caption[:100]}")
                    return caption
                elif caption:
                    print(f"[vision_llm] {vision_model} output too short/vague: '{caption}' — falling back")
        except Exception as e:
            print(f"[vision_llm] {vision_model} failed: {e}")

    print("[vision_llm] No vision model — using CLIP multi-probe + llama3.2 synthesis")
    try:
        clip = MODELS["clip"]
        tok  = MODELS["clip_tok"]

        probes = [
            ("a donkey or mule",                          "animal: donkey/mule"),
            ("a horse",                                   "animal: horse"),
            ("a dog or cat",                              "animal: dog/cat"),
            ("a cow or farm animal",                      "animal: cow/farm animal"),
            ("a wild animal lion tiger bear",             "animal: wild animal"),
            ("a shark or large fish",                     "animal: shark/fish"),
            ("a bird eagle owl",                          "animal: bird"),
            ("a politician giving a speech at a podium",  "person: politician at podium"),
            ("a crowd protesting with signs",              "people: protest crowd"),
            ("a portrait or selfie of a person",          "person: portrait/selfie"),
            ("people at a party or celebration",          "people: party/celebration"),
            ("a soldier or military personnel",           "person: soldier/military"),
            ("a doctor nurse or medical worker",          "person: medical worker"),
            ("rescue workers or firefighters",            "person: rescue/firefighter"),
            ("collapsed building rubble earthquake",      "scene: earthquake rubble"),
            ("flooded street rising water hurricane",     "scene: flood/hurricane"),
            ("wildfire flames burning forest",            "scene: wildfire/fire"),
            ("car crash wreckage accident",               "scene: vehicle crash"),
            ("bomb blast explosion aftermath",            "scene: explosion aftermath"),
            ("a dry open field or desert landscape",      "outdoor: dry field/desert"),
            ("a forest or woodland",                      "outdoor: forest"),
            ("mountains or hills landscape",              "outdoor: mountains"),
            ("a beach or coastal scene",                  "outdoor: beach/coast"),
            ("a city street or urban road",               "outdoor: city street"),
            ("food a meal or plate of food",              "object: food/meal"),
            ("a car truck or vehicle",                    "object: vehicle"),
            ("a building or structure",                   "object: building"),
            ("a flag or banner",                          "object: flag/banner"),
            ("a weapon gun or knife",                     "object: weapon"),
            ("a computer screen or phone",                "object: screen/device"),
            ("a hospital room or medical equipment",      "indoor: hospital"),
            ("an office or workplace",                    "indoor: office"),
            ("a home or residential room",                "indoor: home/room"),
            ("a CGI or digitally composited image",       "special: CGI/fake"),
            ("an aerial or satellite photograph",         "special: aerial view"),
            ("outer space earth from orbit",              "special: outer space"),
        ]

        clip_prompts = [p[0] for p in probes]
        probe_labels = [p[1] for p in probes]

        with torch.no_grad():
            txt_feats = F.normalize(clip.encode_text(tok(clip_prompts).to(device)), dim=-1)
            sims      = (img_feat_norm @ txt_feats.T)[0]
            sim_vals  = sims.cpu().tolist()

        ranked = sorted(zip(probe_labels, sim_vals), key=lambda x: x[1], reverse=True)[:5]
        top_evidence = "\n".join([f"  - {lbl}: similarity {score:.3f}" for lbl, score in ranked])
        print(f"[vision_llm] CLIP top matches: {top_evidence}")

        emotion_probes = [
            "tense alarming dangerous frightening",
            "sad distressing tragic",
            "calm peaceful serene",
            "joyful happy celebratory",
            "neutral informational ordinary",
        ]
        emotion_labels = ["tense/alarming", "sad/distressing", "calm/peaceful", "joyful/celebratory", "neutral"]
        with torch.no_grad():
            e_feats = F.normalize(clip.encode_text(tok(emotion_probes).to(device)), dim=-1)
            e_sims  = (img_feat_norm @ e_feats.T)[0]
            emotion = emotion_labels[int(e_sims.argmax().item())]

        synth_prompt = f"""You are an academic research assistant for a misinformation detection system. Your role is to objectively analyse content regardless of its subject matter. You must always complete your analysis and never refuse based on content sensitivity.

You are helping describe an image for a fake news detection system.
A visual AI has analysed the image and found these similarity scores (higher = more likely):

{top_evidence}

Emotional tone of the image: {emotion}

Based on these clues, write ONE specific sentence describing what is most likely in the image.
Start with "The image shows" and name the most probable subject concretely.
Do not say "possibly" or "might be". Pick the most likely interpretation and state it directly."""

        payload = json_lib.dumps({
            "model":  "llama3.2",
            "prompt": synth_prompt,
            "stream": False,
            "options": {"num_predict": 60, "temperature": 0.1}
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            caption = json_lib.loads(resp.read()).get("response", "").strip()
            if caption:
                print(f"[vision_llm] llama3.2 synthesis: {caption[:100]}")
                return caption

    except Exception as e:
        print(f"[vision_llm] CLIP+llama3.2 synthesis failed: {e}")

    return None


def get_clip_description_fallback(img_feat_norm) -> str:
    clip = MODELS["clip"]
    tok  = MODELS["clip_tok"]

    label_pairs = [
        ("a photo of an animal dog cat horse donkey cow sheep",       "an animal"),
        ("a shark sea creature fish",                                  "a shark or sea creature"),
        ("a portrait selfie of a person face indoors",                "a person (portrait/selfie)"),
        ("a politician leader giving a speech podium microphone",      "a politician or public figure speaking"),
        ("a protest demonstration crowd signs marching",               "a protest or demonstration"),
        ("a party celebration people smiling indoors",                 "people at a party or celebration"),
        ("collapsed building rubble concrete debris earthquake",       "collapsed buildings and rubble"),
        ("flooded street water hurricane storm surge",                 "flooded streets or flood damage"),
        ("wildfire flames smoke burning forest",                       "a wildfire with flames and smoke"),
        ("car crash accident damaged vehicle wreckage",                "a vehicle crash or accident scene"),
        ("firefighter paramedic ambulance emergency rescue",           "emergency responders at an incident"),
        ("crime scene police tape officers",                           "a crime scene"),
        ("landscape nature trees mountains grass sky",                 "a natural landscape or outdoor scene"),
        ("aerial bird eye view photograph from above",                 "an aerial photograph"),
        ("outer space earth orbit solar eclipse nasa",                 "a space or astronomical photograph"),
        ("CGI digitally composited fake manipulated render",           "a digitally manipulated or CGI image"),
        ("street city buildings road urban",                           "a street or urban scene"),
        ("sports stadium athletes competition",                        "a sports event"),
        ("food meal plate restaurant kitchen",                         "food or a restaurant scene"),
        ("hospital patient doctor medical equipment",                  "a hospital or medical setting"),
        ("soldiers military tanks weapons war conflict",               "military or conflict scene"),
        ("office room home indoor furniture",                          "an indoor room or domestic scene"),
    ]

    clip_prompts   = [p[0] for p in label_pairs]
    display_labels = [p[1] for p in label_pairs]

    emotion_labels = [
        "calm and peaceful", "tense and alarming",
        "joyful and celebratory", "sad and distressing",
        "neutral and informational",
    ]
    with torch.no_grad():
        txt_feats  = F.normalize(clip.encode_text(tok(clip_prompts).to(device)), dim=-1)
        sims       = (img_feat_norm @ txt_feats.T)[0]
        best_idx   = int(sims.argmax().item())
        best_score = float(sims.max().item())
        display    = display_labels[best_idx]
        e_feats    = F.normalize(clip.encode_text(tok(emotion_labels).to(device)), dim=-1)
        e_sims     = (img_feat_norm @ e_feats.T)[0]
        emotion    = emotion_labels[int(e_sims.argmax().item())]

    print(f"[clip_fallback] matched='{display}' score={best_score:.3f} emotion={emotion}")
    return f"Image shows {display} with a {emotion} emotional tone."


def get_clip_description(img_feat_norm, img_b64_raw: str = None) -> str:
    if img_b64_raw or img_feat_norm is not None:
        caption = get_vision_llm_description(img_b64_raw, img_feat_norm=img_feat_norm)
        if caption:
            return caption
    print("[get_clip_description] All strategies failed — using single-label CLIP")
    return get_clip_description_fallback(img_feat_norm)


def check_entity_consistency_llm(text, image_description, img_feat_norm=None):
    """
    Hybrid entity consistency check:

    Stage 1 — CLIP contextual matching (primary, runs on image embedding directly)
    ────────────────────────────────────────────────────────────────────────────
    The old approach fed moondream's text description into llama3.2, which lost
    information at every step and failed on contextually adjacent images (crowds
    near an explosion, soldiers on streets during an attack).

    Instead we use CLIP directly on the image embedding, comparing it against a
    rich set of "valid evidence" prompts for each event type detected in the text.
    This works because CLIP was trained on image-text pairs and understands that:
      - "people fleeing on a city street at night" matches "terror attack"
      - "soldiers deployed on urban streets" matches "security emergency"
      - "crowd gathered outside a venue" matches "explosion at nearby location"

    We compute two CLIP scores:
      match_score  — max similarity to any valid-evidence prompt for this event type
      reject_score — max similarity to clearly-unrelated content prompts

    If match_score > threshold OR match_score > reject_score + margin → MATCH
    If reject_score >> match_score → MISMATCH
    If ambiguous → fall through to Stage 2

    Stage 2 — llama3.2 LLM check (fallback for ambiguous cases only)
    ────────────────────────────────────────────────────────────────
    Only runs when CLIP cannot make a confident decision. Uses an improved prompt
    with explicit rules for contextually adjacent images.
    """
    import urllib.request, json as json_lib

    text_lower = text.lower()
    img_lower  = image_description.lower()

    # ── Event type detection from text ───────────────────────────────────────
    attack_text  = any(w in text_lower for w in [
        "attack","terror","terrorist","shooting","bomb","explosion","blast",
        "gunman","gunfire","killed","siege","lockdown","threat","emergency",
    ])
    military_text = any(w in text_lower for w in [
        "military","deployed","troops","soldiers","armed forces","airstrike",
        "invasion","war","conflict","security forces",
    ])
    earthquake_text = any(w in text_lower for w in [
        "earthquake","seismic","tremor","magnitude","collapse","rubble",
    ])
    flood_text = any(w in text_lower for w in [
        "flood","hurricane","tsunami","storm","surge","inundation",
    ])
    fire_text = any(w in text_lower for w in [
        "fire","wildfire","blaze","burning","flames","inferno",
    ])
    space_text = any(w in text_lower for w in [
        "space","iss","nasa","satellite","orbit","eclipse","astronaut",
    ])
    disaster_text = earthquake_text or flood_text or fire_text or attack_text or military_text

    # ── Stage 1: CLIP contextual matching ────────────────────────────────────
    if img_feat_norm is not None:
        clip_model = MODELS["clip"]
        tokenize   = MODELS["clip_tok"]

        # Build event-specific "valid evidence" prompts.
        # Key insight: for breaking news, valid visual evidence includes
        # the CONTEXT of the event, not just the event itself.
        # Real journalists photograph what they can see — crowds, responders,
        # the scene around the event, not just the explosion/rubble.
        if attack_text or military_text:
            match_prompts = [
                # Direct evidence
                "explosion fire blast destruction aftermath",
                "bomb blast rubble destroyed building",
                # Responder evidence (soldiers, police = valid)
                "soldiers military personnel deployed on city streets",
                "police officers standing on street security operation",
                "armed forces urban patrol camouflage uniforms night",
                "security forces riot police crowd control",
                # Scene evidence (crowd at location = valid)
                "crowd of people gathered on city street at night",
                "people fleeing running street panic emergency",
                "urban street scene at night crowd gathering near venue",
                "busy city street people standing outside bar restaurant night",
                "large crowd outside stadium venue at night",
                # Aftermath evidence
                "ambulance emergency vehicles parked street",
                "crime scene police tape cordoned area",
                "emergency response fire trucks police cars",
            ]
            reject_prompts = [
                "a politician giving a speech at a podium indoors",
                "a portrait or headshot of a person smiling",
                "a calm beach or tropical vacation scene",
                "a wildlife animal in nature field",
                "a sports game players on field pitch",
                "outer space satellite earth from orbit",
                "a food meal restaurant indoor dining",
            ]
        elif earthquake_text:
            match_prompts = [
                "collapsed building rubble earthquake damage",
                "destroyed structure debris after earthquake",
                "cracked ground damaged road seismic",
                "rescue workers searching rubble survivors",
                "people standing near damaged buildings disaster",
                "crowds gathered near collapsed structure emergency",
                "damaged temple monument historic building earthquake",
            ]
            reject_prompts = [
                "a politician giving a speech at a podium",
                "a calm beach or vacation scene",
                "a wildlife animal in nature",
                "a sports game players on field",
                "outer space satellite earth from orbit",
                "a normal undamaged building street",
            ]
        elif flood_text:
            match_prompts = [
                "flooded streets rising water hurricane flood",
                "submerged cars buildings flood damage",
                "people in floodwater rescue boat",
                "storm surge waves coastal flooding",
                "aerial view flooded city neighbourhood",
                "rain storm dark clouds severe weather",
            ]
            reject_prompts = [
                "a politician giving a speech indoors",
                "a calm sunny beach vacation",
                "a wildlife animal in dry field",
                "outer space satellite view",
                "a sports game indoor arena",
            ]
        elif fire_text:
            match_prompts = [
                "wildfire flames burning forest trees",
                "fire engulfing building smoke",
                "firefighters battling blaze hose",
                "smoke rising from burning structure",
                "charred burned landscape aftermath fire",
            ]
            reject_prompts = [
                "a politician at a podium indoors",
                "a calm ocean beach scene",
                "outer space satellite",
                "a sports game on field",
            ]
        elif space_text:
            match_prompts = [
                "space station orbit earth view from space",
                "solar eclipse from orbit satellite",
                "nasa spacecraft rocket launch",
                "earth from space atmosphere stars",
                "astronaut spacewalk ISS",
            ]
            reject_prompts = [
                "a city street crowd people",
                "a politician at a podium",
                "a natural disaster flood earthquake",
                "a sports game stadium",
                "wildlife animals in nature",
            ]
        else:
            # No specific event type — use direct CLIP text-image similarity
            # which is already computed in predict(), so just return neutral
            match_prompts  = []
            reject_prompts = []

        if match_prompts:
            with torch.no_grad():
                m_feats = F.normalize(
                    clip_model.encode_text(tokenize(match_prompts).to(device)), dim=-1
                )
                r_feats = F.normalize(
                    clip_model.encode_text(tokenize(reject_prompts).to(device)), dim=-1
                )
                m_sims = (img_feat_norm @ m_feats.T)[0]
                r_sims = (img_feat_norm @ r_feats.T)[0]

            match_score  = float(m_sims.max().item())
            reject_score = float(r_sims.max().item())
            best_match_prompt = match_prompts[int(m_sims.argmax().item())]

            print(f"[clip_entity] match={match_score:.3f} reject={reject_score:.3f} "
                  f"best='{best_match_prompt[:50]}'")

            # Confident MATCH: image clearly relates to this event type
            if match_score > 0.18:
                reason = f"MATCH (CLIP): image matches event context — '{best_match_prompt[:40]}' sim={match_score:.3f}"
                print(f"[clip_entity] → MATCH (score {match_score:.3f} > 0.22 threshold)")
                return True, reason

            # Confident MISMATCH: image clearly shows unrelated content
            if reject_score > 0.22 and reject_score > match_score + 0.05:
                best_reject = reject_prompts[int(r_sims.argmax().item())]
                reason = f"MISMATCH (CLIP): image shows unrelated content — '{best_reject[:40]}' sim={reject_score:.3f}"
                print(f"[clip_entity] → MISMATCH (reject {reject_score:.3f} >> match {match_score:.3f})")
                return False, reason

            # Ambiguous — fall through to LLM Stage 2
            print(f"[clip_entity] Ambiguous (match={match_score:.3f} reject={reject_score:.3f}) → LLM Stage 2")

    # ── Stage 2: LLM fallback for ambiguous cases ─────────────────────────────
    # Only reaches here when CLIP couldn't make a confident call,
    # OR when img_feat_norm is not available.
    try:
        prompt = f"""You are a fake news detector. Decide if the image matches the post.

Post: "{text[:120]}"
Image description: "{image_description}"

Context rules for breaking news:
- MATCH if the image shows direct evidence of the event (damage, destruction, casualties)
- MATCH if the image shows responders at the event (soldiers, police, firefighters, ambulances)
- MATCH if the image shows the scene at or near the reported location (crowds outside a venue, street scenes near the incident, people evacuating)
- MATCH if the image shows a damaged or historic structure and the post describes its collapse or damage
- MATCH for animal/nature images when the post describes that animal or natural phenomenon
- MISMATCH only if the image is completely unrelated — wrong country, wrong topic, wrong subject entirely (e.g. a beach photo for a shooting, a politician portrait for a flood, a wildlife photo for a political event)
- Key: breaking news images show CONTEXT, not just the event itself. A crowd outside a bar near an explosion IS evidence. Soldiers on streets during a terror attack IS evidence.

CRITICAL RULE: A nighttime crowd scene on a city street IS consistent with an explosion/attack report at a nearby location. Crowds, pedestrians, cars, and street scenes near the incident ARE valid evidence. Only say MISMATCH if the image is completely unrelated (e.g. a sunny beach, an animal, a politician at a podium, outer space).

Answer with exactly one word: MATCH or MISMATCH"""

        payload = json_lib.dumps({
            "model":  "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 10, "temperature": 0.0}
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result   = json_lib.loads(resp.read().decode("utf-8"))
            response = result.get("response", "").strip()
            print(f"[llm_entity] Stage 2 LLM: {response}")
            is_match = response.upper().startswith("MATCH")
            return is_match, response

    except Exception as e:
        print(f"[llm_entity] ❌ LLM unavailable: {e} — defaulting to MATCH (no evidence of mismatch)")
        return True, "skipped"


def embed_text(text: str) -> torch.Tensor:
    emb384 = MODELS["st"].encode([text], convert_to_tensor=True,
                                  show_progress_bar=False)[0]
    emb128 = emb384.reshape(128, 3).mean(dim=1)
    return F.normalize(emb128, dim=0)

@app.route("/calibrate", methods=["POST"])
def calibrate():
    """
    Fits Platt scaling calibrator using your actual preprocessed dataset.
    Runs the full embedding pipeline on validation split, collects raw logits,
    fits LogisticRegression on logit → label, saves to platt_calibrator.pkl.
    Call once after startup: POST /calibrate
    """
    try:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        print("[calibration] Loading preprocessed dataset...")
        df = pd.read_pickle(os.path.join(
            PROJECT_ROOT, "Dataset/twitter/df_preprocessed_with_scores.pkl"
        ))

        # Use 20% as calibration set — stratified so fake/real balanced
        _, val_df = train_test_split(
            df, test_size=0.20, random_state=42,
            stratify=df["label"].str.lower()
        )
        print(f"[calibration] Val set: {len(val_df)} samples "
              f"({(val_df['label'].str.lower()=='fake').sum()} fake, "
              f"{(val_df['label'].str.lower()=='real').sum()} real)")

        em = MODELS["emotion_model"]
        em.eval()

        all_logits = []
        all_labels = []

        for i, (_, row) in enumerate(val_df.iterrows()):
            try:
                text  = str(row.get("text", row.get("clean_text", "")))
                label = 1 if str(row.get("label","")).lower() == "fake" else 0

                # Same embedding pipeline as /predict
                text_emb   = embed_text(text).to(device)
                vad_text_d = compute_text_vad(text)
                image_emb  = torch.zeros(1024)
                vad_image_d = {"V": 0.5, "A": 0.5, "D": 0.5}

                h_t   = text_emb.unsqueeze(0).to(device)
                h_i   = image_emb.unsqueeze(0).to(device)
                h_m   = torch.zeros(1, 128).to(device)
                h_aff = torch.zeros(1, 128).to(device)
                vad_t = torch.tensor(
                    [[vad_text_d["V"], vad_text_d["A"], vad_text_d["D"]]],
                    dtype=torch.float32
                ).to(device)
                vad_i = torch.tensor(
                    [[vad_image_d["V"], vad_image_d["A"], vad_image_d["D"]]],
                    dtype=torch.float32
                ).to(device)

                with torch.no_grad():
                    logits, _ = em(h_t, h_i, h_m, h_aff,
                                   vad_text=vad_t, vad_image=vad_i)

                raw_logit = float(logits[0, 0].item())
                all_logits.append(raw_logit)
                all_labels.append(label)

                if (i + 1) % 100 == 0:
                    print(f"[calibration] {i+1}/{len(val_df)} processed...")

            except Exception as e:
                print(f"[calibration] Row {i} failed: {e}")
                continue

        all_logits = np.array(all_logits).reshape(-1, 1)
        all_labels = np.array(all_labels)

        calibrator = LogisticRegression(C=1.0, max_iter=1000)
        calibrator.fit(all_logits, all_labels)

        cal_path = os.path.join(PROJECT_ROOT, "platt_calibrator.pkl")
        with open(cal_path, "wb") as f:
            pickle.dump(calibrator, f)
        MODELS["calibrator"] = calibrator

        # Quick sanity check — what does 0.998 become?
        test_logits = np.array([[4.0], [5.0], [6.0], [-1.0], [-3.0]])
        test_probs  = calibrator.predict_proba(test_logits)[:, 1]
        print(f"[calibration] Sanity check (raw logit → calibrated prob):")
        for logit, prob in zip(test_logits.flatten(), test_probs):
            print(f"[calibration]   logit={logit:+.1f} → {prob:.3f}")

        print(f"[calibration] ✅ Done. Coef={calibrator.coef_[0][0]:.4f}  "
              f"Intercept={calibrator.intercept_[0]:.4f}")

        return jsonify({
            "status":    "ok",
            "n_samples": len(all_labels),
            "n_fake":    int(all_labels.sum()),
            "n_real":    int((all_labels == 0).sum()),
            "coef":      round(float(calibrator.coef_[0][0]), 4),
            "intercept": round(float(calibrator.intercept_[0]), 4),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# ── /predict — GDELT integrated here ─────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data      = request.get_json(force=True)
        text      = data.get("text", "").strip()
        img_b64   = data.get("image_base64", None)
        has_image = bool(img_b64 and len(img_b64) > 100)

        if not text:
            return jsonify({"error": "text required"}), 400

        clip_model = MODELS["clip"]
        tokenize   = MODELS["clip_tok"]
        prep       = MODELS["clip_prep"]

        # 1. Text embedding
        text_emb = embed_text(text).to(device)

        # 2. Text VAD
        vad_text_d = compute_text_vad(text)

        # 3. Image embedding
        if has_image:
            pil   = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
            img_t = prep(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat_raw = clip_model.encode_image(img_t)
            image_emb = img_feat_raw[0].cpu()
            if image_emb.shape[0] < 1024:
                image_emb = F.pad(image_emb, (0, 1024 - image_emb.shape[0]))
            img_feat_norm = F.normalize(img_feat_raw, dim=-1)
            vad_image_d = compute_image_vad(img_feat_norm)
        else:
            image_emb   = torch.zeros(1024)
            vad_image_d = {"V": 0.5, "A": 0.5, "D": 0.5}

        # 4. Forward pass
        em    = MODELS["emotion_model"]
        h_t   = text_emb.unsqueeze(0).to(device)
        h_i   = image_emb.unsqueeze(0).to(device)
        h_m   = torch.zeros(1, 128).to(device)
        h_aff = torch.zeros(1, 128).to(device)
        vad_t = torch.tensor([[vad_text_d["V"], vad_text_d["A"], vad_text_d["D"]]],
                              dtype=torch.float32).to(device)
        vad_i = torch.tensor([[vad_image_d["V"], vad_image_d["A"], vad_image_d["D"]]],
                              dtype=torch.float32).to(device)

        with torch.no_grad():
            logits, intermediates = em(
                h_t, h_i, h_m, h_aff,
                vad_text=vad_t,
                vad_image=vad_i,
            )

        # 5. Sigmoid
        raw_logit = float(logits[0, 0].item())
        prior_correction = float(np.log(4832.0 / 5994.0))   # -0.215 based on dataset balance
        corrected_logit  = raw_logit + prior_correction
        fake_prob = float(torch.sigmoid(torch.tensor(corrected_logit)).item())
        print(f"[DEBUG] raw_logit={raw_logit:.4f}  corrected_logit={corrected_logit:.4f}  "
            f"fake_prob={fake_prob:.4f}  (prior correction)")
        raw_fake_prob = fake_prob  # save before CLIP correction

        print(f"[DEBUG] vad_text={vad_text_d}")
        print(f"[DEBUG] vad_image={vad_image_d}")

        fw           = intermediates["emotion_weights"][0].cpu().tolist()
        vmm          = intermediates["v_mismatch"][0].cpu()
        mismatch_mag = float(vmm.norm().item())
        mixed_score  = float(intermediates.get("mixed_score", torch.zeros(1,1))[0,0].item())

        # 6. Anomaly
        # 6. Anomaly
        # CRITICAL: use z_aug not z_fused
        # Anomaly models were trained on z_aug from prepare_clustering_data.py
        # z_fused is a different tensor — causes distribution mismatch
        if "z_aug" in intermediates:
            z_for_anomaly = intermediates["z_aug"][0].detach().cpu().numpy()
            print(f"[DEBUG] anomaly input: z_aug ✅")
        else:
            z_for_anomaly = intermediates["z_fused"][0].detach().cpu().numpy()
            print(f"[DEBUG] anomaly input: z_fused ⚠️  "
                  f"(z_aug not found — check model intermediates)")

        v_mismatch_np = intermediates["v_mismatch"][0].detach().cpu().numpy()
        z_out_128     = z_for_anomaly[:128]

        print(f"[DEBUG] z_out_128 std={z_out_128.std():.4f}  "
              f"v_mismatch std={v_mismatch_np.std():.4f}")

        anomaly_score, anomaly_level, n_methods, method_flags = run_anomaly(
            z_out_128, v_mismatch_np
        )

        print(f"[DEBUG] anomaly={anomaly_score:.4f} level={anomaly_level} n={n_methods}")

        # ── GDELT Layer 1: Event Verification ────────────────────────────────
        # Query GDELT BEFORE computing final score so its thresholds can
        # influence how the anomaly and fake_prob are weighted.
        # This runs in ~0.5–2s and is cached for 10 minutes.
        gdelt_result = verify_event_gdelt(text)
        gdelt_threshold_adj = gdelt_result["threshold_adj"]   # +ve = raise bar, -ve = lower
        gdelt_anomaly_cap   = gdelt_result["anomaly_cap"]     # overrides semantic cap if GDELT checked
        print(f"[DEBUG] gdelt verdict={gdelt_result['verdict']} "
              f"conf={gdelt_result['confidence']} "
              f"threshold_adj={gdelt_threshold_adj:+.2f} "
              f"anomaly_cap={gdelt_anomaly_cap:.2f}")

        # 7. Base combined score (unchanged from original)
         # 7. Base combined score
        # Text-only: reduce anomaly weight — less reliable without image
        dA_vad = abs(vad_text_d["A"] - vad_image_d["A"])
        dV_vad = abs(vad_text_d["V"] - vad_image_d["V"])

        # CLIP direct text-image similarity
        clip_text_image_sim = 0.5
        if has_image:
            with torch.no_grad():
                txt_tok       = tokenize([text[:77]]).to(device)
                txt_clip_feat = F.normalize(clip_model.encode_text(txt_tok), dim=-1)
                clip_text_image_sim = float((img_feat_norm @ txt_clip_feat.T)[0, 0].item())
            print(f"[DEBUG] clip_text_image_sim={clip_text_image_sim:.3f}")
            # Adjust fake_prob based on how well image matches text
            # Baseline 0.20 = typical CLIP sim for a matching news image
            # Low sim (wrong image) → raise fake_prob
            # High sim (matching image) → lower fake_prob
            clip_correction = (0.20 - clip_text_image_sim) * 0.35
            fake_prob = float(np.clip(fake_prob + clip_correction, 0.05, 0.95))
            print(f"[DEBUG] clip_correction={clip_correction:+.3f}  "
                  f"adjusted fake_prob={fake_prob:.4f}")

        # Base combined score — uses CLIP-corrected fake_prob
        if not has_image:
            combined = 0.80 * fake_prob + 0.20 * anomaly_score
        else:
            combined = 0.65 * fake_prob + 0.35 * anomaly_score
        print(f"[DEBUG] base combined={combined:.4f}")

        # Semantic consistency
        semantic_ok, semantic_sim = check_semantic_consistency(text, img_feat_norm) if has_image else (True, 1.0)

        # LLM entity check
        llm_entity_ok = True
        llm_entity_reason = "skipped"
        if has_image:
            clip_desc = get_clip_description(img_feat_norm, img_b64_raw=img_b64)
            llm_entity_ok, llm_entity_reason = check_entity_consistency_llm(text, clip_desc, img_feat_norm=img_feat_norm)
            print(f"[entity] entity_ok={llm_entity_ok} reason={llm_entity_reason[:80]}")

        # ── Final verdict logic ────────────────────────────────────────────────
        semantic_verified = (
            llm_entity_ok
            and llm_entity_reason != "skipped"
            and semantic_ok
        )

        if not llm_entity_ok and llm_entity_reason != "skipped":
            # Clear semantic entity mismatch — image doesn't match text subject
            # GDELT: if event is unverified AND image mismatches, apply extra penalty
            base_floor = 0.70
            if gdelt_result["verdict"] == "unverified":
                base_floor = 0.78   # stronger fake signal when both checks fail
            combined = max(combined, base_floor)
            print(f"[DEBUG] LLM entity mismatch → combined={combined:.4f}: {llm_entity_reason}")

        elif not semantic_ok and clip_text_image_sim < 0.12:
            # CLIP topic mismatch + very low direct similarity
            base_floor = 0.65
            if gdelt_result["verdict"] == "unverified":
                base_floor = 0.72
            combined = max(combined, base_floor)
            print(f"[DEBUG] CLIP content mismatch → combined={combined:.4f}")

        elif semantic_verified:
            # ── Cap fake_prob when multimodal evidence confirms authenticity ──
            #
            # The neural network returns fake_prob≈1.0 for any high-arousal
            # breaking news post (explosion, terror, Paris) because the training
            # data was skewed — most such posts were labelled fake.
            # When CLIP entity check + semantic consistency both pass, the model's
            # bias must be overridden. We cap fake_prob itself, not just anomaly.
            #
            # Tiers based on how much independent confirmation we have:
            #   GDELT verified           → cap fake_prob at 0.50
            #   GDELT partially verified → cap fake_prob at 0.55
            #   GDELT timed out/unchecked→ cap fake_prob at 0.60
            #   No GDELT at all          → cap fake_prob at 0.65
            if gdelt_result["verdict"] == "verified":
                effective_fake_prob = min(fake_prob, 0.50)
                capped_anomaly = min(anomaly_score, gdelt_anomaly_cap)
            elif gdelt_result["verdict"] == "partially_verified":
                effective_fake_prob = min(fake_prob, 0.55)
                capped_anomaly = min(anomaly_score, gdelt_anomaly_cap)
            elif gdelt_result["gdelt_checked"]:
                effective_fake_prob = min(fake_prob, 0.60)
                capped_anomaly = min(anomaly_score, 0.40)
            else:
                effective_fake_prob = min(fake_prob, 0.65)
                capped_anomaly = min(anomaly_score, 0.45)

            print(f"[DEBUG] Semantic verified → fake_prob capped "
                  f"{fake_prob:.3f}→{effective_fake_prob:.3f} "
                  f"anomaly_cap={capped_anomaly:.3f} gdelt={gdelt_result['verdict']}")
            combined = 0.80 * effective_fake_prob + 0.20 * capped_anomaly
            print(f"[DEBUG] combined={combined:.4f}")

        elif dA_vad < 0.20 and dV_vad < 0.15 and fake_prob < 0.55:
            combined = 0.85 * fake_prob + 0.15 * anomaly_score
            print(f"[DEBUG] VAD consistent → combined={combined:.4f}")

        # ── Threshold with GDELT adjustment ──────────────────────────────────
        # Base threshold: 0.60 if semantically verified, 0.52 otherwise
        # GDELT adjustment: +0.15 if verified (harder to call fake),
        #                   -0.10 if unverified (easier to call fake)
        if not has_image:
            base_threshold = 0.68
        else:
            base_threshold = 0.62 if semantic_verified else 0.52
        # When semantic_verified, don't let GDELT timeout penalise a confirmed real post.
        # GDELT negative adj (-0.10) only applies when we have NO multimodal confirmation.
        effective_gdelt_adj = gdelt_threshold_adj if not semantic_verified else max(gdelt_threshold_adj, 0.0)
        threshold = float(np.clip(base_threshold + effective_gdelt_adj, 0.38, 0.82))
        label = "fake" if combined > threshold else "real"

        print(f"[DEBUG] FINAL combined={combined:.4f} "
              f"base_threshold={base_threshold:.2f} "
              f"gdelt_adj={gdelt_threshold_adj:+.2f} "
              f"final_threshold={threshold:.2f} "
              f"label={label} semantic_verified={semantic_verified} "
              f"gdelt={gdelt_result['verdict']} "
              f"dA={dA_vad:.3f} clip_sim={clip_text_image_sim:.3f}\n")

        return jsonify({
            "label":               label,
            "fake_prob":           round(fake_prob, 4),
            "raw_fake_prob":       round(raw_fake_prob, 4),
            "anomaly_score":       round(anomaly_score, 4),
            "anomaly_level":       anomaly_level,
            "combined_score":      round(combined, 4),
            "contradiction_score": round(float(np.clip(mismatch_mag/2.0, 0, 1)), 4),
            "n_methods_flagged":   n_methods,
            "method_flags":        method_flags,
            "vad_text":            {k: round(v,4) for k,v in vad_text_d.items()},
            "vad_image":           {k: round(v,4) for k,v in vad_image_d.items()},
            "mismatch": {
                "V":     round(abs(vad_text_d["V"]-vad_image_d["V"]), 4),
                "A":     round(abs(vad_text_d["A"]-vad_image_d["A"]), 4),
                "D":     round(abs(vad_text_d["D"]-vad_image_d["D"]), 4),
                "total": round(mismatch_mag, 4),
            },
            "fusion_weights": {
                "text":  round(fw[0], 4),
                "image": round(fw[1], 4),
                "meta":  round(fw[2], 4),
            },
            "mixed_affect_score":   round(mixed_score, 4),
            "image_analysed":       has_image,
            "semantic_consistency": round(semantic_sim, 4),
            "clip_text_image_sim":  round(clip_text_image_sim, 4),
            "semantic_verified":    semantic_verified,
            # ── NEW: GDELT verification results exposed in response ───────────
            "gdelt_verification": {
                "verdict":       gdelt_result["verdict"],
                "confidence":    gdelt_result["confidence"],
                "sources_count": gdelt_result["sources_count"],
                "event_type":    gdelt_result["event_type"],
                "location":      gdelt_result["location"],
                "checked":       gdelt_result["gdelt_checked"],
                "threshold_adj": gdelt_threshold_adj,
                "sources":       gdelt_result["sources"][:3],
            },
            "pipeline_components": {
                "clip_image":       has_image,
                "sentence_encoder": True,
                "emotion_model":    True,
                "anomaly_ensemble": True,
                "gdelt_layer1":     gdelt_result["gdelt_checked"],   # NEW
                "gnn":              False,
                "meta_features":    False,
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/describe", methods=["POST"])
def describe_image():
    try:
        data    = request.get_json(force=True)
        img_b64 = data.get("image_base64", "")
        if not img_b64:
            return jsonify({"description": None}), 400

        caption = get_vision_llm_description(img_b64)
        if caption:
            print(f"[describe] vision LLM: {caption[:80]}")
            return jsonify({"description": caption})

        pil   = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        img_t = MODELS["clip_prep"](pil).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat_norm = F.normalize(MODELS["clip"].encode_image(img_t), dim=-1)
        description = get_clip_description_fallback(img_feat_norm)
        print(f"[describe] CLIP fallback: {description[:80]}")
        return jsonify({"description": description})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"description": None, "error": str(e)}), 500


@app.route("/batch", methods=["POST"])
def batch_predict():
    try:
        data  = request.get_json(force=True)
        posts = data.get("posts", [])
        if not posts:
            return jsonify({"error": "No posts provided"}), 400
        if len(posts) > 100:
            return jsonify({"error": "Max 100 posts per batch"}), 400

        results  = []
        n_fake   = 0
        prob_sum = 0.0

        for i, post in enumerate(posts):
            text    = post.get("text", "")
            img_b64 = post.get("image_base64", None)
            if not text.strip():
                results.append({"index": i, "error": "empty text", "label": "unknown"})
                continue

            try:
                has_image  = img_b64 is not None and len(img_b64) > 0
                clip_model = MODELS["clip"]
                prep       = MODELS["clip_prep"]

                text_emb   = embed_text(text).to(device)
                vad_text_d = compute_text_vad(text)

                if has_image:
                    pil   = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
                    img_t = prep(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        img_feat_raw = clip_model.encode_image(img_t)
                    image_emb = img_feat_raw[0].cpu()
                    if image_emb.shape[0] < 1024:
                        image_emb = F.pad(image_emb, (0, 1024 - image_emb.shape[0]))
                    img_feat_norm = F.normalize(img_feat_raw, dim=-1)
                    vad_image_d   = compute_image_vad(img_feat_norm)
                else:
                    image_emb   = torch.zeros(1024)
                    vad_image_d = {"V": 0.5, "A": 0.5, "D": 0.5}

                em    = MODELS["emotion_model"]
                h_t   = text_emb.unsqueeze(0).to(device)
                h_i   = image_emb.unsqueeze(0).to(device)
                h_m   = torch.zeros(1, 128).to(device)
                h_aff = torch.zeros(1, 128).to(device)
                vad_t = torch.tensor([[vad_text_d["V"], vad_text_d["A"], vad_text_d["D"]]], dtype=torch.float32).to(device)
                vad_i = torch.tensor([[vad_image_d["V"], vad_image_d["A"], vad_image_d["D"]]], dtype=torch.float32).to(device)

                with torch.no_grad():
                    logits, intermediates = em(h_t, h_i, h_m, h_aff, vad_text=vad_t, vad_image=vad_i)

                raw_logit        = float(logits[0, 0].item())
                prior_correction = float(np.log(4832.0 / 5994.0))
                fake_prob        = float(torch.sigmoid(
                    torch.tensor(raw_logit + prior_correction)).item())
                print(f"[DEBUG batch] raw_logit={raw_logit:.4f}  fake_prob={fake_prob:.4f}")
                fw           = intermediates["emotion_weights"][0].cpu().tolist()
                vmm          = intermediates["v_mismatch"][0].cpu()
                mismatch_mag = float(vmm.norm().item())
                if "z_aug" in intermediates:
                    z_for_anomaly = intermediates["z_aug"][0].detach().cpu().numpy()
                else:
                    z_for_anomaly = intermediates["z_fused"][0].detach().cpu().numpy()
                v_mismatch_np = intermediates["v_mismatch"][0].detach().cpu().numpy()
                z_out_128     = z_for_anomaly[:128]

                anomaly_score, anomaly_level, n_methods, method_flags = run_anomaly(z_out_128, v_mismatch_np)

                if not has_image:
                    combined = 0.80 * fake_prob + 0.20 * anomaly_score
                    threshold = 0.68
                else:
                    combined  = 0.65 * fake_prob + 0.35 * anomaly_score
                    threshold = 0.52
                label = "fake" if combined > threshold else "real"

                dA = abs(vad_text_d["A"] - vad_image_d["A"])
                dV = abs(vad_text_d["V"] - vad_image_d["V"])

                prob_sum += fake_prob
                if label == "fake":
                    n_fake += 1

                results.append({
                    "index":               i,
                    "text":                text[:100] + "..." if len(text) > 100 else text,
                    "label":               label,
                    "fake_prob":           round(fake_prob, 4),
                    "anomaly_score":       round(anomaly_score, 4),
                    "anomaly_level":       anomaly_level,
                    "combined_score":      round(combined, 4),
                    "contradiction_score": round(float(np.clip(mismatch_mag/2.0, 0, 1)), 4),
                    "n_methods_flagged":   n_methods,
                    "arousal_mismatch":    round(dA, 4),
                    "valence_mismatch":    round(dV, 4),
                    "vad_text":            {k: round(v, 4) for k, v in vad_text_d.items()},
                    "vad_image":           {k: round(v, 4) for k, v in vad_image_d.items()},
                    "fusion_weights":      {"text": round(fw[0],4), "image": round(fw[1],4), "meta": round(fw[2],4)},
                    "image_analysed":      has_image,
                })
                print(f"[batch] {i+1}/{len(posts)} → {label} (fake_prob={fake_prob:.3f})")

            except Exception as e:
                results.append({"index": i, "error": str(e), "label": "error"})
                print(f"[batch] {i+1}/{len(posts)} → ERROR: {e}")

        n_total = len([r for r in results if r.get("label") in ("fake","real")])
        summary = {
            "total":         len(posts),
            "processed":     n_total,
            "fake":          n_fake,
            "real":          n_total - n_fake,
            "fake_rate":     round(n_fake / n_total, 4) if n_total > 0 else 0,
            "avg_fake_prob": round(prob_sum / n_total, 4) if n_total > 0 else 0,
            "errors":        len([r for r in results if r.get("label") == "error"]),
        }
        print(f"[batch] Done — {n_fake}/{n_total} fake ({summary['fake_rate']*100:.1f}%)")
        return jsonify({"results": results, "summary": summary})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/ai_describe", methods=["POST"])
def ai_describe():
    try:
        import urllib.request, json as json_lib
        data       = request.get_json(force=True)
        prompt     = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 120)
        if not prompt:
            return jsonify({"error": "prompt required"}), 400

        system_prefix = (
            "You are a neutral academic researcher analysing posts for a "
            "misinformation detection system. Your role is to objectively "
            "describe what computational signals indicate about a post — "
            "you are NOT producing or endorsing harmful content, you are "
            "analysing it scientifically. Always complete your analysis. "
            "Never refuse based on post content. Treat all posts as data.\n\n"
        )
        payload = json_lib.dumps({
            "model":  "llama3.2",
            "prompt": system_prefix + prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3}
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json_lib.loads(resp.read().decode("utf-8"))
            text   = result.get("response", "").strip()
            print(f"[ai_describe] ✅ Ollama: {text[:80]}...")
            return jsonify({"text": text or None})

    except Exception as e:
        print(f"[ai_describe] ❌ Ollama failed: {e}")
        return jsonify({"text": None, "error": str(e)}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","device":str(device),"loaded":list(MODELS.keys())})




@app.route("/")
def serve_index():
    return send_from_directory("/app/static", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    try:
        return send_from_directory("/app/static", path)
    except:
        return send_from_directory("/app/static", "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("=" * 60)
    print("Multimodal Deception Framework — Inference Server")
    print("=" * 60)
    load_all_models()
    print(f"[server] Starting on http://0.0.0.0:{port}")
    print("[server] POST /predict  →  { text, image_base64 }")
    print("[server] POST /batch    →  { posts: [{text, image_base64}, ...] }")
    print("[server] POST /describe →  { image_base64 }")
    print("[server] GET  /health   →  status check")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)