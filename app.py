
import base64
import hashlib
import io
import json
import math
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


# ==========================================================
# Pep2Taste Streamlit Rebuild
# Cloud inference is enabled through BITTER_API_URL and UMAMI_API_URL.
# Mock prediction is only used as a fallback when the backend is unavailable.
# ==========================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATABASE_PATH = APP_DIR / "data" / "Database.csv"
VIRTUAL_SCREENING_DIR = DATA_DIR / "VirtualScreening"
USER_SUBMISSION_DIR = DATA_DIR / "UserSubmissions"
CONTACT_FIGURE_DIR = APP_DIR / "figure"
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", re.IGNORECASE)

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_MASS = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "E": 129.1155, "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}
AA_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
PKA = {
    "n_term": 7.50, "c_term": 3.55,
    "C": 8.50, "D": 3.90, "E": 4.10, "H": 6.00, "K": 10.50, "R": 12.50, "Y": 10.10,
}
DATABASE_DISPLAY_COLUMNS = ["Sequence", "Taste", "Mw", "pI", "GRAVY", "Aromaticity", "Stability Index", "Source", "DOI"]
DESCRIPTOR_COLUMNS = ["Mw", "pI", "GRAVY", "Aromaticity", "Stability Index"]
ANALYSIS_COLUMNS = ["Len", *DESCRIPTOR_COLUMNS]
DESCRIPTOR_LABELS = {
    "Len": "Length",
    "Mw": "Mw",
    "pI": "pI",
    "GRAVY": "GRAVY",
    "Aromaticity": "Aromaticity",
    "Stability Index": "Stability Index",
}
SPLIT_LABEL_ORDER = ["train_pos", "train_neg", "test_pos", "test_neg"]

st.set_page_config(
    page_title="Pep2Taste | Peptide Taste Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
    --navy: #0f172a;
    --navy-2: #111827;
    --ink: #172033;
    --muted: #64748b;
    --line: rgba(15,23,42,0.10);
    --bg: #f8fafc;
    --paper: rgba(255,255,255,0.94);
    --bitter: #d97706;
    --bitter-2: #f59e0b;
    --umami: #dc2626;
    --umami-2: #fb7185;
    --blue: #2563eb;
    --green: #16a34a;
    --shadow: 0 18px 55px rgba(15,23,42,0.12);
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
}

.block-container {
    max-width: 1640px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background:
      radial-gradient(circle at 20% 10%, rgba(217,119,6,0.22), transparent 22%),
      radial-gradient(circle at 85% 35%, rgba(220,38,38,0.18), transparent 26%),
      linear-gradient(180deg, #0f172a 0%, #111827 58%, #020617 100%);
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

.sidebar-logo {
    padding: 1rem .3rem .6rem;
}
.sidebar-logo .brand {
    font-weight: 900;
    font-size: 1.55rem;
    letter-spacing: -0.02em;
}
.sidebar-logo .sub {
    display: none;
}

/* Larger sidebar navigation */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] p {
    font-size: 1.02rem !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    padding-top: .22rem !important;
    padding-bottom: .22rem !important;
}

[data-testid="stSidebar"] .stMarkdown {
    font-size: 1rem !important;
}

.hero {
    border-radius: 30px;
    padding: 2.8rem 3.4rem;
    background:
        radial-gradient(circle at 8% 16%, rgba(217,119,6,.24), transparent 28%),
        radial-gradient(circle at 90% 15%, rgba(220,38,38,.20), transparent 30%),
        linear-gradient(135deg, #ffffff 0%, #f8fafc 55%, #fff7ed 100%);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.hero-content {
    position: relative;
    z-index: 1;
    max-width: min(1360px, calc(100vw - 23rem));
}
.hero:after {
    content: "";
    position: absolute;
    right: -90px;
    top: -90px;
    width: 260px;
    height: 260px;
    border-radius: 999px;
    border: 34px solid rgba(15,23,42,.035);
}
.hero h1 {
    margin: 0;
    color: var(--navy);
    font-size: 2.75rem;
    line-height: 1.06;
    letter-spacing: 0;
}
.hero p {
    margin: 1rem 0 0;
    max-width: 1320px;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.6;
}
.hero .actions {
    display:flex;
    flex-wrap:wrap;
    gap:.7rem;
    margin-top: 1.45rem;
}
.badge {
    display:inline-flex;
    padding:.42rem .78rem;
    border-radius:999px;
    background: rgba(255,255,255,.72);
    border:1px solid rgba(15,23,42,.10);
    color:#334155;
    font-weight: 750;
    font-size:.86rem;
}

.section-title {
    display:flex;
    align-items:center;
    gap:.65rem;
    margin: 1.7rem 0 .9rem;
}
.section-title h2 {
    margin:0;
    font-size:1.42rem;
    letter-spacing:-.02em;
    color:#0f172a;
}
.section-title span {
    height:10px;
    width:10px;
    border-radius:999px;
    background: linear-gradient(135deg, var(--bitter), var(--umami));
    box-shadow:0 0 0 6px rgba(217,119,6,.10);
}

.card {
    border:1px solid var(--line);
    border-radius:24px;
    background: var(--paper);
    box-shadow:0 12px 35px rgba(15,23,42,.07);
    padding:1.28rem 1.28rem;
    height:100%;
}
.card h3 {
    margin:0 0 .5rem;
    color:#111827;
    font-size:1.12rem;
}
.card p {
    margin:.3rem 0 0;
    color:#64748b;
    line-height:1.70;
}

.module-link {
    text-decoration: none !important;
    display: block;
    height: 100%;
    color: white !important;
}

.module-card {
    position: relative;
    display: block;
    min-height: 230px;
    padding: 2rem 2rem;
    border-radius: 30px;
    color: white;
    overflow: hidden;
    box-shadow: 0 22px 55px rgba(15,23,42,.18);
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    cursor: pointer;
    box-sizing: border-box;
    isolation: isolate;
}

.module-card:hover {
    transform: translateY(-6px);
    filter: brightness(1.03);
    box-shadow: 0 28px 70px rgba(15,23,42,.24);
}
.module-title {
    position: relative;
    z-index: 1;
    display: block;
    margin:0;
    color:white;
    font-size:1.48rem;
    font-weight: 850;
    line-height: 1.18;
    letter-spacing: 0;
    text-shadow: 0 2px 12px rgba(15,23,42,.25);
}
.module-body {
    position: relative;
    z-index: 1;
    display: block;
    margin:1rem 0 0;
    color:rgba(255,255,255,.98);
    line-height:1.62;
    font-size: 1rem;
    font-weight: 650;
    text-shadow: 0 1px 12px rgba(15,23,42,.28);
}
.module-card:before {
    content:"";
    position:absolute;
    inset:0;
    background:linear-gradient(135deg, rgba(15,23,42,.32), rgba(15,23,42,.08));
    z-index:0;
}
.module-card:after {
    content:"";
    position:absolute;
    right:-44px;
    bottom:-48px;
    width:160px;
    height:160px;
    border-radius:999px;
    background:rgba(255,255,255,.13);
    z-index:0;
}
.module-bitter { background: linear-gradient(135deg, #78350f 0%, #d97706 54%, #f59e0b 100%); }
.module-umami { background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 56%, #fb7185 100%); }
.module-db { background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 58%, #60a5fa 100%); }
.module-help { background: linear-gradient(135deg, #064e3b 0%, #10b981 58%, #6ee7b7 100%); }
.module-download { background: linear-gradient(135deg, #064e3b 0%, #10b981 58%, #6ee7b7 100%); }

.predictor-model-label {
    margin: 1.2rem 0 .35rem;
    color: #111827;
    font-size: 1.12rem;
    font-weight: 900;
}
.example-row {
    display:flex;
    align-items:center;
    gap:.7rem;
    flex-wrap:wrap;
    margin:.1rem 0 .55rem;
    color:#111827;
    font-weight:900;
}
.example-hint {
    color:#0f766e;
    font-weight:800;
}
.note-box {
    border:1px solid rgba(15,23,42,.10);
    background:linear-gradient(180deg,#ffffff 0%, #f8fafc 100%);
    color:#475569;
    padding:1rem 1.1rem;
    border-radius:18px;
    line-height:1.66;
    font-size:.94rem;
}
.note-box strong {
    color:#0f172a;
}

.download-table-header {
    padding:.62rem .25rem .5rem;
    color:#475569;
    font-size:.76rem;
    font-weight:900;
    text-transform:uppercase;
    letter-spacing:.04em;
    border-bottom:1px solid rgba(15,23,42,.12);
}
.download-table-cell,
.download-table-name {
    padding:.5rem .25rem .2rem;
    color:#334155;
    font-size:.92rem;
    font-variant-numeric: tabular-nums;
}
.download-table-name {
    color:#0f172a;
    font-weight:900;
}
.download-table-divider {
    border-top:1px solid rgba(15,23,42,.075);
    margin:.1rem 0 .24rem;
}

.contact-profile-grid {
    display:grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap:1.2rem;
    margin-top:.55rem;
}
.contact-profile-card {
    display:grid;
    grid-template-columns: 190px 1fr;
    gap:1.25rem;
    align-items:stretch;
    border:1px solid rgba(15,23,42,.10);
    background:#ffffff;
    border-radius:8px;
    padding:1rem;
    box-shadow:0 12px 32px rgba(15,23,42,.07);
}
.contact-photo {
    min-height:245px;
    border-radius:8px;
    overflow:hidden;
    background:#eef2f7;
}
.contact-photo img {
    width:100%;
    height:100%;
    object-fit:cover;
    display:block;
}
.contact-profile-body {
    padding:.25rem .25rem .15rem;
}
.contact-role {
    display:inline-flex;
    align-items:center;
    padding:.24rem .55rem;
    border-radius:999px;
    background:#eff6ff;
    color:#1d4ed8;
    font-weight:900;
    font-size:.78rem;
    letter-spacing:.03em;
    text-transform:uppercase;
}
.contact-profile-card.developer .contact-role {
    background:#ecfdf5;
    color:#047857;
}
.contact-profile-body h3 {
    margin:.65rem 0 .18rem;
    color:#0f172a;
    font-size:1.38rem;
    line-height:1.18;
}
.contact-profile-body .contact-subtitle {
    margin:0 0 .8rem;
    color:#64748b;
    font-weight:750;
}
.contact-detail {
    display:grid;
    grid-template-columns: 7.3rem 1fr;
    gap:.65rem;
    padding:.42rem 0;
    border-top:1px solid rgba(15,23,42,.075);
    color:#334155;
    line-height:1.52;
}
.contact-detail span:first-child {
    color:#0f172a;
    font-weight:900;
}
.contact-action-strip {
    display:grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap:1rem;
    margin:.4rem 0 1.2rem;
}
.contact-action {
    border:1px solid rgba(15,23,42,.10);
    border-radius:8px;
    background:linear-gradient(180deg,#ffffff 0%, #f8fafc 100%);
    padding:1rem 1.05rem;
}
.contact-action h3 {
    margin:0 0 .35rem;
    color:#0f172a;
    font-size:1.04rem;
}
.contact-action p {
    margin:0;
    color:#64748b;
    line-height:1.62;
}
@media (max-width: 980px) {
    .contact-profile-grid,
    .contact-action-strip {
        grid-template-columns:1fr;
    }
    .contact-profile-card {
        grid-template-columns:150px 1fr;
    }
}
@media (max-width: 620px) {
    .contact-profile-card {
        grid-template-columns:1fr;
    }
    .contact-photo {
        height:280px;
    }
    .contact-detail {
        grid-template-columns:1fr;
        gap:.15rem;
    }
}

.result {
    padding:1.28rem 1.35rem;
    border-radius:26px;
    color:white;
    box-shadow:0 18px 50px rgba(15,23,42,.15);
}
.result h2 {
    margin:0;
    color:white;
    font-size:1.75rem;
}
.result p {
    color:rgba(255,255,255,.9);
    line-height:1.7;
    margin:.55rem 0 0;
}
.result.bitter { background:linear-gradient(135deg,#78350f,#d97706,#fbbf24); }
.result.umami { background:linear-gradient(135deg,#7f1d1d,#dc2626,#fb7185); }
.result.negative { background:linear-gradient(135deg,#334155,#475569,#64748b); }
.probbar {
    margin-top:1rem;
    height:14px;
    border-radius:999px;
    background:rgba(255,255,255,.28);
    overflow:hidden;
}
.probfill {
    height:14px;
    border-radius:999px;
    background:rgba(255,255,255,.92);
}
.metric-box {
    padding:1.08rem 1.12rem;
    border-radius:22px;
    border:1px solid var(--line);
    background:linear-gradient(180deg,#ffffff 0%, #f8fafc 100%);
    box-shadow:0 10px 28px rgba(15,23,42,.06);
}
.metric-title {
    color:#64748b;
    font-size:.82rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.05em;
}
.metric-value {
    color:#0f172a;
    font-size:1.65rem;
    font-weight:900;
    margin-top:.16rem;
}
.small-muted {
    color:#64748b;
    font-size:.89rem;
    line-height:1.68;
}
.warning-box {
    border:1px solid rgba(217,119,6,.22);
    background:#fff7ed;
    color:#7c2d12;
    padding:1rem 1.1rem;
    border-radius:20px;
    line-height:1.7;
}
.footer {
    text-align:center;
    color:#94a3b8;
    padding:2rem 0 .5rem;
    font-size:.85rem;
}
.stButton > button {
    border-radius:999px !important;
    border:none !important;
    font-weight:850 !important;
    box-shadow:0 10px 24px rgba(15,23,42,.12);
}
textarea, input {
    border-radius:16px !important;
}
[data-testid="stFileUploader"] section {
    border-radius:18px !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------------
# Helpers
# -------------------------
def get_secret_or_env(key: str, default: str = "") -> str:
    try:
        val = st.secrets.get(key, default)
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key, default)


def clean_sequence(seq: str) -> str:
    seq = (seq or "").strip().upper()
    seq = re.sub(r"\s+", "", seq)
    return seq


def validate_sequence(seq: str, min_len: int = 2, max_len: int = 80) -> Tuple[bool, str]:
    seq = clean_sequence(seq)
    if not seq:
        return False, "Sequence is empty."
    if len(seq) < min_len:
        return False, f"Sequence is too short: current length is {len(seq)}, minimum length is {min_len}."
    if len(seq) > max_len:
        return False, f"Sequence is too long: current length is {len(seq)}, maximum recommended length is {max_len}."
    if not AA_PATTERN.match(seq):
        return False, "Sequence contains non-standard amino acid characters. Supported letters: A C D E F G H I K L M N P Q R S T V W Y."
    return True, "OK"


def parse_fasta_or_lines(text: str) -> List[str]:
    text = text or ""
    if not text.strip():
        return []
    seqs, current = [], []
    fasta_like = any(line.strip().startswith(">") for line in text.splitlines())
    if fasta_like:
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    seqs.append(clean_sequence("".join(current)))
                    current = []
            else:
                current.append(line)
        if current:
            seqs.append(clean_sequence("".join(current)))
    else:
        seqs = [clean_sequence(x) for x in re.split(r"[\n,;\t ]+", text) if clean_sequence(x)]
    return seqs


def read_uploaded_sequences(uploaded) -> List[str]:
    if uploaded is None:
        return []
    name = uploaded.name.lower()
    content = uploaded.getvalue()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
        lower = {c.lower(): c for c in df.columns}
        seq_col = None
        for key in ["sequence", "seq", "peptide", "peptides", "肽序列", "序列"]:
            if key in lower:
                seq_col = lower[key]
                break
        if seq_col is None:
            seq_col = df.columns[0]
        return [clean_sequence(x) for x in df[seq_col].astype(str).tolist()]
    text = content.decode("utf-8", errors="ignore")
    return parse_fasta_or_lines(text)


def deterministic_mock_score(seq: str, task: str) -> float:
    """Only for UI testing. Do not report as model result."""
    seq = clean_sequence(seq)
    if not seq:
        return 0.5
    n = len(seq)
    hydrophobic = sum(seq.count(a) for a in "AILMFWYVP") / n
    charged = sum(seq.count(a) for a in "DEKRH") / n
    acidic = sum(seq.count(a) for a in "DE") / n
    glypro = sum(seq.count(a) for a in "GP") / n
    aromatic = sum(seq.count(a) for a in "FWY") / n
    polar = sum(seq.count(a) for a in "STNQ") / n

    seed = int(hashlib.md5((task + seq).encode()).hexdigest()[:8], 16)
    jitter = ((seed % 1000) / 1000 - 0.5) * 0.06

    if task == "bitter":
        score = 0.20 + 0.55 * hydrophobic + 0.15 * aromatic + 0.07 * glypro - 0.05 * acidic + jitter
    else:
        score = 0.18 + 0.36 * acidic + 0.17 * polar + 0.14 * hydrophobic + 0.08 * charged + jitter

    return round(max(0.01, min(0.99, float(score))), 4)


def predict_with_api_or_mock(sequences: List[str], task: str, threshold: float) -> Tuple[List[Dict[str, Any]], str]:
    api_key = "BITTER_API_URL" if task == "bitter" else "UMAMI_API_URL"
    url = get_secret_or_env(api_key, "")
    if url:
        try:
            resp = requests.post(
                url,
                json={"sequences": sequences, "task": task, "threshold": threshold},
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "results" in data:
                return data["results"], "api"
            if isinstance(data, list):
                return data, "api"
            st.error("The backend API returned an unexpected format. Mock prediction is used instead.")
        except Exception as exc:
            st.error(f"Backend API request failed: {exc}. Mock prediction is used instead.")

    positive = "Bitter" if task == "bitter" else "Umami"
    negative = "Non-bitter" if task == "bitter" else "Non-umami"
    results = []
    for seq in sequences:
        prob = deterministic_mock_score(seq, task)
        label = positive if prob >= threshold else negative
        results.append({
            "sequence": seq,
            "length": len(seq),
            "probability": prob,
            "threshold": threshold,
            "label": label,
        })
    return results, "mock"


def render_hero(title: str, subtitle: str, badges: List[str]) -> None:
    badges_html = "".join(f'<span class="badge">{b}</span>' for b in badges)
    actions_html = f'<div class="actions">{badges_html}</div>' if badges_html else ""
    st.markdown(
        f'<div class="hero"><div class="hero-content"><h1>{title}</h1><p>{subtitle}</p>{actions_html}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-title"><span></span><h2>{title}</h2></div>', unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="card"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)


def module_card(css_class: str, title: str, body: str, page: str = "") -> None:
    href = f"?page={page.replace(' ', '%20')}" if page else "#"
    st.markdown(
        f'<a class="module-link module-card {css_class}" href="{href}" target="_self">'
        f'<span class="module-title">{title}</span>'
        f'<span class="module-body">{body}</span>'
        f'</a>',
        unsafe_allow_html=True
    )


def set_session_value(key: str, value: Any) -> None:
    st.session_state[key] = value


def reset_prediction_inputs(text_key: str, threshold_key: str) -> None:
    st.session_state[text_key] = ""
    st.session_state[threshold_key] = 0.50


def metric_box(title: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-box"><div class="metric-title">{title}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def image_data_uri(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_result(res: Dict[str, Any], task: str, threshold: float) -> None:
    prob = float(res.get("probability", 0.0))
    label = str(res.get("label", "Unknown"))
    length = int(res.get("length", len(res.get("sequence", ""))))

    is_positive = prob >= threshold
    cls = "bitter" if task == "bitter" and is_positive else "umami" if task == "umami" and is_positive else "negative"
    prob_name = "Bitter probability" if task == "bitter" else "Umami probability"

    st.markdown(
        f"""
        <div class="result {cls}">
            <h2>{label}</h2>
            <p>Probability is <b>{prob:.3f}</b>, and the active classification threshold is <b>{threshold:.2f}</b>.</p>
            <div class="probbar"><div class="probfill" style="width:{prob*100:.1f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        metric_box("Sequence length", str(length))
    with c2:
        metric_box(prob_name, f"{prob:.3f}")


def embed_html_if_exists(paths: List[Path], height: int = 560) -> bool:
    for p in paths:
        if p.exists() and p.is_file():
            html = p.read_text(encoding="utf-8", errors="ignore")
            components.html(html, height=height, scrolling=True)
            return True
    return False


def split_taste_labels(value: Any) -> List[str]:
    labels = [x.strip() for x in str(value or "").split(";") if x.strip()]
    return labels or ["Unspecified"]


def doi_to_url(value: Any) -> str:
    doi = str(value or "").strip()
    if not doi or doi.lower() == "nan" or doi == "Not reported":
        return ""
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE).strip()
    if doi.startswith("http://") or doi.startswith("https://"):
        return doi
    return f"https://doi.org/{doi}"


def peptide_net_charge(seq: str, ph: float) -> float:
    seq = clean_sequence(seq)
    positive = 1 / (1 + 10 ** (ph - PKA["n_term"]))
    negative = 1 / (1 + 10 ** (PKA["c_term"] - ph))
    for aa in seq:
        if aa in ["H", "K", "R"]:
            positive += 1 / (1 + 10 ** (ph - PKA[aa]))
        elif aa in ["C", "D", "E", "Y"]:
            negative += 1 / (1 + 10 ** (PKA[aa] - ph))
    return positive - negative


def estimate_isoelectric_point(seq: str) -> float:
    low, high = 0.0, 14.0
    for _ in range(50):
        mid = (low + high) / 2
        if peptide_net_charge(seq, mid) > 0:
            low = mid
        else:
            high = mid
    return round((low + high) / 2, 2)


def hydrophobic_moment(seq: str) -> float:
    seq = clean_sequence(seq)
    if not seq:
        return 0.0
    angle = math.radians(100)
    x = sum(AA_HYDROPATHY.get(aa, 0.0) * math.cos(i * angle) for i, aa in enumerate(seq))
    y = sum(AA_HYDROPATHY.get(aa, 0.0) * math.sin(i * angle) for i, aa in enumerate(seq))
    return round(math.sqrt(x * x + y * y) / len(seq), 3)


def peptide_properties(seq: str) -> Dict[str, float]:
    seq = clean_sequence(seq)
    length = len(seq)
    if not length:
        return {
            "Length": 0,
            "Molecular weight (Da)": 0.0,
            "Hydrophobicity": 0.0,
            "Amphipathicity": 0.0,
            "Isoelectric point": 0.0,
        }
    mw = 18.01528 + sum(AA_MASS.get(aa, 0.0) for aa in seq)
    hydrophobicity = sum(AA_HYDROPATHY.get(aa, 0.0) for aa in seq) / length
    return {
        "Length": length,
        "Molecular weight (Da)": round(mw, 2),
        "Hydrophobicity": round(hydrophobicity, 3),
        "Amphipathicity": hydrophobic_moment(seq),
        "Isoelectric point": estimate_isoelectric_point(seq),
    }


def amino_acid_composition(sequences: List[str]) -> pd.DataFrame:
    counts = {aa: 0 for aa in AA_LIST}
    total = 0
    for seq in sequences:
        seq = clean_sequence(seq)
        for aa in seq:
            if aa in counts:
                counts[aa] += 1
                total += 1
    rows = [{"AA": aa, "Count": counts[aa], "Frequency": counts[aa] / total if total else 0.0} for aa in AA_LIST]
    return pd.DataFrame(rows)


def infer_task_from_name(name: str) -> str:
    text = name.lower()
    if "bitter" in text or "btp" in text:
        return "Bitter"
    if "umami" in text or "ump" in text or "umm" in text:
        return "Umami"
    return "Taste"


def infer_split_from_file(path: Path) -> str:
    name = path.stem.lower()
    if "test" in name:
        return "test"
    return "train"


def dataset_display_name(name: str) -> str:
    if name in {"Bitter(Ours)", "Umami(Ours)"}:
        return name
    if name.endswith(" / UMP789_balance"):
        return "UMP789(1:1)"
    if name.endswith(" / UMP789_imbalance"):
        return "UMP789(1:2)"

    folder_name = name.split(" / ", 1)[0]
    match = re.match(r"^(?:Bitter|Umami)\((.+)\)$", folder_name)
    if match:
        return match.group(1)
    return name


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "dataset"


def is_positive_label(label: Any) -> bool:
    text = str(label).strip().lower()
    try:
        numeric = float(text)
        if numeric == 0:
            return True
        if numeric == 1:
            return False
    except Exception:
        pass

    if text in {"0", "pos", "positive", "bioactive", "active", "true", "yes"}:
        return True
    if text in {"1", "neg", "negative", "non-bioactive", "nonbioactive", "inactive", "false", "no"}:
        return False
    return False


def normalized_label_value(label: Any) -> Any:
    text = str(label).strip()
    try:
        numeric = float(text)
        return int(numeric) if numeric.is_integer() else numeric
    except Exception:
        return text


def split_label(split: str, label: Any) -> str:
    positive = is_positive_label(label)
    return f"{split}_{'pos' if positive else 'neg'}"


@st.cache_data(show_spinner=False)
def load_binary_dataset_records() -> pd.DataFrame:
    rows = []
    if not DATA_DIR.exists():
        return pd.DataFrame(columns=["Dataset", "Task", "File", "Split", "Split label", "Sequence", "Label", "Length"])

    for folder in sorted([p for p in DATA_DIR.iterdir() if p.is_dir()]):
        if folder.name.lower() in {"virtualscreening", "virtual_screening", "pre-libraries", "pre_libraries"}:
            continue
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            continue

        has_named_split = any(("train" in p.stem.lower() or "test" in p.stem.lower()) for p in csv_files)
        for path in csv_files:
            dataset_name = folder.name if has_named_split or len(csv_files) == 1 else f"{folder.name} / {path.stem}"
            split = infer_split_from_file(path) if has_named_split else "train"
            try:
                raw = pd.read_csv(path)
            except Exception:
                continue
            columns = {str(c).lower().strip(): c for c in raw.columns}
            if "sequence" not in columns or "label" not in columns:
                continue
            seq_col = columns["sequence"]
            label_col = columns["label"]
            for _, record in raw[[seq_col, label_col]].dropna().iterrows():
                seq = clean_sequence(record[seq_col])
                if not seq:
                    continue
                label = record[label_col]
                rows.append({
                    "Dataset": dataset_name,
                    "Display dataset": dataset_display_name(dataset_name),
                    "Task": infer_task_from_name(dataset_name),
                    "File": path.relative_to(APP_DIR).as_posix(),
                    "Split": split,
                    "Split label": split_label(split, label),
                    "Sequence": seq,
                    "Label": normalized_label_value(label),
                    "Length": len(seq),
                })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def dataset_summary_table() -> pd.DataFrame:
    records = load_binary_dataset_records()
    if records.empty:
        return pd.DataFrame()
    rows = []
    for dataset, group in records.groupby("Dataset", sort=True):
        counts = group["Split label"].value_counts().to_dict()
        rows.append({
            "Dataset": dataset,
            "Display dataset": dataset_display_name(dataset),
            "Task": group["Task"].iloc[0],
            "Sequences": len(group),
            "Min length": int(group["Length"].min()),
            "Max length": int(group["Length"].max()),
            "train_pos": int(counts.get("train_pos", 0)),
            "train_neg": int(counts.get("train_neg", 0)),
            "test_pos": int(counts.get("test_pos", 0)),
            "test_neg": int(counts.get("test_neg", 0)),
            "Files": ", ".join(sorted(group["File"].unique())),
        })
    return pd.DataFrame(rows)


def split_aa_heatmap_data(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLIT_LABEL_ORDER:
        seqs = records.loc[records["Split label"] == split, "Sequence"].tolist()
        if not seqs:
            continue
        comp = amino_acid_composition(seqs)
        comp["Split label"] = split
        rows.extend(comp.to_dict("records"))
    return pd.DataFrame(rows)


def dataset_split_aa_heatmap_data(records: pd.DataFrame, dataset_order: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    row_order = []
    for dataset in dataset_order:
        group = records[records["Dataset"] == dataset]
        if group.empty:
            continue
        display = dataset_display_name(dataset)
        for split in SPLIT_LABEL_ORDER:
            seqs = group.loc[group["Split label"] == split, "Sequence"].tolist()
            if not seqs:
                continue
            row_label = f"{display} | {split}"
            row_order.append(row_label)
            comp = amino_acid_composition(seqs)
            comp["Dataset"] = display
            comp["Split label"] = split
            comp["Dataset split"] = row_label
            rows.extend(comp.to_dict("records"))
    return pd.DataFrame(rows), row_order


def length_violin_data(records: pd.DataFrame, split_order: List[str]) -> pd.DataFrame:
    rows = []
    present_splits = [split for split in split_order if (records["Split label"] == split).any()]
    if records.empty or not present_splits:
        return pd.DataFrame()

    min_len = float(records["Length"].min())
    max_len = float(records["Length"].max())
    if min_len == max_len:
        grid = [min_len - .5, min_len, min_len + .5]
    else:
        steps = 90
        step = (max_len - min_len) / (steps - 1)
        grid = [min_len + i * step for i in range(steps)]

    for split_index, split in enumerate(present_splits):
        values = records.loc[records["Split label"] == split, "Length"].astype(float).tolist()
        n = len(values)
        if not n:
            continue
        series = pd.Series(values)
        std = float(series.std()) if n > 1 else 0.0
        bandwidth = 1.06 * std * (n ** -0.2) if std > 0 else 0.45
        bandwidth = max(bandwidth, 0.35)
        densities = []
        for y_value in grid:
            density = sum(math.exp(-0.5 * ((y_value - value) / bandwidth) ** 2) for value in values)
            density = density / (n * bandwidth * math.sqrt(2 * math.pi))
            densities.append(density)
        max_density = max(densities) if densities else 0.0
        for y_value, density in zip(grid, densities):
            half_width = (density / max_density * .34) if max_density else 0.0
            rows.append({
                "Split label": split,
                "Split index": split_index,
                "Length": y_value,
                "Density": density,
                "Left": split_index - half_width,
                "Right": split_index + half_width,
                "Count": n,
            })
    return pd.DataFrame(rows)


def make_zip_for_files(paths: List[Path]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if path.exists() and path.is_file():
                zf.write(path, arcname=path.relative_to(APP_DIR).as_posix())
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def load_database() -> pd.DataFrame:
    if not DATABASE_PATH.exists():
        return pd.DataFrame(columns=["Sequence", "Taste", "Len", *DESCRIPTOR_COLUMNS, "Source", "DOI"])

    df = pd.read_csv(DATABASE_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["Sequence", "Taste", "Len", *DESCRIPTOR_COLUMNS, "Source", "DOI"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    df["Sequence"] = df["Sequence"].astype(str).map(clean_sequence)
    df["Taste"] = df["Taste"].fillna("Unspecified").astype(str).str.strip().replace("", "Unspecified")
    df["Len"] = pd.to_numeric(df["Len"], errors="coerce").fillna(df["Sequence"].str.len()).astype(int)
    df["Source"] = df["Source"].fillna("Not reported").astype(str).str.strip().replace("", "Not reported")
    df["DOI"] = df["DOI"].fillna("Not reported").astype(str).str.strip().replace("", "Not reported")

    missing_descriptor = False
    for col in DESCRIPTOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        missing_descriptor = missing_descriptor or df[col].isna().any()
    if missing_descriptor:
        props = pd.DataFrame([peptide_properties(seq) for seq in df["Sequence"]])
        fallback = pd.DataFrame({
            "Mw": props["Molecular weight (Da)"],
            "pI": props["Isoelectric point"],
            "GRAVY": props["Hydrophobicity"],
            "Aromaticity": [sum(clean_sequence(seq).count(aa) for aa in "FWY") / len(clean_sequence(seq)) if clean_sequence(seq) else 0.0 for seq in df["Sequence"]],
            "Stability Index": df["Stability Index"].fillna(0.0),
        })
        for col in DESCRIPTOR_COLUMNS:
            df[col] = df[col].fillna(fallback[col]).round(3)

    df["Taste labels"] = df["Taste"].map(split_taste_labels)
    df["DOI link"] = df["DOI"].map(doi_to_url)
    return df[required + ["Taste labels", "DOI link"]]


@st.cache_data(show_spinner=False)
def load_database_with_properties() -> pd.DataFrame:
    return load_database().copy()


# -------------------------
# Pages
# -------------------------
def home_page() -> None:
    render_hero(
        "Pep2Taste",
        "Pep2Taste is an artificial intelligence platform for peptide taste prediction. The system is designed to support sequence-based identification of taste-active peptides, including bitter and umami peptides, by integrating machine learning models, peptide feature representations, and cloud-based inference services. Users can submit single or batch peptide sequences, obtain prediction probabilities and class labels, and access dataset visualization, download templates, and auxiliary analysis tools through a unified web interface.",
        ["Peptide taste prediction", "Bitter peptide", "Umami peptide", "Database", "Download"]
    )

    section("Function Modules")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        module_card(
            "module-bitter",
            "Bitter Peptide Prediction",
            "Predict whether a submitted peptide sequence is likely to exhibit bitter taste. The module supports sequence input, file upload, probability output, and downloadable prediction results.",
            "Bitter Prediction"
        )
    with c2:
        module_card(
            "module-umami",
            "Umami Peptide Prediction",
            "Identify potential umami peptides from user-submitted sequences or uploaded files. The module supports threshold control, batch prediction, and downloadable prediction results.",
            "Umami Prediction"
        )

    st.write("")
    c3, c4 = st.columns(2, gap="large")
    with c3:
        module_card(
            "module-db",
            "Taste Peptide Database",
            "Explore peptide datasets, taste-category statistics, sequence-length distributions, amino acid composition patterns, and interactive visualization outputs.",
            "Database"
        )
    with c4:
        module_card(
            "module-download",
            "Dataset Download",
            "Download CSV and FASTA templates for peptide prediction workflows, and review the standard result table format used by Pep2Taste.",
            "Download"
        )

    st.markdown(
        '<div class="note-box"><strong>Cloud inference is enabled.</strong> Bitter and umami peptide predictions are served through the configured Hugging Face backend API, with local mock output used only when the backend is unavailable.</div>',
        unsafe_allow_html=True,
    )

def prediction_page(task: str) -> None:
    is_bitter = task == "bitter"
    model_name = "Bitter-Fusion" if is_bitter else "Umami-LoRA"
    task_name = "bitter peptide" if is_bitter else "umami peptide"
    subtitle = (
        "Bitter-Fusion predicts bitter peptide candidates from pasted sequences or uploaded FASTA, CSV, and TXT files. The workspace supports example loading, batch prediction, threshold control, and downloadable result tables."
        if is_bitter else
        "Umami-LoRA predicts umami peptide candidates from pasted sequences or uploaded FASTA, CSV, and TXT files. The workspace supports example loading, batch prediction, threshold control, and downloadable result tables."
    )
    badges = [model_name, "Binary classification", "FASTA / CSV / TXT", "Downloadable results"]

    render_hero(model_name, subtitle, badges)
    st.write("")

    example_fasta = (
        ">Positive_F1\nNLLRFF\n>Positive_F2\nRRPPGF\n>Positive_F3\nRRR\n>Positive_F11\nVPPFLE\n>Positive_F12\nLHLPLPLL"
        if is_bitter else
        ">Umami_001\nEEEEEL\n>Umami_002\nDEDEG\n>Umami_003\nEVHEE\n>Umami_004\nEE"
    )
    plain_example = "GLLGFLG\nRRPPGF\nVPPFLE\nLHLPLPLL" if is_bitter else "EEEEEL\nDEDEG\nEVHEE\nEE"
    placeholder = (
        "Paste FASTA records or one peptide sequence per line.\n\n"
        ">pep_001\nGLLGFLG\n>pep_002\nRRPPGF"
        if is_bitter else
        "Paste FASTA records or one peptide sequence per line.\n\n"
        ">pep_001\nEEEEEL\n>pep_002\nDEDEG"
    )
    text_key = f"{task}_sequence_text"
    threshold_key = f"{task}_threshold"
    if text_key not in st.session_state:
        st.session_state[text_key] = ""

    if not is_bitter:
        st.markdown(f'<div class="predictor-model-label">{model_name} Prediction Model:</div>', unsafe_allow_html=True)
        st.selectbox(
            "Prediction model",
            [model_name],
            key=f"{task}_model",
            label_visibility="collapsed",
        )

    section("Prediction Workspace")
    left, right = st.columns([1.12, .88], gap="large")

    with left:
        st.markdown(
            '<div class="example-row"><span>Sequence:</span><span class="example-hint">Load an example or paste your own FASTA/list input.</span></div>',
            unsafe_allow_html=True,
        )
        ex1, ex2 = st.columns(2)
        with ex1:
            st.button(
                "Load FASTA example",
                key=f"{task}_load_fasta",
                use_container_width=True,
                on_click=set_session_value,
                args=(text_key, example_fasta),
            )
        with ex2:
            st.button(
                "Load sequence list",
                key=f"{task}_load_list",
                use_container_width=True,
                on_click=set_session_value,
                args=(text_key, plain_example),
            )
        sequence_text = st.text_area(
            "Peptide sequences",
            height=330,
            placeholder=placeholder,
            key=text_key,
            help="Use FASTA records or one peptide sequence per line. Only the 20 standard amino acid letters are accepted.",
        )

    with right:
        st.subheader("Upload file")
        uploaded = st.file_uploader(
            "Upload CSV / FASTA / TXT",
            type=["csv", "fa", "fasta", "txt"],
            key=f"{task}_upload",
            help="CSV files should include a sequence column, or place peptide sequences in the first column.",
        )
        st.caption("Uploaded file sequences are merged with the pasted sequences before prediction.")
        threshold = st.slider(
            "Classification threshold",
            0.00,
            1.00,
            0.50,
            0.01,
            key=threshold_key,
        )
        b1, b2, b3 = st.columns(3)
        with b1:
            run = st.button("Submit", type="primary", use_container_width=True, key=f"{task}_submit")
        with b2:
            st.button(
                "Clear",
                use_container_width=True,
                key=f"{task}_clear",
                on_click=set_session_value,
                args=(text_key, ""),
            )
        with b3:
            st.button(
                "Reset",
                use_container_width=True,
                key=f"{task}_reset",
                on_click=reset_prediction_inputs,
                args=(text_key, threshold_key),
            )

    section("Prediction Results")
    if run:
        seqs = []
        seqs.extend(parse_fasta_or_lines(sequence_text))
        seqs.extend(read_uploaded_sequences(uploaded))
        seqs = list(dict.fromkeys([clean_sequence(s) for s in seqs if clean_sequence(s)]))

        valid, invalid = [], []
        for s in seqs:
            ok, msg = validate_sequence(s)
            if ok:
                valid.append(s)
            else:
                invalid.append({"sequence": s, "reason": msg})

        if not valid:
            st.error("No valid peptide sequence found. Paste FASTA/list input or upload a CSV, FASTA, or TXT file.")
        else:
            with st.spinner(f"Running {model_name} prediction for {len(valid)} valid sequence(s)..."):
                results, source = predict_with_api_or_mock(valid, task, threshold)
                time.sleep(0.25)

            df = pd.DataFrame(results).drop(columns=["source", "confidence"], errors="ignore")
            preferred_columns = ["sequence", "length", "probability", "threshold", "label"]
            df = df[[c for c in preferred_columns if c in df.columns] + [c for c in df.columns if c not in preferred_columns]]
            st.success(f"Prediction completed. Valid: {len(valid)}, invalid: {len(invalid)}.")
            if len(results) == 1:
                render_result(results[0], task, threshold)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8-sig")
            file_name = "bitter_fusion_prediction_results.csv" if is_bitter else "umami_lora_prediction_results.csv"
            st.download_button(
                "Download results CSV",
                csv,
                file_name=file_name,
                mime="text/csv",
                use_container_width=True,
                key=f"{task}_download_results",
            )
            if source == "mock":
                st.caption("Mock prediction is used because no backend API is configured.")
            if invalid:
                with st.expander("Invalid sequences"):
                    st.dataframe(pd.DataFrame(invalid), use_container_width=True, hide_index=True)
    else:
        st.info("Paste sequences or upload a file, then click Submit.")

    section("Notes")
    st.markdown(
        f"""
        <div class="note-box">
            <strong>Input note.</strong> {model_name} accepts FASTA records, one peptide sequence per line, TXT files, and CSV files. CSV files should include a <code>sequence</code> column; otherwise the first column is used. Supported amino acid letters are A C D E F G H I K L M N P Q R S T V W Y.
            <br><strong>Result note.</strong> Probability ranges from 0 to 1. A sequence is labeled as {task_name} when its probability is greater than or equal to the selected threshold.
        </div>
        """,
        unsafe_allow_html=True,
    )


def database_page() -> None:
    render_hero(
        "Taste Peptide Database",
        "Explore curated taste-active peptide records and compare physicochemical descriptor patterns across taste classes.",
        ["Database explorer", "Physicochemical analysis", "Taste classes", "Descriptor fingerprint"]
    )

    df = load_database_with_properties()
    if df.empty:
        st.error("Database file was not found. Expected path: data/Database.csv")
        return

    all_tastes = sorted({label for labels in df["Taste labels"] for label in labels})
    doi_paper_count = df.loc[df["DOI"] != "Not reported", "DOI"].nunique()
    source_count = df.loc[df["Source"] != "Not reported", "Source"].nunique()

    section("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_box("Peptides", f"{len(df):,}")
    with c2:
        metric_box("Taste classes", str(len(all_tastes)))
    with c3:
        metric_box("Sources", f"{source_count:,}")
    with c4:
        metric_box("DOI papers", f"{doi_paper_count:,}")

    tab_explorer, tab_analysis = st.tabs([
        "Database Explorer",
        "Physicochemical Analysis",
    ])

    with tab_explorer:
        section("Search And Filter")
        f1, f2, f3 = st.columns([1.15, 1.05, .8], gap="large")
        with f1:
            query = st.text_input(
                "Search",
                placeholder="Sequence, taste, source, or DOI",
                key="db_search",
            ).strip()
        with f2:
            selected_tastes = st.multiselect(
                "Taste class",
                all_tastes,
                default=[],
                key="db_taste_filter",
            )
        with f3:
            length_range = st.slider(
                "Length range",
                int(df["Len"].min()),
                int(df["Len"].max()),
                (int(df["Len"].min()), int(df["Len"].max())),
                key="db_length_filter",
            )

        filtered = df[df["Len"].between(length_range[0], length_range[1])].copy()
        if selected_tastes:
            selected = set(selected_tastes)
            filtered = filtered[filtered["Taste labels"].map(lambda labels: bool(selected.intersection(labels)))]
        if query:
            query_l = query.lower()
            haystack = (
                filtered["Sequence"].astype(str) + " " +
                filtered["Taste"].astype(str) + " " +
                filtered["Source"].astype(str) + " " +
                filtered["DOI"].astype(str)
            ).str.lower()
            filtered = filtered[haystack.str.contains(re.escape(query_l), na=False)]

        m1, m2, m3 = st.columns(3)
        with m1:
            metric_box("Filtered records", f"{len(filtered):,}")
        with m2:
            metric_box("Median Mw", f"{filtered['Mw'].median():.1f}" if len(filtered) else "0")
        with m3:
            metric_box("Mean GRAVY", f"{filtered['GRAVY'].mean():.2f}" if len(filtered) else "0")

        st.dataframe(
            filtered[DATABASE_DISPLAY_COLUMNS],
            use_container_width=True,
            hide_index=True,
            height=420,
            column_config={
                "Mw": st.column_config.NumberColumn("Mw", format="%.3f"),
                "pI": st.column_config.NumberColumn("pI", format="%.3f"),
                "GRAVY": st.column_config.NumberColumn("GRAVY", format="%.3f"),
                "Aromaticity": st.column_config.NumberColumn("Aromaticity", format="%.3f"),
                "Stability Index": st.column_config.NumberColumn("Stability Index", format="%.3f"),
            },
        )
        csv = filtered[DATABASE_DISPLAY_COLUMNS].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download filtered database CSV",
            csv,
            file_name="pep2taste_filtered_database.csv",
            mime="text/csv",
            use_container_width=True,
            key="db_download_filtered",
        )

    with tab_analysis:
        section("Physicochemical Analysis")
        left, right = st.columns([.32, .68], gap="large")
        category_options = ["All taste-active peptides"] + all_tastes
        with left:
            selected_category = st.radio(
                "Taste category",
                category_options,
                key="physchem_category",
            )
            if selected_category == "All taste-active peptides":
                analysis_df = df.copy()
            else:
                analysis_df = df[df["Taste labels"].map(lambda labels: selected_category in labels)].copy()

            metric_box("Selected records", f"{len(analysis_df):,}")
            metric_box("Mean length", f"{analysis_df['Len'].mean():.2f}" if len(analysis_df) else "0")
            metric_box("Mean Mw", f"{analysis_df['Mw'].mean():.1f}" if len(analysis_df) else "0")
            st.markdown(
                """
                <div class="note-box">
                    <strong>View logic.</strong> Choose one taste class to inspect its physicochemical fingerprint. Multi-label peptides are counted in every matching taste class.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            minmax = df[ANALYSIS_COLUMNS].agg(["min", "max"])
            selected_label = selected_category.replace("All taste-active peptides", "All taste-active peptides")
            selected_means = analysis_df[ANALYSIS_COLUMNS].mean()
            all_means = df[ANALYSIS_COLUMNS].mean()
            fingerprint_rows = []
            for group, means in [(selected_category, selected_means), ("All peptides", all_means)]:
                for desc in ANALYSIS_COLUMNS:
                    min_v = float(minmax.loc["min", desc])
                    max_v = float(minmax.loc["max", desc])
                    norm = 0.5 if max_v == min_v else (float(means[desc]) - min_v) / (max_v - min_v)
                    fingerprint_rows.append({
                        "Group": group,
                        "Descriptor": DESCRIPTOR_LABELS.get(desc, desc),
                        "Value": float(means[desc]),
                        "Normalized": norm,
                    })
            fingerprint_df = pd.DataFrame(fingerprint_rows)
            st.subheader(f"{selected_label} descriptor fingerprint")
            fingerprint = (
                alt.Chart(fingerprint_df)
                .mark_bar(cornerRadiusEnd=5)
                .encode(
                    y=alt.Y("Descriptor:N", title=None, sort=[DESCRIPTOR_LABELS[c] for c in ANALYSIS_COLUMNS]),
                    x=alt.X("Normalized:Q", title="Normalized descriptor level", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Group:N", scale=alt.Scale(range=["#d97706", "#2563eb"])),
                    tooltip=[
                        alt.Tooltip("Group:N"),
                        alt.Tooltip("Descriptor:N"),
                        alt.Tooltip("Value:Q", format=".3f"),
                    ],
                )
                .properties(height=230)
            )
            st.altair_chart(fingerprint, use_container_width=True)

            st.subheader(f"{selected_label} amino acid composition")
            comp = amino_acid_composition(analysis_df["Sequence"].tolist())
            comp_chart = (
                alt.Chart(comp)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("AA:N", title=None, sort=AA_LIST, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Frequency:Q", title="Frequency"),
                    color=alt.Color("AA:N", legend=None, scale=alt.Scale(scheme="category20")),
                    tooltip=[
                        alt.Tooltip("AA:N"),
                        alt.Tooltip("Count:Q"),
                        alt.Tooltip("Frequency:Q", format=".2%"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(comp_chart, use_container_width=True)

            st.subheader(f"{selected_label} descriptor distributions")
            distribution_colors = {
                "Len": "#64748b",
                "Mw": "#ef4444",
                "pI": "#10b981",
                "GRAVY": "#f97316",
                "Aromaticity": "#6366f1",
                "Stability Index": "#14b8a6",
            }
            for row_start in range(0, len(ANALYSIS_COLUMNS), 3):
                cols = st.columns(3, gap="large")
                for col, desc in zip(cols, ANALYSIS_COLUMNS[row_start:row_start + 3]):
                    values = analysis_df[[desc]].dropna().rename(columns={desc: "Value"})
                    chart = (
                        alt.Chart(values)
                        .mark_bar(color=distribution_colors.get(desc, "#2563eb"), opacity=.84, cornerRadiusEnd=3)
                        .encode(
                            x=alt.X(
                                "Value:Q",
                                bin=alt.Bin(maxbins=22),
                                title=DESCRIPTOR_LABELS.get(desc, desc),
                                scale=alt.Scale(zero=False, nice=True),
                                axis=alt.Axis(labelAngle=0),
                            ),
                            y=alt.Y("count():Q", title="Count"),
                            tooltip=[
                                alt.Tooltip("count():Q", title="Count"),
                            ],
                        )
                        .properties(height=210)
                    )
                    with col:
                        st.altair_chart(chart, use_container_width=True)

            summary = analysis_df[ANALYSIS_COLUMNS].rename(columns=DESCRIPTOR_LABELS).describe().loc[["mean", "std", "min", "50%", "max"]].round(3)
            st.subheader(f"{selected_label} descriptor summary")
            st.dataframe(summary, use_container_width=True)


def download_page() -> None:
    render_hero(
        "Download Datasets",
        "提供呈味肽基准二分类数据集下载，并支持序列长度分布与氨基酸组成分析。",
        ["Dataset download", "Dataset analysis", "Benchmark datasets", "Binary classification"]
    )

    records = load_binary_dataset_records()
    summary = dataset_summary_table()
    if records.empty or summary.empty:
        st.warning("No binary peptide datasets were detected under the data directory.")
        return

    tab_download, tab_analyze = st.tabs(["Dataset Download", "Dataset Analysis"])

    with tab_download:
        all_files = [APP_DIR / f for f in records["File"].drop_duplicates().tolist()]
        title_col, download_all_col = st.columns([.72, .28])
        with title_col:
            section("Dataset Download")
        with download_all_col:
            st.write("")
            st.download_button(
                "Download all datasets ZIP",
                make_zip_for_files(all_files),
                file_name="pep2taste_binary_datasets.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_all_datasets_zip",
            )

        table_widths = [1.55, .72, .78, .78, .9, .9, .82, .82, .95]
        table_headers = [
            "Dataset", "Sequences", "Min length", "Max length",
            "Train positive", "Train negative", "Test positive", "Test negative", "Download",
        ]
        header_cols = st.columns(table_widths)
        for col, header in zip(header_cols, table_headers):
            col.markdown(f'<div class="download-table-header">{header}</div>', unsafe_allow_html=True)

        for idx, row in summary.iterrows():
            row_cols = st.columns(table_widths)
            row_cols[0].markdown(f'<div class="download-table-name">{row["Display dataset"]}</div>', unsafe_allow_html=True)
            for col, value in zip(
                row_cols[1:8],
                [
                    row["Sequences"],
                    row["Min length"],
                    row["Max length"],
                    row["train_pos"],
                    row["train_neg"],
                    row["test_pos"],
                    row["test_neg"],
                ],
            ):
                col.markdown(f'<div class="download-table-cell">{int(value):,}</div>', unsafe_allow_html=True)

            dataset_files = [
                APP_DIR / f
                for f in records.loc[records["Dataset"] == row["Dataset"], "File"].drop_duplicates().tolist()
            ]
            if len(dataset_files) == 1:
                download_bytes = dataset_files[0].read_bytes()
                file_name = dataset_files[0].name
                mime_type = "text/csv"
            else:
                download_bytes = make_zip_for_files(dataset_files)
                file_name = f"{safe_filename(row['Display dataset'])}.zip"
                mime_type = "application/zip"

            with row_cols[8]:
                st.download_button(
                    "Download",
                    download_bytes,
                    file_name=file_name,
                    mime=mime_type,
                    use_container_width=True,
                    key=f"download_dataset_{idx}_{safe_filename(row['Dataset'])}",
                )
            st.markdown('<div class="download-table-divider"></div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="note-box">
                <strong>Dataset note.</strong><br>
                <strong>Max length:</strong> The maximum length of any peptide sequence within the dataset.<br>
                <strong>Min length:</strong> The minimum length of any peptide sequence within the dataset.<br>
                <strong>Train positive:</strong> The number of bioactive peptides used for model training.<br>
                <strong>Train negative:</strong> The number of non-bioactive peptides used for model training.<br>
                <strong>Test positive:</strong> The number of bioactive peptides used for model evaluation.<br>
                <strong>Test negative:</strong> The number of non-bioactive peptides used for model evaluation.<br>
                <strong>Note:</strong> In the provided csv file, peptide bioactivity is encoded as follows:<br>
                &middot; <code>0</code> = Bioactive (Positive)<br>
                &middot; <code>1</code> = Non-bioactive (Negative)
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_analyze:
        section("Dataset Analysis")
        display_to_dataset = dict(zip(summary["Display dataset"], summary["Dataset"]))
        select_col, _ = st.columns([.36, .64])
        with select_col:
            selected_display = st.selectbox(
                "Select dataset for length violin plot",
                summary["Display dataset"].tolist(),
                key="analysis_length_dataset_select",
            )
        selected_analysis = display_to_dataset[selected_display]
        selected_records = records[records["Dataset"] == selected_analysis].copy()

        present_splits = [split for split in SPLIT_LABEL_ORDER if (selected_records["Split label"] == split).any()]
        violin_records = selected_records[selected_records["Split label"].isin(present_splits)].copy()
        if violin_records.empty:
            st.info("No sequence-length records are available for this dataset.")
        else:
            violin_data = length_violin_data(violin_records, SPLIT_LABEL_ORDER)
            axis_label_expr = " : ".join(
                [f"datum.value == {idx} ? '{split}'" for idx, split in enumerate(present_splits)]
            ) + " : ''"
            violin = (
                alt.Chart(violin_data, title=alt.TitleParams(
                    text="The length distribution of peptides",
                    anchor="middle",
                    fontSize=20,
                    fontWeight="bold",
                    offset=14,
                ))
                .mark_area(orient="horizontal", opacity=.76, interpolate="monotone")
                .encode(
                    y=alt.Y(
                        "Length:Q",
                        title="Peptide length",
                        axis=alt.Axis(labelAngle=0),
                        scale=alt.Scale(nice=True),
                    ),
                    x=alt.X(
                        "Left:Q",
                        title=None,
                        axis=alt.Axis(
                            values=list(range(len(present_splits))),
                            labelExpr=axis_label_expr,
                            labelAngle=0,
                            labelPadding=8,
                        ),
                        scale=alt.Scale(domain=[-.55, max(len(present_splits) - .45, .55)]),
                    ),
                    x2="Right:Q",
                    color=alt.Color(
                        "Split label:N",
                        sort=present_splits,
                        scale=alt.Scale(range=["#16a34a", "#64748b", "#2563eb", "#dc2626"]),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=[
                        alt.Tooltip("Split label:N"),
                        alt.Tooltip("Count:Q", format=","),
                        alt.Tooltip("Length:Q", format=".1f"),
                        alt.Tooltip("Density:Q", format=".4f"),
                    ],
                )
                .properties(width=720, height=360)
            )
            v_left, v_center, v_right = st.columns([.17, .66, .17])
            with v_center:
                st.altair_chart(violin, use_container_width=False)

        heatmap_data, heatmap_row_order = dataset_split_aa_heatmap_data(records, summary["Dataset"].tolist())
        if heatmap_data.empty:
            st.info("No amino-acid composition data are available.")
        else:
            heatmap_height = max(360, min(760, 22 * len(heatmap_row_order)))
            heatmap = (
                alt.Chart(heatmap_data, title=alt.TitleParams(
                    text="The amino acid composition of taste peptides",
                    anchor="middle",
                    fontSize=20,
                    fontWeight="bold",
                    offset=14,
                ))
                .mark_rect(cornerRadius=2)
                .encode(
                    x=alt.X("AA:N", title=None, sort=AA_LIST, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(
                        "Dataset split:N",
                        title=None,
                        sort=heatmap_row_order,
                        axis=alt.Axis(labelLimit=240, labelPadding=8, labelFontSize=11),
                    ),
                    color=alt.Color("Frequency:Q", scale=alt.Scale(scheme="yelloworangered"), title="Frequency"),
                    tooltip=[
                        alt.Tooltip("Dataset:N"),
                        alt.Tooltip("Split label:N"),
                        alt.Tooltip("AA:N"),
                        alt.Tooltip("Count:Q"),
                        alt.Tooltip("Frequency:Q", format=".2%"),
                    ],
                )
                .properties(width=820, height=heatmap_height)
            )
            h_left, h_center, h_right = st.columns([.12, .76, .12])
            with h_center:
                st.altair_chart(heatmap, use_container_width=False)


def virtual_screening_page() -> None:
    render_hero(
        "Virtual Screening",
        "A future workspace for virtual hydrolysate peptide libraries, model-based probability screening, high-confidence candidate downloads, terminal preference analysis, and MEME motif outputs.",
        ["Virtual hydrolysis", "Probability screening", "Candidate download", "Motif analysis"]
    )

    manifest_path = VIRTUAL_SCREENING_DIR / "virtual_screening_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = pd.DataFrame(columns=[
            "Task", "Library", "Enzyme(s)", "Probability group", "Unique sequences",
            "Pool file", "Filtered file", "N/C terminal analysis", "MEME output", "Status",
        ])

    section("Screening Overview")
    st.markdown(
        """
        <div class="note-box">
            <strong>Why a separate page?</strong> Virtual digestion screening can grow into a large workflow: millions of candidate fragments, multiple enzymes, bitter/umami model scores, probability thresholds, high-confidence peptide downloads, N/C-terminal enrichment, and MEME motif reports. Keeping it separate from Dataset Download avoids a crowded page and makes the future screening results easier to navigate.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_box("Tasks", str(manifest["Task"].nunique()) if not manifest.empty else "0")
    with c2:
        metric_box("Libraries", str(manifest["Library"].nunique()) if not manifest.empty else "0")
    with c3:
        metric_box("Screening groups", f"{len(manifest):,}")
    with c4:
        ready = int((manifest.get("Status", pd.Series(dtype=str)).astype(str).str.lower() != "placeholder").sum()) if not manifest.empty else 0
        metric_box("Ready files", str(ready))

    tab_pools, tab_analysis, tab_download = st.tabs([
        "Prediction Pools",
        "High-confidence Analysis",
        "Downloads",
    ])

    with tab_pools:
        section("Prediction Pools")
        task_filter = st.multiselect(
            "Task",
            sorted(manifest["Task"].dropna().unique().tolist()) if not manifest.empty else [],
            default=[],
            key="virtual_task_filter",
        )
        view = manifest.copy()
        if task_filter:
            view = view[view["Task"].isin(task_filter)]
        st.dataframe(view, use_container_width=True, hide_index=True, height=360)
        st.caption("Placeholder rows will become downloadable prediction pools after virtual digestion and model scoring files are added under data/VirtualScreening/.")

    with tab_analysis:
        section("High-confidence Candidate Analysis")
        p1, p2, p3 = st.columns(3)
        with p1:
            card("N/C-terminal enrichment", "Future panels will summarize enriched N-terminal and C-terminal residues among high-confidence bitter and umami candidates.")
        with p2:
            card("MEME motif output", "MEME motif reports can be linked or embedded here after filtered candidate FASTA files are generated.")
        with p3:
            card("Descriptor profile", "Filtered candidates can be compared by length, amino acid composition, GRAVY, aromaticity, and other descriptors.")
        st.info("No virtual screening result files have been added yet. The page is ready for future prediction-pool files.")

    with tab_download:
        section("Screening Downloads")
        if manifest.empty:
            st.info("No manifest is available yet.")
        else:
            for _, row in manifest.iterrows():
                file_path = VIRTUAL_SCREENING_DIR / str(row.get("Filtered file", ""))
                label = f"{row.get('Task', 'Task')} · {row.get('Probability group', 'Group')}"
                if file_path.exists() and file_path.is_file():
                    st.download_button(
                        label,
                        file_path.read_bytes(),
                        file_name=file_path.name,
                        use_container_width=True,
                        key=f"download_virtual_{file_path.as_posix()}",
                    )
                else:
                    st.button(f"{label} (pending)", disabled=True, use_container_width=True, key=f"pending_{label}_{row.name}")


def tools_page() -> None:
    render_hero(
        "Tools",
        "Auxiliary tools for peptide sequence validation, amino acid composition and basic sequence statistics.",
        ["Sequence check", "AAC", "Batch parser"]
    )

    tab_check, tab_aac = st.tabs(["Sequence validation", "Amino acid composition"])
    with tab_check:
        seq = st.text_area("Input sequences", height=180, placeholder="One sequence per line or FASTA format")
        if st.button("Validate sequences", use_container_width=True):
            seqs = parse_fasta_or_lines(seq)
            rows = []
            for s in seqs:
                ok, msg = validate_sequence(s)
                rows.append({"sequence": s, "length": len(s), "valid": ok, "message": msg})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.warning("No sequence found.")

    with tab_aac:
        seq = st.text_input("Peptide sequence", value="GLLGFLG")
        seq = clean_sequence(seq)
        ok, msg = validate_sequence(seq)
        if not ok:
            st.error(msg)
        else:
            aa = list("ACDEFGHIKLMNPQRSTVWY")
            data = pd.DataFrame({"AA": aa, "Frequency": [seq.count(a) / len(seq) for a in aa]})
            st.bar_chart(data.set_index("AA"))
            st.dataframe(data, use_container_width=True, hide_index=True)


def help_page() -> None:
    render_hero(
        "Help",
        "Guidelines for using the Pep2Taste Streamlit platform.",
        ["Input guide", "Prediction explanation", "Deployment"]
    )

    section("How to use")
    st.markdown(
        """
        1. 在侧边栏选择 **苦味肽预测** 或 **鲜味肽预测**。
        2. 输入单条肽序列，或上传 CSV/FASTA/TXT 文件进行批量预测。
        3. 点击 Predict / Run batch prediction。
        4. 查看预测类别、概率、置信度，并下载 CSV 结果。

        **注意：当前版本暂未接入最终模型，预测输出为模拟结果，仅用于页面开发。**
        """
    )

    section("Backend API connection")
    st.markdown("后续接入真实模型时，在 `.streamlit/secrets.toml` 中配置：")
    st.code(
        """BITTER_API_URL = "https://your-bitter-api/predict"
UMAMI_API_URL = "https://your-umami-api/predict" """,
        language="toml",
    )

    section("API format")
    st.code(
        json.dumps({
            "request": {"sequences": ["GLLGFLG", "EEEEE"], "task": "bitter", "threshold": 0.5},
            "response": {
                "results": [
                    {"sequence": "GLLGFLG", "length": 7, "probability": 0.8123, "label": "Bitter"}
                ]
            }
        }, indent=2, ensure_ascii=False),
        language="json"
    )


def legacy_contact_page() -> None:
    render_hero(
        "Contact",
        "Contact information and project description. This page corresponds to the original contact.html module.",
        ["Research use", "Peptide taste", "Web platform"]
    )

    section("Project")
    card(
        "Pep2Taste Streamlit Rebuild",
        "本项目为原 Pep2Taste HTML/CSS/JavaScript Web 应用的 Streamlit 复刻版本，面向苦味肽与鲜味肽预测任务进行重新设计。"
    )

    section("Contact form preview")
    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        msg = st.text_area("Message", height=150)
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("This is a frontend-only form preview. No email has been sent.")


def contact_page() -> None:
    render_hero(
        "Contact",
        "Contact the Pep2Taste supervisor and developer, share suggestions for improving the platform, or contribute newly reported taste-active peptide sequences for future database curation.",
        []
    )

    liang_photo = image_data_uri(CONTACT_FIGURE_DIR / "liangPic.png")
    yang_photo = image_data_uri(CONTACT_FIGURE_DIR / "yang.jpg")

    section("Team Contacts")
    st.markdown(
        f"""
        <div class="contact-profile-grid">
            <div class="contact-profile-card supervisor">
                <div class="contact-photo"><img src="{liang_photo}" alt="Professor Guizhao Liang"></div>
                <div class="contact-profile-body">
                    <span class="contact-role">Supervisor</span>
                    <h3>Professor Guizhao Liang</h3>
                    <p class="contact-subtitle">College of Bioengineering, Chongqing University</p>
                    <div class="contact-detail"><span>E-mail</span><div>gzliang@cqu.edu.cn</div></div>
                    <div class="contact-detail"><span>Tel</span><div>(86)2365102507</div></div>
                    <div class="contact-detail"><span>Address</span><div>Room 519, College of Bioengineering, Chongqing University,<br>No.174 Shazhengjie, Shapingba, Chongqing, 400044, China</div></div>
                </div>
            </div>
            <div class="contact-profile-card developer">
                <div class="contact-photo"><img src="{yang_photo}" alt="Pep2Taste developer"></div>
                <div class="contact-profile-body">
                    <span class="contact-role">Developer</span>
                    <h3>Yanlong Yang</h3>
                    <p class="contact-subtitle">Master's Student, Pep2Taste development</p>
                    <div class="contact-detail"><span>Student ID</span><div>202319021092T</div></div>
                    <div class="contact-detail"><span>E-mail</span><div>202319021092T@stu.cqu.edu.cn</div></div>
                    <div class="contact-detail"><span>Address</span><div>Room 509, College of Bioengineering, Chongqing University,<br>No.174 Shazhengjie, Shapingba, Chongqing, 400044, China</div></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Feedback And Peptide Submission")
    st.markdown(
        """
        <div class="contact-action-strip">
            <div class="contact-action">
                <h3>Platform feedback</h3>
                <p>Report interface issues, suggest new analysis modules, or describe model and database features that would make Pep2Taste more useful.</p>
            </div>
            <div class="contact-action">
                <h3>New peptide records</h3>
                <p>Submit taste-active peptide sequences with optional taste labels, literature sources, DOI information, or batch files for later manual curation.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("contact_form", clear_on_submit=False):
        left, right = st.columns([.48, .52], gap="large")
        with left:
            st.subheader("Feedback")
            submitter_name = st.text_input("Name", placeholder="Your name")
            submitter_email = st.text_input("E-mail", placeholder="name@example.com")
            affiliation = st.text_input("Affiliation", placeholder="University, institute, or company")
            feedback_type = st.selectbox(
                "Feedback type",
                ["General suggestion", "Database correction", "Prediction issue", "Dataset contribution", "Other"],
            )
            message = st.text_area(
                "Message",
                height=170,
                placeholder="Describe your suggestion, question, or correction.",
            )

        with right:
            st.subheader("Peptide contribution")
            peptide_sequences = st.text_area(
                "Peptide sequences",
                height=150,
                placeholder="One sequence per line, or paste FASTA records.",
            )
            taste_label = st.text_input("Taste annotation", placeholder="Bitter, umami, sweet, salty, sour, kokumi, etc.")
            source_or_doi = st.text_input("Source / DOI", placeholder="Publication source, DOI, or database note")
            uploaded_records = st.file_uploader(
                "Upload peptide file",
                type=["csv", "txt", "fa", "fasta", "xlsx"],
                accept_multiple_files=True,
                help="Accepted formats: CSV, TXT, FASTA, and XLSX.",
            )

        submitted = st.form_submit_button("Submit contribution", type="primary", use_container_width=True)
        if submitted:
            has_content = bool(message.strip() or peptide_sequences.strip() or uploaded_records)
            if not has_content:
                st.warning("Please provide a message, peptide sequences, or an uploaded file before submitting.")
            else:
                USER_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
                upload_dir = USER_SUBMISSION_DIR / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)

                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                digest = hashlib.sha1(f"{timestamp}|{submitter_email}|{time.time()}".encode("utf-8")).hexdigest()[:8]
                submission_id = f"{timestamp}_{digest}"
                saved_files = []
                for uploaded in uploaded_records or []:
                    clean_name = safe_filename(uploaded.name)
                    target = upload_dir / f"{submission_id}_{clean_name}"
                    target.write_bytes(uploaded.getvalue())
                    saved_files.append(target.relative_to(APP_DIR).as_posix())

                row = pd.DataFrame([{
                    "submission_id": submission_id,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "name": submitter_name.strip(),
                    "email": submitter_email.strip(),
                    "affiliation": affiliation.strip(),
                    "feedback_type": feedback_type,
                    "message": message.strip(),
                    "peptide_sequences": peptide_sequences.strip(),
                    "taste_annotation": taste_label.strip(),
                    "source_or_doi": source_or_doi.strip(),
                    "uploaded_files": "; ".join(saved_files),
                }])
                log_path = USER_SUBMISSION_DIR / "contact_submissions.csv"
                if log_path.exists():
                    existing = pd.read_csv(log_path)
                    row = pd.concat([existing, row], ignore_index=True)
                row.to_csv(log_path, index=False, encoding="utf-8-sig")

                st.success(f"Submission saved. Reference ID: {submission_id}")
                st.caption("Saved locally under data/UserSubmissions/.")


def sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="sidebar-logo">
            <div class="brand">🧬 Pep2Taste</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "Home",
        "Bitter Prediction",
        "Umami Prediction",
        "Database",
        "Download",
        "Virtual Screening",
        "Tools",
        "Help",
        "Contact",
    ]

    default_page = "Home"
    try:
        query_page = st.query_params.get("page", "Home")
        if isinstance(query_page, list):
            query_page = query_page[0]
        if query_page in pages:
            default_page = query_page
    except Exception:
        default_page = "Home"

    page = st.sidebar.radio(
        "Navigation",
        pages,
        index=pages.index(default_page),
    )
    return page


page = sidebar()

if page == "Home":
    home_page()
elif page == "Bitter Prediction":
    prediction_page("bitter")
elif page == "Umami Prediction":
    prediction_page("umami")
elif page == "Database":
    database_page()
elif page == "Download":
    download_page()
elif page == "Virtual Screening":
    virtual_screening_page()
elif page == "Tools":
    tools_page()
elif page == "Help":
    help_page()
elif page == "Contact":
    contact_page()

st.markdown('<div class="footer">Pep2Taste · AI-powered peptide taste prediction platform · Research use only</div>', unsafe_allow_html=True)
