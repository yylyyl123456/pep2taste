
import base64
import hashlib
import html
import io
import json
import math
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


# ==========================================================
# Pep2Taste Streamlit Rebuild
# Mock prediction is only used as a fallback when the backend is unavailable.
# ==========================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATABASE_PATH = APP_DIR / "data" / "Database.csv"
VIRTUAL_SCREENING_DIR = DATA_DIR / "VirtualScreening"
USER_SUBMISSION_DIR = DATA_DIR / "UserSubmissions"
CONTACT_FIGURE_DIR = APP_DIR / "figure"
MODEL_FIGURE_DIR = CONTACT_FIGURE_DIR / "model_architectures"
HELP_WORKFLOW_FIGURE = CONTACT_FIGURE_DIR / "pep2taste_workflow.png"
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
MODEL_DISPLAY_NAMES = {
    "Bitter_Stacking": "BPPred",
    "Umami_LoRA": "UPPred",
}

st.set_page_config(
    page_title="Pep2Taste | Peptide Taste Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="auto",
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
    max-width: 100%;
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
.virtual-overview-note {
    border:1px solid rgba(47,154,214,.28);
    border-left:5px solid #2F9AD6;
    background:linear-gradient(180deg,#ffffff 0%, #f3f9fd 100%);
    color:#1f2937;
    padding:1.05rem 1.2rem;
    border-radius:12px;
    line-height:1.72;
    font-size:1.04rem;
    font-weight:720;
    margin:.35rem 0 1.1rem;
}
.virtual-overview-note strong {
    color:#0f172a;
    font-weight:900;
}
.highconf-note {
    border:1px solid rgba(15,23,42,.10);
    border-left:5px solid #ef4444;
    background:linear-gradient(180deg,#ffffff 0%, #fff7f7 100%);
    color:#1f2937;
    padding:1rem 1.15rem;
    border-radius:12px;
    line-height:1.68;
    font-size:1rem;
    font-weight:680;
    margin:.35rem 0 1rem;
}
.highconf-note strong {
    color:#111827;
    font-weight:900;
}
.analysis-status {
    border:1px dashed rgba(15,23,42,.18);
    background:#f8fafc;
    color:#475569;
    padding:.86rem 1rem;
    border-radius:10px;
    line-height:1.58;
    font-size:.94rem;
    margin:.35rem 0 .9rem;
}
.analysis-status strong {
    color:#0f172a;
}
div[data-testid="stVegaLiteChart"] {
    display:flex;
    justify-content:center;
}
div[data-testid="stVegaLiteChart"] > div {
    margin-left:auto;
    margin-right:auto;
}
div[data-testid="stFullScreenFrame"] {
    display:flex;
    justify-content:center;
    overflow:visible;
}
div[data-testid="stFullScreenFrame"] > div {
    flex:0 0 auto;
}
.centered-image-row {
    display:flex;
    justify-content:center;
    align-items:center;
    margin:.65rem 0 1.45rem;
}
.centered-image-block {
    width:min(100%, 360px);
    text-align:center;
}
.centered-image-block img {
    max-width:240px;
    width:100%;
    height:auto;
}
.centered-image-caption {
    color:#475569;
    font-size:.9rem;
    line-height:1.45;
    margin-top:.45rem;
}
.meme-logo-grid-wrap {
    width:100%;
    display:flex;
    justify-content:center;
    margin:1.1rem 0 1.7rem;
}
.meme-logo-grid {
    display:grid;
    grid-template-columns:repeat(4, minmax(180px, 230px));
    justify-content:center;
    align-items:start;
    gap:1.15rem;
    width:min(100%, 1010px);
}
.meme-logo-item {
    text-align:center;
}
.meme-logo-item img {
    display:block;
    margin:0 auto;
    max-width:210px;
    max-height:240px;
    width:auto;
    height:auto;
}
.meme-logo-caption {
    color:#475569;
    font-size:.82rem;
    line-height:1.42;
    margin-top:.45rem;
}
@media (max-width: 900px) {
    .meme-logo-grid {
        grid-template-columns:repeat(2, minmax(140px, 170px));
        width:min(100%, 420px);
    }
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
.virtual-download-task {
    margin:1.25rem 0 .55rem;
    padding:.82rem 1rem;
    border-left:5px solid #2F9AD6;
    background:#f4fbff;
    border-radius:10px;
    color:#0f172a;
    font-weight:900;
}
.virtual-download-task span {
    color:#64748b;
    font-size:.9rem;
    font-weight:700;
    margin-left:.35rem;
}
.virtual-download-header {
    min-height:2.55rem;
    display:flex;
    align-items:center;
    padding:.55rem .65rem;
    border:1px solid rgba(47,154,214,.18);
    background:#eaf7fb;
    color:#334155;
    font-size:.78rem;
    font-weight:900;
    text-transform:uppercase;
    letter-spacing:.035em;
}
.virtual-download-cell,
.virtual-download-method,
.virtual-download-file,
.virtual-download-muted {
    min-height:3rem;
    display:flex;
    align-items:center;
    padding:.46rem .65rem;
    border-bottom:1px solid rgba(47,154,214,.16);
    color:#334155;
    font-size:.92rem;
    line-height:1.35;
    font-variant-numeric:tabular-nums;
}
.virtual-download-method {
    color:#0f172a;
    font-weight:850;
}
.virtual-download-file {
    color:#2563eb;
    font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size:.84rem;
    word-break:break-all;
}
.virtual-download-muted {
    color:#94a3b8;
}
div[role="radiogroup"][aria-label="Virtual download task"],
div[role="radiogroup"][aria-label="High-confidence analysis target"] {
    display:flex;
    justify-content:center;
    gap:.85rem;
    flex-wrap:nowrap;
    width:100%;
}
div[role="radiogroup"][aria-label="Virtual download task"] label,
div[role="radiogroup"][aria-label="High-confidence analysis target"] label {
    min-width:190px;
    justify-content:center;
    border:1px solid rgba(15,23,42,.14);
    border-radius:10px;
    padding:.72rem 1.05rem;
    background:#ffffff;
    box-shadow:0 10px 24px rgba(15,23,42,.05);
    transition:all .16s ease;
}
div[role="radiogroup"][aria-label="Virtual download task"] label:hover,
div[role="radiogroup"][aria-label="High-confidence analysis target"] label:hover {
    border-color:rgba(47,154,214,.45);
    background:#f8fcff;
}
div[role="radiogroup"][aria-label="Virtual download task"] label > div:first-child,
div[role="radiogroup"][aria-label="High-confidence analysis target"] label > div:first-child {
    display:none;
}
div[role="radiogroup"][aria-label="Virtual download task"] label p,
div[role="radiogroup"][aria-label="High-confidence analysis target"] label p {
    margin:0;
    color:#334155;
    font-size:.98rem;
    font-weight:850;
    text-align:center;
}
div[role="radiogroup"][aria-label="Virtual download task"] label:has(input:checked),
div[role="radiogroup"][aria-label="High-confidence analysis target"] label:has(input:checked) {
    border-color:#2F9AD6;
    background:linear-gradient(180deg,#e8f7fd 0%, #ffffff 100%);
    box-shadow:0 12px 28px rgba(47,154,214,.18);
}
div[role="radiogroup"][aria-label="Virtual download task"] label:has(input:checked) p,
div[role="radiogroup"][aria-label="High-confidence analysis target"] label:has(input:checked) p {
    color:#0f172a;
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
.help-workflow-figure {
    margin:.5rem 0 1.2rem;
    border:1px solid rgba(15,23,42,.10);
    border-radius:8px;
    overflow:hidden;
    background:#ffffff;
    box-shadow:0 12px 32px rgba(15,23,42,.07);
}
.help-workflow-figure img {
    display:block;
    width:100%;
    height:auto;
}
.help-workflow-caption {
    padding:.7rem 1rem .8rem;
    color:#64748b;
    font-size:.9rem;
    line-height:1.5;
    border-top:1px solid rgba(15,23,42,.08);
}
.help-guide-grid {
    display:grid;
    grid-template-columns:repeat(2, minmax(0, 1fr));
    gap:1rem;
    margin:.35rem 0 1.15rem;
}
.help-guide-item {
    border:1px solid rgba(15,23,42,.10);
    border-left:5px solid var(--guide-color, #2563eb);
    border-radius:8px;
    background:#ffffff;
    padding:1rem 1.05rem;
}
.help-guide-item h3 {
    margin:0 0 .35rem;
    color:#0f172a;
    font-size:1.05rem;
}
.help-guide-item p {
    margin:0;
    color:#64748b;
    line-height:1.62;
}
.help-reference-grid {
    display:grid;
    grid-template-columns:repeat(3, minmax(0, 1fr));
    gap:1rem;
    margin:.45rem 0 1rem;
}
.help-reference-item {
    border-top:3px solid var(--ref-color, #2563eb);
    background:#f8fafc;
    padding:.9rem 1rem;
    min-height:150px;
}
.help-reference-item h4 {
    margin:0 0 .45rem;
    color:#0f172a;
    font-size:1rem;
}
.help-reference-item p {
    margin:.25rem 0;
    color:#475569;
    line-height:1.55;
    font-size:.92rem;
}
.help-reference-item code {
    color:#0f172a;
    font-weight:750;
}
@media (max-width: 980px) {
    .contact-profile-grid,
    .contact-action-strip,
    .help-guide-grid,
    .help-reference-grid {
        grid-template-columns:1fr;
    }
    .contact-profile-card {
        grid-template-columns:150px 1fr;
    }
}
@media (max-width: 620px) {
    .block-container {
        padding:1rem .85rem 2rem;
    }
    .hero {
        padding:1.7rem 1.35rem;
        border-radius:18px;
    }
    .hero h1 {
        font-size:2.05rem;
        line-height:1.12;
    }
    .hero p {
        font-size:.95rem;
        line-height:1.55;
    }
    .hero .actions {
        gap:.5rem;
        margin-top:1.1rem;
    }
    .badge {
        font-size:.78rem;
        padding:.36rem .62rem;
    }
    .module-card {
        min-height:190px;
        padding:1.45rem;
        border-radius:18px;
    }
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


def predict_with_api_or_mock(
    sequences: List[str],
    task: str,
    threshold: float,
    method: str | None = None,
) -> Tuple[List[Dict[str, Any]], str]:
    api_key = "BITTER_API_URL" if task == "bitter" else "UMAMI_API_URL"
    url = get_secret_or_env(api_key, "")
    if url:
        try:
            payload = {"sequences": sequences, "task": task, "threshold": threshold}
            if method:
                payload["method"] = method
            resp = requests.post(
                url,
                json=payload,
                timeout=600,
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
            "method": method,
        })
        if task == "bitter" and method == "Bitter_Stacking":
            branch_seed = deterministic_mock_score(seq + "AA", task)
            fp_seed = deterministic_mock_score(seq + "FP", task)
            plm_seed = deterministic_mock_score(seq + "PLM", task)
            results[-1].update({
                "aa_lgbm_probability": branch_seed,
                "fp_catboost_probability": fp_seed,
                "esm2_t33_mlp_probability": plm_seed,
                "bitter_stacking_probability": prob,
            })
        if task == "umami" and method == "Umami_LoRA":
            esm_seed = deterministic_mock_score(seq + "ESM2", task)
            pep_seed = deterministic_mock_score(seq + "PepBERT", task)
            prott5_seed = deterministic_mock_score(seq + "ProtT5", task)
            results[-1].update({
                "esm2_probability": esm_seed,
                "pepbert_probability": pep_seed,
                "prott5_probability": prott5_seed,
                "umami_lora_probability": prob,
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


def normalize_doi_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "Not reported"
    text = re.sub(r"\bDOI\d*\s*:\s*", "", text, flags=re.IGNORECASE)
    doi_values = re.findall(r"10\.\d{4,9}/[^\s;,]+", text)
    return "; ".join(dict.fromkeys(doi_values)) if doi_values else text


def doi_to_url(value: Any) -> str:
    doi = normalize_doi_text(value)
    if not doi or doi.lower() == "nan" or doi == "Not reported":
        return ""
    doi = doi.split(";", 1)[0].strip()
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
    if name == "Bitter(Ours)":
        return "BTP1160"
    if name == "Umami(Ours)":
        return "UMP1916"
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
    df["DOI"] = df["DOI"].map(normalize_doi_text)

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
        "Pep2Taste is an integrated platform for the computational discovery of taste peptides. It combines bitter and umami peptide prediction with a curated database, benchmark datasets, physicochemical analysis, and virtual-hydrolysate screening in one research workspace.",
        ["Bitter peptide", "Umami peptide", "Virtual hydrolysis", "Database", "Download"]
    )

    section("Function Modules")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        module_card(
            "module-bitter",
            "Bitter Peptide Prediction",
            "Classify bitter and non-bitter peptides with BPPred or its three constituent feature branches, using pasted sequences or batch files.",
            "Bitter Prediction"
        )
    with c2:
        module_card(
            "module-umami",
            "Umami Peptide Prediction",
            "Screen umami peptide candidates with UPPred or individual ESM2-LoRA, PepBERT-LoRA, and ProtT5-LoRA branches.",
            "Umami Prediction"
        )

    st.write("")
    c3, c4 = st.columns(2, gap="large")
    with c3:
        module_card(
            "module-db",
            "Taste Peptide Database",
            "Search curated taste peptide records and examine sequence, source, literature, and physicochemical patterns across taste classes.",
            "Database"
        )
    with c4:
        module_card(
            "module-download",
            "Dataset Download",
            "Download taste peptide binary-classification benchmarks and compare their peptide-length and amino acid composition profiles.",
            "Download"
        )

def prediction_page(task: str) -> None:
    is_bitter = task == "bitter"
    model_name = "BPPred" if is_bitter else "UPPred"
    method_options = (
        ["ESM2_t33_MLP", "AA_LGBM", "FP_CatBoost", "Bitter_Stacking"]
        if is_bitter else
        ["ESM2_LoRA", "PepBERT_LoRA", "ProtT5_LoRA", "Umami_LoRA"]
    )
    default_method_index = 3
    task_name = "bitter peptide" if is_bitter else "umami peptide"
    subtitle = (
        "BPPred combines ESM-2 sequence representations, amino acid descriptors, and FCFP4 fingerprints through a probability-stacking classifier. Submit individual or batch peptide sequences for binary prediction and downloadable results."
        if is_bitter else
        "UPPred integrates ESM2-LoRA, PepBERT-LoRA, and ProtT5-LoRA branch probabilities through weighted soft voting. Submit individual or batch peptide sequences for binary prediction and downloadable results."
    )
    badges = ["Binary classification", "FASTA / CSV / TXT", "Downloadable results"]

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
    selected_method = None

    architecture_path = MODEL_FIGURE_DIR / ("bppred_architecture.jpg" if is_bitter else "uppred_architecture.jpg")
    with st.expander("Model architecture", expanded=False):
        if architecture_path.exists():
            st.image(
                str(architecture_path),
                caption=f"{model_name} model architecture",
                use_container_width=True,
            )
        else:
            st.info(f"{model_name} architecture figure is not available.")

    st.markdown('<div class="predictor-model-label">Prediction Model:</div>', unsafe_allow_html=True)
    selected_method = st.selectbox(
        "Prediction model",
        method_options,
        index=default_method_index,
        key=f"{task}_model",
        label_visibility="collapsed",
        format_func=lambda method: MODEL_DISPLAY_NAMES.get(method, method),
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
            run_name = MODEL_DISPLAY_NAMES.get(selected_method, selected_method or model_name)
            with st.spinner(f"Running {run_name} prediction for {len(valid)} valid sequence(s)..."):
                results, source = predict_with_api_or_mock(valid, task, threshold, selected_method)
                time.sleep(0.25)

            df = pd.DataFrame(results).drop(
                columns=[
                    "source",
                    "confidence",
                    "softvoting_gridoof_mcc_probability",
                    "logistic_stacking_probability",
                    "softvoting_equal_probability",
                ],
                errors="ignore",
            )
            df = df.dropna(axis=1, how="all")
            if "method" in df.columns:
                df["method"] = df["method"].replace(MODEL_DISPLAY_NAMES)
            if is_bitter and selected_method == "Bitter_Stacking":
                preferred_columns = [
                    "sequence",
                    "length",
                    "probability",
                    "threshold",
                    "label",
                    "method",
                    "aa_lgbm_probability",
                    "fp_catboost_probability",
                    "esm2_t33_mlp_probability",
                    "bitter_stacking_probability",
                ]
            elif is_bitter:
                preferred_columns = ["sequence", "length", "probability", "threshold", "label", "method"]
            elif selected_method == "Umami_LoRA":
                preferred_columns = [
                    "sequence",
                    "length",
                    "probability",
                    "threshold",
                    "label",
                    "method",
                    "esm2_probability",
                    "pepbert_probability",
                    "prott5_probability",
                    "umami_lora_probability",
                ]
            else:
                preferred_columns = ["sequence", "length", "probability", "threshold", "label", "method"]
            df = df[[c for c in preferred_columns if c in df.columns] + [c for c in df.columns if c not in preferred_columns]]
            st.success(f"Prediction completed. Valid: {len(valid)}, invalid: {len(invalid)}.")
            if len(results) == 1:
                render_result(results[0], task, threshold)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8-sig")
            file_name = "bppred_prediction_results.csv" if is_bitter else "uppred_prediction_results.csv"
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
        "Search curated taste peptide sequences with reported taste classes, biological or food sources, literature references, and computed physicochemical descriptors, then compare sequence-property patterns across taste categories.",
        ["Database explorer", "Physicochemical analysis", "Taste peptide classes", "Descriptor fingerprint"]
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
        category_options = ["All taste peptides"] + all_tastes
        with left:
            selected_category = st.radio(
                "Taste category",
                category_options,
                key="physchem_category",
            )
            if selected_category == "All taste peptides":
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
            selected_label = selected_category
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
        "Download benchmark binary-classification datasets for taste peptide research and inspect their peptide-length distributions and amino acid composition patterns.",
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
        dataset_display_options = summary["Display dataset"].tolist()
        selector_key = "analysis_length_dataset_select"
        if selector_key not in st.session_state or st.session_state[selector_key] not in dataset_display_options:
            st.session_state[selector_key] = dataset_display_options[0]
        selected_display = st.session_state[selector_key]
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
                .properties(width=620, height=360)
            )
            v_left, v_center, v_right = st.columns([.20, .60, .20])
            with v_center:
                st.altair_chart(violin, use_container_width=True)

        selector_left, selector_center, selector_right = st.columns([.33, .34, .33])
        with selector_center:
            st.selectbox(
                "Dataset",
                dataset_display_options,
                key=selector_key,
                label_visibility="collapsed",
            )

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
                .properties(width=680, height=heatmap_height)
            )
            h_left, h_center, h_right = st.columns([.14, .72, .14])
            with h_center:
                st.altair_chart(heatmap, use_container_width=True)


def _virtual_screening_page_legacy() -> None:
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


def virtual_screening_page() -> None:
    render_hero(
        "Virtual Screening",
        "Explore bitter and umami peptide candidates identified from virtual enzymatic hydrolysates, compare screening outcomes, analyze high-confidence sequence patterns, and download filtered peptide libraries.",
        ["Bitter peptide", "Umami peptide", "High-confidence analysis", "Candidate downloads"],
    )

    manifest_path = VIRTUAL_SCREENING_DIR / "virtual_screening_manifest.csv"
    umami_root = VIRTUAL_SCREENING_DIR / "umami" / "virtual_digest"
    bitter_root = VIRTUAL_SCREENING_DIR / "bitter" / "virtual_digest"
    umami_tables_dir = umami_root / "tables"
    bitter_tables_dir = bitter_root / "tables"
    figures_dir = umami_root / "figures"
    method_summary_path = umami_tables_dir / "hydrolysis_method_umami_summary.csv"
    reported_summary_path = umami_tables_dir / "hydrolysis_method_umami_reported_summary.csv"
    reported_download_manifest_path = umami_tables_dir / "reported_download_manifest.csv"
    bitter_method_summary_path = bitter_tables_dir / "hydrolysis_method_bitter_summary.csv"
    bitter_reported_summary_path = bitter_tables_dir / "hydrolysis_method_bitter_reported_summary.csv"
    source_count_path = umami_tables_dir / "hydrolysis_source_count_umami_summary.csv"
    bitter_source_count_path = bitter_tables_dir / "hydrolysis_source_count_bitter_summary.csv"

    def load_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def load_json_file(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def normalize_virtual_manifest(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "Task" not in df.columns:
            return df
        normalized = df.copy()
        if not isinstance(normalized.index, pd.RangeIndex):
            reset = normalized.reset_index()
            index_col = reset.columns[0]
            task_values = set(reset[index_col].dropna().astype(str).str.strip().unique().tolist())
            if task_values and task_values.issubset({"Umami", "Bitter"}):
                normalized = reset.rename(columns={index_col: "Task_Source"})
                if "Dataset" in normalized.columns:
                    left = normalized["Task"].fillna("").astype(str).str.strip()
                    right = normalized["Dataset"].fillna("").astype(str).str.strip()
                    has_right = right.ne("") & right.str.lower().ne("nan")
                    normalized["Dataset"] = left.mask(has_right, left + "," + right)
                normalized["Task"] = normalized["Task_Source"].astype(str).str.strip()
                normalized = normalized.drop(columns=["Task_Source"])
        return normalized

    manifest = load_csv(manifest_path)
    manifest = normalize_virtual_manifest(manifest)
    method_summary = load_csv(method_summary_path)
    reported_summary = load_csv(reported_summary_path)
    reported_download_manifest = load_csv(reported_download_manifest_path)
    bitter_method_summary = load_csv(bitter_method_summary_path)
    bitter_reported_summary = load_csv(bitter_reported_summary_path)
    source_count_summary = load_csv(source_count_path)
    bitter_source_count_summary = load_csv(bitter_source_count_path)

    numeric_cols = [
        "Count",
        "Unique sequences",
        "Mean_Probability",
        "Median_Probability",
        "Mean_Length",
    ]
    for col in numeric_cols:
        if col in manifest.columns:
            manifest[col] = pd.to_numeric(manifest[col], errors="coerce")

    if not manifest.empty and "Status" in manifest.columns:
        ready_manifest = manifest[manifest["Status"].astype(str).str.lower().eq("ready")].copy()
    else:
        ready_manifest = pd.DataFrame(columns=manifest.columns)

    protein_count = 21249
    species_count = 60
    ready_download_count = len(ready_manifest)
    if "method_id" in method_summary.columns:
        method_count = method_summary["method_id"].nunique()
    elif "method_id" in bitter_method_summary.columns:
        method_count = bitter_method_summary["method_id"].nunique()
    else:
        method_count = 9
    peptide_pool = int(source_count_summary["N"].sum()) if "N" in source_count_summary.columns else 0
    if not peptide_pool and "N" in bitter_source_count_summary.columns:
        peptide_pool = int(bitter_source_count_summary["N"].sum())
    umami_999_ready = "Final_gte_0_999" in source_count_summary.columns
    umami_high_conf_999 = int(source_count_summary["Final_gte_0_999"].sum()) if umami_999_ready else 0

    umami_confidence_options = {
        "Reported": "Reported",
        "No reported": "No_reported",
        "Pro < 0.50": "Final_lt_0_50",
        "Pro >= 0.50": "Final_gte_0_50",
        "Pro >= 0.90": "Final_gte_0_90",
        "Pro >= 0.95": "Final_gte_0_95",
    }
    bitter_confidence_options = {
        "Reported": "Reported",
        "No reported": "No_reported",
        "Pro < 0.50": "Final_lt_0_50",
        "Pro >= 0.50": "Final_gte_0_50",
        "Pro >= 0.85": "Final_gte_0_85",
        "Pro >= 0.90": "Final_gte_0_90",
    }
    enzyme_display_names = {
        "Chymotrypsin + Trypsin + Pepsin (pH=1.3)": "Chymotrypsin+Trypsin+Pepsin",
        "Thermolysin + Papain": "Thermolysin+Papain",
        "Pepsin + Pancreatic": "Pepsin+Pancreatic",
        "Proteinase K": "Proteinase K",
        "Thermolysin": "Thermolysin",
        "Papain": "Papain",
        "Pepsin (pH=1.3)": "Pepsin",
        "Chymotrypsin": "Chymotrypsin",
        "Trypsin": "Trypsin",
    }
    enzyme_order_source = [
        "Chymotrypsin + Trypsin + Pepsin (pH=1.3)",
        "Thermolysin + Papain",
        "Pepsin + Pancreatic",
        "Proteinase K",
        "Thermolysin",
        "Papain",
        "Pepsin (pH=1.3)",
        "Chymotrypsin",
        "Trypsin",
    ]
    fixed_enzyme_order = [enzyme_display_names[name] for name in enzyme_order_source]

    def file_cache_signature(path: Path) -> str:
        if not path.exists():
            return "missing"
        try:
            stat = path.stat()
        except OSError:
            return "unreadable"
        return f"{stat.st_size}:{int(stat.st_mtime)}"

    @st.cache_data(show_spinner=False)
    def load_high_confidence_analysis(csv_path_text: str, file_signature: str = "") -> Dict[str, pd.DataFrame]:
        csv_path = Path(csv_path_text)
        if not csv_path.exists():
            return {}

        wanted_cols = {
            "sequence",
            "length",
            "Final_Prob",
            "hydrolysis_count",
            "from_ChyTryPep",
            "from_ThePap",
            "from_PepPan",
            "from_ProteinaseK",
            "from_Thermolysin",
            "from_Papain",
            "from_Pepsin",
            "from_Chymotrypsin",
            "from_Trypsin",
        }
        try:
            raw = pd.read_csv(csv_path, usecols=lambda col: col in wanted_cols)
        except (OSError, ValueError, pd.errors.ParserError):
            return {}
        if raw.empty or "sequence" not in raw.columns:
            return {}

        raw["sequence"] = raw["sequence"].astype(str).str.upper().str.strip()
        raw = raw[raw["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", na=False)].copy()
        if raw.empty:
            return {}

        raw["length"] = pd.to_numeric(raw.get("length", raw["sequence"].str.len()), errors="coerce").fillna(raw["sequence"].str.len()).astype(int)
        raw["Final_Prob"] = pd.to_numeric(raw.get("Final_Prob", pd.Series(dtype=float)), errors="coerce")
        sequences = raw["sequence"].tolist()

        residue_counts = {aa: 0 for aa in AA_LIST}
        dipeptide_counts = {a + b: 0 for a in AA_LIST for b in AA_LIST}
        total_residues = 0
        total_dipeptides = 0
        for seq in sequences:
            total_residues += len(seq)
            for aa in seq:
                residue_counts[aa] += 1
            for i in range(len(seq) - 1):
                dipeptide = seq[i:i + 2]
                if dipeptide in dipeptide_counts:
                    dipeptide_counts[dipeptide] += 1
                    total_dipeptides += 1

        aa_df = pd.DataFrame({
            "Residue": AA_LIST,
            "Count": [residue_counts[aa] for aa in AA_LIST],
        })
        aa_df["Frequency (%)"] = aa_df["Count"] / max(total_residues, 1) * 100

        dipeptide_df = pd.DataFrame([
            {
                "First residue": dipeptide[0],
                "Second residue": dipeptide[1],
                "Dipeptide": dipeptide,
                "Count": count,
                "Frequency per 1,000": count / max(total_dipeptides, 1) * 1000,
            }
            for dipeptide, count in dipeptide_counts.items()
        ])

        terminal_rows = []
        for terminal_name, residues in {
            "N-terminal": raw["sequence"].str[0],
            "C-terminal": raw["sequence"].str[-1],
        }.items():
            counts = residues.value_counts()
            total = int(counts.sum())
            for aa in AA_LIST:
                count = int(counts.get(aa, 0))
                terminal_rows.append({
                    "Terminal": terminal_name,
                    "Residue": aa,
                    "Count": count,
                    "Frequency (%)": count / max(total, 1) * 100,
                })
        terminal_df = pd.DataFrame(terminal_rows)

        length_df = raw.groupby("length", as_index=False).size().rename(columns={"size": "Count", "length": "Length"})
        length_df = length_df.sort_values("Length")
        length_df["Length label"] = length_df["Length"].astype(int).astype(str)

        prob = raw["Final_Prob"].dropna()
        if prob.empty:
            probability_df = pd.DataFrame(columns=["Probability bin", "Count", "Bin order"])
        else:
            probability_bins = pd.cut(prob, bins=12, include_lowest=True, duplicates="drop")
            probability_counts = probability_bins.value_counts(sort=False)
            probability_df = pd.DataFrame({
                "Probability bin": [
                    f"{interval.left:.6f}-{interval.right:.6f}"
                    for interval in probability_counts.index
                ],
                "Count": probability_counts.values.astype(int),
                "Bin order": list(range(len(probability_counts))),
            })

        method_id_to_label = {
            "ChyTryPep": "Chymotrypsin+Trypsin+Pepsin",
            "ThePap": "Thermolysin+Papain",
            "PepPan": "Pepsin+Pancreatic",
            "ProteinaseK": "Proteinase K",
            "Thermolysin": "Thermolysin",
            "Papain": "Papain",
            "Pepsin": "Pepsin",
            "Chymotrypsin": "Chymotrypsin",
            "Trypsin": "Trypsin",
        }
        method_rows = []
        for method_id, label in method_id_to_label.items():
            col = f"from_{method_id}"
            if col in raw.columns:
                method_flags = raw[col].map(
                    lambda value: str(value).strip().lower() in {"true", "1", "yes"}
                    if not isinstance(value, bool) else value
                )
                count = int(method_flags.sum())
                method_rows.append({"Enzyme": label, "Count": count})
        method_df = pd.DataFrame(method_rows)
        if not method_df.empty:
            method_df["Enzyme"] = pd.Categorical(method_df["Enzyme"], categories=fixed_enzyme_order, ordered=True)
            method_df = method_df.sort_values("Enzyme")

        if "hydrolysis_count" in raw.columns:
            overlap_df = (
                pd.to_numeric(raw["hydrolysis_count"], errors="coerce")
                .dropna()
                .astype(int)
                .value_counts()
                .rename_axis("Hydrolysis methods per peptide")
                .reset_index(name="Count")
                .sort_values("Hydrolysis methods per peptide")
            )
            overlap_df["Method count label"] = overlap_df["Hydrolysis methods per peptide"].astype(str)
        else:
            overlap_df = pd.DataFrame(columns=["Hydrolysis methods per peptide", "Count", "Method count label"])

        summary_df = pd.DataFrame([{
            "Peptides": len(raw),
            "Mean probability": float(raw["Final_Prob"].mean()) if "Final_Prob" in raw.columns else 0.0,
            "Mean length": float(raw["length"].mean()) if "length" in raw.columns else 0.0,
        }])

        return {
            "summary": summary_df,
            "amino_acids": aa_df,
            "dipeptides": dipeptide_df,
            "terminals": terminal_df,
            "lengths": length_df,
            "probabilities": probability_df,
            "methods": method_df,
            "overlap": overlap_df,
        }

    def nice_axis_limit(value: float) -> int:
        if not value or pd.isna(value) or value <= 0:
            return 100
        raw = value * 1.16
        exponent = math.floor(math.log10(raw))
        fraction = raw / (10 ** exponent)
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10
        return int(nice_fraction * (10 ** exponent))

    def render_confidence_chart(
        title: str,
        summary_df: pd.DataFrame,
        reported_df: pd.DataFrame,
        tier_label: str,
        color: str,
        confidence_options: Dict[str, str],
        show_title: bool = True,
    ) -> None:
        tier_col = confidence_options[tier_label]
        if show_title and title:
            st.markdown(f"#### {title}")
        source_df = reported_df if tier_label in {"Reported", "No reported"} else summary_df
        if source_df.empty or tier_col not in source_df.columns or "method_name" not in source_df.columns:
            st.info(f"{title} prediction output has not been added yet. It will use the same virtual hydrolysate peptide library.")
            return

        chart_df = source_df[["method_name", tier_col]].copy()
        chart_df[tier_col] = pd.to_numeric(chart_df[tier_col], errors="coerce").fillna(0).astype(int)
        chart_df = chart_df.rename(columns={"method_name": "Enzyme full name", tier_col: "Count"})
        chart_df["Enzyme"] = chart_df["Enzyme full name"].map(lambda name: enzyme_display_names.get(str(name), str(name)))
        chart_df["Confidence tier"] = tier_label
        extra_enzymes = [name for name in chart_df["Enzyme"].tolist() if name not in fixed_enzyme_order]
        enzyme_order = fixed_enzyme_order + extra_enzymes
        chart_df["Enzyme"] = pd.Categorical(chart_df["Enzyme"], categories=enzyme_order, ordered=True)
        chart_df = chart_df.sort_values("Enzyme")

        if tier_label.startswith("Pro"):
            tier_cols = [
                col for label, col in confidence_options.items()
                if label.startswith("Pro") and col in summary_df.columns
            ]
            all_tier_max = pd.to_numeric(summary_df[tier_cols].stack(), errors="coerce").max() if tier_cols else chart_df["Count"].max()
            y_limit = nice_axis_limit(float(all_tier_max))
        else:
            y_limit = nice_axis_limit(float(chart_df["Count"].max()))

        max_count = chart_df["Count"].max()
        min_count = chart_df["Count"].min()
        label_df = pd.concat([
            chart_df[chart_df["Count"].eq(max_count)].head(1),
            chart_df[chart_df["Count"].eq(min_count)].head(1),
        ]).drop_duplicates(subset=["Enzyme"])
        mean_value = float(chart_df["Count"].mean()) if not chart_df.empty else 0.0
        mean_df = pd.DataFrame({"Mean": [mean_value], "Enzyme": [enzyme_order[-1]], "Mean label": [f"{mean_value:.2f}"]})
        axis_label_df = pd.DataFrame([
            {"Enzyme": enzyme_order[0], "Count": y_limit, "Label": "Count", "Kind": "y"},
            {"Enzyme": enzyme_order[-1], "Count": 0, "Label": "Enzyme", "Kind": "x"},
        ])

        bars = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, color=color, size=46)
            .encode(
                x=alt.X(
                    "Enzyme:N",
                    sort=enzyme_order,
                    title=None,
                    scale=alt.Scale(paddingInner=0.42, paddingOuter=0.22),
                    axis=alt.Axis(
                        labelAngle=-45,
                        labelLimit=460,
                        labelOverlap=False,
                        labelFontSize=15,
                        tickSize=7,
                    ),
                ),
                y=alt.Y(
                    "Count:Q",
                    title=None,
                    scale=alt.Scale(domain=[0, y_limit], nice=False),
                    axis=alt.Axis(format=",.0f", tickCount=6, grid=True, labelFontSize=15),
                ),
                tooltip=[
                    alt.Tooltip("Enzyme full name:N", title="Enzyme"),
                    alt.Tooltip("Count:Q", format=","),
                    alt.Tooltip("Confidence tier:N"),
                ],
            )
        )
        mean_rule = (
            alt.Chart(mean_df)
            .mark_rule(stroke=color, strokeDash=[6, 4], opacity=0.75)
            .encode(y=alt.Y("Mean:Q", scale=alt.Scale(domain=[0, y_limit], nice=False)))
        )
        mean_text = (
            alt.Chart(mean_df)
            .mark_text(align="left", baseline="middle", dx=54, color=color, fontSize=15)
            .encode(
                x=alt.X("Enzyme:N", sort=enzyme_order),
                y=alt.Y("Mean:Q", scale=alt.Scale(domain=[0, y_limit], nice=False)),
                text="Mean label:N",
            )
        )
        extrema_text = (
            alt.Chart(label_df)
            .mark_text(align="center", baseline="bottom", dy=-10, color=color, fontSize=16, fontWeight="bold")
            .encode(
                x=alt.X("Enzyme:N", sort=enzyme_order),
                y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, y_limit], nice=False)),
                text=alt.Text("Count:Q", format=","),
            )
        )
        count_title = (
            alt.Chart(axis_label_df[axis_label_df["Kind"].eq("y")])
            .mark_text(align="right", baseline="bottom", dx=-48, dy=-12, color="#2F2F2F", fontSize=17)
            .encode(
                x=alt.X("Enzyme:N", sort=enzyme_order),
                y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, y_limit], nice=False)),
                text="Label:N",
            )
        )
        enzyme_title = (
            alt.Chart(axis_label_df[axis_label_df["Kind"].eq("x")])
            .mark_text(align="left", baseline="middle", dx=76, dy=0, color="#2F2F2F", fontSize=17)
            .encode(
                x=alt.X("Enzyme:N", sort=enzyme_order),
                y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, y_limit], nice=False)),
                text="Label:N",
            )
        )
        chart = (
            bars + mean_rule + mean_text + extrema_text + count_title + enzyme_title
        ).properties(
            height=620,
            padding={"left": 74, "right": 110, "bottom": 138, "top": 42},
        ).configure_view(strokeWidth=0)
        st.altair_chart(chart, use_container_width=True)

    def render_centered_chart(
        chart: alt.Chart,
        width: int = 0,
        use_container_width: bool = False,
    ) -> None:
        prepared_chart = chart.properties(width=width) if width and not use_container_width else chart
        st.altair_chart(prepared_chart, use_container_width=use_container_width)

    def chart_heading(title: str, caption: str = "") -> None:
        st.markdown(f"##### {title}")
        if caption:
            st.caption(caption)

    def render_centered_image(path: Path, caption: str, alt_text: str = "") -> None:
        uri = image_data_uri(path)
        if not uri:
            return
        safe_caption = html.escape(caption)
        safe_alt = html.escape(alt_text or caption)
        st.markdown(
            f"""
            <div class="centered-image-row">
                <div class="centered-image-block">
                    <img src="{uri}" alt="{safe_alt}">
                    <div class="centered-image-caption">{safe_caption}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_meme_logo_grid(items: List[Tuple[Path, str, str]]) -> None:
        blocks = []
        for path, caption, alt_text in items:
            uri = image_data_uri(path)
            if not uri:
                continue
            safe_caption = html.escape(caption)
            safe_alt = html.escape(alt_text or caption)
            blocks.append(
                f'<div class="meme-logo-item"><img src="{uri}" alt="{safe_alt}">'
                f'<div class="meme-logo-caption">{safe_caption}</div></div>'
            )
        if not blocks:
            st.info("MEME motif logo files have not been added yet.")
            return
        st.markdown(
            f'<div class="meme-logo-grid-wrap"><div class="meme-logo-grid">{"".join(blocks)}</div></div>',
            unsafe_allow_html=True,
        )

    def render_high_confidence_panel(
        task_label: str,
        high_conf_dir: Path,
        analysis_file_name: str,
        summary_file_name: str,
        cutoff_label: str,
        primary_color: str,
        secondary_color: str,
        minlen_summary_file_name: Optional[str] = None,
    ) -> None:
        task_lower = task_label.lower()
        meme_output_dir = high_conf_dir / "meme_output"
        summary = load_json_file(high_conf_dir / summary_file_name)
        minlen_summary = load_json_file(high_conf_dir / minlen_summary_file_name) if minlen_summary_file_name else {}
        meme_motif_summary = load_csv(meme_output_dir / "meme_motif_summary.csv")
        analysis_file = high_conf_dir / analysis_file_name
        analysis_data = load_high_confidence_analysis(str(analysis_file), file_cache_signature(analysis_file))

        summary_stats = analysis_data.get("summary", pd.DataFrame())
        computed_rows = int(summary_stats["Peptides"].iloc[0]) if not summary_stats.empty and "Peptides" in summary_stats.columns else 0
        high_conf_rows = int(summary.get("high_confidence_rows", computed_rows or 0))
        meme_ready_rows = int(
            minlen_summary.get(
                "high_confidence_rows",
                summary.get("high_confidence_minlen8_rows", 0),
            )
            or 0
        )
        mean_prob = float(summary.get("mean_final_prob", summary_stats["Mean probability"].iloc[0] if not summary_stats.empty else 0.0) or 0.0)
        mean_len = float(summary.get("mean_length", summary_stats["Mean length"].iloc[0] if not summary_stats.empty else 0.0) or 0.0)

        st.markdown(f"##### {task_label} peptide high-confidence set")
        u1, u2, u3, u4 = st.columns(4)
        with u1:
            metric_box("Confidence cutoff", cutoff_label)
        with u2:
            metric_box("High-confidence peptides", f"{high_conf_rows:,}" if high_conf_rows else "Pending")
        with u3:
            metric_box("MEME-ready peptides", f"{meme_ready_rows:,}" if meme_ready_rows else "Pending")
        with u4:
            metric_box("Mean length", f"{mean_len:.2f} aa" if mean_len else "Pending")

        if not analysis_data:
            st.info(f"High-confidence {task_lower} analysis data are not available yet.")
            return

        aa_df = analysis_data["amino_acids"]
        aa_chart = (
            alt.Chart(aa_df)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=primary_color, size=28)
            .encode(
                x=alt.X(
                    "Residue:N",
                    sort=AA_LIST,
                    title="Amino acid",
                    axis=alt.Axis(labelFontSize=13, titleFontSize=15, labelAngle=0),
                ),
                y=alt.Y(
                    "Frequency (%):Q",
                    title="Frequency (%)",
                    axis=alt.Axis(format=".1f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                ),
                tooltip=[
                    alt.Tooltip("Residue:N"),
                    alt.Tooltip("Count:Q", format=","),
                    alt.Tooltip("Frequency (%):Q", format=".2f"),
                ],
            )
            .properties(height=340)
            .configure_view(strokeWidth=0)
        )
        chart_heading("Amino acid composition", f"Residue frequencies among high-confidence {task_lower} peptides.")
        render_centered_chart(aa_chart, width=760)

        st.divider()
        dipeptide_df = analysis_data["dipeptides"]
        dipeptide_chart = (
            alt.Chart(dipeptide_df)
            .mark_rect()
            .encode(
                x=alt.X(
                    "Second residue:N",
                    sort=AA_LIST,
                    title="Second residue",
                    axis=alt.Axis(labelAngle=0, labelFontSize=12, titleFontSize=14),
                ),
                y=alt.Y(
                    "First residue:N",
                    sort=AA_LIST,
                    title="First residue",
                    axis=alt.Axis(labelFontSize=12, titleFontSize=14),
                ),
                color=alt.Color(
                    "Frequency per 1,000:Q",
                    title="Frequency per 1,000",
                    scale=alt.Scale(scheme="blues" if task_label == "Bitter" else "reds"),
                ),
                tooltip=[
                    alt.Tooltip("Dipeptide:N"),
                    alt.Tooltip("Count:Q", format=","),
                    alt.Tooltip("Frequency per 1,000:Q", format=".2f"),
                ],
            )
            .properties(height=620)
            .configure_view(strokeWidth=0)
        )
        chart_heading("Dipeptide composition", "Heatmap of all 400 amino-acid pairs in the high-confidence set.")
        render_centered_chart(dipeptide_chart, width=620)

        st.divider()
        terminal_df = analysis_data["terminals"]
        terminal_chart = (
            alt.Chart(terminal_df)
            .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X(
                    "Residue:N",
                    sort=AA_LIST,
                    title="Terminal residue",
                    axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=15),
                ),
                xOffset=alt.XOffset("Terminal:N"),
                y=alt.Y(
                    "Frequency (%):Q",
                    title="Frequency (%)",
                    axis=alt.Axis(format=".1f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                ),
                color=alt.Color(
                    "Terminal:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["N-terminal", "C-terminal"],
                        range=[primary_color, secondary_color],
                    ),
                    legend=alt.Legend(orient="top", labelFontSize=13),
                ),
                tooltip=[
                    alt.Tooltip("Terminal:N"),
                    alt.Tooltip("Residue:N"),
                    alt.Tooltip("Count:Q", format=","),
                    alt.Tooltip("Frequency (%):Q", format=".2f"),
                ],
            )
            .properties(height=350)
            .configure_view(strokeWidth=0)
        )
        chart_heading("N/C-terminal residue preference", "Paired frequencies of residues at the N-terminus and C-terminus.")
        render_centered_chart(terminal_chart, width=760)

        st.divider()
        length_df = analysis_data["lengths"]
        length_labels = length_df["Length label"].tolist()
        length_chart = (
            alt.Chart(length_df)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#39A78E", size=30)
            .encode(
                x=alt.X(
                    "Length label:N",
                    sort=length_labels,
                    title="Peptide length (aa)",
                    axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=15),
                ),
                y=alt.Y(
                    "Count:Q",
                    title="Count",
                    axis=alt.Axis(format=",.0f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                ),
                tooltip=[
                    alt.Tooltip("Length:Q", format=".0f"),
                    alt.Tooltip("Count:Q", format=","),
                ],
            )
            .properties(height=330)
            .configure_view(strokeWidth=0)
        )
        chart_heading("Peptide length distribution", f"Length distribution of all high-confidence {task_lower} peptides.")
        render_centered_chart(length_chart, width=700)

        st.divider()
        probability_df = analysis_data["probabilities"]
        if not probability_df.empty:
            probability_chart = (
                alt.Chart(probability_df)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#6C8AE4")
                .encode(
                    x=alt.X(
                        "Probability bin:N",
                        sort=probability_df["Probability bin"].tolist(),
                        title="Final_Prob range",
                        axis=alt.Axis(labelAngle=-35, labelFontSize=10, titleFontSize=15, labelLimit=120),
                    ),
                    y=alt.Y(
                        "Count:Q",
                        title="Count",
                        axis=alt.Axis(format=",.0f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                    ),
                    tooltip=[
                        alt.Tooltip("Probability bin:N"),
                        alt.Tooltip("Count:Q", format=","),
                    ],
                )
                .properties(height=330)
                .configure_view(strokeWidth=0)
            )
            chart_heading("Prediction probability distribution", f"Final model probabilities within the high-confidence set; mean Final_Prob = {mean_prob:.6f}.")
            render_centered_chart(probability_chart, width=760)

        st.divider()
        method_df = analysis_data["methods"]
        if not method_df.empty:
            method_y_limit = nice_axis_limit(float(method_df["Count"].max()))
            method_bars = (
                alt.Chart(method_df)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=primary_color, size=36)
                .encode(
                    x=alt.X(
                        "Enzyme:N",
                        sort=fixed_enzyme_order,
                        title="Enzyme",
                        axis=alt.Axis(labelAngle=-35, labelFontSize=12, titleFontSize=15, labelLimit=260),
                    ),
                    y=alt.Y(
                        "Count:Q",
                        title="Count",
                        scale=alt.Scale(domain=[0, method_y_limit], nice=False),
                        axis=alt.Axis(format=",.0f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                    ),
                    tooltip=[
                        alt.Tooltip("Enzyme:N"),
                        alt.Tooltip("Count:Q", format=","),
                    ],
                )
            )
            method_text = (
                alt.Chart(method_df)
                .mark_text(align="center", baseline="bottom", dy=-8, color="#1f2937", fontSize=12, fontWeight="bold")
                .encode(
                    x=alt.X("Enzyme:N", sort=fixed_enzyme_order),
                    y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, method_y_limit], nice=False)),
                    text=alt.Text("Count:Q", format=","),
                )
            )
            method_chart = (method_bars + method_text).properties(height=390).configure_view(strokeWidth=0)
            chart_heading("Hydrolysis-source contribution", "Number of high-confidence peptides generated by each hydrolysis method.")
            render_centered_chart(method_chart, width=760)

        st.divider()
        overlap_df = analysis_data["overlap"]
        if not overlap_df.empty:
            overlap_y_limit = nice_axis_limit(float(overlap_df["Count"].max()))
            overlap_bars = (
                alt.Chart(overlap_df)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=secondary_color, size=38)
                .encode(
                    x=alt.X(
                        "Method count label:N",
                        sort=overlap_df["Method count label"].tolist(),
                        title="Hydrolysis methods per peptide",
                        axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=15),
                    ),
                    y=alt.Y(
                        "Count:Q",
                        title="Count",
                        scale=alt.Scale(domain=[0, overlap_y_limit], nice=False),
                        axis=alt.Axis(format=",.0f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                    ),
                    tooltip=[
                        alt.Tooltip("Hydrolysis methods per peptide:Q", format=".0f"),
                        alt.Tooltip("Count:Q", format=","),
                    ],
                )
            )
            overlap_text = (
                alt.Chart(overlap_df)
                .mark_text(align="center", baseline="bottom", dy=-8, color="#1f2937", fontSize=12, fontWeight="bold")
                .encode(
                    x=alt.X("Method count label:N", sort=overlap_df["Method count label"].tolist()),
                    y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, overlap_y_limit], nice=False)),
                    text=alt.Text("Count:Q", format=","),
                )
            )
            overlap_chart = (overlap_bars + overlap_text).properties(height=330).configure_view(strokeWidth=0)
            chart_heading("Hydrolysis-source overlap", "How many hydrolysis methods produced the same high-confidence peptide.")
            render_centered_chart(overlap_chart, width=560)

        st.divider()
        chart_heading(
            "MEME motif logos",
            f"Non-redundant MEME motif logos from high-confidence {task_lower} peptides with length >= 8 aa.",
        )
        if meme_motif_summary.empty:
            logo_items = [
                (logo_path, logo_path.stem, f"MEME motif logo {logo_path.stem}")
                for logo_path in sorted(meme_output_dir.glob("*.png"), key=lambda path: (len(path.stem), path.stem))
            ]
            render_meme_logo_grid(logo_items)
            return

        motif_view = meme_motif_summary.copy()
        motif_view["e_value"] = pd.to_numeric(motif_view.get("e_value"), errors="coerce")
        motif_view["sites"] = pd.to_numeric(motif_view.get("sites"), errors="coerce")
        motif_view["width"] = pd.to_numeric(motif_view.get("width"), errors="coerce")
        motif_view = motif_view.sort_values(["e_value", "regular_expression"]).drop_duplicates("regular_expression")

        logo_items = []
        for _, motif_row in motif_view.iterrows():
            motif_name = str(motif_row.get("regular_expression") or motif_row.get("motif_name") or "").strip()
            if not motif_name:
                continue
            logo_path = meme_output_dir / f"{motif_name}.png"
            if not logo_path.exists():
                continue
            e_value = motif_row.get("e_value")
            sites = motif_row.get("sites")
            width = motif_row.get("width")
            e_label = f"{float(e_value):.2g}" if pd.notna(e_value) else "NA"
            sites_label = f"{int(sites)}" if pd.notna(sites) else "NA"
            width_label = f"{int(width)}" if pd.notna(width) else "NA"
            caption = f"{motif_name} | width {width_label} | sites {sites_label} | E-value {e_label}"
            logo_items.append((logo_path, caption, f"MEME motif logo {motif_name}"))
        render_meme_logo_grid(logo_items)

    section("Screening Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_box("Protein sources", f"{protein_count:,}")
    with c2:
        metric_box("Species", f"{species_count:,}")
    with c3:
        metric_box("Hydrolysis methods", f"{method_count:,}")
    with c4:
        metric_box("Unique peptide pool", f"{peptide_pool:,}" if peptide_pool else "Pending")

    tab_overview, tab_high_conf, tab_downloads = st.tabs(
        ["Overview", "High-confidence Analysis", "Downloads"]
    )

    with tab_overview:
        st.markdown(
            """
            <div class="virtual-overview-note">
                <strong>Virtual hydrolysis screening library.</strong>
                This page summarizes a peptide library generated by simulating enzymatic hydrolysis of
                <strong>21,249 proteins from 60 species</strong>. Different enzymes and enzyme combinations
                are used to generate hydrolyzed fragments, illegal sequences are removed, and the remaining
                peptides serve as the shared screening pool for both umami and bitter prediction models.
                <strong>Reported</strong> indicates peptides already matched to Database.csv records for the
                corresponding taste task; <strong>No reported</strong> indicates candidates not found in those
                reported task-specific records.
            </div>
            """,
            unsafe_allow_html=True,
        )

        bitter_default_tier = "No reported"
        bitter_tier_labels = list(bitter_confidence_options.keys())
        bitter_default_tier_index = bitter_tier_labels.index(bitter_default_tier) if bitter_default_tier in bitter_tier_labels else len(bitter_tier_labels) - 1

        st.markdown("#### Bitter Prediction")
        bitter_filter_left, bitter_filter_center, bitter_filter_right = st.columns([0.23, 0.54, 0.23])
        with bitter_filter_center:
            bitter_tier = st.radio(
                "Bitter chart filter",
                bitter_tier_labels,
                index=bitter_default_tier_index,
                horizontal=True,
                label_visibility="collapsed",
                key="virtual_overview_bitter_confidence_tier",
            )
        render_confidence_chart("Bitter Prediction", bitter_method_summary, bitter_reported_summary, bitter_tier, "#2F9AD6", bitter_confidence_options, show_title=False)

        st.divider()
        umami_default_tier = "No reported"
        umami_tier_labels = list(umami_confidence_options.keys())
        umami_default_tier_index = umami_tier_labels.index(umami_default_tier) if umami_default_tier in umami_tier_labels else len(umami_tier_labels) - 1

        st.markdown("#### Umami Prediction")
        umami_filter_left, umami_filter_center, umami_filter_right = st.columns([0.23, 0.54, 0.23])
        with umami_filter_center:
            umami_tier = st.radio(
                "Umami chart filter",
                umami_tier_labels,
                index=umami_default_tier_index,
                horizontal=True,
                label_visibility="collapsed",
                key="virtual_overview_umami_confidence_tier",
            )
        render_confidence_chart("Umami Prediction", method_summary, reported_summary, umami_tier, "#D65A66", umami_confidence_options, show_title=False)

        with st.expander("Workflow used for this release", expanded=False):
            st.markdown(
                """
                1. Simulate hydrolysis of 21,249 proteins from 60 species with different enzymes and enzyme combinations.
                2. Remove illegal peptide sequences and merge repeated fragments.
                3. Score the resulting peptide fragments with taste prediction models.
                4. Summarize candidate counts by hydrolysis method and prediction confidence tier.
                5. Present high-confidence statistics and figures separately from download buttons.
                """
            )

    with tab_high_conf:
        st.markdown("#### High-confidence Candidate Analysis")
        target_left, target_center, target_right = st.columns([0.14, 0.72, 0.14])
        with target_center:
            selected_analysis_target = st.radio(
                "High-confidence analysis target",
                ["Bitter peptide", "Umami peptide"],
                index=1,
                horizontal=True,
                label_visibility="collapsed",
                key="virtual_high_confidence_target",
            )

        if selected_analysis_target == "Umami peptide":
            high_conf_dir = umami_root / "high_confidence_umami"
            meme_output_dir = high_conf_dir / "meme_output"
            umami_summary = load_json_file(high_conf_dir / "high_confidence_umami_gte_0.999_summary.json")
            umami_minlen8_summary = load_json_file(high_conf_dir / "high_confidence_umami_gte_0.999_minlen8_summary.json")
            meme_motif_summary = load_csv(meme_output_dir / "meme_motif_summary.csv")
            analysis_file = high_conf_dir / "high_confidence_umami_gte_0.999.csv.gz"
            analysis_data = load_high_confidence_analysis(str(analysis_file), file_cache_signature(analysis_file))

            summary_stats = analysis_data.get("summary", pd.DataFrame())
            computed_rows = int(summary_stats["Peptides"].iloc[0]) if not summary_stats.empty and "Peptides" in summary_stats.columns else 0
            high_conf_rows = int(umami_summary.get("high_confidence_rows", computed_rows or umami_high_conf_999 or 0))
            meme_ready_rows = int(umami_minlen8_summary.get("high_confidence_rows", 0))
            mean_prob = float(umami_summary.get("mean_final_prob", summary_stats["Mean probability"].iloc[0] if not summary_stats.empty else 0.0) or 0.0)
            mean_len = float(umami_summary.get("mean_length", summary_stats["Mean length"].iloc[0] if not summary_stats.empty else 0.0) or 0.0)

            st.markdown("##### Umami peptide high-confidence set")
            u1, u2, u3, u4 = st.columns(4)
            with u1:
                metric_box("Confidence cutoff", "Final_Prob >= 0.999")
            with u2:
                metric_box("High-confidence peptides", f"{high_conf_rows:,}" if high_conf_rows else "Pending")
            with u3:
                metric_box("MEME-ready peptides", f"{meme_ready_rows:,}" if meme_ready_rows else "Pending")
            with u4:
                metric_box("Mean length", f"{mean_len:.2f} aa" if mean_len else "Pending")

            if not analysis_data:
                st.info("High-confidence umami analysis data are not available yet.")
            else:
                aa_df = analysis_data["amino_acids"]
                aa_chart = (
                    alt.Chart(aa_df)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#D65A66", size=28)
                    .encode(
                        x=alt.X(
                            "Residue:N",
                            sort=AA_LIST,
                            title="Amino acid",
                            axis=alt.Axis(labelFontSize=13, titleFontSize=15, labelAngle=0),
                        ),
                        y=alt.Y(
                            "Frequency (%):Q",
                            title="Frequency (%)",
                            axis=alt.Axis(format=".1f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                        ),
                        tooltip=[
                            alt.Tooltip("Residue:N"),
                            alt.Tooltip("Count:Q", format=","),
                            alt.Tooltip("Frequency (%):Q", format=".2f"),
                        ],
                    )
                    .properties(height=340)
                    .configure_view(strokeWidth=0)
                )
                chart_heading("Amino acid composition", "Residue frequencies among high-confidence umami peptides.")
                render_centered_chart(aa_chart, width=760)

                st.divider()
                dipeptide_df = analysis_data["dipeptides"]
                dipeptide_chart = (
                    alt.Chart(dipeptide_df)
                    .mark_rect()
                    .encode(
                        x=alt.X(
                            "Second residue:N",
                            sort=AA_LIST,
                            title="Second residue",
                            axis=alt.Axis(labelAngle=0, labelFontSize=12, titleFontSize=14),
                        ),
                        y=alt.Y(
                            "First residue:N",
                            sort=AA_LIST,
                            title="First residue",
                            axis=alt.Axis(labelFontSize=12, titleFontSize=14),
                        ),
                        color=alt.Color(
                            "Frequency per 1,000:Q",
                            title="Frequency per 1,000",
                            scale=alt.Scale(scheme="reds"),
                        ),
                        tooltip=[
                            alt.Tooltip("Dipeptide:N"),
                            alt.Tooltip("Count:Q", format=","),
                            alt.Tooltip("Frequency per 1,000:Q", format=".2f"),
                        ],
                    )
                    .properties(height=620)
                    .configure_view(strokeWidth=0)
                )
                chart_heading("Dipeptide composition", "Heatmap of all 400 amino-acid pairs in the high-confidence set.")
                render_centered_chart(dipeptide_chart, width=620)

                st.divider()
                terminal_df = analysis_data["terminals"]
                terminal_chart = (
                    alt.Chart(terminal_df)
                    .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                    .encode(
                        x=alt.X(
                            "Residue:N",
                            sort=AA_LIST,
                            title="Terminal residue",
                            axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=15),
                        ),
                        xOffset=alt.XOffset("Terminal:N"),
                        y=alt.Y(
                            "Frequency (%):Q",
                            title="Frequency (%)",
                            axis=alt.Axis(format=".1f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                        ),
                        color=alt.Color(
                            "Terminal:N",
                            title=None,
                            scale=alt.Scale(
                                domain=["N-terminal", "C-terminal"],
                                range=["#D65A66", "#F59E0B"],
                            ),
                            legend=alt.Legend(orient="top", labelFontSize=13),
                        ),
                        tooltip=[
                            alt.Tooltip("Terminal:N"),
                            alt.Tooltip("Residue:N"),
                            alt.Tooltip("Count:Q", format=","),
                            alt.Tooltip("Frequency (%):Q", format=".2f"),
                        ],
                    )
                    .properties(height=350)
                    .configure_view(strokeWidth=0)
                )
                chart_heading("N/C-terminal residue preference", "Paired frequencies of residues at the N-terminus and C-terminus.")
                render_centered_chart(terminal_chart, width=760)

                st.divider()
                length_df = analysis_data["lengths"]
                length_labels = length_df["Length label"].tolist()
                length_chart = (
                    alt.Chart(length_df)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#39A78E", size=30)
                    .encode(
                        x=alt.X(
                            "Length label:N",
                            sort=length_labels,
                            title="Peptide length (aa)",
                            axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=15),
                        ),
                        y=alt.Y(
                            "Count:Q",
                            title="Count",
                            axis=alt.Axis(format=",.0f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                        ),
                        tooltip=[
                            alt.Tooltip("Length:Q", format=".0f"),
                            alt.Tooltip("Count:Q", format=","),
                        ],
                    )
                    .properties(height=330)
                    .configure_view(strokeWidth=0)
                )
                chart_heading("Peptide length distribution", "Length distribution of all high-confidence umami peptides.")
                render_centered_chart(length_chart, width=700)

                st.divider()
                probability_df = analysis_data["probabilities"]
                if not probability_df.empty:
                    probability_chart = (
                        alt.Chart(probability_df)
                        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#6C8AE4")
                        .encode(
                            x=alt.X(
                                "Probability bin:N",
                                sort=probability_df["Probability bin"].tolist(),
                                title="Final_Prob range",
                                axis=alt.Axis(labelAngle=-35, labelFontSize=10, titleFontSize=15, labelLimit=120),
                            ),
                            y=alt.Y(
                                "Count:Q",
                                title="Count",
                                axis=alt.Axis(format=",.0f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                            ),
                            tooltip=[
                                alt.Tooltip("Probability bin:N"),
                                alt.Tooltip("Count:Q", format=","),
                            ],
                        )
                        .properties(height=330)
                        .configure_view(strokeWidth=0)
                    )
                    chart_heading("Prediction probability distribution", f"Final model probabilities within the high-confidence set; mean Final_Prob = {mean_prob:.6f}.")
                    render_centered_chart(probability_chart, width=760)

                st.divider()
                method_df = analysis_data["methods"]
                if not method_df.empty:
                    method_y_limit = nice_axis_limit(float(method_df["Count"].max()))
                    method_bars = (
                        alt.Chart(method_df)
                        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#D65A66", size=36)
                        .encode(
                            x=alt.X(
                                "Enzyme:N",
                                sort=fixed_enzyme_order,
                                title="Enzyme",
                                axis=alt.Axis(labelAngle=-35, labelFontSize=12, titleFontSize=15, labelLimit=260),
                            ),
                            y=alt.Y(
                                "Count:Q",
                                title="Count",
                                scale=alt.Scale(domain=[0, method_y_limit], nice=False),
                                axis=alt.Axis(format=",.0f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                            ),
                            tooltip=[
                                alt.Tooltip("Enzyme:N"),
                                alt.Tooltip("Count:Q", format=","),
                            ],
                        )
                    )
                    method_text = (
                        alt.Chart(method_df)
                        .mark_text(align="center", baseline="bottom", dy=-8, color="#1f2937", fontSize=12, fontWeight="bold")
                        .encode(
                            x=alt.X("Enzyme:N", sort=fixed_enzyme_order),
                            y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, method_y_limit], nice=False)),
                            text=alt.Text("Count:Q", format=","),
                        )
                    )
                    method_chart = (method_bars + method_text).properties(height=390).configure_view(strokeWidth=0)
                    chart_heading("Hydrolysis-source contribution", "Number of high-confidence peptides generated by each hydrolysis method.")
                    render_centered_chart(method_chart, width=760)

                st.divider()
                overlap_df = analysis_data["overlap"]
                if not overlap_df.empty:
                    overlap_y_limit = nice_axis_limit(float(overlap_df["Count"].max()))
                    overlap_bars = (
                        alt.Chart(overlap_df)
                        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#F59E0B", size=38)
                        .encode(
                            x=alt.X(
                                "Method count label:N",
                                sort=overlap_df["Method count label"].tolist(),
                                title="Hydrolysis methods per peptide",
                                axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=15),
                            ),
                            y=alt.Y(
                                "Count:Q",
                                title="Count",
                                scale=alt.Scale(domain=[0, overlap_y_limit], nice=False),
                                axis=alt.Axis(format=",.0f", tickCount=6, labelFontSize=13, titleFontSize=15, grid=True),
                            ),
                            tooltip=[
                                alt.Tooltip("Hydrolysis methods per peptide:Q", format=".0f"),
                                alt.Tooltip("Count:Q", format=","),
                            ],
                        )
                    )
                    overlap_text = (
                        alt.Chart(overlap_df)
                        .mark_text(align="center", baseline="bottom", dy=-8, color="#1f2937", fontSize=12, fontWeight="bold")
                        .encode(
                            x=alt.X("Method count label:N", sort=overlap_df["Method count label"].tolist()),
                            y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, overlap_y_limit], nice=False)),
                            text=alt.Text("Count:Q", format=","),
                        )
                    )
                    overlap_chart = (overlap_bars + overlap_text).properties(height=330).configure_view(strokeWidth=0)
                    chart_heading("Hydrolysis-source overlap", "How many hydrolysis methods produced the same high-confidence peptide.")
                    render_centered_chart(overlap_chart, width=560)

                st.divider()
                chart_heading(
                    "MEME motif logos",
                    "Non-redundant MEME motif logos from high-confidence umami peptides with length >= 8 aa.",
                )
                if meme_motif_summary.empty:
                    st.info("MEME motif logo files have not been added yet.")
                else:
                    motif_view = meme_motif_summary.copy()
                    motif_view["e_value"] = pd.to_numeric(motif_view.get("e_value"), errors="coerce")
                    motif_view["sites"] = pd.to_numeric(motif_view.get("sites"), errors="coerce")
                    motif_view["width"] = pd.to_numeric(motif_view.get("width"), errors="coerce")
                    motif_view = motif_view.sort_values(["e_value", "regular_expression"]).drop_duplicates("regular_expression")

                    logo_items = []
                    for _, motif_row in motif_view.iterrows():
                        motif_name = str(motif_row.get("regular_expression") or motif_row.get("motif_name") or "").strip()
                        if not motif_name:
                            continue
                        logo_path = meme_output_dir / f"{motif_name}.png"
                        if not logo_path.exists():
                            continue
                        e_value = motif_row.get("e_value")
                        sites = motif_row.get("sites")
                        width = motif_row.get("width")
                        e_label = f"{float(e_value):.2g}" if pd.notna(e_value) else "NA"
                        sites_label = f"{int(sites)}" if pd.notna(sites) else "NA"
                        width_label = f"{int(width)}" if pd.notna(width) else "NA"
                        caption = f"{motif_name} | width {width_label} | sites {sites_label} | E-value {e_label}"
                        logo_items.append((logo_path, caption, f"MEME motif logo {motif_name}"))
                    render_meme_logo_grid(logo_items)
        else:
            render_high_confidence_panel(
                task_label="Bitter",
                high_conf_dir=bitter_root / "high_confidence_bitter",
                analysis_file_name="high_confidence_bitter_gte_0.93.csv.gz",
                summary_file_name="high_confidence_bitter_gte_0.93_summary.json",
                cutoff_label="Final_Prob >= 0.93",
                primary_color="#2F9AD6",
                secondary_color="#EF6F6C",
            )

    with tab_downloads:
        st.markdown("#### Download Center")
        st.caption("All available virtual screening output files are listed by prediction task and hydrolysis method.")

        def resolve_download_path(row: pd.Series) -> Path | None:
            for column in ["Filtered file", "Download_File", "Pool file"]:
                value = row.get(column)
                if pd.notna(value) and str(value).strip():
                    return VIRTUAL_SCREENING_DIR / str(value).strip()
            return None

        def format_count(value: Any) -> str:
            try:
                if pd.isna(value):
                    return "-"
                return f"{int(float(value)):,}"
            except (TypeError, ValueError):
                return "-"

        def render_virtual_download_table(download_df: pd.DataFrame) -> None:
            table_widths = [1.25, .55, .68, 1.55, .88]
            headers = ["Enzyme(s)", "Probability", "Unique sequences", "Prediction pools", "Download"]
            header_cols = st.columns(table_widths)
            for header_col, header in zip(header_cols, headers):
                header_col.markdown(f'<div class="virtual-download-header">{header}</div>', unsafe_allow_html=True)

            previous_method = None
            for row_number, (_, row) in enumerate(download_df.iterrows()):
                method_name = str(row.get("Hydrolysis_Method", row.get("Enzyme(s)", ""))).strip()
                if not method_name or method_name.lower() == "nan":
                    method_name = str(row.get("Enzyme(s)", "Hydrolysis method")).strip()
                probability = str(row.get("Probability group", row.get("Probability_Tier", ""))).strip()
                if not probability or probability.lower() == "nan":
                    probability = "Coming soon"
                sequence_count = format_count(row.get("Unique sequences", row.get("Count", 0)))
                file_path = resolve_download_path(row)
                file_name = file_path.name if file_path and file_path.exists() else "Pending"
                display_method = method_name if method_name != previous_method else ""
                previous_method = method_name

                row_cols = st.columns(table_widths)
                row_cols[0].markdown(
                    f'<div class="virtual-download-method">{html.escape(display_method)}</div>',
                    unsafe_allow_html=True,
                )
                row_cols[1].markdown(
                    f'<div class="virtual-download-cell">{html.escape(probability)}</div>',
                    unsafe_allow_html=True,
                )
                row_cols[2].markdown(
                    f'<div class="virtual-download-cell">{sequence_count}</div>',
                    unsafe_allow_html=True,
                )
                file_class = "virtual-download-file" if file_path and file_path.exists() else "virtual-download-muted"
                row_cols[3].markdown(
                    f'<div class="{file_class}">{html.escape(file_name)}</div>',
                    unsafe_allow_html=True,
                )
                with row_cols[4]:
                    if file_path and file_path.exists() and file_path.is_file():
                        st.download_button(
                            "Download",
                            data=file_path.read_bytes(),
                            file_name=file_path.name,
                            mime="application/gzip" if file_path.suffix == ".gz" else "application/octet-stream",
                            use_container_width=True,
                            key=f"virtual_download_{row_number}_{file_path.name}",
                        )
                    else:
                        st.button(
                            "Pending",
                            disabled=True,
                            use_container_width=True,
                            key=f"virtual_pending_{row_number}_{method_name}_{probability}",
                        )

        selector_left, selector_center, selector_right = st.columns([0.14, 0.72, 0.14])
        with selector_center:
            selected_download_target = st.radio(
                "Virtual download task",
                ["Bitter peptide", "Umami peptide"],
                index=1,
                horizontal=True,
                label_visibility="collapsed",
                key="virtual_download_task_target",
            )

        if manifest.empty and reported_download_manifest.empty:
            st.info("No virtual screening download records are available yet.")
        else:
            manifest_parts = [manifest] if not manifest.empty else []
            manifest_has_reported = (
                not manifest.empty
                and "Task" in manifest.columns
                and "Probability group" in manifest.columns
                and manifest["Task"].astype(str).eq("Umami").any()
                and manifest["Probability group"].astype(str).isin(["Reported", "No reported"]).any()
            )
            if not reported_download_manifest.empty and not manifest_has_reported:
                manifest_parts.append(reported_download_manifest)
            download_manifest = pd.concat(manifest_parts, ignore_index=True, sort=False)
            download_manifest["Task"] = download_manifest["Task"].fillna("Task").astype(str)
            method_order_map = {name: order for order, name in enumerate(enzyme_order_source)}
            tier_order_map = {
                "Reported": 0,
                "No reported": 1,
                "< 0.50": 2,
                ">= 0.50": 3,
                ">= 0.85": 4,
                ">= 0.90": 5,
                ">= 0.95": 6,
                "Coming soon": 99,
            }
            task_order_map = {"Bitter": 0, "Umami": 1}
            download_manifest["Task order"] = download_manifest["Task"].map(task_order_map).fillna(20)
            download_manifest["Method order"] = download_manifest["Hydrolysis_Method"].map(method_order_map).fillna(50)
            download_manifest["Tier order"] = download_manifest["Probability group"].map(tier_order_map).fillna(50)
            download_manifest = download_manifest.sort_values(["Task order", "Method order", "Tier order", "Hydrolysis_Method"])

            selected_task = "Bitter" if selected_download_target == "Bitter peptide" else "Umami"
            download_manifest = download_manifest[download_manifest["Task"].eq(selected_task)].copy()
            task_labels = {
                "Umami": "Umami peptide prediction",
                "Bitter": "Bitter peptide prediction",
            }
            if download_manifest.empty:
                st.info(f"{task_labels.get(selected_task, selected_task)} download files have not been added yet.")
            for task_name, task_df in download_manifest.groupby("Task", sort=False):
                ready_rows = [
                    row for _, row in task_df.iterrows()
                    if (path := resolve_download_path(row)) and path.exists() and path.is_file()
                ]
                task_title = task_labels.get(task_name, task_name)
                st.markdown(
                    f'<div class="virtual-download-task">{html.escape(task_title)}'
                    f'<span>{len(ready_rows):,} ready file(s)</span></div>',
                    unsafe_allow_html=True,
                )
                render_virtual_download_table(task_df)

def help_page() -> None:
    render_hero(
        "Help & Documentation",
        "A practical guide to peptide prediction, database exploration, benchmark dataset analysis, virtual screening, and result interpretation across Pep2Taste.",
        ["Quick start", "Input formats", "Result guide", "Platform map"]
    )

    section("Prediction Workflow")
    workflow_uri = image_data_uri(HELP_WORKFLOW_FIGURE)
    if workflow_uri:
        st.markdown(
            f"""
            <div class="help-workflow-figure">
                <img src="{workflow_uri}" alt="Four-step Pep2Taste peptide prediction workflow">
                <div class="help-workflow-caption">From peptide input to a probability-based classification and downloadable result table.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("The prediction workflow figure is not available.")

    section("Platform Guide")
    st.markdown(
        """
        <div class="help-guide-grid">
            <div class="help-guide-item" style="--guide-color:#d97706">
                <h3>Bitter Prediction</h3>
                <p>Use BPPred or one of its constituent models to classify bitter and non-bitter peptide sequences.</p>
            </div>
            <div class="help-guide-item" style="--guide-color:#dc2626">
                <h3>Umami Prediction</h3>
                <p>Use UPPred or an individual PLM-LoRA branch to classify umami and non-umami peptide sequences.</p>
            </div>
            <div class="help-guide-item" style="--guide-color:#2563eb">
                <h3>Database & Physicochemical Analysis</h3>
                <p>Search curated taste peptide records, filter taste classes, and compare computed sequence descriptors.</p>
            </div>
            <div class="help-guide-item" style="--guide-color:#16a34a">
                <h3>Datasets & Virtual Screening</h3>
                <p>Download benchmark datasets or inspect enzyme-level screening results, high-confidence candidates, and filtered libraries.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Input And Result Reference")
    input_tab, result_tab, screening_tab = st.tabs([
        "Prediction input",
        "Prediction results",
        "Virtual screening",
    ])
    with input_tab:
        st.markdown(
            """
            <div class="help-reference-grid">
                <div class="help-reference-item" style="--ref-color:#2563eb">
                    <h4>Sequence list</h4>
                    <p>Enter one peptide per line using the 20 standard amino acid letters.</p>
                    <p><code>GLLGFLG</code><br><code>RRPPGF</code></p>
                </div>
                <div class="help-reference-item" style="--ref-color:#d97706">
                    <h4>FASTA</h4>
                    <p>Use a header beginning with <code>&gt;</code>, followed by the peptide sequence.</p>
                    <p><code>&gt;pep_001</code><br><code>EEEEEL</code></p>
                </div>
                <div class="help-reference-item" style="--ref-color:#16a34a">
                    <h4>CSV / TXT</h4>
                    <p>CSV files should contain a <code>sequence</code> column. TXT files may contain FASTA records or one sequence per line.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Valid sequences contain 2-80 residues selected from A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, and Y.")

    with result_tab:
        st.markdown(
            """
            <div class="help-reference-grid">
                <div class="help-reference-item" style="--ref-color:#dc2626">
                    <h4>Probability</h4>
                    <p>A continuous model score from 0 to 1 for the selected taste class.</p>
                </div>
                <div class="help-reference-item" style="--ref-color:#64748b">
                    <h4>Threshold & label</h4>
                    <p>A peptide is assigned the positive class when its probability is greater than or equal to the selected threshold.</p>
                </div>
                <div class="help-reference-item" style="--ref-color:#7c3aed">
                    <h4>Branch probabilities</h4>
                    <p>Ensemble results also report constituent-model probabilities so the final decision can be inspected.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Prediction scores support candidate prioritization and should be interpreted together with experimental validation.")

    with screening_tab:
        st.markdown(
            """
            1. Use **Overview** to compare candidate counts by enzyme and probability tier.
            2. Use **High-confidence Analysis** to examine composition, terminal preference, length, probability, and motif summaries.
            3. Use **Downloads** to retrieve filtered bitter or umami peptide libraries by hydrolysis method.
            """
        )

    section("Common Questions")
    with st.expander("Which prediction model should I use?", expanded=False):
        st.markdown("Use **BPPred** for the final bitter peptide ensemble and **UPPred** for the final umami peptide ensemble. Select an individual branch when you need model-level comparison.")
    with st.expander("Why did the same sequence receive a different label after I changed the threshold?", expanded=False):
        st.markdown("The probability does not change, but the classification boundary does. Raising the threshold applies a stricter positive-class criterion.")
    with st.expander("Where can I inspect the model structures?", expanded=False):
        st.markdown("Open **Model architecture** above the prediction-model selector on either prediction page.")


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
        "Contact the Pep2Taste supervisor and developer, share suggestions for improving the platform, or contribute newly reported taste peptide sequences for future database curation.",
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
                <p>Submit taste peptide sequences with optional taste labels, literature sources, DOI information, or batch files for later manual curation.</p>
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
elif page == "Help":
    help_page()
elif page == "Contact":
    contact_page()

st.markdown('<div class="footer">Pep2Taste · AI-powered peptide taste prediction platform · Research use only</div>', unsafe_allow_html=True)
