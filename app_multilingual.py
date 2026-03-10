"""
NorthShield Insurance Association — Member Intelligence Platform
Multilingual Portfolio Demo App (English · Français · العربية)
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import glob
from pathlib import Path
from translations import t

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NorthShield — Member Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Language selection (must be first widget) ─────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# ── Custom CSS ────────────────────────────────────────────────────────────
def get_css(lang):
    rtl = "direction: rtl; text-align: right;" if lang == "ar" else ""
    arabic_font = "@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');" if lang == "ar" else ""
    body_font = "'Cairo', sans-serif" if lang == "ar" else "'DM Sans', sans-serif"
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
{arabic_font}

html, body, [class*="css"] {{
    font-family: {body_font};
    {rtl}
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0F1B35 0%, #1B3A6B 60%, #0F1B35 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}}
[data-testid="stSidebar"] * {{ color: #E8F0FF !important; }}
[data-testid="stSidebar"] .stRadio label {{
    font-family: {body_font};
    font-size: 0.92rem;
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
    transition: background 0.2s;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(255,255,255,0.08);
}}

.metric-card {{
    background: linear-gradient(135deg, #1B3A6B 0%, #2E4A8C 100%);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 4px 24px rgba(27,58,107,0.3);
}}
.metric-card .label {{
    font-size: 0.78rem;
    color: #A8C4E8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    margin-bottom: 0.3rem;
}}
.metric-card .value {{
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #FFFFFF;
    line-height: 1.1;
}}
.metric-card .sub {{
    font-size: 0.78rem;
    color: #7EC8A0;
    margin-top: 0.3rem;
    font-weight: 500;
}}

.result-card {{
    background: #FAFBFF;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid #E2E8F4;
    margin-bottom: 0.8rem;
    border-left: 4px solid #1B3A6B;
}}
.result-card.green {{ border-left-color: #1E6B3C; background: #F0FAF4; }}
.result-card.amber {{ border-left-color: #7D4E00; background: #FFFAF0; }}
.result-card.red   {{ border-left-color: #7B1A1A; background: #FFF5F5; }}

.section-header {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #1B3A6B;
    margin-bottom: 0.2rem;
}}

.code-block {{
    background: #1E1E2E;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #A6E22E;
    line-height: 1.6;
    border: 1px solid rgba(255,255,255,0.08);
    overflow-x: auto;
    direction: ltr;
    text-align: left;
}}

.prop-bar-bg {{
    background: #E8ECF4;
    border-radius: 4px;
    height: 8px;
    width: 100%;
}}

.page-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #1B3A6B;
    line-height: 1.15;
}}
.page-title span {{ color: #4A6FA5; font-style: italic; }}

.fancy-divider {{
    height: 2px;
    background: linear-gradient(90deg, #1B3A6B, #4A90D9, transparent);
    border: none;
    margin: 1.2rem 0;
    border-radius: 2px;
}}

.lang-btn {{
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    margin: 0.1rem;
}}
</style>
"""

# ── Helpers ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
lang = st.session_state.lang

def prop_bar(value):
    pct = min(int(value * 100), 100)
    color = "#1E6B3C" if value > 0.5 else "#1B3A6B" if value > 0.25 else "#7D4E00"
    return f"""
    <div class="prop-bar-bg">
      <div style="width:{pct}%;background:{color};border-radius:4px;height:8px"></div>
    </div>
    <small style="color:#6B7A99;font-size:0.72rem">{value:.3f}</small>
    """

# ── Data loading ──────────────────────────────────────────────────────────
@st.cache_data
def load_households():
    p = ROOT / "data/processed/households.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data
def load_features():
    p = ROOT / "data/processed/features.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data
def load_recommendations():
    p = ROOT / "data/processed/recommendations.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data
def load_model_metadata():
    meta = {}
    for f in glob.glob(str(ROOT / "models/artifacts/*_metadata.json")):
        with open(f) as fp:
            d = json.load(fp)
            meta[d.get("label", Path(f).stem)] = d
    return meta

@st.cache_resource
def load_classifiers():
    models = {}
    patterns = {
        "FSV CMSI": "fsv_cmsi_*.pkl", "FSV Credit Card": "fsv_credit_card_*.pkl",
        "FSV ID Theft": "fsv_id_theft_*.pkl", "INS Client": "ins_client_*.pkl",
        "TRV Globalware": "trv_globalware_*.pkl", "New Mover": "new_mover_*.pkl",
    }
    for name, pat in patterns.items():
        matches = glob.glob(str(ROOT / "models/artifacts" / pat))
        if matches:
            try:
                models[name] = joblib.load(sorted(matches)[0])
            except Exception:
                pass
    return models

df_hh   = load_households()
df_feat = load_features()
df_rec  = load_recommendations()
meta    = load_model_metadata()
models  = load_classifiers()

# ── Inject CSS ────────────────────────────────────────────────────────────
st.markdown(get_css(lang), unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    # Language switcher
    st.markdown(f"""
    <div style="padding:0.8rem 0 0.5rem 0">
      <div style="font-size:0.75rem;color:#A8C4E8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem">
        {t('sidebar_language', lang)}
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🇺🇸 EN", use_container_width=True,
                     type="primary" if lang == "en" else "secondary"):
            st.session_state.lang = "en"
            st.rerun()
    with col2:
        if st.button("🇫🇷 FR", use_container_width=True,
                     type="primary" if lang == "fr" else "secondary"):
            st.session_state.lang = "fr"
            st.rerun()
    with col3:
        if st.button("🇩🇿 AR", use_container_width=True,
                     type="primary" if lang == "ar" else "secondary"):
            st.session_state.lang = "ar"
            st.rerun()

    st.markdown("""<hr style="border:none;border-top:1px solid rgba(255,255,255,0.12);margin:0.8rem 0">""",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:0.5rem 0 1rem 0">
      <div style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:#FFFFFF;line-height:1.2">
        {t('app_title', lang)}
      </div>
      <div style="font-size:0.75rem;color:#A8C4E8;margin-top:0.3rem;letter-spacing:0.04em">
        {t('app_subtitle', lang)}
      </div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.12);margin:0 0 0.8rem 0">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Nav",
        [t("nav_overview", lang), t("nav_performance", lang), t("nav_segments", lang),
         t("nav_lookup", lang), t("nav_technical", lang)],
        label_visibility="collapsed"
    )

    st.markdown(f"""
    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.12);margin:1rem 0">
    <div style="font-size:0.72rem;color:#7090B0;line-height:1.8">
      <b style="color:#A8C4E8">{t('sidebar_dataset', lang)}</b><br>
      3,511 {"ménages" if lang=="fr" else "أسرة" if lang=="ar" else "households"}<br>
      21,344 {"enregistrements" if lang=="fr" else "سجلاً" if lang=="ar" else "raw records"}<br>
      140 {"variables" if lang=="fr" else "ميزة" if lang=="ar" else "features"}<br><br>
      <b style="color:#A8C4E8">{t('sidebar_models', lang)}</b><br>
      6 {"classificateurs" if lang=="fr" else "مصنفات" if lang=="ar" else "classifiers"}<br>
      1 {"régresseur" if lang=="fr" else "منحدر" if lang=="ar" else "regressor"}<br>
      1 K-Means<br><br>
      <b style="color:#A8C4E8">{t('sidebar_stack', lang)}</b><br>
      Python 3.11 · scikit-learn<br>
      XGBoost · pandas · Streamlit
K-Means · Huber Regressor
    </div>
    """, unsafe_allow_html=True)

# Refresh lang after sidebar interaction
lang = st.session_state.lang

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == t("nav_overview", lang):
    st.markdown(f'<div class="page-title">{t("ov_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.9rem;color:#6B7A99;margin-bottom:1.2rem">{t("ov_subtitle", lang)}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label_k, val, sub_k in [
        (c1, "kpi_households", "3,511", "kpi_households_sub"),
        (c2, "kpi_lift",       "5.71×", "kpi_lift_sub"),
        (c3, "kpi_r2",         "0.59",  "kpi_r2_sub"),
        (c4, "kpi_segments",   "6 / 8", "kpi_segments_sub"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
              <div class="label">{t(label_k, lang)}</div>
              <div class="value">{val}</div>
              <div class="sub">{t(sub_k, lang)}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(f"### {t('ov_what_title', lang)}")
        st.markdown(t("ov_what_body", lang))

    with col_r:
        st.markdown(f"### {t('ov_rec_title', lang)}")
        if df_rec is not None:
            dist = df_rec["recommended_product"].value_counts().reset_index()
            dist.columns = ["Product", "Households"]
            dist["Share"] = (dist["Households"] / dist["Households"].sum() * 100).round(1)
            for _, row in dist.iterrows():
                color = "#1E6B3C" if "Nurture" not in row["Product"] else "#7D4E00"
                pct = int(row["Share"])
                st.markdown(f"""
                <div style="margin-bottom:0.8rem">
                  <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem">
                    <span style="font-size:0.85rem;font-weight:500;color:#1B3A6B">{row['Product']}</span>
                    <span style="font-size:0.85rem;color:#6B7A99">{row['Households']:,}</span>
                  </div>
                  <div style="background:#E8ECF4;border-radius:4px;height:10px">
                    <div style="width:{pct}%;background:{color};border-radius:4px;height:10px"></div>
                  </div>
                  <div style="font-size:0.72rem;color:#6B7A99;margin-top:0.1rem">{row['Share']}%</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(t("ov_run_first", lang))

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"### {t('ov_pipeline_title', lang)}")
    pipeline_text = {
        "en": """Raw CSV (21,344 rows)
    ↓  aggregate costs per member
    ↓  filter cancelled members
    ↓  encode income bands to numbers
    ↓  deduplicate + join aggregated costs
    ↓  aggregate to household level
    ↓  engineer 50+ features
    └─ build model matrix (140 features, 0 nulls)

Feature Matrix (3,511 × 140)
    ├─→  6 Classification models  →  AUC 0.82–0.89
    ├─→  1 Regression model       →  R² = 0.59
    └─→  Clustering + Action Table → 8 segments""",
        "fr": """CSV brut (21 344 lignes)
    ↓  agrégation des coûts par membre
    ↓  filtrage des membres annulés
    ↓  encodage des tranches de revenus
    ↓  déduplication + jointure des coûts
    ↓  agrégation au niveau du ménage
    ↓  ingénierie de 50+ variables
    └─ construction de la matrice (140 variables, 0 valeurs manquantes)

Matrice de caractéristiques (3 511 × 140)
    ├─→  6 modèles de classification  →  AUC 0,82–0,89
    ├─→  1 modèle de régression       →  R² = 0,59
    └─→  Clustering + Table d'action  → 8 segments""",
        "ar": """CSV الخام (21,344 صف)
    ↓  تجميع التكاليف لكل عضو
    ↓  تصفية الأعضاء الملغاة
    ↓  ترميز شرائح الدخل
    ↓  إزالة التكرار + ربط التكاليف
    ↓  التجميع على مستوى الأسرة
    ↓  هندسة 50+ ميزة
    └─ بناء مصفوفة النموذج (140 ميزة، 0 قيم مفقودة)

مصفوفة الميزات (3,511 × 140)
    ├─→  6 نماذج تصنيف   →  AUC 0.82–0.89
    ├─→  نموذج انحدار    →  R² = 0.59
    └─→  تجميع + جدول إجراءات → 8 شرائح"""
    }
    st.markdown(f'<div class="code-block">{pipeline_text[lang]}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == t("nav_performance", lang):
    st.markdown(f'<div class="page-title">{t("mp_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.9rem;color:#6B7A99;margin-bottom:1.2rem">{t("mp_subtitle", lang)}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([t("mp_tab_class", lang), t("mp_tab_reg", lang), t("mp_tab_feat", lang)])

    with tab1:
        st.markdown(f"#### {t('mp_class_title', lang)}")
        st.markdown(t("mp_class_desc", lang))

        results = [
            {"Product":"FSV CMSI",        "Model":"Random Forest","AUC":0.8915,"F1":0.5217,"Lift":5.714,"active":True},
            {"Product":"FSV Credit Card",  "Model":"XGBoost",     "AUC":0.8610,"F1":0.3066,"Lift":3.661,"active":True},
            {"Product":"FSV ID Theft",     "Model":"Random Forest","AUC":0.8200,"F1":0.2692,"Lift":3.429,"active":True},
            {"Product":"INS Client",       "Model":"Random Forest","AUC":0.8711,"F1":0.6594,"Lift":2.638,"active":True},
            {"Product":"New Mover",        "Model":"Random Forest","AUC":0.8491,"F1":0.2783,"Lift":4.229,"active":True},
            {"Product":"TRV Globalware",   "Model":"Random Forest","AUC":0.8544,"F1":0.5263,"Lift":2.790,"active":True},
            {"Product":"FSV Deposit",      "Model":"—",            "AUC":None,  "F1":None,  "Lift":None, "active":False},
            {"Product":"FSV Home Equity",  "Model":"—",            "AUC":None,  "F1":None,  "Lift":None, "active":False},
            {"Product":"FSV Mortgage",     "Model":"—",            "AUC":None,  "F1":None,  "Lift":None, "active":False},
        ]

        for r in results:
            if not r["active"]:
                st.markdown(f"""<div class="result-card amber">
                  <strong style="color:#7D4E00">{r['Product']}</strong><br>
                  <span style="font-size:0.82rem;color:#7D4E00">{t('mp_skipped', lang)}</span>
                </div>""", unsafe_allow_html=True)
                continue
            c1, c2, c3, c4 = st.columns([3,2,2,3])
            with c1:
                st.markdown(f"**{r['Product']}**")
                st.caption(r["Model"])
            with c2:
                st.metric("AUC", f"{r['AUC']:.4f}")
            with c3:
                st.metric("F1", f"{r['F1']:.4f}")
            with c4:
                lift_color = "#1E6B3C" if r["Lift"] >= 4 else "#2E4A8C" if r["Lift"] >= 2.5 else "#7D4E00"
                st.markdown(f"""
                <div style="margin-top:0.3rem">
                  <span style="font-size:0.75rem;color:#6B7A99">Lift@10%</span><br>
                  <span style="font-size:1.8rem;font-family:'DM Serif Display',serif;color:{lift_color}">{r['Lift']:.2f}×</span>
                  <span style="font-size:0.72rem;color:#6B7A99"> {t('mp_lift_label', lang)}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('<hr style="border:none;border-top:1px solid #E8ECF4;margin:0.3rem 0">', unsafe_allow_html=True)

        st.info(t("mp_lift_info", lang))

    with tab2:
        st.markdown(f"#### {t('mp_reg_title', lang)}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="metric-card" style="margin-bottom:1rem">
              <div class="label">{t('mp_best_model', lang)}</div>
              <div class="value" style="font-size:1.4rem">Huber Regressor</div>
              <div class="sub">{t('mp_robust', lang)}</div>
            </div>""", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("R²", "0.5918")
            m2.metric("RMSE", "$118.82")
            m3.metric("MAE", "$81.37")

        with col2:
            st.markdown(f"#### {t('mp_leakage', lang)}")
            leakage_rounds = {
                "en": [
                    ("Round 1", "R² = 0.98", "🔴", "Annual cost columns were components of the target. Removed."),
                    ("Round 2", "R² = 0.55", "🟡", "cost_trend = last_year − first_year was still present. Removed."),
                    ("Round 3", "R² = 0.59", "🟢", "All top features confirmed as counts and demographics. Accepted."),
                ],
                "fr": [
                    ("Tour 1", "R² = 0,98", "🔴", "Les colonnes de coûts annuels faisaient partie de la cible. Supprimées."),
                    ("Tour 2", "R² = 0,55", "🟡", "cost_trend = dernière − première année était encore présent. Supprimé."),
                    ("Tour 3", "R² = 0,59", "🟢", "Toutes les variables confirmées comme comptages et démographie. Accepté."),
                ],
                "ar": [
                    ("الجولة 1", "R² = 0.98", "🔴", "أعمدة التكلفة السنوية كانت مكونات من الهدف. تمت إزالتها."),
                    ("الجولة 2", "R² = 0.55", "🟡", "cost_trend = آخر سنة − أول سنة كان لا يزال موجوداً. تمت إزالته."),
                    ("الجولة 3", "R² = 0.59", "🟢", "جميع المتغيرات الأعلى مؤكدة كعدادات وبيانات ديموغرافية. مقبولة."),
                ],
            }
            for rnd, r2, icon, desc in leakage_rounds[lang]:
                card = "red" if icon == "🔴" else "amber" if icon == "🟡" else "green"
                st.markdown(f"""<div class="result-card {card}">
                  <strong>{rnd}</strong> — <code>{r2}</code> {icon}<br>
                  <span style="font-size:0.82rem">{desc}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"#### {t('mp_r2_accept', lang)}")
        r2_body = {
            "en": "Emergency events are inherently random. No demographic data predicts these with high accuracy. The model explains **59% of cost variance** through legitimate behavioral signal — primarily claims call frequency. The remaining **41%** is genuinely random and cannot be predicted from member data.",
            "fr": "Les événements d'urgence sont intrinsèquement aléatoires. Aucune donnée démographique ne peut les prédire avec une grande précision. Le modèle explique **59% de la variance des coûts** via un signal comportemental légitime — principalement la fréquence des appels. Les **41%** restants sont vraiment aléatoires.",
            "ar": "الأحداث الطارئة عشوائية بطبيعتها. لا يمكن لأي بيانات ديموغرافية التنبؤ بها بدقة عالية. يفسر النموذج **59% من تباين التكلفة** من خلال إشارة سلوكية حقيقية — بشكل أساسي تكرار مكالمات المطالبات. الـ**41%** المتبقية عشوائية حقيقية.",
        }
        st.markdown(r2_body[lang])

    with tab3:
        st.markdown(f"#### {t('mp_feat_title', lang)}")
        avail_meta = {k: v for k, v in meta.items() if "ers_cost" not in k}
        if avail_meta:
            selected = st.selectbox(t("mp_feat_select", lang), list(avail_meta.keys()))
            m = avail_meta[selected]
            feats = m.get("feature_names", [])
            if feats:
                top_n = min(15, len(feats))
                for i, f in enumerate(feats[:top_n]):
                    bar_pct = max(5, 100 - i * 6)
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.3rem">
                      <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#6B7A99;width:1.5rem;text-align:right;direction:ltr">{i+1}</span>
                      <span style="font-size:0.82rem;color:#1B3A6B;width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;direction:ltr">{f}</span>
                      <div style="flex:1;background:#E8ECF4;border-radius:3px;height:6px">
                        <div style="width:{bar_pct}%;background:#2E4A8C;border-radius:3px;height:6px"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(t("mp_no_meta", lang))


# ══════════════════════════════════════════════════════════════
# PAGE 3 — SEGMENT EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == t("nav_segments", lang):
    st.markdown(f'<div class="page-title">{t("seg_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.9rem;color:#6B7A99;margin-bottom:1.2rem">{t("seg_subtitle", lang)}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    filter_options = [t("seg_all",lang), t("seg_high",lang), t("seg_second",lang), t("seg_nurture",lang)]
    filter_priority = st.selectbox(t("seg_filter", lang), filter_options)

    seg_names = {
        "en": {7:"High Claims Members", 3:"Long-Tenure Single-Product", 1:"High-Income Prospects",
               2:"Long-Tenure Single-Product", 5:"Active Claims Members", 4:"Long-Tenure Single-Product",
               0:"Long-Tenure Single-Product", 6:"High Claims Members"},
        "fr": {7:"Membres à sinistres élevés", 3:"Fidèles mono-produit", 1:"Prospects à revenus élevés",
               2:"Fidèles mono-produit", 5:"Membres sinistres actifs", 4:"Fidèles mono-produit",
               0:"Fidèles mono-produit", 6:"Membres à sinistres élevés"},
        "ar": {7:"أعضاء المطالبات العالية", 3:"عضو قديم أحادي المنتج", 1:"آفاق عالية الدخل",
               2:"عضو قديم أحادي المنتج", 5:"أعضاء مطالبات نشطون", 4:"عضو قديم أحادي المنتج",
               0:"عضو قديم أحادي المنتج", 6:"أعضاء المطالبات العالية"},
    }
    priority_labels = {
        "en": {7:"🔴 Highest Priority", 3:"🔴 Highest Priority", 1:"🔴 Highest Priority",
               2:"🟡 Second Wave", 5:"🟡 Second Wave", 4:"🟡 Second Wave",
               0:"⏸ Nurture", 6:"⏸ Nurture"},
        "fr": {7:"🔴 Priorité maximale", 3:"🔴 Priorité maximale", 1:"🔴 Priorité maximale",
               2:"🟡 Deuxième vague", 5:"🟡 Deuxième vague", 4:"🟡 Deuxième vague",
               0:"⏸ Nurture", 6:"⏸ Nurture"},
        "ar": {7:"🔴 أعلى أولوية", 3:"🔴 أعلى أولوية", 1:"🔴 أعلى أولوية",
               2:"🟡 الموجة الثانية", 5:"🟡 الموجة الثانية", 4:"🟡 الموجة الثانية",
               0:"⏸ تغذية", 6:"⏸ تغذية"},
    }

    segments = [
        {"id":7,"size":177, "product":"INS Client",    "propensity":0.7514,"income":110487,"tenure":28.8,"products":2.24,"ers":0.972},
        {"id":3,"size":257, "product":"FSV Credit Card","propensity":0.6965,"income":89819, "tenure":42.2,"products":2.39,"ers":0.661},
        {"id":1,"size":265, "product":"INS Client",    "propensity":0.6717,"income":108596,"tenure":35.5,"products":2.18,"ers":0.657},
        {"id":2,"size":497, "product":"INS Client",    "propensity":0.4467,"income":76609, "tenure":41.5,"products":1.00,"ers":0.630},
        {"id":5,"size":463, "product":"INS Client",    "propensity":0.4276,"income":119823,"tenure":30.9,"products":1.01,"ers":0.873},
        {"id":4,"size":265, "product":"INS Client",    "propensity":0.3887,"income":85036, "tenure":36.7,"products":1.00,"ers":0.457},
        {"id":0,"size":1071,"product":"Nurture",       "propensity":0.0,   "income":86632, "tenure":39.6,"products":0.00,"ers":0.516},
        {"id":6,"size":516, "product":"Nurture",       "propensity":0.0,   "income":116817,"tenure":32.4,"products":0.00,"ers":0.870},
    ]

    for seg in segments:
        priority = priority_labels[lang][seg["id"]]
        seg_name = seg_names[lang][seg["id"]]
        is_nurture = seg["product"] == "Nurture"

        # Filter
        if filter_priority != t("seg_all", lang):
            if filter_priority == t("seg_high", lang) and "🔴" not in priority: continue
            if filter_priority == t("seg_second", lang) and "🟡" not in priority: continue
            if filter_priority == t("seg_nurture", lang) and "⏸" not in priority: continue

        with st.expander(f"{t('nav_segments', lang).split('  ')[1] if '  ' in t('nav_segments',lang) else ''} {seg_name}  ·  {seg['size']:,}  ·  {priority}", expanded=seg["propensity"] > 0.6):
            c1, c2, c3 = st.columns([2, 2, 3])
            with c1:
                st.markdown(f"**{t('seg_profile', lang)}**")
                st.markdown(f"- {t('seg_income', lang)}: **${seg['income']:,.0f}**")
                st.markdown(f"- {t('seg_tenure', lang)}: **{seg['tenure']:.1f} {t('seg_years', lang)}**")
                st.markdown(f"- {t('seg_products', lang)}: **{seg['products']:.2f}**")
                st.markdown(f"- {t('seg_ers', lang)}: **{seg['ers']*100:.0f}%**")
            with c2:
                st.markdown(f"**{t('seg_action', lang)}**")
                if is_nurture:
                    st.markdown(f"""<div class="result-card amber">
                      <strong>{t('seg_no_offer', lang)}</strong><br>
                      <span style="font-size:0.82rem">{t('seg_nurture_desc', lang)}</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="result-card green">
                      <strong>{t('seg_lead', lang)} {seg['product']}</strong><br>
                      <span style="font-size:0.82rem">{t('seg_contact', lang)} {seg['size']} {t('seg_hh', lang)}</span>
                    </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"**{t('seg_propensity', lang)}**")
                st.markdown(prop_bar(seg["propensity"]), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**{t('seg_why', lang)}**")
                st.caption(t(f"seg_reason_{seg['id']}", lang))


# ══════════════════════════════════════════════════════════════
# PAGE 4 — HOUSEHOLD LOOKUP
# ══════════════════════════════════════════════════════════════
elif page == t("nav_lookup", lang):
    st.markdown(f'<div class="page-title">{t("lu_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.9rem;color:#6B7A99;margin-bottom:1.2rem">{t("lu_subtitle", lang)}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    if df_feat is None or df_hh is None:
        st.warning(t("lu_no_data", lang))
    elif not models:
        st.warning(t("lu_no_models", lang))
    else:
        if "Household Key" in df_hh.columns:
            hh_keys = sorted(df_hh["Household Key"].unique().tolist())
        elif df_hh.index.name == "Household Key":
            hh_keys = sorted(df_hh.index.unique().tolist())
        else:
            hh_keys = list(range(len(df_hh)))

        col_sel, col_rand = st.columns([3, 1])
        with col_sel:
            selected_hh = st.selectbox(t("lu_select", lang), hh_keys[:500])
        with col_rand:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(t("lu_random", lang)):
                selected_hh = np.random.choice(hh_keys)

        try:
            if df_feat.index.name == "Household Key":
                feat_row = df_feat.loc[[selected_hh]]
            else:
                idx = hh_keys.index(selected_hh)
                feat_row = df_feat.iloc[[idx]]

            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"### {t('lu_scores', lang)} `{selected_hh}`")

            scores = {}
            for name, model in models.items():
                try:
                    scores[name] = model.predict_proba(feat_row)[0, 1]
                except Exception:
                    scores[name] = None

            scores_sorted = sorted([(k,v) for k,v in scores.items() if v is not None], key=lambda x: -x[1])

            c1, c2 = st.columns(2)
            for i, (product, score) in enumerate(scores_sorted):
                col = c1 if i % 2 == 0 else c2
                tier = t("lu_high",lang) if score > 0.5 else t("lu_medium",lang) if score > 0.25 else t("lu_low",lang)
                border = "#1E6B3C" if score > 0.5 else "#2E4A8C" if score > 0.25 else "#7D4E00"
                with col:
                    st.markdown(f"""
                    <div style="background:#FAFBFF;border-radius:10px;padding:1rem;margin-bottom:0.8rem;
                                border:1px solid #E2E8F4;border-left:4px solid {border}">
                      <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem">
                        <span style="font-weight:600;color:#1B3A6B;font-size:0.9rem;direction:ltr">{product}</span>
                        <span style="font-size:0.75rem">{tier}</span>
                      </div>
                      {prop_bar(score)}
                    </div>
                    """, unsafe_allow_html=True)

            if scores_sorted:
                top_product, top_score = scores_sorted[0]
                rec_text = t("lu_lead", lang) if top_score > 0.15 else t("lu_nurture", lang)
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1B3A6B,#2E4A8C);border-radius:12px;
                            padding:1.2rem 1.5rem;margin-top:1rem;border:1px solid rgba(255,255,255,0.1)">
                  <div style="font-size:0.75rem;color:#A8C4E8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.3rem">
                    {t('lu_top_rec', lang)}
                  </div>
                  <div style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:#FFFFFF;direction:ltr">{top_product}</div>
                  <div style="font-size:0.82rem;color:#7EC8A0;margin-top:0.2rem">{rec_text} — {top_score:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — TECHNICAL DEEP DIVE
# ══════════════════════════════════════════════════════════════
elif page == t("nav_technical", lang):
    st.markdown(f'<div class="page-title">{t("td_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.9rem;color:#6B7A99;margin-bottom:1.2rem">{t("td_subtitle", lang)}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([t("td_tab_arch",lang), t("td_tab_test",lang), t("td_tab_design",lang)])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### {t('td_repo', lang)}")
            st.markdown("""<div class="code-block">nia-member-analysis/
├── configs/settings.yaml
├── src/
│   ├── config.py
│   ├── features/preprocessing.py
│   ├── models/
│   │   ├── classifier.py
│   │   ├── regressor.py
│   │   └── clustering.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── plots.py
│   └── pipelines/train.py
├── tests/
├── notebooks/
├── app.py
└── requirements.txt</div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"#### {t('td_decisions', lang)}")
            decisions = {
                "en": [
                    ("YAML config", "Every constant lives in settings.yaml — one file to change everything."),
                    ("Typed dataclasses", "cfg.training.test_size catches typos at import time, not runtime."),
                    ("Pure functions", "Every function takes explicit inputs and returns explicit outputs. Fully testable."),
                    ("Stage-by-stage CLI", "--stage classify runs only model training. No re-running preprocessing."),
                    ("Parquet over CSV", "5–10× smaller, dtype preservation, sub-second reads."),
                    ("Absolute paths", "str(cfg.paths.abs('model_dir')) prevents silent failures."),
                ],
                "fr": [
                    ("Config YAML", "Chaque constante est dans settings.yaml — un seul fichier à modifier."),
                    ("Dataclasses typées", "cfg.training.test_size détecte les fautes de frappe à l'import."),
                    ("Fonctions pures", "Chaque fonction prend des entrées explicites et retourne des sorties explicites."),
                    ("CLI par étapes", "--stage classify exécute uniquement l'entraînement des modèles."),
                    ("Parquet vs CSV", "5 à 10× plus petit, préservation des types, lectures en moins d'une seconde."),
                    ("Chemins absolus", "str(cfg.paths.abs('model_dir')) empêche les échecs silencieux."),
                ],
                "ar": [
                    ("إعداد YAML", "كل ثابت في settings.yaml — ملف واحد لتغيير كل شيء."),
                    ("فئات بيانات مكتوبة", "cfg.training.test_size يكتشف الأخطاء عند الاستيراد."),
                    ("دوال نقية", "كل دالة تأخذ مدخلات صريحة وتعيد مخرجات صريحة. قابلة للاختبار بالكامل."),
                    ("CLI مرحلي", "--stage classify يشغل تدريب النموذج فقط."),
                    ("Parquet بدل CSV", "أصغر بـ5-10 أضعاف، حفظ أنواع البيانات، قراءة فورية."),
                    ("مسارات مطلقة", "str(cfg.paths.abs('model_dir')) يمنع الإخفاقات الصامتة."),
                ],
            }
            for title, desc in decisions[lang]:
                st.markdown(f"""<div class="result-card" style="margin-bottom:0.5rem">
                  <strong style="color:#1B3A6B">{title}</strong><br>
                  <span style="font-size:0.82rem">{desc}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"#### {t('td_commands', lang)}")
        st.markdown("""<div class="code-block">python3 -m src.pipelines.train --stage preprocess   # 4s
python3 -m src.pipelines.train --stage classify     # 7s
python3 -m src.pipelines.train --stage regress      # 2s
python3 -m src.pipelines.train --stage cluster      # 1s
python3 -m src.pipelines.train --stage all          # 15s</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown(f"#### {t('td_tests', lang)}")
        st.markdown(t("td_tests_desc", lang))
        test_files = {
            "en": [
                ("test_config.py", 13, "settings.yaml loads, all 9 products present, income map correct"),
                ("test_preprocessing.py", 49, "Ordinal encoding, NaN handling, product flags, zero nulls in output"),
                ("test_metrics.py", 12, "Perfect classifier AUC=1.0, random lift≈1.0, RMSE=0 for perfect predictions"),
                ("test_clustering.py", 10, "Inertia decreases with k, silhouette in [-1,1], action table products valid"),
            ],
            "fr": [
                ("test_config.py", 13, "Chargement de settings.yaml, 9 produits présents, carte des revenus correcte"),
                ("test_preprocessing.py", 49, "Encodage ordinal, gestion NaN, drapeaux produits, zéro valeurs manquantes"),
                ("test_metrics.py", 12, "Classificateur parfait AUC=1,0, lift aléatoire≈1,0, RMSE=0 pour prédictions parfaites"),
                ("test_clustering.py", 10, "Inertie décroissante avec k, silhouette dans [-1,1], produits du tableau d'action valides"),
            ],
            "ar": [
                ("test_config.py", 13, "تحميل settings.yaml، جميع المنتجات التسعة موجودة، خريطة الدخل صحيحة"),
                ("test_preprocessing.py", 49, "الترميز الترتيبي، معالجة القيم المفقودة، أعلام المنتجات، صفر قيم مفقودة في المخرجات"),
                ("test_metrics.py", 12, "المصنف المثالي AUC=1.0، الرفع العشوائي≈1.0، RMSE=0 للتنبؤات المثالية"),
                ("test_clustering.py", 10, "القصور الذاتي يتناقص مع k، silhouette في [-1,1]، منتجات جدول الإجراءات صالحة"),
            ],
        }
        for fname, n, desc in test_files[lang]:
            st.markdown(f"""<div style="background:#FAFBFF;border-radius:8px;padding:0.8rem 1rem;
                              margin-bottom:0.6rem;border:1px solid #E2E8F4">
              <div style="display:flex;justify-content:space-between">
                <code style="color:#2E4A8C;font-size:0.85rem;direction:ltr">{fname}</code>
                <span style="background:#D6EFE1;color:#1E6B3C;border-radius:20px;
                             padding:0.1rem 0.6rem;font-size:0.72rem;font-weight:600">{n}</span>
              </div>
              <div style="font-size:0.8rem;color:#6B7A99;margin-top:0.3rem">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown(f"#### {t('td_before_after', lang)}")
        comparisons = {
            "en": [
                ("Clustering input", "Binary ownership flags — model found who ALREADY owned products", "6 propensity scores — model finds who WILL BUY products"),
                ("Regression R²", "R² = 0.98 — model was doing arithmetic on cost components", "R² = 0.59 — genuine signal after 3-round leakage investigation"),
                ("Feature selection", "All 140 features passed to every model", "Leakage patterns excluded per stage"),
                ("Error handling", "Silent except: pass swallowed all errors", "Explicit log.error with exception type"),
                ("Path resolution", "Relative path — glob found nothing silently", "Absolute path — failure is loud and immediate"),
                ("Return values", "stage_cluster() did not return the result", "return run_clustering() — every result returned"),
            ],
            "fr": [
                ("Entrée clustering", "Drapeaux binaires — le modèle trouvait qui POSSÉDAIT déjà des produits", "6 scores de propension — le modèle trouve qui VA ACHETER"),
                ("R² régression", "R² = 0,98 — le modèle faisait de l'arithmétique sur les coûts", "R² = 0,59 — signal réel après investigation en 3 tours"),
                ("Sélection variables", "140 variables passées à tous les modèles", "Motifs de fuite exclus par étape"),
                ("Gestion d'erreurs", "except: pass silencieux avalait toutes les erreurs", "log.error explicite avec type d'exception"),
                ("Résolution chemins", "Chemin relatif — glob ne trouvait rien silencieusement", "Chemin absolu — l'échec est visible et immédiat"),
                ("Valeurs de retour", "stage_cluster() ne retournait pas le résultat", "return run_clustering() — chaque résultat retourné"),
            ],
            "ar": [
                ("مدخل التجميع", "أعلام ملكية ثنائية — النموذج وجد من يمتلك المنتجات بالفعل", "6 درجات ميل — النموذج يجد من سيشتري"),
                ("R² الانحدار", "R² = 0.98 — النموذج كان يجري حسابات على مكونات التكلفة", "R² = 0.59 — إشارة حقيقية بعد تحقيق ثلاثي الجولات"),
                ("اختيار المتغيرات", "140 متغير لجميع النماذج", "أنماط التسرب مستبعدة لكل مرحلة"),
                ("معالجة الأخطاء", "except: pass الصامت يبتلع جميع الأخطاء", "log.error صريح مع نوع الاستثناء"),
                ("تحليل المسارات", "مسار نسبي — glob لم يجد شيئاً بصمت", "مسار مطلق — الفشل واضح وفوري"),
                ("قيم الإرجاع", "stage_cluster() لم تُعد النتيجة", "return run_clustering() — كل نتيجة مُعادة"),
            ],
        }
        for topic, before, after in comparisons[lang]:
            st.markdown(f"""
            <div style="background:#FAFBFF;border-radius:10px;padding:0.9rem 1rem;
                        margin-bottom:0.7rem;border:1px solid #E2E8F4">
              <div style="font-weight:600;color:#1B3A6B;font-size:0.88rem;margin-bottom:0.5rem">{topic}</div>
              <div style="display:flex;gap:1rem">
                <div style="flex:1;background:#FEF2F2;border-radius:6px;padding:0.5rem 0.7rem;border-left:3px solid #7B1A1A">
                  <div style="font-size:0.68rem;font-weight:600;color:#7B1A1A;text-transform:uppercase;margin-bottom:0.2rem">
                    {t('td_before', lang)}
                  </div>
                  <div style="font-size:0.78rem;color:#7B1A1A">{before}</div>
                </div>
                <div style="flex:1;background:#ECFDF5;border-radius:6px;padding:0.5rem 0.7rem;border-left:3px solid #1E6B3C">
                  <div style="font-size:0.68rem;font-weight:600;color:#1E6B3C;text-transform:uppercase;margin-bottom:0.2rem">
                    {t('td_after', lang)}
                  </div>
                  <div style="font-size:0.78rem;color:#1E6B3C">{after}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
