"""
RailGuard AI — Aplicação Principal Streamlit
============================================
Plataforma de Compliance Preditivo Ferroviário
MVP demonstrativo — dados simulados.

Execução:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Any

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import database as db
import ml_model as ml
import models as m
import reports as rp
import risk_engine as risk_engine
import seed_data

# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RailGuard AI",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# PALETA — TEMA CORPORATIVO FERROVIÁRIO DARK
# ═══════════════════════════════════════════════════════════════

PRIMARY = "#0A1628"
SECONDARY = "#102A4C"
ACCENT = "#E63946"
ACCENT2 = "#1D7EC2"
SURFACE = "#0F1F38"
CARD_BG = "#132035"
BORDER = "#1E3A5F"
TEXT_PRI = "#E8EDF2"
TEXT_SEC = "#8FA8BF"
TEXT_MUT = "#4A6580"

C_BAIXO = "#00C48C"
C_MEDIO = "#FFB020"
C_ALTO = "#FF6B35"
C_CRITICO = "#E63946"
C_PURPLE = "#9B59B6"
C_TEAL = "#1ABC9C"

RISK_COLORS_PLOTLY = {
    "Baixo": C_BAIXO,
    "Médio": C_MEDIO,
    "Alto": C_ALTO,
    "Crítico": C_CRITICO,
}

RCRS_COLORS_PLOTLY = {
    "Conforme": C_BAIXO,
    "Atenção": C_MEDIO,
    "Não conformidade potencial": C_ALTO,
    "Crítico": C_CRITICO,
}

ESG_COLORS_PLOTLY = {
    "Baixa": C_BAIXO,
    "Média": ACCENT2,
    "Alta": C_ALTO,
    "Crítica": C_CRITICO,
}

# ═══════════════════════════════════════════════════════════════
# CSS GLOBAL
# ═══════════════════════════════════════════════════════════════

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] {{ font-family:'Inter', sans-serif !important; }}
.stApp {{
    background:{PRIMARY};
    background-image:
      linear-gradient(rgba(30,58,95,0.12) 1px, transparent 1px),
      linear-gradient(90deg, rgba(30,58,95,0.12) 1px, transparent 1px);
    background-size:40px 40px;
}}
.main .block-container {{ padding:1.5rem 2rem 3rem 2rem; max-width:1600px; }}
[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,{SECONDARY} 0%,{PRIMARY} 100%);
    border-right:1px solid {BORDER};
}}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
    color:{TEXT_PRI}!important; font-size:0.88rem!important; font-weight:500;
}}
div[data-testid="stSidebarNav"] {{ display:none; }}
.rg-page-header {{ display:flex; align-items:center; gap:14px; padding:20px 0 16px 0; border-bottom:2px solid {BORDER}; margin-bottom:24px; }}
.rg-page-icon {{ width:44px; height:44px; background:linear-gradient(135deg,{ACCENT2},#0D5C8A); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.3rem; box-shadow:0 4px 15px rgba(29,126,194,0.4); }}
.rg-page-title {{ font-size:1.6rem; font-weight:800; color:{TEXT_PRI}; letter-spacing:-0.02em; line-height:1.2; }}
.rg-page-subtitle {{ font-size:0.8rem; color:{TEXT_SEC}; font-weight:400; margin-top:2px; }}
.kpi-card {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:14px; padding:20px 22px; position:relative; overflow:hidden; min-height:124px; }}
.kpi-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }}
.kpi-primary::before {{ background:linear-gradient(90deg,{ACCENT2},{PRIMARY}); }}
.kpi-success::before {{ background:linear-gradient(90deg,{C_BAIXO},{PRIMARY}); }}
.kpi-warning::before {{ background:linear-gradient(90deg,{C_MEDIO},{PRIMARY}); }}
.kpi-danger::before {{ background:linear-gradient(90deg,{C_CRITICO},{PRIMARY}); }}
.kpi-info::before {{ background:linear-gradient(90deg,{C_PURPLE},{PRIMARY}); }}
.kpi-label {{ font-size:0.72rem; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:{TEXT_MUT}; margin-bottom:8px; }}
.kpi-value {{ font-size:2.05rem; font-weight:800; color:{TEXT_PRI}; line-height:1; font-variant-numeric:tabular-nums; }}
.kpi-delta {{ font-size:0.77rem; margin-top:8px; font-weight:500; color:{TEXT_SEC}; }}
.kpi-delta.up {{ color:{C_CRITICO}; }} .kpi-delta.down {{ color:{C_BAIXO}; }} .kpi-delta.neutral {{ color:{TEXT_SEC}; }}
.kpi-icon {{ position:absolute; right:18px; top:18px; font-size:2rem; opacity:0.12; }}
.rg-card {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:14px; padding:20px 22px; margin-bottom:16px; }}
.rg-card-title {{ font-size:0.82rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:{TEXT_SEC}; margin-bottom:14px; display:flex; align-items:center; gap:8px; }}
.rg-card-title::before {{ content:''; display:inline-block; width:3px; height:14px; background:{ACCENT2}; border-radius:2px; }}
.rbadge {{ display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; }}
.rbadge-baixo {{ background:rgba(0,196,140,0.18); color:{C_BAIXO}; border:1px solid rgba(0,196,140,0.4); }}
.rbadge-medio {{ background:rgba(255,176,32,0.18); color:{C_MEDIO}; border:1px solid rgba(255,176,32,0.4); }}
.rbadge-alto {{ background:rgba(255,107,53,0.18); color:{C_ALTO}; border:1px solid rgba(255,107,53,0.4); }}
.rbadge-critico {{ background:rgba(230,57,70,0.18); color:{C_CRITICO}; border:1px solid rgba(230,57,70,0.4); }}
.alert-item {{ display:flex; align-items:flex-start; gap:14px; padding:14px 16px; border-radius:10px; margin-bottom:10px; border:1px solid; }}
.alert-critico {{ background:rgba(230,57,70,0.08); border-color:rgba(230,57,70,0.35); }}
.alert-alto {{ background:rgba(255,107,53,0.08); border-color:rgba(255,107,53,0.35); }}
.alert-medio {{ background:rgba(255,176,32,0.08); border-color:rgba(255,176,32,0.35); }}
.alert-pulse {{ width:10px; height:10px; border-radius:50%; margin-top:3px; flex-shrink:0; animation:pulse 1.8s infinite; }}
.pulse-critico {{ background:{C_CRITICO}; box-shadow:0 0 0 0 rgba(230,57,70,0.5); }}
.pulse-alto {{ background:{C_ALTO}; box-shadow:0 0 0 0 rgba(255,107,53,0.5); }}
.pulse-medio {{ background:{C_MEDIO}; box-shadow:0 0 0 0 rgba(255,176,32,0.5); }}
@keyframes pulse {{ 0% {{ box-shadow:0 0 0 0 rgba(230,57,70,0.5); }} 70% {{ box-shadow:0 0 0 8px rgba(230,57,70,0); }} 100% {{ box-shadow:0 0 0 0 rgba(230,57,70,0); }} }}
.alert-body {{ flex:1; }} .alert-titulo {{ font-size:0.85rem; font-weight:700; color:{TEXT_PRI}; }}
.alert-msg {{ font-size:0.78rem; color:{TEXT_SEC}; margin-top:3px; line-height:1.5; }}
.alert-meta {{ font-size:0.70rem; color:{TEXT_MUT}; margin-top:5px; font-family:'JetBrains Mono',monospace; }}
.alert-time {{ font-size:0.68rem; color:{TEXT_MUT}; white-space:nowrap; font-family:'JetBrains Mono',monospace; }}
.section-divider {{ display:flex; align-items:center; gap:12px; margin:28px 0 18px 0; }}
.section-divider-line {{ flex:1; height:1px; background:{BORDER}; }}
.section-divider-text {{ font-size:0.70rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:{TEXT_MUT}; padding:0 8px; }}
[data-testid="metric-container"] {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px; padding:16px; }}
[data-testid="stMetricValue"] {{ color:{TEXT_PRI}!important; font-weight:700!important; }}
[data-testid="stMetricLabel"] {{ color:{TEXT_SEC}!important; font-size:0.78rem!important; }}
[data-testid="stForm"] {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:14px; padding:20px!important; }}
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stTextArea>div>div>textarea {{ background:{SURFACE}!important; border:1px solid {BORDER}!important; color:{TEXT_PRI}!important; border-radius:8px!important; }}
.stSelectbox>div>div, .stMultiSelect>div>div, .stDateInput>div>div {{ background:{SURFACE}!important; border:1px solid {BORDER}!important; border-radius:8px!important; color:{TEXT_PRI}!important; }}
label {{ color:{TEXT_SEC}!important; font-size:0.82rem!important; font-weight:500!important; }}
[data-testid="stTabs"] [role="tablist"] {{ background:{SURFACE}; border-radius:10px; padding:4px; border:1px solid {BORDER}; gap:4px; }}
[data-testid="stTabs"] [role="tab"] {{ color:{TEXT_SEC}!important; border-radius:7px!important; font-size:0.82rem!important; font-weight:600!important; padding:8px 16px!important; border:none!important; }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ background:{ACCENT2}!important; color:white!important; }}
[data-testid="stTabs"] [role="tabpanel"] {{ padding-top:20px; }}
.stButton>button {{ background:linear-gradient(135deg,{ACCENT2},#0D5C8A)!important; color:white!important; border:none!important; border-radius:8px!important; font-weight:600!important; font-size:0.84rem!important; padding:10px 20px!important; box-shadow:0 4px 15px rgba(29,126,194,0.3)!important; }}
.stButton>button:hover {{ box-shadow:0 6px 20px rgba(29,126,194,0.5)!important; transform:translateY(-1px)!important; }}
.stDownloadButton>button {{ background:linear-gradient(135deg,#27AE60,#1A8A48)!important; color:white!important; border:none!important; border-radius:8px!important; font-weight:600!important; box-shadow:0 4px 15px rgba(39,174,96,0.3)!important; }}
[data-testid="stExpander"] {{ background:{CARD_BG}!important; border:1px solid {BORDER}!important; border-radius:10px!important; }}
[data-testid="stExpander"] summary {{ color:{TEXT_PRI}!important; font-weight:600!important; }}
.sidebar-logo {{ padding:20px 16px 8px; border-bottom:1px solid {BORDER}; margin-bottom:16px; }}
.sidebar-brand {{ font-size:1.1rem; font-weight:800; color:{TEXT_PRI}; letter-spacing:-0.01em; }}
.sidebar-tagline {{ font-size:0.68rem; color:{TEXT_MUT}; letter-spacing:0.06em; text-transform:uppercase; margin-top:2px; }}
.info-box {{ background:rgba(29,126,194,0.12); border-left:4px solid {ACCENT2}; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; font-size:0.84rem; color:{TEXT_SEC}; }}
.warn-box {{ background:rgba(255,176,32,0.10); border-left:4px solid {C_MEDIO}; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; font-size:0.84rem; color:{TEXT_SEC}; }}
.danger-box {{ background:rgba(230,57,70,0.10); border-left:4px solid {C_CRITICO}; border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; font-size:0.84rem; color:{TEXT_SEC}; }}
.esg-bar-wrap {{ margin-bottom:14px; }}
.esg-bar-label {{ display:flex; justify-content:space-between; font-size:0.78rem; color:{TEXT_SEC}; margin-bottom:5px; }}
.esg-bar-track {{ height:8px; background:{BORDER}; border-radius:4px; overflow:hidden; }}
.esg-bar-fill {{ height:100%; border-radius:4px; transition:width 0.6s ease; }}
.exec-report-header {{ background:linear-gradient(135deg,{SECONDARY},{SURFACE}); border:1px solid {BORDER}; border-radius:16px; padding:28px 32px; margin-bottom:24px; position:relative; overflow:hidden; }}
.exec-title {{ font-size:1.4rem; font-weight:800; color:{TEXT_PRI}; letter-spacing:-0.02em; }}
.exec-subtitle {{ font-size:0.82rem; color:{TEXT_SEC}; margin-top:4px; }}
.exec-badge {{ display:inline-block; margin-top:14px; padding:6px 14px; border-radius:20px; font-size:0.75rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; }}
.small-note {{ font-size:0.76rem; color:{TEXT_MUT}; line-height:1.55; }}
code, pre {{ font-family:'JetBrains Mono', monospace !important; }}
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{PRIMARY}; }}
::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background:{ACCENT2}; }}
#MainMenu {{ visibility:hidden; }} footer {{ visibility:hidden; }} header {{ visibility:hidden; }}
</style>
"""

# ═══════════════════════════════════════════════════════════════
# INICIALIZAÇÃO
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def inicializar() -> bool:
    db.init_database()
    if db.check_db_empty():
        seed_data.seed_database()
    return True

inicializar()
st.markdown(CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HELPERS DE UI E DADOS
# ═══════════════════════════════════════════════════════════════

def _pt(fig: go.Figure, h: int | None = None) -> go.Figure:
    """Aplica tema dark aos gráficos Plotly."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=TEXT_SEC, size=12),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_MUT, size=11)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_MUT, size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=11), bordercolor=BORDER, borderwidth=1),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(family="Inter", color=TEXT_PRI, size=12)),
        margin=dict(t=40, b=30, l=10, r=10),
        colorway=[ACCENT2, C_BAIXO, C_MEDIO, C_ALTO, C_CRITICO, C_PURPLE, C_TEAL],
    )
    if h:
        fig.update_layout(height=h)
    return fig


def gauge_risco(score: float, nivel: str, h: int = 245) -> go.Figure:
    cor = RISK_COLORS_PLOTLY.get(nivel, TEXT_SEC)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        number={"suffix": "/100", "font": {"size": 28, "color": cor, "family": "Inter"}},
        title={"text": f"<b>{nivel}</b>", "font": {"size": 13, "color": cor, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BORDER, "tickfont": {"color": TEXT_MUT, "size": 10}},
            "bar": {"color": cor, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": BORDER,
            "steps": [
                {"range": [0, 25], "color": "rgba(0,196,140,0.12)"},
                {"range": [25, 50], "color": "rgba(255,176,32,0.12)"},
                {"range": [50, 75], "color": "rgba(255,107,53,0.12)"},
                {"range": [75, 100], "color": "rgba(230,57,70,0.12)"},
            ],
            "threshold": {"line": {"color": cor, "width": 3}, "thickness": 0.75, "value": float(score)},
        },
    ))
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, b=10, l=20, r=20), font=dict(family="Inter"))
    return fig


def ph(icon: str, title: str, subtitle: str = "") -> None:
    sub = f'<div class="rg-page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="rg-page-header">
            <div class="rg-page-icon">{icon}</div>
            <div>
                <div class="rg-page-title">{title}</div>
                {sub}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label: str, value: Any, delta: str = "", ddir: str = "neutral", variant: str = "primary", icon: str = "") -> None:
    delta_html = f'<div class="kpi-delta {ddir}">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi-card kpi-{variant}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sdiv(label: str) -> None:
    st.markdown(
        f"""
        <div class="section-divider">
            <div class="section-divider-line"></div>
            <div class="section-divider-text">{label}</div>
            <div class="section-divider-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info(msg: str) -> None:
    st.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)


def warn(msg: str) -> None:
    st.markdown(f'<div class="warn-box">{msg}</div>', unsafe_allow_html=True)


def danger(msg: str) -> None:
    st.markdown(f'<div class="danger-box">{msg}</div>', unsafe_allow_html=True)


def card_title(title: str) -> None:
    st.markdown(f'<div class="rg-card-title">{title}</div>', unsafe_allow_html=True)


def badge(label: str) -> str:
    cls = str(label).lower().replace("é", "e").replace("í", "i").replace(" ", "-")
    if cls in {"baixo", "baixa", "conforme"}:
        cls = "baixo"
    elif cls in {"medio", "media", "atencao", "atenção"}:
        cls = "medio"
    elif "alto" in cls or "alta" in cls or "potencial" in cls:
        cls = "alto"
    elif "critico" in cls or "critica" in cls:
        cls = "critico"
    else:
        cls = "medio"
    return f'<span class="rbadge rbadge-{cls}">{label}</span>'


def row_kv(lbl: str, val: Any) -> str:
    return f"""
    <div style="display:flex;justify-content:space-between;gap:16px;padding:9px 0;border-bottom:1px solid {BORDER};">
        <span style="font-size:0.78rem;color:{TEXT_MUT};">{lbl}</span>
        <span style="font-size:0.80rem;color:{TEXT_PRI};font-weight:500;font-family:'JetBrains Mono',monospace;text-align:right;">{val}</span>
    </div>
    """


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].fillna("")
    return out


def df_download_button(df: pd.DataFrame, filename: str, label: str = "⬇️ Baixar CSV") -> None:
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False, sep=";", encoding="utf-8-sig")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


def safe_insert(action, success_msg: str, audit: tuple[str, str, str, str] | None = None) -> None:
    try:
        action()
        if audit:
            db.insert_auditoria(*audit)
        st.success(success_msg)
        st.rerun()
    except Exception as exc:
        st.error(f"Não foi possível concluir a operação: {exc}")


def latest_risk_per_asset(riscos: pd.DataFrame) -> pd.DataFrame:
    if riscos.empty:
        return riscos
    tmp = riscos.copy()
    if "data_calculo" in tmp.columns:
        tmp["data_calculo"] = pd.to_datetime(tmp["data_calculo"], errors="coerce")
        tmp = tmp.sort_values("data_calculo", ascending=False)
    return tmp.drop_duplicates("ativo_id", keep="first")

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-logo">
            <div class="sidebar-brand">Rail<span style="color:{ACCENT}">Guard</span> AI</div>
            <div class="sidebar-tagline">Compliance Preditivo Ferroviário</div>
            <div style="display:flex;align-items:center;gap:6px;margin-top:10px;font-size:0.72rem;color:{C_BAIXO};">
                <div style="width:6px;height:6px;border-radius:50%;background:{C_BAIXO};box-shadow:0 0 6px {C_BAIXO};animation:pulse 2s infinite;"></div>
                Sistema Operacional
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    NAV_ITEMS = {
        "📊  Dashboard": "Dashboard",
        "🛤️  Trechos": "Trechos",
        "⚙️  Ativos": "Ativos",
        "🔍  Inspeções": "Inspecoes",
        "🤖  Modelo Preditivo": "ML",
        "📋  Compliance & RCRS": "Compliance",
        "🗂️  Auditoria": "Auditoria",
        "🌿  ESG": "ESG",
        "📄  Relatórios": "Relatorios",
        "⚙️  Configurações": "Config",
    }

    nav_sel = st.radio("MENU PRINCIPAL", list(NAV_ITEMS.keys()), label_visibility="visible")
    pagina = NAV_ITEMS[nav_sel]

    sb = db.get_dashboard_stats()
    st.markdown(
        f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:14px 16px;margin-top:20px;margin-bottom:16px;">
            <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:{TEXT_MUT};font-weight:700;margin-bottom:10px;">VISÃO RÁPIDA</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div><div style="font-size:1.3rem;font-weight:800;color:{TEXT_PRI};">{sb.get('total_ativos', 0)}</div><div style="font-size:0.68rem;color:{TEXT_MUT};">Ativos</div></div>
                <div><div style="font-size:1.3rem;font-weight:800;color:{C_CRITICO};">{sb.get('risco_critico', 0)}</div><div style="font-size:0.68rem;color:{TEXT_MUT};">Críticos</div></div>
                <div><div style="font-size:1.3rem;font-weight:800;color:{TEXT_PRI};">{sb.get('total_inspecoes', 0)}</div><div style="font-size:0.68rem;color:{TEXT_MUT};">Inspeções</div></div>
                <div><div style="font-size:1.3rem;font-weight:800;color:{C_MEDIO};">{sb.get('total_alertas_abertos', 0)}</div><div style="font-size:0.68rem;color:{TEXT_MUT};">Alertas</div></div>
            </div>
        </div>
        <div style="font-size:0.65rem;color:{TEXT_MUT};text-align:center;padding:0 8px;">
            MVP v0.2 · {datetime.now().strftime('%d/%m/%Y %H:%M')}<br>
            ⚠️ Dados simulados — uso demonstrativo
        </div>
        """,
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# PÁGINAS
# ═══════════════════════════════════════════════════════════════

def page_dashboard() -> None:
    ph("📊", "Dashboard", "Visão geral da plataforma RailGuard AI")
    stats = db.get_dashboard_stats()
    riscos = db.get_all_riscos()
    alertas = db.get_all_alertas()
    ativos = db.get_all_ativos()
    inspecoes = db.get_all_inspecoes()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Trechos", stats.get("total_trechos", 0), "Segmentos cadastrados", "neutral", "primary", "🛤️")
    with c2:
        kpi("Ativos", stats.get("total_ativos", 0), "Infraestrutura monitorada", "neutral", "success", "⚙️")
    with c3:
        kpi("Inspeções", stats.get("total_inspecoes", 0), "Registros técnicos", "neutral", "info", "🔍")
    with c4:
        kpi("Alertas abertos", stats.get("total_alertas_abertos", 0), "Exigem acompanhamento", "up", "warning", "🚨")

    sdiv("Mapa executivo de risco")
    col_a, col_b = st.columns([1.35, 1])
    with col_a:
        card_title("Distribuição por nível de risco")
        if riscos.empty:
            warn("Nenhum risco calculado ainda. Registre uma inspeção para gerar o primeiro score.")
        else:
            contagem = riscos["nivel_risco"].value_counts().reset_index()
            contagem.columns = ["Nível de risco", "Quantidade"]
            fig = px.bar(contagem, x="Nível de risco", y="Quantidade", color="Nível de risco", color_discrete_map=RISK_COLORS_PLOTLY, text="Quantidade")
            fig.update_traces(textposition="outside")
            st.plotly_chart(_pt(fig, 360), use_container_width=True)
    with col_b:
        card_title("Último score crítico")
        if riscos.empty:
            info("Sem score disponível.")
        else:
            r = riscos.sort_values("score_risco", ascending=False).iloc[0]
            st.plotly_chart(gauge_risco(float(r["score_risco"]), str(r["nivel_risco"])), use_container_width=True)
            st.markdown(
                row_kv("Ativo", r.get("ativo_codigo", "—"))
                + row_kv("Tipo", r.get("tipo_ativo", "—"))
                + row_kv("RCRS", f"{float(r.get('score_rcrs', 0)):.1f}/100")
                + row_kv("Classificação", r.get("classificacao_rcrs", "—")),
                unsafe_allow_html=True,
            )

    col_c, col_d = st.columns([1, 1])
    with col_c:
        card_title("Ativos por tipo")
        if ativos.empty:
            warn("Nenhum ativo cadastrado.")
        else:
            dft = ativos["tipo_ativo"].value_counts().reset_index()
            dft.columns = ["Tipo", "Quantidade"]
            fig = px.pie(dft, names="Tipo", values="Quantidade", hole=0.55)
            st.plotly_chart(_pt(fig, 330), use_container_width=True)
    with col_d:
        card_title("Inspeções nos últimos meses")
        if inspecoes.empty:
            warn("Nenhuma inspeção registrada.")
        else:
            dfi = inspecoes.copy()
            dfi["data_inspecao"] = pd.to_datetime(dfi["data_inspecao"], errors="coerce")
            dfi = dfi.dropna(subset=["data_inspecao"])
            dfi["Mês"] = dfi["data_inspecao"].dt.to_period("M").astype(str)
            serie = dfi.groupby("Mês").size().reset_index(name="Inspeções")
            fig = px.line(serie, x="Mês", y="Inspeções", markers=True)
            st.plotly_chart(_pt(fig, 330), use_container_width=True)

    sdiv("Alertas operacionais")
    if alertas.empty:
        info("Não há alertas registrados no momento.")
    else:
        abertos = alertas[alertas["status"].astype(str).str.lower() == "aberto"].head(6)
        if abertos.empty:
            info("Todos os alertas registrados estão resolvidos.")
        for _, al in abertos.iterrows():
            urg = str(al.get("nivel_urgencia", "Médio"))
            css = "critico" if urg in ["Urgente", "Crítico", "Critico"] else "alto" if urg == "Alta" else "medio"
            st.markdown(
                f"""
                <div class="alert-item alert-{css}">
                    <div class="alert-pulse pulse-{css}"></div>
                    <div class="alert-body">
                        <div class="alert-titulo">{al.get('tipo_alerta', 'Alerta')} · {al.get('ativo_codigo', '—')}</div>
                        <div class="alert-msg">{al.get('mensagem', '')}</div>
                        <div class="alert-meta">Trecho: {al.get('trecho_codigo', '—')} · Status: {al.get('status', '—')}</div>
                    </div>
                    <div class="alert-time">{str(al.get('data_alerta', ''))[:16]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def page_trechos() -> None:
    ph("🛤️", "Trechos Ferroviários", "Cadastro, consulta e análise da malha monitorada")
    trechos = db.get_all_trechos()
    ativos = db.get_all_ativos()
    riscos = latest_risk_per_asset(db.get_all_riscos())

    if not trechos.empty:
        total_km = (trechos["km_final"].astype(float) - trechos["km_inicial"].astype(float)).abs().sum()
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Trechos", len(trechos), icon="🛤️")
        with c2: kpi("Extensão", f"{total_km:.1f} km", icon="📏", variant="info")
        with c3: kpi("Estados", trechos["estado"].nunique(), icon="🗺️", variant="success")
        with c4: kpi("Críticos", int((trechos["criticidade_operacional"] == "Crítica").sum()), icon="⚠️", variant="danger")

    tab1, tab2, tab3 = st.tabs(["📋 Consultar trechos", "➕ Cadastrar trecho", "📈 Análise por trecho"])

    with tab1:
        if trechos.empty:
            warn("Nenhum trecho cadastrado.")
        else:
            filtros = st.columns(3)
            with filtros[0]:
                estados = st.multiselect("Filtrar por estado", sorted(trechos["estado"].dropna().unique()))
            with filtros[1]:
                crits = st.multiselect("Filtrar por criticidade", m.CRITICIDADE_OPERACIONAL)
            with filtros[2]:
                ferrovias = st.multiselect("Filtrar por ferrovia", sorted(trechos["ferrovia"].dropna().unique()))
            df = trechos.copy()
            if estados: df = df[df["estado"].isin(estados)]
            if crits: df = df[df["criticidade_operacional"].isin(crits)]
            if ferrovias: df = df[df["ferrovia"].isin(ferrovias)]
            st.dataframe(clean_df(df), use_container_width=True, hide_index=True)
            df_download_button(df, "railguard_trechos.csv")

    with tab2:
        with st.form("form_trecho", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                codigo = st.text_input("Código do trecho", placeholder="TRC-009")
                ferrovia = st.selectbox("Ferrovia", m.FERROVIAS)
                estado = st.selectbox("Estado", m.ESTADOS, index=m.ESTADOS.index("PA") if "PA" in m.ESTADOS else 0)
                tipo_via = st.selectbox("Tipo de via", m.TIPOS_VIA)
            with c2:
                km_inicial = st.number_input("KM inicial", min_value=0.0, step=0.1, format="%.1f")
                km_final = st.number_input("KM final", min_value=0.0, step=0.1, format="%.1f")
                criticidade = st.selectbox("Criticidade operacional", m.CRITICIDADE_OPERACIONAL)
                observacoes = st.text_area("Observações", height=110)
            submitted = st.form_submit_button("Cadastrar trecho")
            if submitted:
                if not codigo.strip():
                    st.error("Informe o código do trecho.")
                elif km_final <= km_inicial:
                    st.error("O KM final deve ser maior que o KM inicial.")
                else:
                    def action():
                        db.insert_trecho(codigo.strip(), ferrovia, km_inicial, km_final, estado, tipo_via, criticidade, observacoes)
                    safe_insert(action, "Trecho cadastrado com sucesso.", ("INSERÇÃO", "Usuário", f"Trecho {codigo}", f"Trecho {codigo} cadastrado manualmente."))

    with tab3:
        if trechos.empty:
            warn("Sem dados para análise.")
        else:
            df = trechos.copy()
            df["comprimento_km"] = (df["km_final"].astype(float) - df["km_inicial"].astype(float)).abs()
            fig = px.bar(df.sort_values("comprimento_km", ascending=False), x="codigo", y="comprimento_km", color="criticidade_operacional", color_discrete_map={"Baixa": C_BAIXO, "Média": C_MEDIO, "Alta": C_ALTO, "Crítica": C_CRITICO})
            st.plotly_chart(_pt(fig, 360), use_container_width=True)
            if not ativos.empty and not riscos.empty:
                merge = ativos[["id", "trecho_id", "codigo"]].merge(riscos[["ativo_id", "score_risco", "nivel_risco"]], left_on="id", right_on="ativo_id", how="left")
                resumo = merge.groupby("trecho_id").agg(ativos=("id", "count"), risco_medio=("score_risco", "mean")).reset_index()
                resumo = trechos[["id", "codigo", "ferrovia"]].merge(resumo, left_on="id", right_on="trecho_id", how="left")
                st.dataframe(clean_df(resumo[["codigo", "ferrovia", "ativos", "risco_medio"]]), use_container_width=True, hide_index=True)


def page_ativos() -> None:
    ph("⚙️", "Ativos Ferroviários", "Cadastro, consulta e visão técnica dos ativos monitorados")
    ativos = db.get_all_ativos()
    trechos = db.get_all_trechos()
    riscos = latest_risk_per_asset(db.get_all_riscos())

    if not ativos.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Ativos", len(ativos), icon="⚙️")
        with c2: kpi("Tipos", ativos["tipo_ativo"].nunique(), icon="🧩", variant="info")
        with c3: kpi("Idade média", f"{ativos['idade_anos'].mean():.1f} anos", icon="⏱️", variant="warning")
        with c4:
            criticos = 0 if riscos.empty else int((riscos["nivel_risco"] == "Crítico").sum())
            kpi("Risco crítico", criticos, icon="🔴", variant="danger")

    tab1, tab2, tab3 = st.tabs(["📋 Consultar ativos", "➕ Cadastrar ativo", "🔬 Perfil de risco"])

    with tab1:
        if ativos.empty:
            warn("Nenhum ativo cadastrado.")
        else:
            df = ativos.copy()
            if not riscos.empty:
                df = df.merge(riscos[["ativo_id", "score_risco", "nivel_risco", "score_rcrs", "classificacao_rcrs"]], left_on="id", right_on="ativo_id", how="left")
            c1, c2, c3 = st.columns(3)
            with c1: tipo = st.multiselect("Tipo de ativo", sorted(df["tipo_ativo"].dropna().unique()))
            with c2: cond = st.multiselect("Condição visual", m.CONDICAO_VISUAL)
            with c3: trecho = st.multiselect("Trecho", sorted(df["trecho_codigo"].dropna().unique()))
            if tipo: df = df[df["tipo_ativo"].isin(tipo)]
            if cond: df = df[df["condicao_visual"].isin(cond)]
            if trecho: df = df[df["trecho_codigo"].isin(trecho)]
            st.dataframe(clean_df(df), use_container_width=True, hide_index=True)
            df_download_button(df, "railguard_ativos.csv")

    with tab2:
        if trechos.empty:
            warn("Cadastre um trecho antes de cadastrar ativos.")
        else:
            trecho_opcoes = {f'{row["codigo"]} — {row["ferrovia"]} ({row["estado"]})': int(row["id"]) for _, row in trechos.iterrows()}
            with st.form("form_ativo", clear_on_submit=False):
                c1, c2 = st.columns(2)
                with c1:
                    codigo = st.text_input("Código do ativo", placeholder="ATI-023")
                    tipo_ativo = st.selectbox("Tipo de ativo", m.TIPOS_ATIVO)
                    trecho_label = st.selectbox("Trecho associado", list(trecho_opcoes.keys()))
                with c2:
                    idade_anos = st.number_input("Idade do ativo em anos", min_value=0.0, max_value=100.0, step=0.5)
                    data_ultima_manutencao = st.date_input("Data da última manutenção", value=datetime.now().date() - timedelta(days=90))
                    condicao_visual = st.selectbox("Condição visual", m.CONDICAO_VISUAL, index=1)
                observacoes = st.text_area("Observações", height=100)
                submitted = st.form_submit_button("Cadastrar ativo")
                if submitted:
                    if not codigo.strip():
                        st.error("Informe o código do ativo.")
                    else:
                        def action():
                            db.insert_ativo(codigo.strip(), tipo_ativo, trecho_opcoes[trecho_label], idade_anos, str(data_ultima_manutencao), condicao_visual, observacoes)
                        safe_insert(action, "Ativo cadastrado com sucesso.", ("INSERÇÃO", "Usuário", f"Ativo {codigo}", f"Ativo {codigo} cadastrado manualmente."))

    with tab3:
        if ativos.empty:
            warn("Sem ativos para análise.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                dft = ativos["tipo_ativo"].value_counts().reset_index()
                dft.columns = ["Tipo", "Quantidade"]
                fig = px.bar(dft, x="Tipo", y="Quantidade", text="Quantidade")
                st.plotly_chart(_pt(fig, 330), use_container_width=True)
            with c2:
                dfc = ativos["condicao_visual"].value_counts().reset_index()
                dfc.columns = ["Condição", "Quantidade"]
                fig = px.pie(dfc, names="Condição", values="Quantidade", hole=0.45)
                st.plotly_chart(_pt(fig, 330), use_container_width=True)
            if not riscos.empty:
                top = ativos.merge(riscos[["ativo_id", "score_risco", "nivel_risco"]], left_on="id", right_on="ativo_id", how="inner")
                top = top.sort_values("score_risco", ascending=False).head(10)
                fig = px.bar(top, x="codigo", y="score_risco", color="nivel_risco", color_discrete_map=RISK_COLORS_PLOTLY, hover_data=["tipo_ativo", "trecho_codigo"])
                st.plotly_chart(_pt(fig, 360), use_container_width=True)


def page_inspecoes() -> None:
    ph("🔍", "Inspeções", "Registro técnico, anomalias e cálculo automático de risco operacional")
    inspecoes = db.get_all_inspecoes()
    ativos = db.get_all_ativos()
    riscos = db.get_all_riscos()

    tab1, tab2, tab3 = st.tabs(["📋 Consultar inspeções", "➕ Registrar inspeção", "🧠 Explicabilidade"])

    with tab1:
        if inspecoes.empty:
            warn("Nenhuma inspeção registrada.")
        else:
            df = inspecoes.copy()
            if not riscos.empty:
                df = df.merge(riscos[["inspecao_id", "score_risco", "nivel_risco", "score_rcrs", "classificacao_rcrs"]], left_on="id", right_on="inspecao_id", how="left")
            c1, c2, c3 = st.columns(3)
            with c1: tipos = st.multiselect("Tipo", sorted(df["tipo_inspecao"].dropna().unique()))
            with c2: ativos_f = st.multiselect("Ativo", sorted(df["ativo_codigo"].dropna().unique()))
            with c3: resp = st.multiselect("Responsável", sorted(df["responsavel"].dropna().unique()))
            if tipos: df = df[df["tipo_inspecao"].isin(tipos)]
            if ativos_f: df = df[df["ativo_codigo"].isin(ativos_f)]
            if resp: df = df[df["responsavel"].isin(resp)]
            st.dataframe(clean_df(df), use_container_width=True, hide_index=True)
            df_download_button(df, "railguard_inspecoes.csv")

    with tab2:
        if ativos.empty:
            warn("Cadastre um ativo antes de registrar inspeções.")
        else:
            ativo_opcoes = {f'{row["codigo"]} — {row["tipo_ativo"]} · {row["trecho_codigo"]}': int(row["id"]) for _, row in ativos.iterrows()}
            with st.form("form_inspecao", clear_on_submit=False):
                c1, c2 = st.columns(2)
                with c1:
                    ativo_label = st.selectbox("Ativo inspecionado", list(ativo_opcoes.keys()))
                    data_inspecao = st.date_input("Data da inspeção", value=datetime.now().date())
                    responsavel = st.text_input("Responsável", value="Usuário")
                    tipo_inspecao = st.selectbox("Tipo de inspeção", m.TIPOS_INSPECAO)
                    temperatura = st.number_input("Temperatura medida (°C)", value=30.0, step=0.5)
                with c2:
                    st.markdown("**Anomalias detectadas**")
                    fissura = st.checkbox("Fissura detectada")
                    desgaste = st.checkbox("Desgaste detectado")
                    corrosao = st.checkbox("Corrosão detectada")
                    falha_fixacao = st.checkbox("Falha de fixação")
                    nivel_vibracao = st.slider("Nível de vibração", 0.0, 10.0, 3.0, 0.1)
                    carga_operacional = st.slider("Carga operacional (%)", 0.0, 100.0, 60.0, 0.5)
                observacoes = st.text_area("Observações", height=100)
                submitted = st.form_submit_button("Registrar inspeção e calcular risco")
                if submitted:
                    ativo_id = ativo_opcoes[ativo_label]
                    ativo = db.get_ativo_by_id(ativo_id)
                    if not ativo:
                        st.error("Ativo não encontrado.")
                    else:
                        trecho = db.get_trecho_by_id(ativo["trecho_id"])
                        dias_manutencao = max((pd.to_datetime(str(data_inspecao)) - pd.to_datetime(ativo.get("data_ultima_manutencao"))).days, 0)
                        score, nivel, contrib = risk_engine.calcular_risco_operacional(
                            idade_anos=float(ativo["idade_anos"]),
                            dias_desde_manutencao=int(dias_manutencao),
                            criticidade_operacional=trecho["criticidade_operacional"] if trecho else "Média",
                            fissura=fissura,
                            desgaste=desgaste,
                            corrosao=corrosao,
                            falha_fixacao=falha_fixacao,
                            nivel_vibracao=float(nivel_vibracao),
                            carga_operacional=float(carga_operacional),
                            condicao_visual=ativo["condicao_visual"],
                        )
                        rcrs, class_rcrs, recomendacao = risk_engine.calcular_rcrs(
                            score_risco_operacional=score,
                            criticidade_trecho=trecho["criticidade_operacional"] if trecho else "Média",
                            historico_falhas=int(fissura) + int(desgaste) + int(corrosao) + int(falha_fixacao),
                            impacto_regulatorio=score * 0.8,
                            risco_esg=score * 0.55,
                            confiabilidade_dado=85,
                        )
                        try:
                            inspecao_id = db.insert_inspecao(ativo_id, str(data_inspecao), responsavel, tipo_inspecao, fissura, desgaste, corrosao, falha_fixacao, nivel_vibracao, temperatura, carga_operacional, observacoes)
                            db.insert_risco(inspecao_id, ativo_id, score, nivel, rcrs, class_rcrs, recomendacao)
                            db.insert_auditoria("INSERÇÃO", responsavel, f"Inspeção {ativo['codigo']}", f"Inspeção registrada com risco {nivel}, score {score:.1f} e RCRS {class_rcrs}.")
                            st.success(f"Inspeção registrada. Risco: {nivel} | Score: {score:.1f}/100 | RCRS: {rcrs:.1f}/100")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Não foi possível registrar a inspeção: {exc}")

    with tab3:
        if riscos.empty or inspecoes.empty or ativos.empty:
            warn("Registre inspeções com risco para visualizar a explicabilidade.")
        else:
            insp_map = inspecoes.merge(ativos[["id", "codigo", "idade_anos", "condicao_visual", "criticidade_operacional", "data_ultima_manutencao"]], left_on="ativo_id", right_on="id", how="left", suffixes=("", "_ativo"))
            riscos_view = riscos.merge(insp_map, left_on="inspecao_id", right_on="id", how="left", suffixes=("", "_insp"))
            riscos_view = riscos_view.sort_values("score_risco", ascending=False)
            label_map = {f"{row.get('ativo_codigo', row.get('codigo', 'Ativo'))} · Inspeção {int(row['inspecao_id'])} · {row['nivel_risco']} ({row['score_risco']:.1f})": idx for idx, row in riscos_view.iterrows()}
            label = st.selectbox("Selecione uma inspeção calculada", list(label_map.keys()))
            row = riscos_view.loc[label_map[label]]
            dias = 365
            try:
                dias = max((pd.to_datetime(row["data_inspecao"]) - pd.to_datetime(row["data_ultima_manutencao"])).days, 0)
            except Exception:
                pass
            score, nivel, contrib = risk_engine.calcular_risco_operacional(
                idade_anos=float(row.get("idade_anos", 0)),
                dias_desde_manutencao=int(dias),
                criticidade_operacional=row.get("criticidade_operacional", "Média"),
                fissura=bool(row.get("fissura", 0)),
                desgaste=bool(row.get("desgaste", 0)),
                corrosao=bool(row.get("corrosao", 0)),
                falha_fixacao=bool(row.get("falha_fixacao", 0)),
                nivel_vibracao=float(row.get("nivel_vibracao", 0)),
                carga_operacional=float(row.get("carga_operacional", 0)),
                condicao_visual=row.get("condicao_visual", "Bom"),
            )
            c1, c2 = st.columns([0.8, 1.2])
            with c1:
                st.plotly_chart(gauge_risco(score, nivel), use_container_width=True)
            with c2:
                texto = risk_engine.gerar_explicabilidade(score, nivel, contrib, bool(row.get("fissura", 0)), bool(row.get("desgaste", 0)), bool(row.get("corrosao", 0)), bool(row.get("falha_fixacao", 0)), float(row.get("nivel_vibracao", 0)), float(row.get("idade_anos", 0)), int(dias), row.get("criticidade_operacional", "Média"), row.get("condicao_visual", "Bom"))
                info(texto.replace("\n", "<br>"))
                contrib_df = pd.DataFrame([{"Fator": k, "Contribuição": v} for k, v in contrib.items()]).sort_values("Contribuição", ascending=True)
                fig = px.bar(contrib_df, x="Contribuição", y="Fator", orientation="h")
                st.plotly_chart(_pt(fig, 300), use_container_width=True)


@st.cache_resource(show_spinner=False)
def treinar_modelo_cache(n_amostras: int = 900):
    X, y = ml.gerar_dados_sinteticos(n_amostras)
    return ml.treinar_modelo(X, y)


def page_ml() -> None:
    ph("🤖", "Modelo Preditivo", "RandomForest para previsão de risco operacional")
    inspecoes = db.get_all_inspecoes()
    ativos = db.get_all_ativos()
    trechos = db.get_all_trechos()

    tab1, tab2, tab3 = st.tabs(["📊 Treinamento", "🎯 Predição interativa", "🧾 Dados do modelo"])

    with tab1:
        usar_reais = st.toggle("Tentar usar dados reais do banco quando houver volume suficiente", value=True)
        X_real, y_real = (None, None)
        if usar_reais and not inspecoes.empty and not ativos.empty and not trechos.empty:
            X_real, y_real = ml.preparar_features_reais(inspecoes, ativos, trechos)
        if X_real is not None and y_real is not None:
            X, y = X_real, y_real
            origem = "Dados reais cadastrados no banco"
        else:
            X, y = ml.gerar_dados_sinteticos(900)
            origem = "Dados sintéticos gerados pelo motor de risco"
        model, accuracy, X_test, y_test, report = ml.treinar_modelo(X, y)
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Acurácia", f"{accuracy*100:.1f}%", icon="🎯", variant="success")
        with c2: kpi("Amostras", len(X), icon="🧪", variant="info")
        with c3: kpi("Features", X.shape[1], icon="🧬")
        with c4: kpi("Origem", "Real" if X_real is not None else "Sintética", icon="🗃️", variant="warning")
        info(f"Origem do treinamento: <b>{origem}</b>. O MVP usa o motor de risco como oráculo para gerar ou rotular exemplos.")
        fi = ml.get_feature_importance(model)
        fig = px.bar(fi.sort_values("Importância", ascending=True), x="Importância", y="Variável", orientation="h")
        st.plotly_chart(_pt(fig, 430), use_container_width=True)
        with st.expander("Relatório de classificação"):
            st.code(report)

    with tab2:
        model, accuracy, X_test, y_test, report = treinar_modelo_cache(900)
        c1, c2 = st.columns([1, 1])
        with c1:
            idade_anos = st.number_input("Idade do ativo (anos)", 0.0, 60.0, 12.0, 0.5)
            dias_desde_manutencao = st.number_input("Dias desde última manutenção", 0, 1500, 180, 10)
            criticidade = st.selectbox("Criticidade operacional", m.CRITICIDADE_OPERACIONAL, index=2)
            condicao = st.selectbox("Condição visual", m.CONDICAO_VISUAL, index=2)
            nivel_vibracao = st.slider("Nível de vibração", 0.0, 10.0, 4.0, 0.1)
            carga_operacional = st.slider("Carga operacional (%)", 0.0, 100.0, 70.0, 0.5)
        with c2:
            fissura = st.checkbox("Fissura")
            desgaste = st.checkbox("Desgaste", value=True)
            corrosao = st.checkbox("Corrosão")
            falha_fixacao = st.checkbox("Falha de fixação")
            params = {
                "idade_anos": idade_anos,
                "dias_desde_manutencao": dias_desde_manutencao,
                "criticidade_operacional": m.CRITICIDADE_PESO.get(criticidade, 2),
                "fissura": int(fissura),
                "desgaste": int(desgaste),
                "corrosao": int(corrosao),
                "falha_fixacao": int(falha_fixacao),
                "nivel_vibracao": nivel_vibracao,
                "carga_operacional": carga_operacional,
                "condicao_visual": m.CONDICAO_PESO.get(condicao, 2),
            }
            pred, probs = ml.prever_risco(model, params)
            score_rule, nivel_rule, contrib = risk_engine.calcular_risco_operacional(idade_anos, dias_desde_manutencao, criticidade, fissura, desgaste, corrosao, falha_fixacao, nivel_vibracao, carga_operacional, condicao)
            st.plotly_chart(gauge_risco(score_rule, nivel_rule), use_container_width=True)
            st.markdown(f"### Predição ML: {badge(pred)}", unsafe_allow_html=True)
            prob_df = pd.DataFrame({"Classe": ["Baixo", "Médio", "Alto", "Crítico"], "Probabilidade": probs})
            fig = px.bar(prob_df, x="Classe", y="Probabilidade", color="Classe", color_discrete_map=RISK_COLORS_PLOTLY)
            st.plotly_chart(_pt(fig, 280), use_container_width=True)

    with tab3:
        X, y = ml.gerar_dados_sinteticos(600)
        preview = X.copy()
        preview["nivel_risco"] = y.map(ml.INT_TO_RISK)
        st.dataframe(preview.head(100), use_container_width=True, hide_index=True)
        df_download_button(preview, "railguard_dataset_ml_sintetico.csv")


def page_compliance() -> None:
    ph("📋", "Compliance & RCRS", "Railway Compliance Risk Score e não conformidades potenciais")
    riscos = db.get_all_riscos()
    trechos = db.get_all_trechos()

    if not riscos.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("RCRS médio", f"{riscos['score_rcrs'].mean():.1f}", icon="📋")
        with c2: kpi("Conforme", int((riscos["classificacao_rcrs"] == "Conforme").sum()), icon="✅", variant="success")
        with c3: kpi("Atenção", int((riscos["classificacao_rcrs"] == "Atenção").sum()), icon="⚠️", variant="warning")
        with c4: kpi("Crítico", int((riscos["classificacao_rcrs"] == "Crítico").sum()), icon="🔴", variant="danger")

    tab1, tab2 = st.tabs(["📊 Painel RCRS", "🧮 Cálculo manual"])
    with tab1:
        if riscos.empty:
            warn("Nenhum RCRS calculado ainda.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                dfr = riscos["classificacao_rcrs"].value_counts().reset_index()
                dfr.columns = ["Classificação", "Quantidade"]
                fig = px.pie(dfr, names="Classificação", values="Quantidade", hole=0.45, color="Classificação", color_discrete_map=RCRS_COLORS_PLOTLY)
                st.plotly_chart(_pt(fig, 350), use_container_width=True)
            with c2:
                top = riscos.sort_values("score_rcrs", ascending=False).head(10)
                fig = px.bar(top, x="ativo_codigo", y="score_rcrs", color="classificacao_rcrs", color_discrete_map=RCRS_COLORS_PLOTLY, hover_data=["recomendacao"])
                st.plotly_chart(_pt(fig, 350), use_container_width=True)
            cols = ["ativo_codigo", "tipo_ativo", "score_risco", "nivel_risco", "score_rcrs", "classificacao_rcrs", "recomendacao", "data_calculo"]
            st.dataframe(clean_df(riscos[[c for c in cols if c in riscos.columns]]), use_container_width=True, hide_index=True)
            df_download_button(riscos, "railguard_compliance_rcrs.csv")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            score_risco_operacional = st.slider("Score de risco operacional", 0.0, 100.0, 45.0, 0.5)
            criticidade_trecho = st.selectbox("Criticidade do trecho", m.CRITICIDADE_OPERACIONAL, index=2)
            historico_falhas = st.number_input("Histórico de falhas", 0, 20, 2)
        with c2:
            impacto_regulatorio = st.slider("Impacto regulatório", 0.0, 100.0, 50.0, 0.5)
            risco_esg = st.slider("Risco ESG", 0.0, 100.0, 40.0, 0.5)
            confiabilidade_dado = st.slider("Confiabilidade do dado", 0.0, 100.0, 85.0, 0.5)
        rcrs, classificacao, recomendacao = risk_engine.calcular_rcrs(score_risco_operacional, criticidade_trecho, int(historico_falhas), impacto_regulatorio, risco_esg, confiabilidade_dado)
        st.markdown(f"### Resultado: {badge(classificacao)}", unsafe_allow_html=True)
        st.progress(int(rcrs), text=f"RCRS: {rcrs:.1f}/100")
        info(recomendacao)


def page_auditoria() -> None:
    ph("🗂️", "Auditoria", "Rastreabilidade das ações executadas na plataforma")
    auditoria = db.get_all_auditoria()
    if auditoria.empty:
        warn("Nenhum registro de auditoria encontrado.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Eventos", len(auditoria), icon="🗂️")
    with c2: kpi("Usuários", auditoria["usuario"].nunique(), icon="👤", variant="info")
    with c3: kpi("Tipos de ação", auditoria["tipo_acao"].nunique(), icon="🔧", variant="success")
    with c4: kpi("Última ação", str(auditoria.iloc[0]["data_hora"])[:10], icon="🕘", variant="warning")

    f1, f2, f3 = st.columns(3)
    with f1: tipos = st.multiselect("Tipo de ação", sorted(auditoria["tipo_acao"].dropna().unique()))
    with f2: usuarios = st.multiselect("Usuário", sorted(auditoria["usuario"].dropna().unique()))
    with f3: termo = st.text_input("Buscar em item/descrição")
    df = auditoria.copy()
    if tipos: df = df[df["tipo_acao"].isin(tipos)]
    if usuarios: df = df[df["usuario"].isin(usuarios)]
    if termo:
        mask = df["item_alterado"].astype(str).str.contains(termo, case=False, na=False) | df["descricao"].astype(str).str.contains(termo, case=False, na=False)
        df = df[mask]
    col_a, col_b = st.columns([1, 1])
    with col_a:
        fig = px.bar(df["tipo_acao"].value_counts().reset_index().rename(columns={"tipo_acao": "Tipo", "count": "Quantidade"}), x="Tipo", y="Quantidade") if not df.empty else go.Figure()
        st.plotly_chart(_pt(fig, 320), use_container_width=True)
    with col_b:
        dft = df.copy()
        dft["data_hora"] = pd.to_datetime(dft["data_hora"], errors="coerce")
        dft["Dia"] = dft["data_hora"].dt.date.astype(str)
        serie = dft.groupby("Dia").size().reset_index(name="Eventos") if not dft.empty else pd.DataFrame(columns=["Dia", "Eventos"])
        fig = px.line(serie, x="Dia", y="Eventos", markers=True)
        st.plotly_chart(_pt(fig, 320), use_container_width=True)
    st.dataframe(clean_df(df), use_container_width=True, hide_index=True)
    df_download_button(df, "railguard_auditoria.csv")


def page_esg() -> None:
    ph("🌿", "ESG", "Indicadores ambientais, impacto de paralisação e eficiência de manutenção")
    esg = db.get_all_esg()
    trechos = db.get_all_trechos()
    ativos = db.get_all_ativos()
    riscos = latest_risk_per_asset(db.get_all_riscos())

    tab1, tab2 = st.tabs(["📊 Indicadores ESG", "➕ Calcular indicador"])
    with tab1:
        if esg.empty:
            warn("Nenhum indicador ESG calculado.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1: kpi("Registros ESG", len(esg), icon="🌿")
            with c2: kpi("CO₂ estimado", f"{esg['emissao_co2_estimada'].sum():.1f} t", icon="🏭", variant="warning")
            with c3: kpi("Área impacto", f"{esg['area_impacto_km2'].sum():.1f} km²", icon="🗺️", variant="info")
            with c4: kpi("Prioridade crítica", int((esg["prioridade_esg"] == "Crítica").sum()), icon="🔴", variant="danger")
            c1, c2 = st.columns([1, 1])
            with c1:
                dfe = esg["prioridade_esg"].value_counts().reset_index()
                dfe.columns = ["Prioridade", "Quantidade"]
                fig = px.pie(dfe, names="Prioridade", values="Quantidade", hole=0.45, color="Prioridade", color_discrete_map=ESG_COLORS_PLOTLY)
                st.plotly_chart(_pt(fig, 350), use_container_width=True)
            with c2:
                cols = ["risco_ambiental", "impacto_paralisacao", "eficiencia_manutencao"]
                mean_vals = esg[cols].mean().reset_index()
                mean_vals.columns = ["Indicador", "Valor"]
                fig = px.bar(mean_vals, x="Indicador", y="Valor", text="Valor")
                st.plotly_chart(_pt(fig, 350), use_container_width=True)
            st.dataframe(clean_df(esg), use_container_width=True, hide_index=True)
            df_download_button(esg, "railguard_esg.csv")

    with tab2:
        if trechos.empty:
            warn("Cadastre trechos antes de calcular indicadores ESG.")
        else:
            trecho_opcoes = {f'{row["codigo"]} — {row["ferrovia"]}': int(row["id"]) for _, row in trechos.iterrows()}
            label = st.selectbox("Trecho", list(trecho_opcoes.keys()))
            tid = trecho_opcoes[label]
            trecho = db.get_trecho_by_id(tid)
            comprimento = abs(float(trecho["km_final"]) - float(trecho["km_inicial"])) if trecho else 0
            ativos_t = ativos[ativos["trecho_id"] == tid] if not ativos.empty else pd.DataFrame()
            ncrit = 0
            if not ativos_t.empty and not riscos.empty:
                ncrit = int(riscos[riscos["ativo_id"].isin(ativos_t["id"])]["nivel_risco"].isin(["Crítico", "Alto"]).sum())
            media_dias = st.slider("Média estimada de dias desde última manutenção", 0.0, 730.0, 180.0, 5.0)
            ncrit = st.number_input("Número de ativos críticos/altos no trecho", 0, 100, int(ncrit))
            if st.button("Calcular e salvar ESG"):
                esg_calc = risk_engine.calcular_esg_indicators(trecho["codigo"], trecho["criticidade_operacional"], int(ncrit), comprimento, media_dias)
                try:
                    db.insert_esg(tid, esg_calc["risco_ambiental"], esg_calc["impacto_paralisacao"], esg_calc["eficiencia_manutencao"], esg_calc["prioridade_esg"], esg_calc["emissao_co2_estimada"], esg_calc["area_impacto_km2"], esg_calc["recomendacoes"])
                    db.insert_auditoria("INSERÇÃO", "Usuário", f"ESG {trecho['codigo']}", f"Indicador ESG calculado com prioridade {esg_calc['prioridade_esg']}.")
                    st.success("Indicador ESG calculado e salvo com sucesso.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Não foi possível salvar o indicador ESG: {exc}")


def page_relatorios() -> None:
    ph("📄", "Relatórios", "Relatórios de compliance por ativo ou por trecho com exportação CSV")
    ativos = db.get_all_ativos()
    trechos = db.get_all_trechos()
    tab1, tab2, tab3 = st.tabs(["🚆 Relatório por ativo", "🛤️ Relatório por trecho", "📦 Exportação geral"])

    with tab1:
        if ativos.empty:
            warn("Nenhum ativo disponível para relatório.")
        else:
            ativo_opcoes = {f'{row["codigo"]} — {row["tipo_ativo"]} · {row["trecho_codigo"]}': int(row["id"]) for _, row in ativos.iterrows()}
            label = st.selectbox("Selecione o ativo", list(ativo_opcoes.keys()))
            rel = rp.gerar_relatorio_ativo(ativo_opcoes[label])
            if rel is None:
                st.error("Relatório não encontrado.")
            else:
                ativo = rel["ativo"]
                trecho = rel["trecho"] or {}
                ult_risco = rel["riscos"].iloc[0] if not rel["riscos"].empty else None
                nivel_badge = badge(ult_risco["nivel_risco"]) if ult_risco is not None else badge("Sem risco")
                st.markdown(
                    f"""
                    <div class="exec-report-header">
                        <div class="exec-title">Relatório de Compliance — {ativo.get('codigo', 'Ativo')}</div>
                        <div class="exec-subtitle">{ativo.get('tipo_ativo', '—')} · Trecho {trecho.get('codigo', '—')} · Gerado em {rel['data_relatorio']}</div>
                        <div class="exec-badge">{nivel_badge}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    card_title("Dados do ativo")
                    st.markdown(row_kv("Código", ativo.get("codigo", "—")) + row_kv("Tipo", ativo.get("tipo_ativo", "—")) + row_kv("Idade", f"{ativo.get('idade_anos', 0)} anos") + row_kv("Condição", ativo.get("condicao_visual", "—")), unsafe_allow_html=True)
                with c2:
                    card_title("Trecho associado")
                    st.markdown(row_kv("Código", trecho.get("codigo", "—")) + row_kv("Ferrovia", trecho.get("ferrovia", "—")) + row_kv("UF", trecho.get("estado", "—")) + row_kv("Criticidade", trecho.get("criticidade_operacional", "—")), unsafe_allow_html=True)
                with c3:
                    card_title("Último risco")
                    if ult_risco is not None:
                        st.plotly_chart(gauge_risco(float(ult_risco["score_risco"]), str(ult_risco["nivel_risco"]), h=220), use_container_width=True)
                    else:
                        info("Sem risco calculado.")
                st.subheader("Inspeções")
                st.dataframe(clean_df(rel["inspecoes"]), use_container_width=True, hide_index=True)
                st.subheader("Scores de risco e RCRS")
                st.dataframe(clean_df(rel["riscos"]), use_container_width=True, hide_index=True)
                st.subheader("Alertas e auditoria relacionada")
                st.dataframe(clean_df(rel["alertas"]), use_container_width=True, hide_index=True)
                st.dataframe(clean_df(rel["auditoria"]), use_container_width=True, hide_index=True)
                csv = rp.relatorio_para_csv(rel)
                st.download_button("⬇️ Baixar relatório do ativo em CSV", data=csv, file_name=f"relatorio_{ativo.get('codigo', 'ativo')}.csv", mime="text/csv")

    with tab2:
        if trechos.empty:
            warn("Nenhum trecho disponível para relatório.")
        else:
            trecho_opcoes = {f'{row["codigo"]} — {row["ferrovia"]}': int(row["id"]) for _, row in trechos.iterrows()}
            label = st.selectbox("Selecione o trecho", list(trecho_opcoes.keys()))
            tid = trecho_opcoes[label]
            trecho = db.get_trecho_by_id(tid)
            st.markdown(row_kv("Trecho", trecho.get("codigo", "—")) + row_kv("Ferrovia", trecho.get("ferrovia", "—")) + row_kv("Extensão", f"{abs(float(trecho['km_final'])-float(trecho['km_inicial'])):.1f} km") + row_kv("Criticidade", trecho.get("criticidade_operacional", "—")), unsafe_allow_html=True)
            csv = rp.relatorio_trecho_para_csv(tid)
            st.download_button("⬇️ Baixar relatório do trecho em CSV", data=csv, file_name=f"relatorio_{trecho.get('codigo', 'trecho')}.csv", mime="text/csv")
            ativos_t = ativos[ativos["trecho_id"] == tid] if not ativos.empty else pd.DataFrame()
            st.dataframe(clean_df(ativos_t), use_container_width=True, hide_index=True)

    with tab3:
        bases = {
            "Trechos": db.get_all_trechos(),
            "Ativos": db.get_all_ativos(),
            "Inspeções": db.get_all_inspecoes(),
            "Riscos": db.get_all_riscos(),
            "Alertas": db.get_all_alertas(),
            "Auditoria": db.get_all_auditoria(),
            "ESG": db.get_all_esg(),
        }
        escolha = st.selectbox("Base para exportar", list(bases.keys()))
        st.dataframe(clean_df(bases[escolha]), use_container_width=True, hide_index=True)
        df_download_button(bases[escolha], f"railguard_{escolha.lower()}.csv")


def page_config() -> None:
    ph("⚙️", "Configurações", "Informações do sistema, ambiente e orientações de operação")
    stats = db.get_dashboard_stats()
    tab1, tab2, tab3 = st.tabs(["🧭 Sistema", "🗄️ Banco de dados", "🚀 Deploy Streamlit Cloud"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Versão MVP", "0.2", icon="🚂")
        with c2: kpi("Streamlit", st.__version__, icon="🌐", variant="info")
        with c3: kpi("Python", sys.version.split()[0], icon="🐍", variant="success")
        with c4: kpi("Registros", sum([stats.get("total_trechos", 0), stats.get("total_ativos", 0), stats.get("total_inspecoes", 0)]), icon="🧾", variant="warning")
        info("Este MVP usa dados simulados para demonstração acadêmica. Não há integração real com ANTT, operadores ferroviários ou sensores IoT.")
        st.markdown("""
        **Arquivos esperados no projeto:** `app.py`, `database.py`, `models.py`, `risk_engine.py`, `ml_model.py`, `reports.py`, `seed_data.py`, `requirements.txt`.
        """)

    with tab2:
        db_path = getattr(db, "DB_PATH", "data/railguard.db")
        exists = os.path.exists(db_path)
        st.markdown(row_kv("Banco", "SQLite") + row_kv("Caminho", db_path) + row_kv("Arquivo existe", "Sim" if exists else "Não") + row_kv("Tamanho", f"{os.path.getsize(db_path)/1024:.1f} KB" if exists else "—"), unsafe_allow_html=True)
        warn("Para evitar perda de dados, esta tela não apaga nem recria o banco automaticamente. Se quiser resetar a demo, remova o arquivo `data/railguard.db` no repositório/ambiente e reinicie o app.")
        st.code("""# Migração futura para PostgreSQL
# 1) Instalar psycopg2-binary e sqlalchemy
# 2) Substituir get_connection() em database.py
# 3) Migrar as tabelas: trechos, ativos, inspecoes, riscos, alertas, auditoria, indicadores_esg""", language="python")

    with tab3:
        st.markdown("""
        Para rodar no **Streamlit Community Cloud**:

        1. Mantenha `app.py` na raiz do repositório.
        2. Garanta que `requirements.txt` contenha `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn` e `python-dateutil`.
        3. Não suba ambientes virtuais (`venv/`) nem caches (`__pycache__/`).
        4. Se o banco não existir, o app cria `data/railguard.db` e popula os dados de demonstração automaticamente.
        """)
        st.code("streamlit run app.py", language="bash")

# ═══════════════════════════════════════════════════════════════
# ROTEAMENTO FINAL
# ═══════════════════════════════════════════════════════════════

if pagina == "Dashboard":
    page_dashboard()
elif pagina == "Trechos":
    page_trechos()
elif pagina == "Ativos":
    page_ativos()
elif pagina == "Inspecoes":
    page_inspecoes()
elif pagina == "ML":
    page_ml()
elif pagina == "Compliance":
    page_compliance()
elif pagina == "Auditoria":
    page_auditoria()
elif pagina == "ESG":
    page_esg()
elif pagina == "Relatorios":
    page_relatorios()
elif pagina == "Config":
    page_config()
