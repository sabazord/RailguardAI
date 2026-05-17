"""
RailGuard AI — Aplicação Principal (UI Redesign v2)
=====================================================
Plataforma de Compliance Preditivo Ferroviário
MVP demonstrativo — dados simulados.

Execução:
    streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

import database as db
import models as m
import risk_engine as re_
import ml_model as ml
import reports as rp
import seed_data

# ═══════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO DA PÁGINA
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RailGuard AI",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#  PALETA — TEMA CORPORATIVO FERROVIÁRIO DARK
# ═══════════════════════════════════════════════════════════════

PRIMARY    = "#0A1628"
SECONDARY  = "#102A4C"
ACCENT     = "#E63946"
ACCENT2    = "#1D7EC2"
SURFACE    = "#0F1F38"
CARD_BG    = "#132035"
BORDER     = "#1E3A5F"
TEXT_PRI   = "#E8EDF2"
TEXT_SEC   = "#8FA8BF"
TEXT_MUT   = "#4A6580"

C_BAIXO   = "#00C48C"
C_MEDIO   = "#FFB020"
C_ALTO    = "#FF6B35"
C_CRITICO = "#E63946"

RISK_COLORS_PLOTLY = {
    "Baixo":   C_BAIXO,
    "Médio":   C_MEDIO,
    "Alto":    C_ALTO,
    "Crítico": C_CRITICO,
}


# ═══════════════════════════════════════════════════════════════
#  CSS GLOBAL
# ═══════════════════════════════════════════════════════════════

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif!important}}
.stApp{{background:{PRIMARY};background-image:linear-gradient(rgba(30,58,95,0.12) 1px,transparent 1px),linear-gradient(90deg,rgba(30,58,95,0.12) 1px,transparent 1px);background-size:40px 40px}}
.main .block-container{{padding:1.5rem 2rem 3rem 2rem;max-width:1600px}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,{SECONDARY} 0%,{PRIMARY} 100%);border-right:1px solid {BORDER}}}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p{{color:{TEXT_PRI}!important;font-size:0.88rem!important;font-weight:500}}
div[data-testid="stSidebarNav"]{{display:none}}
.rg-page-header{{display:flex;align-items:center;gap:14px;padding:20px 0 16px 0;border-bottom:2px solid {BORDER};margin-bottom:24px}}
.rg-page-icon{{width:44px;height:44px;background:linear-gradient(135deg,{ACCENT2},#0D5C8A);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;box-shadow:0 4px 15px rgba(29,126,194,0.4)}}
.rg-page-title{{font-size:1.6rem;font-weight:800;color:{TEXT_PRI};letter-spacing:-0.02em;line-height:1.2}}
.rg-page-subtitle{{font-size:0.8rem;color:{TEXT_SEC};font-weight:400;margin-top:2px}}
.kpi-card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:20px 22px;position:relative;overflow:hidden}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px}}
.kpi-card.kpi-primary::before{{background:linear-gradient(90deg,{ACCENT2},{PRIMARY})}}
.kpi-card.kpi-success::before{{background:linear-gradient(90deg,{C_BAIXO},{PRIMARY})}}
.kpi-card.kpi-warning::before{{background:linear-gradient(90deg,{C_MEDIO},{PRIMARY})}}
.kpi-card.kpi-danger::before{{background:linear-gradient(90deg,{C_CRITICO},{PRIMARY})}}
.kpi-card.kpi-info::before{{background:linear-gradient(90deg,#9B59B6,{PRIMARY})}}
.kpi-label{{font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:{TEXT_MUT};margin-bottom:8px}}
.kpi-value{{font-size:2.1rem;font-weight:800;color:{TEXT_PRI};line-height:1;font-variant-numeric:tabular-nums}}
.kpi-delta{{font-size:0.77rem;margin-top:6px;font-weight:500}}
.kpi-delta.up{{color:{C_CRITICO}}}.kpi-delta.down{{color:{C_BAIXO}}}.kpi-delta.neutral{{color:{TEXT_SEC}}}
.kpi-icon{{position:absolute;right:18px;top:18px;font-size:2rem;opacity:0.12}}
.rg-card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:20px 22px;margin-bottom:16px}}
.rg-card-title{{font-size:0.82rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{TEXT_SEC};margin-bottom:14px;display:flex;align-items:center;gap:8px}}
.rg-card-title::before{{content:'';display:inline-block;width:3px;height:14px;background:{ACCENT2};border-radius:2px}}
.rbadge{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase}}
.rbadge-baixo{{background:rgba(0,196,140,0.18);color:{C_BAIXO};border:1px solid rgba(0,196,140,0.4)}}
.rbadge-medio{{background:rgba(255,176,32,0.18);color:{C_MEDIO};border:1px solid rgba(255,176,32,0.4)}}
.rbadge-alto{{background:rgba(255,107,53,0.18);color:{C_ALTO};border:1px solid rgba(255,107,53,0.4)}}
.rbadge-critico{{background:rgba(230,57,70,0.18);color:{C_CRITICO};border:1px solid rgba(230,57,70,0.4)}}
.alert-item{{display:flex;align-items:flex-start;gap:14px;padding:14px 16px;border-radius:10px;margin-bottom:10px;border:1px solid}}
.alert-critico{{background:rgba(230,57,70,0.08);border-color:rgba(230,57,70,0.35)}}
.alert-alto{{background:rgba(255,107,53,0.08);border-color:rgba(255,107,53,0.35)}}
.alert-medio{{background:rgba(255,176,32,0.08);border-color:rgba(255,176,32,0.35)}}
.alert-pulse{{width:10px;height:10px;border-radius:50%;margin-top:3px;flex-shrink:0;animation:pulse 1.8s infinite}}
.pulse-critico{{background:{C_CRITICO};box-shadow:0 0 0 0 rgba(230,57,70,0.5)}}
.pulse-alto{{background:{C_ALTO};box-shadow:0 0 0 0 rgba(255,107,53,0.5)}}
.pulse-medio{{background:{C_MEDIO};box-shadow:0 0 0 0 rgba(255,176,32,0.5)}}
@keyframes pulse{{0%{{box-shadow:0 0 0 0 rgba(230,57,70,0.5)}}70%{{box-shadow:0 0 0 8px rgba(230,57,70,0)}}100%{{box-shadow:0 0 0 0 rgba(230,57,70,0)}}}}
.alert-body{{flex:1}}.alert-titulo{{font-size:0.85rem;font-weight:700;color:{TEXT_PRI}}}
.alert-msg{{font-size:0.78rem;color:{TEXT_SEC};margin-top:3px;line-height:1.5}}
.alert-meta{{font-size:0.7rem;color:{TEXT_MUT};margin-top:5px;font-family:'JetBrains Mono',monospace}}
.alert-time{{font-size:0.68rem;color:{TEXT_MUT};white-space:nowrap;font-family:'JetBrains Mono',monospace}}
.section-divider{{display:flex;align-items:center;gap:12px;margin:28px 0 18px 0}}
.section-divider-line{{flex:1;height:1px;background:{BORDER}}}
.section-divider-text{{font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{TEXT_MUT};padding:0 8px}}
[data-testid="metric-container"]{{background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;padding:16px}}
[data-testid="stMetricValue"]{{color:{TEXT_PRI}!important;font-weight:700!important}}
[data-testid="stMetricLabel"]{{color:{TEXT_SEC}!important;font-size:0.78rem!important}}
[data-testid="stForm"]{{background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:20px!important}}
.stTextInput>div>div>input,.stNumberInput>div>div>input,.stTextArea>div>div>textarea{{background:{SURFACE}!important;border:1px solid {BORDER}!important;color:{TEXT_PRI}!important;border-radius:8px!important}}
.stSelectbox>div>div,.stMultiSelect>div>div{{background:{SURFACE}!important;border:1px solid {BORDER}!important;border-radius:8px!important;color:{TEXT_PRI}!important}}
label{{color:{TEXT_SEC}!important;font-size:0.82rem!important;font-weight:500!important}}
[data-testid="stTabs"] [role="tablist"]{{background:{SURFACE};border-radius:10px;padding:4px;border:1px solid {BORDER};gap:4px}}
[data-testid="stTabs"] [role="tab"]{{color:{TEXT_SEC}!important;border-radius:7px!important;font-size:0.82rem!important;font-weight:600!important;padding:8px 16px!important;border:none!important}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{{background:{ACCENT2}!important;color:white!important}}
[data-testid="stTabs"] [role="tabpanel"]{{padding-top:20px}}
.stButton>button{{background:linear-gradient(135deg,{ACCENT2},#0D5C8A)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;font-size:0.84rem!important;padding:10px 20px!important;box-shadow:0 4px 15px rgba(29,126,194,0.3)!important}}
.stButton>button:hover{{box-shadow:0 6px 20px rgba(29,126,194,0.5)!important;transform:translateY(-1px)!important}}
.stButton>button[kind="secondary"]{{background:{SURFACE}!important;border:1px solid {BORDER}!important;box-shadow:none!important;color:{TEXT_SEC}!important}}
.stDownloadButton>button{{background:linear-gradient(135deg,#27AE60,#1A8A48)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;box-shadow:0 4px 15px rgba(39,174,96,0.3)!important}}
[data-testid="stExpander"]{{background:{CARD_BG}!important;border:1px solid {BORDER}!important;border-radius:10px!important}}
[data-testid="stExpander"] summary{{color:{TEXT_PRI}!important;font-weight:600!important}}
::-webkit-scrollbar{{width:6px;height:6px}}::-webkit-scrollbar-track{{background:{PRIMARY}}}::-webkit-scrollbar-thumb{{background:{BORDER};border-radius:3px}}::-webkit-scrollbar-thumb:hover{{background:{ACCENT2}}}
.exec-report-header{{background:linear-gradient(135deg,{SECONDARY},{SURFACE});border:1px solid {BORDER};border-radius:16px;padding:28px 32px;margin-bottom:24px;position:relative;overflow:hidden}}
.exec-title{{font-size:1.4rem;font-weight:800;color:{TEXT_PRI};letter-spacing:-0.02em}}
.exec-subtitle{{font-size:0.82rem;color:{TEXT_SEC};margin-top:4px}}
.exec-badge{{display:inline-block;margin-top:14px;padding:6px 14px;border-radius:20px;font-size:0.75rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase}}
.esg-bar-wrap{{margin-bottom:14px}}
.esg-bar-label{{display:flex;justify-content:space-between;font-size:0.78rem;color:{TEXT_SEC};margin-bottom:5px}}
.esg-bar-track{{height:8px;background:{BORDER};border-radius:4px;overflow:hidden}}
.esg-bar-fill{{height:100%;border-radius:4px;transition:width 0.6s ease}}
.sidebar-logo{{padding:20px 16px 8px;border-bottom:1px solid {BORDER};margin-bottom:16px}}
.sidebar-brand{{font-size:1.1rem;font-weight:800;color:{TEXT_PRI};letter-spacing:-0.01em}}
.sidebar-tagline{{font-size:0.68rem;color:{TEXT_MUT};letter-spacing:0.06em;text-transform:uppercase;margin-top:2px}}
.info-box{{background:rgba(29,126,194,0.12);border-left:4px solid {ACCENT2};border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;font-size:0.84rem;color:{TEXT_SEC}}}
.warn-box{{background:rgba(255,176,32,0.10);border-left:4px solid {C_MEDIO};border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;font-size:0.84rem;color:{TEXT_SEC}}}
.danger-box{{background:rgba(230,57,70,0.10);border-left:4px solid {C_CRITICO};border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;font-size:0.84rem;color:{TEXT_SEC}}}
#MainMenu{{visibility:hidden}}footer{{visibility:hidden}}header{{visibility:hidden}}
</style>
"""

# ═══════════════════════════════════════════════════════════════
#  INICIALIZAÇÃO
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def inicializar():
    db.init_database()
    if db.check_db_empty():
        seed_data.seed_database()

inicializar()
st.markdown(CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  TEMA PLOTLY DARK
# ═══════════════════════════════════════════════════════════════

def _pt(fig, h=None):
    """Apply dark plotly theme."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=TEXT_SEC, size=12),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_MUT, size=11)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_MUT, size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SEC, size=11), bordercolor=BORDER, borderwidth=1),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(family="Inter", color=TEXT_PRI, size=12)),
        margin=dict(t=40, b=30, l=10, r=10),
        colorway=[ACCENT2, C_BAIXO, C_MEDIO, C_ALTO, C_CRITICO, "#9B59B6", "#1ABC9C"],
    )
    if h:
        fig.update_layout(height=h)
    return fig

def gauge_risco(score, nivel, h=240):
    cor = RISK_COLORS_PLOTLY.get(nivel, TEXT_SEC)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix":"/100","font":{"size":28,"color":cor,"family":"Inter"}},
        title={"text":f"<b>{nivel}</b>","font":{"size":13,"color":cor,"family":"Inter"}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":BORDER,"tickfont":{"color":TEXT_MUT,"size":10}},
            "bar":{"color":cor,"thickness":0.25},
            "bgcolor":"rgba(0,0,0,0)","bordercolor":BORDER,
            "steps":[
                {"range":[0,25],"color":"rgba(0,196,140,0.12)"},
                {"range":[25,50],"color":"rgba(255,176,32,0.12)"},
                {"range":[50,75],"color":"rgba(255,107,53,0.12)"},
                {"range":[75,100],"color":"rgba(230,57,70,0.12)"},
            ],
            "threshold":{"line":{"color":cor,"width":3},"thickness":0.75,"value":score},
        },
    ))
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=30,b=10,l=20,r=20), font=dict(family="Inter"))
    return fig

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f'''
    <div class="sidebar-logo">
        <div class="sidebar-brand">Rail<span style="color:{ACCENT}">Guard</span> AI</div>
        <div class="sidebar-tagline">Compliance Preditivo Ferroviário</div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:10px;font-size:0.72rem;color:{C_BAIXO};">
            <div style="width:6px;height:6px;border-radius:50%;background:{C_BAIXO};box-shadow:0 0 6px {C_BAIXO};animation:pulse 2s infinite;"></div>
            Sistema Operacional
        </div>
    </div>
    ''', unsafe_allow_html=True)

    NAV_ITEMS = {
        "📊  Dashboard":          "Dashboard",
        "🛤️  Trechos":            "Trechos",
        "⚙️  Ativos":             "Ativos",
        "🔍  Inspeções":          "Inspecoes",
        "🤖  Modelo Preditivo":   "ML",
        "📋  Compliance & RCRS":  "Compliance",
        "🗂️  Auditoria":          "Auditoria",
        "🌿  ESG":                "ESG",
        "📄  Relatórios":         "Relatorios",
        "⚙️  Configurações":      "Config",
    }

    nav_sel = st.radio("MENU PRINCIPAL", list(NAV_ITEMS.keys()), label_visibility="visible")
    pagina  = NAV_ITEMS[nav_sel]

    sb = db.get_dashboard_stats()
    st.markdown(f'''
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                padding:14px 16px;margin-top:20px;margin-bottom:16px;">
        <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:{TEXT_MUT};font-weight:700;margin-bottom:10px;">VISÃO RÁPIDA</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div><div style="font-size:1.3rem;font-weight:800;color:{TEXT_PRI};">{sb["total_ativos"]}</div>
                 <div style="font-size:0.68rem;color:{TEXT_MUT};">Ativos</div></div>
            <div><div style="font-size:1.3rem;font-weight:800;color:{C_CRITICO};">{sb["risco_critico"]}</div>
                 <div style="font-size:0.68rem;color:{TEXT_MUT};">Críticos</div></div>
            <div><div style="font-size:1.3rem;font-weight:800;color:{TEXT_PRI};">{sb["total_inspecoes"]}</div>
                 <div style="font-size:0.68rem;color:{TEXT_MUT};">Inspeções</div></div>
            <div><div style="font-size:1.3rem;font-weight:800;color:{C_MEDIO};">{sb["total_alertas_abertos"]}</div>
                 <div style="font-size:0.68rem;color:{TEXT_MUT};">Alertas</div></div>
        </div>
    </div>
    <div style="font-size:0.65rem;color:{TEXT_MUT};text-align:center;padding:0 8px;">
        MVP v0.2 · {datetime.now().strftime("%d/%m/%Y %H:%M")}<br>
        ⚠️ Dados simulados — uso demonstrativo
    </div>
    ''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def ph(icon, title, subtitle=""):
    """Page header."""
    st.markdown(f'''
    <div class="rg-page-header">
        <div class="rg-page-icon">{icon}</div>
        <div>
            <div class="rg-page-title">{title}</div>
            {"<div class=\'rg-page-subtitle\'>" + subtitle + "</div>" if subtitle else ""}
        </div>
    </div>
    ''', unsafe_allow_html=True)

def kpi(label, value, delta="", ddir="neutral", variant="primary", icon=""):
    st.markdown(f'''
    <div class="kpi-card kpi-{variant}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {"<div class=\'kpi-delta " + ddir + "\'>" + delta + "</div>" if delta else ""}
    </div>
    ''', unsafe_allow_html=True)

def sdiv(label):
    st.markdown(f'''
    <div class="section-divider">
        <div class="section-divider-line"></div>
        <div class="section-divider-text">{label}</div>
        <div class="section-divider-line"></div>
    </div>
    ''', unsafe_allow_html=True)

def info(msg):
    st.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)

def warn(msg):
    st.markdown(f'<div class="warn-box">{msg}</div>', unsafe_allow_html=True)

def danger(msg):
    st.markdown(f'<div class="danger-box">{msg}</div>', unsafe_allow_html=True)

def row_kv(lbl, val):
    return f'''<div style="display:flex;justify-content:space-between;padding:9px 0;
    border-bottom:1px solid {BORDER};">
    <span style="font-size:0.78rem;color:{TEXT_MUT};">{lbl}</span>
    <span style="font-size:0.8rem;color:{TEXT_PRI};font-weight:500;
    font-family:'JetBrains Mono',monospace;">{val}</span></div>'''

