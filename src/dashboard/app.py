"""SENTINEL-AI — Security Intelligence Command Center Dashboard v3.0"""

import streamlit as st, requests, json, time, plotly.graph_objects as go
from collections import Counter

st.set_page_config(
    page_title="SENTINEL-AI", page_icon="\U0001f6e1\ufe0f", layout="wide"
)
API = "http://localhost:8000"

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Space+Mono:wght@400;700&family=IBM+Plex+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');
:root{
--bg-void:#020409;--bg-base:#060c18;--bg-surface:rgba(10,22,40,0.55);
--bg-elevated:rgba(15,30,56,0.6);--bg-glass:rgba(12,20,40,0.45);
--accent-cyan:#00d4ff;--accent-purple:#7c3aed;--accent-blue:#2563eb;
--accent-green:#00ff88;--accent-red:#ff2d55;--accent-amber:#ffaa00;
--text-primary:#e8f4ff;--text-secondary:#7a9cc0;--text-tertiary:#3a5a7a;
--text-mono:#00d4ff;
--border-subtle:rgba(0,212,255,0.06);--border-active:rgba(0,212,255,0.25);
--glow-cyan:0 0 40px rgba(0,212,255,0.12);
--glow-red:0 0 50px rgba(255,45,85,0.18);
--glow-green:0 0 50px rgba(0,255,136,0.15);
--blur:blur(16px);--blur-sm:blur(8px)
}

/* ANIMATED GRADIENT MESH BG */
@keyframes meshDrift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"],.main{
  background:linear-gradient(135deg,#020409 0%,#060c18 25%,#0a0f2e 50%,#060c18 75%,#020409 100%)!important;
  background-size:400% 400%!important;animation:meshDrift 25s ease infinite!important;
  color:var(--text-primary)!important;font-family:'Inter',sans-serif!important}
.block-container{max-width:1400px!important;padding-top:2rem!important}

/* FLOATING ORBS */
@keyframes orbFloat1{0%,100%{transform:translate(0,0) scale(1)}25%{transform:translate(80px,-60px) scale(1.1)}50%{transform:translate(-40px,-120px) scale(0.9)}75%{transform:translate(-80px,-30px) scale(1.05)}}
@keyframes orbFloat2{0%,100%{transform:translate(0,0) scale(1)}25%{transform:translate(-100px,50px) scale(1.15)}50%{transform:translate(60px,100px) scale(0.85)}75%{transform:translate(90px,-40px) scale(1.1)}}
@keyframes orbFloat3{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(70px,80px) scale(1.2)}66%{transform:translate(-90px,-50px) scale(0.9)}}
.orb-container{position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;overflow:hidden}
.orb{position:absolute;border-radius:50%;filter:blur(80px);opacity:.08}
.orb-1{width:600px;height:600px;background:radial-gradient(circle,var(--accent-cyan),transparent 70%);top:10%;left:20%;animation:orbFloat1 30s ease-in-out infinite}
.orb-2{width:500px;height:500px;background:radial-gradient(circle,var(--accent-purple),transparent 70%);top:50%;right:10%;animation:orbFloat2 35s ease-in-out infinite}
.orb-3{width:450px;height:450px;background:radial-gradient(circle,var(--accent-blue),transparent 70%);bottom:10%;left:40%;animation:orbFloat3 28s ease-in-out infinite}

/* DOT GRID OVERLAY */
[data-testid="stAppViewContainer"]::before{content:'';position:fixed;top:0;left:0;width:100vw;height:100vh;
  background-image:radial-gradient(circle,rgba(0,212,255,0.04) 1px,transparent 1px);
  background-size:30px 30px;pointer-events:none;z-index:1}

/* NOISE */
[data-testid="stAppViewContainer"]::after{content:'';position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:1;opacity:.025;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");background-repeat:repeat;background-size:200px}

/* SCAN LINE */
@keyframes scanLine{0%{top:-2px;opacity:1}80%{opacity:.5}100%{top:100vh;opacity:0}}
.scan-overlay{position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:99999;overflow:hidden}
.scan-overlay::before{content:'';position:absolute;left:0;width:100%;height:2px;
  background:linear-gradient(90deg,transparent 5%,var(--accent-cyan) 30%,rgba(124,58,237,0.8) 50%,var(--accent-cyan) 70%,transparent 95%);
  box-shadow:0 0 30px var(--accent-cyan),0 0 80px rgba(0,212,255,0.3);animation:scanLine 1.8s ease-out forwards}

/* SIDEBAR — FROSTED GLASS */
[data-testid="stSidebar"]{background:rgba(4,8,20,0.85)!important;backdrop-filter:var(--blur)!important;-webkit-backdrop-filter:var(--blur)!important;border-right:1px solid rgba(0,212,255,0.06)!important;overflow:hidden}
[data-testid="stSidebar"]::before{content:'01001 10110 00101 11010 01101 10011 00110 11001 01010 10101 00011 11100 01111 10000 00100 11011 01000 10111 00010 11110 01011 10010 00111 11101';position:absolute;top:0;left:0;right:0;bottom:0;font-family:'IBM Plex Mono',monospace;font-size:9px;color:var(--accent-cyan);opacity:.03;word-wrap:break-word;line-height:2.2;padding:80px 10px;pointer-events:none;overflow:hidden;animation:matrixScroll 25s linear infinite;z-index:0}
@keyframes matrixScroll{0%{transform:translateY(0)}100%{transform:translateY(-50%)}}
[data-testid="stSidebar"]>div{position:relative;z-index:1}

/* PULSE DOT */
@keyframes pulse{0%,100%{transform:scale(1);box-shadow:0 0 0 0 rgba(0,255,136,.4)}50%{transform:scale(1.3);box-shadow:0 0 16px 6px rgba(0,255,136,0)}}
@keyframes pulseOff{0%,100%{transform:scale(1);box-shadow:0 0 0 0 rgba(255,45,85,.4)}50%{transform:scale(1.2);box-shadow:0 0 12px 4px rgba(255,45,85,0)}}
.sdot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:10px;vertical-align:middle}
.sdot.on{background:var(--accent-green);animation:pulse 2s ease-in-out infinite}
.sdot.off{background:var(--accent-red);animation:pulseOff 2.5s ease-in-out infinite}

/* NAV */
[data-testid="stSidebar"] [data-testid="stRadio"] label{font-family:'Space Mono',monospace!important;font-size:13px!important;letter-spacing:1.5px!important;color:var(--text-secondary)!important;padding:10px 16px!important;border-left:2px solid transparent!important;transition:all .3s cubic-bezier(.4,0,.2,1)!important;margin:2px 0!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{color:var(--text-primary)!important;background:rgba(0,212,255,.03)!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){color:var(--accent-cyan)!important;border-left-color:var(--accent-cyan)!important;background:rgba(0,212,255,.06)!important;text-shadow:0 0 20px rgba(0,212,255,.3)!important}
[data-testid="stSidebar"] [data-testid="stRadio"]>div[role="radiogroup"]>label>div:first-child{display:none!important}

/* GLASSMORPHISM HEADER */
.cmd-hdr{position:relative;background:var(--bg-glass);backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);border:1px solid var(--border-subtle);padding:36px 44px;margin-bottom:28px;overflow:hidden}
.cmd-hdr::before,.cmd-hdr::after{content:'';position:absolute;width:24px;height:24px;border-color:var(--accent-cyan);border-style:solid;opacity:.6}
.cmd-hdr::before{top:-1px;left:-1px;border-width:2px 0 0 2px}
.cmd-hdr::after{top:-1px;right:-1px;border-width:2px 2px 0 0}
.cmd-inner{position:relative}
.cmd-inner::before,.cmd-inner::after{content:'';position:absolute;width:24px;height:24px;border-color:var(--accent-cyan);border-style:solid;opacity:.6}
.cmd-inner::before{bottom:-36px;left:-44px;border-width:0 0 2px 2px}
.cmd-inner::after{bottom:-36px;right:-44px;border-width:0 2px 2px 0}
.grad-line{position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent-cyan),var(--accent-purple),var(--accent-cyan),transparent);background-size:200% 100%;animation:gradMove 4s linear infinite}
@keyframes gradMove{0%{background-position:0% 50%}100%{background-position:200% 50%}}
.hdr-glow{position:absolute;top:-50%;left:30%;width:40%;height:200%;background:radial-gradient(ellipse,rgba(0,212,255,0.04),transparent 70%);pointer-events:none}

/* GLITCH TITLE */
.glitch{font-family:'Orbitron',sans-serif;font-size:36px;font-weight:800;color:var(--accent-cyan);letter-spacing:6px;text-transform:uppercase;cursor:default;display:inline-block;text-shadow:0 0 30px rgba(0,212,255,.2)}
.glitch:hover{animation:glitch .3s ease-in-out}
@keyframes glitch{0%,100%{text-shadow:0 0 30px rgba(0,212,255,.2)}20%{text-shadow:-3px 0 #ff2d55,3px 0 #00d4ff}40%{text-shadow:3px 0 #ff2d55,-3px 0 #00d4ff}60%{text-shadow:-2px 0 #ff2d55,2px 0 #00d4ff}80%{text-shadow:2px 0 #ff2d55,-2px 0 #00d4ff}}

/* VERDICT — GLASS + GLOW */
@keyframes verdictIn{0%{opacity:0;transform:translateY(16px) scale(.98)}100%{opacity:1;transform:translateY(0) scale(1)}}
.vbanner{position:relative;padding:36px 44px;margin:24px 0;animation:verdictIn .5s cubic-bezier(.4,0,.2,1);backdrop-filter:var(--blur-sm);-webkit-backdrop-filter:var(--blur-sm);overflow:hidden}
.vbanner::before,.vbanner::after{content:'';position:absolute;width:24px;height:24px;border-style:solid;opacity:.7}
.vbanner::before{top:-1px;left:-1px;border-width:2px 0 0 2px}
.vbanner::after{top:-1px;right:-1px;border-width:2px 2px 0 0}
.v-glow{position:absolute;top:-30%;left:20%;width:60%;height:160%;border-radius:50%;filter:blur(60px);opacity:.12;pointer-events:none}
.vbanner.v-fake{background:rgba(40,8,16,0.5);border:1px solid rgba(255,45,85,.4)}.vbanner.v-fake::before,.vbanner.v-fake::after{border-color:var(--accent-red)}.vbanner.v-fake .v-glow{background:var(--accent-red)}
.vbanner.v-real{background:rgba(5,30,15,0.5);border:1px solid rgba(0,255,136,.35)}.vbanner.v-real::before,.vbanner.v-real::after{border-color:var(--accent-green)}.vbanner.v-real .v-glow{background:var(--accent-green)}
.vbanner.v-satire{background:rgba(40,28,5,0.5);border:1px solid rgba(255,170,0,.35)}.vbanner.v-satire::before,.vbanner.v-satire::after{border-color:var(--accent-amber)}.vbanner.v-satire .v-glow{background:var(--accent-amber)}
.vbanner.v-uncertain{background:rgba(15,20,35,0.5);border:1px solid rgba(71,85,105,.4)}.vbanner.v-uncertain::before,.vbanner.v-uncertain::after{border-color:#475569}.vbanner.v-uncertain .v-glow{background:#475569}
.vlbl{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:4px;text-transform:uppercase;margin-bottom:10px;opacity:.7}
.vtitle{font-family:'Orbitron',sans-serif;font-size:34px;font-weight:800;letter-spacing:3px;margin:0;line-height:1.2}
.vmeta{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--text-secondary);margin-top:14px;letter-spacing:.5px}
.v-fake .vlbl,.v-fake .vtitle{color:#ff6b8a}.v-real .vlbl,.v-real .vtitle{color:#5cffb1}.v-satire .vlbl,.v-satire .vtitle{color:#ffc44d}.v-uncertain .vlbl,.v-uncertain .vtitle{color:#94a3b8}

/* METRIC CARDS — GLASS */
.smc{background:var(--bg-glass);backdrop-filter:var(--blur-sm);-webkit-backdrop-filter:var(--blur-sm);border:1px solid var(--border-subtle);padding:24px 16px;text-align:center;position:relative;transition:all .35s cubic-bezier(.4,0,.2,1);overflow:hidden}
.smc::before,.smc::after{content:'';position:absolute;width:14px;height:14px;border-color:var(--accent-cyan);border-style:solid;opacity:.3;transition:opacity .3s}
.smc::before{top:-1px;left:-1px;border-width:1px 0 0 1px}
.smc::after{bottom:-1px;right:-1px;border-width:0 1px 1px 0}
.smc:hover{border-color:var(--border-active);transform:translateY(-3px);box-shadow:0 8px 32px rgba(0,0,0,.3),var(--glow-cyan)}
.smc:hover::before,.smc:hover::after{opacity:.7}
.smc .mv{font-family:'IBM Plex Mono',monospace;font-size:26px;font-weight:600;color:var(--text-mono);text-shadow:0 0 20px rgba(0,212,255,.15)}
.smc .ml{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2.5px;text-transform:uppercase;color:var(--text-tertiary);margin-top:8px}
.smc .mbar{position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent-cyan),var(--accent-purple));opacity:.4}

/* TABS */
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border-subtle)!important;gap:4px!important}
[data-testid="stTabs"] button{font-family:'Space Mono',monospace!important;font-size:11px!important;letter-spacing:2px!important;text-transform:uppercase!important;color:var(--text-tertiary)!important;background:transparent!important;border:none!important;padding:12px 20px!important;transition:all .3s!important}
[data-testid="stTabs"] button:hover{color:var(--text-secondary)!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent-cyan)!important;text-shadow:0 0 20px rgba(0,212,255,.3)!important;border-bottom:2px solid var(--accent-cyan)!important}

/* TEXT AREA */
div[data-testid="stTextArea"] textarea{background:var(--bg-glass)!important;backdrop-filter:var(--blur-sm)!important;border:1px solid var(--border-subtle)!important;color:var(--text-primary)!important;font-family:'IBM Plex Mono',monospace!important;font-size:14px!important;transition:all .3s!important}
div[data-testid="stTextArea"] textarea:focus{border-color:rgba(0,212,255,.3)!important;box-shadow:0 0 0 1px rgba(0,212,255,.1),0 0 40px rgba(0,212,255,.08)!important}

/* BUTTONS */
@keyframes btnGlow{0%,100%{box-shadow:0 0 20px rgba(0,212,255,.08)}50%{box-shadow:0 0 40px rgba(0,212,255,.2),0 0 80px rgba(124,58,237,.1)}}
.stButton>button{font-family:'Space Mono',monospace!important;letter-spacing:2.5px!important;text-transform:uppercase!important;border:1px solid rgba(0,212,255,0.25)!important;border-radius:3px!important;transition:all .35s cubic-bezier(.4,0,.2,1)!important;position:relative!important;color:#e8f4ff!important;background:rgba(10,22,40,0.6)!important;backdrop-filter:blur(8px)!important;-webkit-backdrop-filter:blur(8px)!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 30px rgba(0,0,0,.3),0 0 30px rgba(0,212,255,0.1)!important;border-color:rgba(0,212,255,0.5)!important;color:#00d4ff!important;background:rgba(0,212,255,0.08)!important}
.stButton>button:active{transform:translateY(0)!important;box-shadow:0 0 15px rgba(0,212,255,0.2)!important}
/* PRIMARY ANALYZE BUTTON — special accent style */
.stButton>button[kind="primary"],div[data-testid="stHorizontalBlock"]:has(+ div[data-testid="stHorizontalBlock"]) .stButton>button:only-child{background:linear-gradient(135deg,rgba(0,212,255,0.15),rgba(124,58,237,0.15))!important;border-color:rgba(0,212,255,0.4)!important;color:#00d4ff!important;animation:btnGlow 3s ease-in-out infinite!important}
/* TOGGLE STYLING */
[data-testid="stCheckbox"] label,[data-testid="stToggle"] label{color:var(--text-secondary)!important}
[data-testid="stToggle"] span[data-testid="stMarkdownContainer"]{color:var(--text-secondary)!important;font-family:'Space Mono',monospace!important;font-size:11px!important;letter-spacing:1px!important}

/* TOGGLES */
[data-testid="stCheckbox"] label span{font-family:'Space Mono',monospace!important;font-size:11px!important;color:var(--text-secondary)!important;letter-spacing:1px!important}

/* ENTITY CHIPS */
.ec{display:inline-block;padding:5px 14px;border-radius:20px;font-size:11px;font-family:'IBM Plex Mono',monospace;margin:3px;border:1px solid;backdrop-filter:var(--blur-sm);transition:all .2s}
.ec:hover{transform:translateY(-1px)}
.ec-PER{background:rgba(124,58,237,.15);border-color:rgba(124,58,237,.4);color:#a78bfa}
.ec-ORG{background:rgba(14,165,233,.15);border-color:rgba(14,165,233,.4);color:#38bdf8}
.ec-LOC{background:rgba(22,163,74,.15);border-color:rgba(22,163,74,.4);color:#4ade80}
.ec-MISC{background:rgba(217,119,6,.15);border-color:rgba(217,119,6,.4);color:#fbbf24}

/* TABLE */
[data-testid="stTable"] table{background:var(--bg-glass)!important;backdrop-filter:var(--blur-sm)!important;border-collapse:collapse}
[data-testid="stTable"] th{background:rgba(15,30,56,.6)!important;color:var(--accent-cyan)!important;font-family:'Space Mono',monospace!important;font-size:11px!important;letter-spacing:1.5px!important;text-transform:uppercase!important;border-bottom:1px solid var(--border-active)!important;padding:14px 16px!important}
[data-testid="stTable"] td{color:var(--text-primary)!important;font-family:'IBM Plex Mono',monospace!important;font-size:13px!important;border-bottom:1px solid var(--border-subtle)!important;padding:12px 16px!important}
[data-testid="stTable"] tr:nth-child(even) td{background:rgba(10,22,40,.3)!important}
[data-testid="stTable"] tr:hover td{background:rgba(0,212,255,.03)!important}

/* DOWNLOAD */
[data-testid="stDownloadButton"] button{background:transparent!important;border:1px solid rgba(0,212,255,.25)!important;color:var(--accent-cyan)!important;font-family:'Space Mono',monospace!important;backdrop-filter:var(--blur-sm)!important}
[data-testid="stDownloadButton"] button:hover{background:rgba(0,212,255,.06)!important;border-color:rgba(0,212,255,.5)!important}

/* EXPANDER */
[data-testid="stExpander"]{border:1px solid var(--border-subtle)!important;background:var(--bg-glass)!important;backdrop-filter:var(--blur-sm)!important}
[data-testid="stExpander"] summary span{font-family:'IBM Plex Mono',monospace!important;color:var(--text-secondary)!important;font-size:13px!important}

/* MISC */
.slbl{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:4px;text-transform:uppercase;color:var(--text-tertiary);margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid var(--border-subtle)}
.sstat{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--text-secondary);line-height:2.4}
.sstat b{color:var(--text-primary);font-weight:500}
hr{border-color:var(--border-subtle)!important;opacity:.5!important}
.stSpinner>div{border-top-color:var(--accent-cyan)!important}
[data-testid="stMetric"]{background:var(--bg-glass)!important;backdrop-filter:var(--blur-sm)!important;border:1px solid var(--border-subtle)!important;padding:20px!important;border-radius:3px!important}
[data-testid="stMetric"] [data-testid="stMetricValue"]{font-family:'IBM Plex Mono',monospace!important;color:var(--text-mono)!important}
[data-testid="stMetric"] [data-testid="stMetricLabel"]{font-family:'Space Mono',monospace!important;color:var(--text-tertiary)!important;text-transform:uppercase!important;letter-spacing:1.5px!important}
.sub-t{font-family:'Space Mono',monospace;font-size:14px;color:var(--text-secondary);letter-spacing:2px;margin-top:6px}
.sbar{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--text-tertiary);margin-top:18px;letter-spacing:1px}
.sbar span{color:var(--accent-green);text-shadow:0 0 10px rgba(0,255,136,.2)}
[data-testid="stJson"]{background:var(--bg-glass)!important;backdrop-filter:var(--blur-sm)!important}

/* GRADIENT DIVIDER */
.gdiv{height:1px;background:linear-gradient(90deg,transparent,rgba(0,212,255,.15),rgba(124,58,237,.1),transparent);margin:24px 0;border:none}
</style>"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="scan-overlay"></div><div class="orb-container"><div class="orb orb-1"></div><div class="orb orb-2"></div><div class="orb orb-3"></div></div>',
    unsafe_allow_html=True,
)


# ===================== HELPERS =====================
def api_get(path):
    try:
        return requests.get(f"{API}{path}", timeout=5).json()
    except:
        return None


def api_post(path, data):
    try:
        return requests.post(f"{API}{path}", json=data, timeout=120).json()
    except:
        return None


def verdict_class(cls):
    c = str(cls).upper()
    if "FAKE" in c or "CONTRADICT" in c:
        return "fake"
    if "REAL" in c or "SUPPORT" in c:
        return "real"
    if "SATIRE" in c:
        return "satire"
    return "uncertain"


def verdict_title(cls):
    m = {
        "REAL_NEWS": "VERIFIED REAL",
        "FAKE_NEWS": "FAKE NEWS DETECTED",
        "SATIRE": "SATIRE CONTENT",
        "SPAM": "SPAM DETECTED",
        "LIKELY_FAKE": "LIKELY FAKE",
        "LIKELY_REAL": "LIKELY REAL",
        "UNCERTAIN": "UNCERTAIN",
        "SUPPORTED": "SUPPORTED",
        "CONTRADICTED": "CONTRADICTED",
        "UNVERIFIABLE": "UNVERIFIABLE",
    }
    return m.get(str(cls).upper(), str(cls).upper())


def metric_card(value, label, color="#00d4ff"):
    return f'<div class="smc"><div class="mv" style="color:{color}">{value}</div><div class="ml">{label}</div><div class="mbar"></div></div>'


def render_countup_js():
    st.components.v1.html(
        """<script>
    setTimeout(()=>{try{const d=window.parent.document;d.querySelectorAll('.smc .mv').forEach(el=>{
    const txt=el.textContent.trim();const m=txt.match(/([\\.\\d]+)/);if(!m)return;
    const target=parseFloat(m[1]);const suffix=txt.replace(m[1],'');const dec=(m[1].includes('.'))?m[1].split('.')[1].length:0;
    const t0=performance.now();el.textContent='0'+suffix;
    function u(now){const p=Math.min((now-t0)/900,1);const e=1-Math.pow(1-p,3);
    el.textContent=(target*e).toFixed(dec)+suffix;if(p<1)requestAnimationFrame(u)}
    requestAnimationFrame(u)})}catch(e){}},250);</script>""",
        height=0,
    )


def typing_component(text, speed_ms=22):
    safe = (
        text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    h = max(100, len(text) // 2)
    st.components.v1.html(
        f"""
    <div style="background:rgba(10,22,40,0.5);backdrop-filter:blur(8px);border-left:3px solid #00d4ff;padding:24px 28px;font-size:14px;color:#cbd5e1;line-height:1.9;font-family:Inter,sans-serif;min-height:40px">
    <span id="tp"></span><span id="tc" style="border-right:2px solid #00d4ff;animation:bk 1s infinite">&#8203;</span></div>
    <style>@keyframes bk{{0%,100%{{opacity:1}}50%{{opacity:0}}}}</style>
    <script>const t=`{safe}`;let i=0;const e=document.getElementById('tp'),c=document.getElementById('tc');
    const iv=setInterval(()=>{{e.textContent+=t[i++];if(i>=t.length){{clearInterval(iv);c.style.display='none'}}}},{speed_ms});</script>""",
        height=h,
    )


def svg_progress(value, max_val=1.0, label="", size=130):
    pct = min(value / max_val, 1.0) if max_val else 0
    circ = 2 * 3.14159 * 42
    off = circ * (1 - pct)
    clr = "#00ff88" if pct > 0.75 else ("#ffaa00" if pct > 0.5 else "#ff2d55")
    st.components.v1.html(
        f"""<div style="text-align:center;padding:12px">
    <svg width="{size}" height="{size}" viewBox="0 0 100 100">
    <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(15,30,56,0.5)" stroke-width="5"/>
    <circle cx="50" cy="50" r="42" fill="none" stroke="{clr}" stroke-width="5"
      stroke-dasharray="{circ}" stroke-dashoffset="{off}" stroke-linecap="round"
      transform="rotate(-90 50 50)" style="transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1);filter:drop-shadow(0 0 6px {clr})"/>
    <text x="50" y="46" text-anchor="middle" fill="#e8f4ff" font-family="IBM Plex Mono"
      font-size="18" font-weight="600">{value:.2f}</text>
    <text x="50" y="62" text-anchor="middle" fill="#7a9cc0" font-family="Space Mono"
      font-size="7" letter-spacing="1.5">{label.upper()}</text></svg></div>""",
        height=size + 30,
    )


# ===================== SIDEBAR =====================
health = api_get("/health")
with st.sidebar:
    st.markdown(
        '<p style="font-family:Orbitron,sans-serif;font-size:22px;font-weight:800;color:#00d4ff;letter-spacing:6px;margin-bottom:2px;text-shadow:0 0 30px rgba(0,212,255,.2)">SENTINEL-AI</p>',
        unsafe_allow_html=True,
    )
    if health and health.get("status") == "healthy":
        st.markdown(
            '<div style="margin:8px 0 16px"><span class="sdot on"></span><span style="font-family:Space Mono,monospace;font-size:10px;color:#00ff88;letter-spacing:3px;text-shadow:0 0 10px rgba(0,255,136,.2)">ONLINE</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="margin:8px 0 16px"><span class="sdot off"></span><span style="font-family:Space Mono,monospace;font-size:10px;color:#ff2d55;letter-spacing:3px">OFFLINE</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["\u2b21 ANALYZE", "\u2b21 MONITOR", "\u2b21 SYSTEM"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
    if health:
        ml = health.get("models_loaded", 0)
        pct_bar = "\u2588" * ml + "\u2591" * (13 - ml)
        kb = health.get("kb_size", 0)
        dev = health.get("device", "cpu").upper()
        st.markdown(
            f"""<div class="sstat">
        <b>MODELS</b>  {ml}/13  <span style="color:#00d4ff;font-size:9px">{pct_bar}</span> {ml*100//13}%<br>
        <b>KB</b>      {kb:,} chunks<br>
        <b>DEVICE</b>  {dev}<br>
        <b>BUILD</b>   v3.0.0
        </div>""",
            unsafe_allow_html=True,
        )


# ===================== PAGE 1: ANALYZE =====================
if "ANALYZE" in page:
    st.markdown(
        """<div class="cmd-hdr"><div class="cmd-inner">
    <div class="grad-line"></div><div class="hdr-glow"></div>
    <p class="glitch">SENTINEL-AI</p>
    <p class="sub-t">REAL-TIME CONTENT INTELLIGENCE PLATFORM</p>
    <div class="sbar">STATUS: <span>OPERATIONAL</span>  \u00b7  RAG: <span>ACTIVE</span>  \u00b7  <span>"""
        + (health.get("device", "CPU").upper() if health else "CPU")
        + """</span></div></div></div>""",
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "Input",
        height=150,
        placeholder="// INPUT THREAT VECTOR \u2014 PASTE CONTENT FOR ANALYSIS...",
        label_visibility="collapsed",
        value=st.session_state.get("_example_text", ""),
    )

    c1, c2, c3, c4 = st.columns(4)
    use_rag = c1.toggle("RAG FACT-CHECK", value=True)
    use_shap = c2.toggle("SHAP EXPLAIN", value=True)
    use_summary = c3.toggle("SUMMARIZE", value=True)
    use_ner = c4.toggle("ENTITY EXTRACT", value=True)

    analyze = st.button(
        "\u2b21  ANALYZE THREAT VECTOR", use_container_width=True, type="primary"
    )

    ec1, ec2, ec3 = st.columns(3)
    examples = [
        "BREAKING: Scientists confirm 5G towers cause COVID-19 infections. WHO refuses to comment on mounting evidence.",
        "The Federal Reserve announced a quarter-point interest rate cut today, citing concerns about slowing economic growth.",
        "SHOCKING: Local man discovers that WiFi routers are secretly recording all conversations. Government cover-up exposed!",
    ]
    if ec1.button("\u26a0 FAKE NEWS SAMPLE", use_container_width=True):
        st.session_state["_example_text"] = examples[0]
        st.rerun()
    if ec2.button("\u2713 REAL NEWS SAMPLE", use_container_width=True):
        st.session_state["_example_text"] = examples[1]
        st.rerun()
    if ec3.button("\u25c8 SATIRE SAMPLE", use_container_width=True):
        st.session_state["_example_text"] = examples[2]
        st.rerun()

    if analyze and text and len(text) >= 20:
        st.session_state["_example_text"] = text
        with st.spinner("\u2b21 ANALYZING THREAT VECTOR..."):
            payload = {
                "text": text,
                "include_rag": use_rag,
                "include_explanation": use_shap,
                "include_summary": use_summary,
                "include_ner": use_ner,
            }
            if use_rag:
                data = api_post("/fact-check/full-analysis", payload)
            else:
                data = api_post("/classify", payload)

        if data:
            cls = data.get("fused_verdict", data.get("classification", "UNKNOWN"))
            conf = data.get("ml_confidence", data.get("confidence", 0))
            method = data.get("fusion_method", data.get("method", ""))
            human = data.get("human_review_required", False)
            vc = verdict_class(cls)
            icon = {
                "fake": "\u2716",
                "real": "\u2714",
                "satire": "\u25c8",
                "uncertain": "?",
            }.get(vc, "?")
            meta_parts = [f"{conf:.1%} ML CONFIDENCE", method.upper() if method else ""]
            if human:
                meta_parts.append("\u26a0 HUMAN REVIEW REQUIRED")
            meta_str = "  \u00b7  ".join([p for p in meta_parts if p])

            st.markdown(
                f"""<div class="vbanner v-{vc}">
            <div class="v-glow"></div>
            <div class="vlbl">\u2b21 THREAT ASSESSMENT</div>
            <p class="vtitle">{icon}  {verdict_title(cls)}</p>
            <div class="vmeta">{meta_str}</div>
            </div>""",
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4, m5 = st.columns(5)
            rag_sim = data.get("avg_retrieval_similarity", 0)
            anomaly = data.get("anomaly_score")
            manip = data.get("manipulation_score")
            pt = data.get("processing_time_ms", 0)
            m1.markdown(
                metric_card(f"{conf:.1%}", "ML CONFIDENCE"), unsafe_allow_html=True
            )
            m2.markdown(
                metric_card(f"{rag_sim:.2f}" if rag_sim else "N/A", "RETRIEVAL SIM"),
                unsafe_allow_html=True,
            )
            m3.markdown(
                metric_card(f"{anomaly:.3f}" if anomaly else "N/A", "ANOMALY"),
                unsafe_allow_html=True,
            )
            m4.markdown(
                metric_card(f"{manip:.2f}" if manip else "N/A", "MANIPULATION"),
                unsafe_allow_html=True,
            )
            m5.markdown(metric_card(f"{pt:.0f}ms", "PROC TIME"), unsafe_allow_html=True)
            render_countup_js()

            tabs = st.tabs(["EVIDENCE", "SIGNALS", "ENTITIES", "SUMMARY", "RAW"])

            with tabs[0]:
                reasoning = data.get("rag_reasoning", "")
                if reasoning:
                    st.markdown(
                        '<div class="slbl">\u2b21 RAG INTELLIGENCE REASONING</div>',
                        unsafe_allow_html=True,
                    )
                    typing_component(reasoning, speed_ms=22)
                claim = data.get("primary_claim", "")
                if claim:
                    st.markdown(
                        '<div class="slbl">\u2b21 EXTRACTED CLAIM</div>',
                        unsafe_allow_html=True,
                    )
                    st.code(claim, language=None)
                if rag_sim:
                    st.markdown(
                        '<div class="slbl">\u2b21 RETRIEVAL SIMILARITY</div>',
                        unsafe_allow_html=True,
                    )
                    svg_progress(rag_sim, max_val=1.0, label="SIMILARITY", size=130)
                docs = data.get("retrieved_docs", [])
                if docs:
                    st.markdown(
                        '<div class="slbl">\u2b21 RETRIEVED DOCUMENTS</div>',
                        unsafe_allow_html=True,
                    )
                    for i, doc in enumerate(docs):
                        src = doc.get("source", "unknown")
                        vlbl = doc.get("verdict_label", "")
                        rel = doc.get("relevance_score", doc.get("similarity", 0))
                        vlbl_color = (
                            "#ff6b8a"
                            if "false" in vlbl.lower() or "fake" in vlbl.lower()
                            else (
                                "#5cffb1"
                                if "true" in vlbl.lower() or "real" in vlbl.lower()
                                else "#ffc44d"
                            )
                        )
                        doc_card = f"""<div style="display:flex;gap:16px;align-items:center;padding:12px 16px;
                            background:rgba(10,22,40,0.4);border:1px solid rgba(0,212,255,0.08);
                            margin-bottom:6px;backdrop-filter:blur(6px);transition:all 0.2s;">
                            <span style="color:#00d4ff;font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:13px;
                                min-width:70px;">DOC-{i+1:03d}</span>
                            <span style="color:#7a9cc0;font-family:'IBM Plex Mono',monospace;font-size:12px;
                                min-width:100px;">{src}</span>
                            <span style="color:{vlbl_color};font-family:'IBM Plex Mono',monospace;font-size:12px;
                                font-weight:500;min-width:80px;">{vlbl}</span>
                            <span style="color:#3a5a7a;font-family:'IBM Plex Mono',monospace;font-size:11px;
                                margin-left:auto;">relevance: {rel:.3f}</span>
                        </div>"""
                        st.markdown(doc_card, unsafe_allow_html=True)
                        with st.expander(f"\u25b8 View excerpt — DOC-{i+1:03d}"):
                            st.markdown(
                                f'<p style="font-family:Inter,sans-serif;font-size:13px;color:#cbd5e1;line-height:1.8;padding:4px 0;">{doc.get("excerpt", "")}</p>',
                                unsafe_allow_html=True,
                            )

            with tabs[1]:
                if use_shap:
                    expl = data.get("explanation")
                    if expl:
                        st.markdown(
                            '<div class="slbl">\u2b21 FEATURE SIGNAL ANALYSIS</div>',
                            unsafe_allow_html=True,
                        )
                        # Use actual SHAP data if available, fallback to structured visualization
                        if isinstance(expl, dict) and expl.get("top_features"):
                            features = expl["top_features"]
                            labels = [
                                f.get("feature", f"Feature {i}")
                                for i, f in enumerate(features)
                            ]
                            vals = [f.get("importance", 0) for f in features]
                        elif isinstance(expl, list):
                            labels = [
                                (
                                    f.get("feature", f"Feature {i}")
                                    if isinstance(f, dict)
                                    else f"Feature {i}"
                                )
                                for i, f in enumerate(expl[:15])
                            ]
                            vals = [
                                (
                                    f.get("importance", f.get("value", 0))
                                    if isinstance(f, dict)
                                    else float(f)
                                )
                                for f in expl[:15]
                            ]
                        else:
                            labels = [f"Feature {i}" for i in range(10)]
                            vals = [
                                0.1 * (10 - i) * (-1 if i % 3 == 0 else 1)
                                for i in range(10)
                            ]
                        colors = ["#ff2d55" if v < 0 else "#00d4ff" for v in vals]
                        fig = go.Figure(
                            go.Bar(
                                y=labels,
                                x=vals,
                                orientation="h",
                                marker_color=colors,
                                marker=dict(line=dict(width=0)),
                            )
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(10,22,40,0.3)",
                            font=dict(color="#e8f4ff", family="IBM Plex Mono", size=11),
                            height=400,
                            margin=dict(l=10, r=10, t=10, b=30),
                            xaxis=dict(
                                gridcolor="rgba(0,212,255,0.05)",
                                zeroline=True,
                                zerolinecolor="#3a5a7a",
                                zerolinewidth=1,
                            ),
                            yaxis=dict(gridcolor="rgba(0,212,255,0.03)"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(
                            '<p style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#3a5a7a;text-align:center;letter-spacing:2px">\u2190 PUSHES TOWARD REAL  \u00b7  PUSHES TOWARD FAKE \u2192</p>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info(
                            "\u2b21 SHAP explanation was requested but no data was returned by the API."
                        )
                else:
                    st.info(
                        "\u2b21 SHAP explanation was not requested. Enable the SHAP EXPLAIN toggle and re-analyze."
                    )

            with tabs[2]:
                if use_ner:
                    ents = data.get("entities", [])
                    if ents:
                        st.markdown(
                            '<div class="slbl">\u2b21 NAMED ENTITY EXTRACTION</div>',
                            unsafe_allow_html=True,
                        )
                        chips = ""
                        for e in ents:
                            eg = e.get("entity_group", "MISC")
                            chips += f'<span class="ec ec-{eg}">{e.get("word","")} ({eg})</span> '
                        st.markdown(chips, unsafe_allow_html=True)
                        counts = Counter(e.get("entity_group", "MISC") for e in ents)
                        cmap = {
                            "PER": "#a78bfa",
                            "ORG": "#38bdf8",
                            "LOC": "#4ade80",
                            "MISC": "#fbbf24",
                        }
                        fig = go.Figure(
                            go.Pie(
                                labels=list(counts.keys()),
                                values=list(counts.values()),
                                marker=dict(
                                    colors=[
                                        cmap.get(k, "#94a3b8") for k in counts.keys()
                                    ]
                                ),
                                hole=0.55,
                                textfont=dict(family="IBM Plex Mono", size=12),
                            )
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e8f4ff", family="IBM Plex Mono"),
                            height=300,
                            margin=dict(l=20, r=20, t=20, b=20),
                            showlegend=True,
                            legend=dict(
                                font=dict(color="#7a9cc0", family="Space Mono", size=11)
                            ),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(
                            "\u2b21 Entity extraction was requested but no entities were found."
                        )
                else:
                    st.info(
                        "\u2b21 Entity extraction was not requested. Enable the ENTITY EXTRACT toggle and re-analyze."
                    )

            with tabs[3]:
                if use_summary:
                    summary = data.get("summary")
                    if summary:
                        st.markdown(
                            '<div class="slbl">\u2b21 BART SUMMARY</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div style="background:rgba(124,58,237,0.08);backdrop-filter:blur(8px);border-left:3px solid #7c3aed;padding:20px 24px;color:#cbd5e1;font-size:14px;line-height:1.9;font-family:Inter,sans-serif">{summary}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info(
                            "\u2b21 Summary was requested but no data was returned by the API."
                        )
                else:
                    st.info(
                        "\u2b21 Summary was not requested. Enable the SUMMARIZE toggle and re-analyze."
                    )
                sent = data.get("sentiment")
                if sent:
                    fb = sent.get("finbert", {})
                    slbl = fb.get("label", "N/A")
                    sscore = fb.get("score", 0)
                    sclr = (
                        "#00ff88"
                        if "pos" in slbl.lower()
                        else ("#ff2d55" if "neg" in slbl.lower() else "#ffaa00")
                    )
                    st.markdown(
                        f"""<div style="margin-top:20px">
                    <div class="slbl">\u2b21 SENTIMENT</div>
                    <p style="font-family:IBM Plex Mono,monospace;font-size:14px;color:{sclr};text-shadow:0 0 15px {sclr}44">{slbl.upper()} \u2014 {sscore:.2%}</p>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                if manip is not None:
                    st.markdown(
                        '<div class="slbl" style="margin-top:20px">\u2b21 MANIPULATION SCORE</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(min(manip, 1.0), text=f"Manipulation: {manip:.2f}")

            with tabs[4]:
                st.markdown(
                    '<div class="slbl">\u2b21 RAW API RESPONSE</div>',
                    unsafe_allow_html=True,
                )
                st.json(data)
                st.download_button(
                    "\u2b21  DOWNLOAD JSON",
                    json.dumps(data, indent=2),
                    "sentinel_result.json",
                    "application/json",
                )


# ===================== PAGE 2: MONITOR =====================
elif "MONITOR" in page:
    st.markdown(
        """<div class="cmd-hdr"><div class="cmd-inner">
    <div class="grad-line"></div><div class="hdr-glow"></div>
    <p class="glitch" style="font-size:28px">SYSTEM MONITOR</p>
    <p class="sub-t">REAL-TIME PLATFORM METRICS</p>
    </div></div>""",
        unsafe_allow_html=True,
    )

    grafana_url = "http://localhost:3000/d/sentinel-main/sentinel-ai-dashboard?orgId=1&refresh=10s"
    st.components.v1.iframe(grafana_url, height=600, scrolling=True)

    if health:
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            metric_card(f"{health.get('models_loaded',0)}/13", "MODELS LOADED"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            metric_card(f"{health.get('kb_size',0):,}", "KB CHUNKS"),
            unsafe_allow_html=True,
        )
        c3.markdown(
            metric_card(health.get("device", "cpu").upper(), "DEVICE"),
            unsafe_allow_html=True,
        )

    if st.button("\u2b21  REFRESH METRICS", use_container_width=True):
        st.rerun()

# ===================== PAGE 3: SYSTEM =====================
elif "SYSTEM" in page:
    st.markdown(
        """<div class="cmd-hdr"><div class="cmd-inner">
    <div class="grad-line"></div><div class="hdr-glow"></div>
    <p class="glitch" style="font-size:28px">SYSTEM INFORMATION</p>
    <p class="sub-t">ARCHITECTURE & COURSE COVERAGE</p>
    </div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="slbl">\u2b21 MODULE ARCHITECTURE</div>', unsafe_allow_html=True
    )
    arch = [
        [
            "A",
            "Classical NLP",
            "TextCleaner, Tokenizer, PosParser, FeatureExtractor",
            "CSR322 Units I-III",
        ],
        [
            "B",
            "Deep Learning",
            "MLPClassifier, BiLSTMAttention, TextCNN, ImageCNN",
            "CSR311 Units I-IV",
        ],
        [
            "C",
            "Transformers",
            "BERT/RoBERTa/DistilBERT, BART, T5, NER, QA, Sentiment",
            "CSR322 Units IV-VI",
        ],
        [
            "D",
            "Generative / XAI",
            "GAN Augmentor, VAE Explorer, SHAP, LIME, FedAvg",
            "CSR311 Units V-VI",
        ],
        [
            "E",
            "API + Dashboard",
            "FastAPI, Streamlit, Prometheus, Grafana",
            "INT377 Units I-V",
        ],
        [
            "F",
            "Infrastructure",
            "Docker, Kubernetes, Terraform, GitHub Actions",
            "INT377 Units III-V",
        ],
        [
            "G",
            "RAG Fact-Check",
            "ClaimExtractor, Retriever, RAG Chain, ChromaDB, LangChain",
            "CSR322 CO4-CO6",
        ],
    ]
    st.table(
        {
            "Module": [a[0] for a in arch],
            "Name": [a[1] for a in arch],
            "Components": [a[2] for a in arch],
            "Coverage": [a[3] for a in arch],
        }
    )

    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="slbl">\u2b21 TEAM</div>', unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    t1.markdown(
        """<div class="smc" style="text-align:left;padding:28px 30px">
    <div class="ml" style="margin-bottom:10px">DEPARTMENT</div>
    <div class="mv" style="font-size:18px">B.Tech CSE (AI/ML)</div>
    <div style="font-family:IBM Plex Mono,monospace;font-size:12px;color:#7a9cc0;margin-top:10px">Lovely Professional University</div>
    <div class="mbar"></div></div>""",
        unsafe_allow_html=True,
    )
    t2.markdown(
        """<div class="smc" style="text-align:left;padding:28px 30px">
    <div class="ml" style="margin-bottom:10px">SESSION</div>
    <div class="mv" style="font-size:18px">2025-26</div>
    <div style="font-family:IBM Plex Mono,monospace;font-size:12px;color:#7a9cc0;margin-top:10px">Capstone Project \u00b7 Semester VI</div>
    <div class="mbar"></div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#3a5a7a;letter-spacing:1.5px">SENTINEL-AI v3.0.0  \u00b7  BUILD 2025.05  \u00b7  PYTHON 3.11+  \u00b7  TORCH 2.x  \u00b7  CUDA READY</p>',
        unsafe_allow_html=True,
    )
