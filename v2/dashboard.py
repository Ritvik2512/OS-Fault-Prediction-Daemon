# dashboard.py — section 9: Streamlit view of the daemon's event file, plus a launcher that
# starts the daemon with any of the evaluated models.
#
#   streamlit run v2/dashboard.py

import html
import json
import re
import subprocess
import sys
import time

import altair as alt
import pandas as pd
import requests
import streamlit as st

import common as C
from detector import available_models, model_spec

cfg = C.load_config()
RES = C.path(cfg, "results_dir")
EVENTS = RES / "live_events.jsonl"
SEV = {"low": "#3DD68C", "medium": "#F2A541", "high": "#FF6B4A", "critical": "#E5484D"}
LEVEL = {"INFO": "#7DA2C9", "WARNING": "#F2A541", "ERROR": "#FF6B4A", "SEVERE": "#FF6B4A",
         "FATAL": "#E5484D", "FAILURE": "#E5484D"}
MODEL_LABEL = {"distilbert": "DistilBERT", "roberta": "RoBERTa", "electra": "ELECTRA", "hgb": "Gradient-boosted trees"}

st.set_page_config(page_title="Fault-prediction daemon", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0D1220; border-right: 1px solid #1F2937; }
.hero { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap;
        padding: 1.2rem 1.4rem; border-radius: 16px; margin-bottom: 1rem;
        background: radial-gradient(1200px 200px at 0% 0%, rgba(139,123,255,.20), transparent 60%),
                    linear-gradient(135deg, #111827 0%, #0F172A 100%);
        border: 1px solid #1F2937; }
.hero h1 { margin:0; font-size: 1.75rem; font-weight: 700; letter-spacing: -.02em;
           background: linear-gradient(90deg, #FFFFFF, #B9B0FF); -webkit-background-clip: text; color: transparent; }
.hero p { margin:.25rem 0 0 0; color:#9CA3AF; font-size:.92rem; }
.banner { display:inline-flex; align-items:center; gap:.5rem; padding:.4rem .8rem; border-radius:999px;
          background: rgba(139,123,255,.12); border:1px solid rgba(139,123,255,.35); color:#C9C2FF; font-size:.82rem; }
.chips { display:flex; gap:.5rem; flex-wrap:wrap; margin: .2rem 0 1rem 0; }
.chip { padding:.28rem .7rem; border-radius:8px; background:#111827; border:1px solid #1F2937; color:#D1D5DB;
        font-size:.8rem; font-family:'JetBrains Mono', monospace; }
.chip b { color:#fff; font-weight:600; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .8rem; margin-bottom: 1rem; }
.kpi { background:#111827; border:1px solid #1F2937; border-radius:14px; padding: .9rem 1rem; }
.kpi .label { color:#9CA3AF; font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }
.kpi .value { color:#F9FAFB; font-size:1.6rem; font-weight:700; margin-top:.2rem; font-variant-numeric: tabular-nums; }
.kpi .sub { color:#6B7280; font-size:.78rem; margin-top:.15rem; }
.bar { height:6px; background:#1F2937; border-radius:999px; overflow:hidden; margin-top:.5rem; }
.bar > div { height:100%; border-radius:999px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:.45rem; vertical-align:middle; }
.dot.live { background:#3DD68C; box-shadow: 0 0 0 0 rgba(61,214,140,.7); animation: pulse 1.6s infinite; }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(61,214,140,.6);} 70% { box-shadow:0 0 0 10px rgba(61,214,140,0);} 100% { box-shadow:0 0 0 0 rgba(61,214,140,0);} }
.section { color:#E5E7EB; font-weight:600; font-size:1rem; margin: .2rem 0 .5rem 0; }
.panel { background:#111827; border:1px solid #1F2937; border-radius:14px; padding: .8rem 1rem; }
.term { background:#05070D; border:1px solid #1F2937; border-radius:12px; padding:.7rem .9rem; max-height: 360px;
        overflow-y:auto; font-family:'JetBrains Mono', monospace; font-size:.74rem; line-height:1.55; }
.term .ln { white-space: nowrap; overflow:hidden; text-overflow: ellipsis; color:#C7CDD8; }
.term .ts { color:#6B7280; } .term .node { color:#8B95A7; } .term .comp { color:#B9B0FF; }
.card { background:#111827; border:1px solid #1F2937; border-left: 4px solid var(--sev); border-radius:12px;
        padding:.8rem 1rem; margin-bottom:.7rem; }
.card .top { display:flex; justify-content:space-between; align-items:center; gap:.5rem; }
.pill { padding:.12rem .55rem; border-radius:999px; font-size:.7rem; font-weight:700; letter-spacing:.05em;
        color:#0A0E17; background: var(--sev); }
.card .comp { color:#E5E7EB; font-weight:600; font-size:.85rem; }
.card .sum { color:#E5E7EB; font-size:.9rem; margin:.45rem 0; }
.card .act { color:#C9C2FF; font-size:.85rem; }
.card .meta { color:#6B7280; font-size:.74rem; margin-top:.5rem; font-family:'JetBrains Mono', monospace; }
.empty { color:#6B7280; font-size:.85rem; padding: 1.2rem; text-align:center; border:1px dashed #1F2937; border-radius:12px; }
.side-title { color:#9CA3AF; font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; margin: .8rem 0 .3rem 0; }
.stat { display:flex; justify-content:space-between; font-size:.84rem; padding:.22rem 0; border-bottom:1px solid #1A2232; }
.stat span:first-child { color:#9CA3AF; } .stat span:last-child { color:#F3F4F6; font-variant-numeric: tabular-nums; }
.note { color:#6B7280; font-size:.74rem; margin-top:.5rem; }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] { background:#111827; border:1px solid #1F2937; border-radius:10px; padding:.35rem .9rem; }
.stTabs [aria-selected="true"] { border-color:#8B7BFF; }
.stMarkdown table { font-size:.8rem; display:block; overflow-x:auto; }
</style>
""", unsafe_allow_html=True)

e = html.escape


# ------------------------------------------------------------------ data helpers

def load_events():
    if not EVENTS.exists():
        return []
    out = []
    for ln in EVENTS.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:       # a line the daemon is still writing
            pass
    return out


@st.cache_data(ttl=10)
def comparison():
    p = RES / "model_comparison.csv"
    return pd.read_csv(p).set_index("model") if p.exists() else None


@st.cache_data(ttl=10)
def ollama_up():
    try:
        return requests.get("http://localhost:11434/api/version", timeout=0.5).ok
    except requests.RequestException:
        return False


@st.cache_resource
def registry():
    """Daemon process started from this dashboard (shared across browser sessions)."""
    return {"proc": None, "launched_at": 0.0, "model": None}


def daemon_running():
    p = registry()["proc"]
    return p is not None and p.poll() is None


def stop_daemon():
    reg = registry()
    if daemon_running():
        reg["proc"].terminate()
        reg["proc"].wait(timeout=10)


def start_daemon(model, slice_mode, speed, llm):
    stop_daemon()
    cmd = [sys.executable, str(C.ROOT / "v2" / "daemon.py"), "--demo", "--model", model,
           "--slice-mode", slice_mode, "--speed", str(speed)] + ([] if llm else ["--no-llm"])
    log = open(RES / "daemon_dashboard.log", "w", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    reg = registry()
    reg.update(proc=subprocess.Popen(cmd, cwd=C.ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags),
               launched_at=time.time(), model=model)


# ------------------------------------------------------------------ sidebar: launcher + model card

def sidebar():
    sb = st.sidebar
    sb.markdown("### ⚡ Demo control")
    models = available_models(cfg)
    sel = C.read_json(C.path(cfg, "models_dir", "selected.json"))
    label = lambda m: f"{MODEL_LABEL.get(m, m)}  ·  selected" if m == sel["name"] else MODEL_LABEL.get(m, m)
    model = sb.selectbox("Model", models, index=models.index(sel["name"]), format_func=label)
    spec = model_spec(cfg, model)
    slice_mode = sb.radio("Replay slice", ["own", "shared"], horizontal=True,
                          format_func=lambda s: "Model's showcase" if s == "own" else "Shared demo",
                          help="Showcase: a test fault this model's gate warned about offline (picked with ground "
                               "truth, so it fires). Shared: the same slice for every model, for a fair side-by-side.")
    speed = sb.slider("Speed (log lines / second)", 50, 1000, int(cfg["daemon"]["speed"]), step=50)
    llm = sb.toggle("Write LLM incident cards", value=True)
    if llm and not ollama_up():
        sb.warning("Ollama is not reachable on localhost:11434 - cards will fall back. Start it with `ollama serve`.")
    c1, c2 = sb.columns(2)
    if c1.button("▶ Start demo", type="primary", width="stretch"):
        start_daemon(model, slice_mode, speed, llm)
        st.rerun()
    if c2.button("■ Stop", width="stretch", disabled=not daemon_running()):
        stop_daemon()
        st.rerun()
    if daemon_running():
        sb.caption(f"Started **{MODEL_LABEL.get(registry()['model'])}** from the dashboard (pid {registry()['proc'].pid}). Status is shown on the Live tab.")
    sb.markdown(f'<div class="side-title">Operating point · {e(MODEL_LABEL.get(model, model))}</div>'
                f'<div class="stat"><span>Threshold τ</span><span>{spec["tau"]:.4f}</span></div>'
                f'<div class="stat"><span>Gate N</span><span>{spec["N"]} window(s)</span></div>'
                f'<div class="stat"><span>Cooldown</span><span>{spec["cooldown"]} windows</span></div>',
                unsafe_allow_html=True)
    df = comparison()
    if df is not None and model in df.index:
        r = df.loc[model]
        rows = [("Window precision", r.test_precision), ("Window recall", r.test_recall),
                ("Incident precision", r.test_incident_precision), ("Episode recall", r.test_episode_recall),
                ("Test PR-AUC", r.test_pr_auc)]
        sb.markdown('<div class="side-title">Held-out test set</div>' + "".join(
            f'<div class="stat"><span>{k}</span><span>{float(v):.3f}</span></div>' for k, v in rows) +
            '<div class="note">Window-level and incident-level are different units - see the Model comparison tab. '
            'Incident cards are LLM-generated summaries, not verified diagnoses.</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ live view

def kpi(label, value, sub="", bar=None, color="#8B7BFF"):
    b = f'<div class="bar"><div style="width:{max(0, min(bar, 1)) * 100:.1f}%;background:{color}"></div></div>' if bar is not None else ""
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div>{b}</div>'


def log_line(ln):
    p = ln.split(None, 8)
    if len(p) < 9:
        return f'<div class="ln">{e(ln)}</div>'
    m = re.match(r"\d{4}-\d{2}-\d{2}-(\d{2})\.(\d{2})\.(\d{2})", p[3])
    ts = f"{m.group(1)}:{m.group(2)}:{m.group(3)}" if m else p[3]
    col = LEVEL.get(p[7], "#9CA3AF")
    return (f'<div class="ln"><span class="ts">{e(ts)}</span> <span class="node">{e(p[2])}</span> '
            f'<span class="comp">{e(p[6])}</span> <span style="color:{col};font-weight:600">{e(p[7])}</span> {e(p[8])}</div>')


def chart(windows, fired, tau):
    df = pd.DataFrame({"window": [w["window_idx"] for w in windows], "confidence": [w["confidence"] for w in windows]})
    x = alt.X("window:Q", title="Window (20 log lines each)")
    y = alt.Y("confidence:Q", title="P(pre-fault)", scale=alt.Scale(domain=[0, 1]))
    zone = alt.Chart(pd.DataFrame({"y": [tau], "y2": [1]})).mark_rect(color="#E5484D", opacity=0.07).encode(y="y:Q", y2="y2:Q")
    area = alt.Chart(df).mark_area(
        line={"color": "#8B7BFF", "strokeWidth": 2}, interpolate="monotone",
        color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                           stops=[alt.GradientStop(color="rgba(139,123,255,0.02)", offset=0),
                                  alt.GradientStop(color="rgba(139,123,255,0.45)", offset=1)])
    ).encode(x=x, y=y, tooltip=["window", alt.Tooltip("confidence", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"tau": [tau]})).mark_rule(color="#E5484D", strokeDash=[6, 4]).encode(y="tau:Q")
    text = alt.Chart(pd.DataFrame({"tau": [tau], "label": [f"τ = {tau:.3f}"]})).mark_text(
        align="left", dx=4, dy=-7, color="#FF8A8D", fontSize=11).encode(y="tau:Q", text="label:N", x=alt.value(0))
    layers = [zone, area, rule, text]
    if fired:
        marks = pd.DataFrame({"window": [f["window_range"][1] for f in fired], "confidence": [f["confidence"] for f in fired]})
        layers.append(alt.Chart(marks).mark_point(shape="triangle-down", size=220, filled=True, color="#E5484D",
                                                  stroke="#fff", strokeWidth=1)
                      .encode(x="window:Q", y="confidence:Q", tooltip=["window", alt.Tooltip("confidence", format=".3f")]))
    return (alt.layer(*layers).properties(height=320)
            .configure(background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor="#9CA3AF", titleColor="#9CA3AF", gridColor="#1F2937", domainColor="#374151",
                            labelFont="Inter", titleFont="Inter"))


def card_html(inc):
    c = inc["card"]
    sev = c["severity"] if c["severity"] in SEV else "medium"
    return (f'<div class="card" style="--sev:{SEV[sev]}"><div class="top"><span class="comp">{e(c["component"])}</span>'
            f'<span class="pill">{e(sev.upper())}</span></div><div class="sum">{e(c["summary"])}</div>'
            f'<div class="act">→ {e(c["suggested_action"])}</div>'
            f'<div class="meta">windows {inc["window_range"][0]}-{inc["window_range"][1]} · conf {inc["confidence"]:.3f}'
            f' · LLM {inc["llm_seconds"]:.1f}s</div></div>')


@st.fragment(run_every="1s")
def live():
    reg = registry()
    events = load_events()
    starting = daemon_running() and not any(ev["type"] == "start" and ev["t"] >= reg["launched_at"] for ev in events)
    if starting:
        st.markdown(f'<div class="empty">Loading <b>{e(MODEL_LABEL.get(reg["model"], reg["model"]))}</b> and warming up '
                    'the LLM… the stream starts in a few seconds.</div>', unsafe_allow_html=True)
        return
    start = next((ev for ev in reversed(events) if ev["type"] == "start"), None)
    windows = [ev for ev in events if ev["type"] == "window"]
    if not start or not windows:
        st.markdown('<div class="empty">No stream yet. Pick a model in the sidebar and press <b>▶ Start demo</b> '
                    '(or run <code>python v2/daemon.py --demo</code>).</div>', unsafe_allow_html=True)
        return
    fired = [ev for ev in events if ev["type"] == "incident_fired"]
    cards = [ev for ev in events if ev["type"] == "incident"]
    done = any(ev["type"] == "end" for ev in events)
    tau, total = start["tau"], max(start.get("windows", len(windows)), 1)
    name = MODEL_LABEL.get(start["model"], start["model"])

    sl = start.get("slice")
    st.markdown('<div class="chips">' + "".join(f'<span class="chip">{k} <b>{e(str(v))}</b></span>' for k, v in [
        ("model", name), ("τ", f"{tau:.3f}"), ("N", start["N"]), ("cooldown", start["cooldown"]),
        ("slice", f"{sl[0]:,}–{sl[1]:,}" if sl else "-"), ("mode", start.get("slice_mode", "-"))]) + "</div>",
        unsafe_allow_html=True)

    conf = windows[-1]["confidence"]
    status = ('<span class="dot live"></span>Streaming' if not done else
              '<span class="dot" style="background:#6B7280"></span>Finished')
    st.markdown('<div class="kpis">' +
                kpi("Status", status, f"{len(windows) * cfg['window']['W']:,} log lines read") +
                kpi("Windows scored", f"{len(windows)}<span style='color:#6B7280;font-size:1rem'> / {total}</span>",
                    "progress through the slice", len(windows) / total) +
                kpi("Latest confidence", f"{conf:.3f}", "above τ - pre-fault" if conf >= tau else "below τ",
                    conf, "#E5484D" if conf >= tau else "#8B7BFF") +
                kpi("Incidents predicted", f"{len(fired)}", f"{len(cards)} card(s) written") + "</div>",
                unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown('<div class="section">Pre-fault confidence per window</div>', unsafe_allow_html=True)
        st.altair_chart(chart(windows, fired, tau), width="stretch")
        st.markdown('<div class="section">Live log feed</div>', unsafe_allow_html=True)
        feed = [ln for w in windows[-2:] for ln in w["lines"]][-30:]
        st.markdown('<div class="term">' + "".join(log_line(ln) for ln in reversed(feed)) + "</div>",
                    unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section">Incident cards</div>', unsafe_allow_html=True)
        pending = len(fired) - len(cards)
        if pending > 0:
            st.caption(f"✍️ {pending} card(s) being written by the local LLM…" if not done else
                       f"{pending} incident(s) without a card (run ended or ran without the LLM).")
        if not fired:
            st.markdown('<div class="empty">No incident predicted yet.</div>', unsafe_allow_html=True)
        st.markdown("".join(card_html(c) for c in reversed(cards)), unsafe_allow_html=True)


def render_md(path):
    """Markdown report with its ![..](file.png) lines rendered as images."""
    if not path.exists():
        st.info(f"{path.name} not generated yet.")
        return
    chunk = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", ln.strip())
        if m:
            st.markdown("\n".join(chunk))
            chunk = []
            if (RES / m.group(1)).exists():
                st.image(str(RES / m.group(1)), width="stretch")
        else:
            chunk.append(ln)
    st.markdown("\n".join(chunk))


# ------------------------------------------------------------------ page

st.markdown('<div class="hero"><div><h1>OS fault-prediction daemon</h1>'
            '<p>Predicts the run-up to faults in Blue Gene/L system logs and writes an incident card with a local LLM.</p></div>'
            '<span class="banner">● Simulated live stream from held-out BGL logs.</span></div>', unsafe_allow_html=True)
sidebar()
tab_live, tab_models, tab_report = st.tabs(["Live", "Model comparison", "Evaluation report"])
with tab_live:
    live()
with tab_models:
    render_md(RES / "model_report.md")
with tab_report:
    render_md(RES / "RUN_REPORT.md")
    if (RES / "pr_curves.png").exists():
        st.image(str(RES / "pr_curves.png"), caption="Window-level PR curves on the validation set")
