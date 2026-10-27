# make_deck.py — builds the Review 2 product-demo deck from the files in results/, models/ and data/v2/.
# Every number on a slide is read from those files; anything missing becomes a red MISSING box.
#
#   python v2/make_deck.py            # build results/Review2_Fault_Prediction_Daemon.pptx + QA
#   python v2/make_deck.py --render   # also export PDF + PNG previews through PowerPoint (Windows)

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from lxml import etree
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
RES, MODELS, DATA = ROOT / "results", ROOT / "models", ROOT / "data" / "v2"
SHOTS = ROOT / "screenshots"
OUT = RES / "Review2_Fault_Prediction_Daemon.pptx"

NAVY, PANEL_NAVY, TEAL, LTEAL = "0E1B2A", "13293D", "1C7293", "2E8BC0"
AMBER, GREEN, RED, INK, MUTED = "E8A13A", "2E9E5B", "D64545", "1B2A3A", "50647A"
LIGHT, BORDER, WHITE = "F4F6F8", "D7DEE6", "FFFFFF"
FONT = "Calibri"
SW, SH, M = 13.333, 7.5, 0.5
LABEL = {"distilbert": "DistilBERT", "roberta": "RoBERTa", "electra": "ELECTRA", "hgb": "Gradient-boosted trees"}
FORBIDDEN = ["v1", "baseline", "0.193", "0.297", "0.649", "true_label", "comparative study", "not a benchmark",
             "labelled pre-fault", "improved from", "before/after"]

_FONTS = {(b, i): ImageFont.truetype(f"C:/Windows/Fonts/{f}", 100)
          for (b, i), f in {(False, False): "calibri.ttf", (True, False): "calibrib.ttf",
                            (False, True): "calibrii.ttf", (True, True): "calibriz.ttf"}.items()}


def rgb(h):
    return RGBColor.from_string(h)


def label(m):
    return LABEL.get(m, m)


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


# ======================================================================= data

class Data:
    """Loads every input once and records where each value came from."""

    def __init__(self):
        self.missing, self.sources = [], defaultdict(set)

    def file(self, rel):
        p = ROOT / rel
        if not p.exists():
            self.missing.append(rel)
            return None
        return p

    def load(self):
        f = self.file("models/selected.json")
        self.sel = json.loads(f.read_text()) if f else None
        f = self.file("results/model_comparison.csv")
        self.comp = None
        if f:
            c = pd.read_csv(f)
            c = c[c["type"].notna()].set_index("model")          # model rows only
            order = sorted(c.index, key=lambda m: (m != (self.sel or {}).get("name"), -c.loc[m, "val_pr_auc"]))
            self.comp = c.loc[order]
        self.cfg = yaml.safe_load(open(ROOT / "v2/config.yaml", encoding="utf-8"))
        f = self.file("data/v2/default/meta.json")
        self.meta = json.loads(f.read_text()) if f else None
        f = self.file("data/v2/parse_meta.json")
        self.parse = json.loads(f.read_text()) if f else None
        f = self.file("results/ablation.csv")
        self.abl = pd.read_csv(f) if f else None
        f = self.file("results/gate_sweep.csv")
        self.gate = pd.read_csv(f) if f else None
        f = self.file("results/RUN_REPORT.md")
        self.kpi = self._run_report(f.read_text(encoding="utf-8")) if f else None
        self.models = list(self.comp.index) if self.comp is not None else []
        self.train = {}
        for m in self.models:
            f = self.file(f"results/train_{m}.json")
            self.train[m] = json.loads(f.read_text()) if f else None
        s = (self.sel or {}).get("name")
        f = self.file(f"results/eval_{s}.json") if s else None
        self.eval_sel = json.loads(f.read_text()) if f else None
        f = self.file(f"results/demo_runs/{s}_shared.jsonl") if s else None
        self.demo = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()] if f else None
        f = self.file("results/model_report.md")
        self.demo_row = self._demo_row(f.read_text(encoding="utf-8"), s) if f and s else None

    def _run_report(self, t):
        k = {}
        m = re.search(r"Incident-level test precision ([0-9.]+), episode recall ([0-9.]+) \((\d+) of (\d+) episodes", t)
        if m:
            k.update(incident_precision=float(m[1]), episode_recall=float(m[2]), caught=int(m[3]), episodes=int(m[4]))
        m = re.search(r"Median lead time: ([0-9.]+) s \(IQR ([0-9.]+)-([0-9.]+) s\)", t)
        if m:
            k.update(lead_median=float(m[1]), lead_q1=float(m[2]), lead_q3=float(m[3]))
        m = re.search(r"LLM latency \(([^)]+)\): (\d+) incident card\(s\) in the last daemon run, median ([0-9.]+) s", t)
        if m:
            k.update(llm=m[1], llm_cards=int(m[2]), llm_median=float(m[3]))
        for key in ("incident_precision", "lead_median", "llm_median"):
            if key not in k:
                self.missing.append(f"results/RUN_REPORT.md ({key})")
        return k

    @staticmethod
    def _demo_row(t, name):
        sec = t.split("## 5.")[1].split("## 6.")[0] if "## 5." in t else ""
        rows = [r for r in sec.splitlines() if r.startswith("|")]
        if len(rows) < 3:
            return None
        head = [c.strip() for c in rows[0].strip("|").split("|")]
        for r in rows[2:]:
            cells = [c.strip() for c in r.strip("|").split("|")]
            if cells[0] == name:
                return dict(zip(head, cells))
        return None

    def src(self, slide, what, rel):
        self.sources[slide].add(f"{what} <- {rel}")


# ======================================================================= deck builder

class Deck:
    def __init__(self, data):
        self.d = data
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(SW), Inches(SH)
        self.reg = []            # (slide_no, id, x, y, w, h, parent, kind, name)
        self.texts = []          # (slide_no, name, paragraphs, w, h)
        self.placeholders = []
        self.n = 0
        self._id = 0

    # ---------------------------------------------------------- primitives
    def slide(self, title=None, dark=False, notes=None):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(NAVY if dark else WHITE)
        if title:
            self.text(s, title, M + 0.1, M, SW - 2 * M - 0.2, 0.8, size=34, bold=True,
                      color=WHITE if dark else INK, anchor=MSO_ANCHOR.TOP, name="title")
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def _register(self, x, y, w, h, parent, kind, name):
        self._id += 1
        self.reg.append((self.n, self._id, x, y, w, h, parent, kind, name))
        return self._id

    def box(self, s, x, y, w, h, fill=LIGHT, line=BORDER, lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
            radius=0.08, parent=None, kind="box", name="", dash=False):
        sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        sh.shadow.inherit = False
        if fill is None:
            sh.fill.background()
        else:
            sh.fill.solid()
            sh.fill.fore_color.rgb = rgb(fill)
        if line is None:
            sh.line.fill.background()
        else:
            sh.line.color.rgb = rgb(line)
            sh.line.width = Pt(lw)
            if dash:
                sh.line.dash_style = MSO_LINE.DASH
        if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
            sh.adjustments[0] = radius
        sh.text_frame.text = ""
        return sh, self._register(x, y, w, h, parent, kind, name)

    def text(self, s, content, x, y, w, h, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, italic=False, parent=None, name="", kind="text", spacing=1.0):
        """content: str, or list of paragraphs; a paragraph is str or dict(text, size, bold, color, italic)."""
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap, tf.auto_size, tf.vertical_anchor = True, MSO_AUTO_SIZE.NONE, anchor
        tf.margin_left = tf.margin_right = Inches(0.05)
        tf.margin_top = tf.margin_bottom = Inches(0.03)
        paras = content if isinstance(content, list) else [content]
        norm = []
        for i, p in enumerate(paras):
            p = {"text": p} if isinstance(p, str) else dict(p)
            p.setdefault("size", size); p.setdefault("bold", bold); p.setdefault("color", color)
            p.setdefault("italic", italic)
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = p.get("align", align)
            para.line_spacing = spacing
            if i:
                para.space_before = Pt(p.get("space", 4))
            r = para.add_run()
            r.text = p["text"]
            f = r.font
            f.name, f.size, f.bold, f.italic = FONT, Pt(p["size"]), p["bold"], p["italic"]
            f.color.rgb = rgb(p["color"])
            norm.append(p)
        self.texts.append((self.n, name or content if isinstance(content, str) else name, norm, w, h, spacing))
        self._register(x, y, w, h, parent, kind, name)
        return tb

    def arrow(self, s, x1, y1, x2, y2, color=MUTED, w=1.75, head=True, dash=False):
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        c.line.color.rgb, c.line.width = rgb(color), Pt(w)
        if dash:
            c.line.dash_style = MSO_LINE.DASH
        if head:
            ln = c.line._get_or_add_ln()
            t = etree.SubElement(ln, qn("a:tailEnd"))
            t.set("type", "triangle"); t.set("w", "med"); t.set("len", "med")
        self._register(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1), None, "conn", "arrow")
        return c

    def placeholder(self, s, x, y, w, h, what, parent=None):
        self.box(s, x, y, w, h, fill=RED, line="9E2B2B", lw=2, radius=0.04, parent=parent, kind="placeholder")
        self.text(s, [{"text": "MISSING", "size": 20, "bold": True},
                      {"text": what, "size": 14}], x + 0.15, y + 0.15, w - 0.3, h - 0.3, color=WHITE,
                  align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="placeholder")
        self.placeholders.append((self.n, what))

    def picture(self, s, path, x, y, w, h, parent=None):
        """Insert an image scaled to fit (x, y, w, h), aspect ratio kept, centred."""
        iw, ih = Image.open(path).size
        sc = min(w / iw, h / ih)
        pw, ph = iw * sc, ih * sc
        px, py = x + (w - pw) / 2, y + (h - ph) / 2
        s.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
        self._register(px, py, pw, ph, parent, "picture", Path(path).name)

    def terminal(self, s, x, y, w, h, img, caption, name=""):
        """Screenshot inside a dark terminal frame (three dots + italic caption in the top bar)."""
        _, fid = self.box(s, x, y, w, h, fill=NAVY, line=PANEL_NAVY, lw=1, radius=0.04, name="terminal")
        for i, c in enumerate((RED, AMBER, GREEN)):
            self.box(s, x + 0.18 + i * 0.24, y + 0.15, 0.14, 0.14, fill=c, line=None, shape=MSO_SHAPE.OVAL,
                     parent=fid, kind="dot")
        self.text(s, caption, x + 0.95, y + 0.06, w - 1.1, 0.34, size=11, italic=True, color="B8C7D6",
                  anchor=MSO_ANCHOR.MIDDLE, parent=fid, name="term-caption")
        ix, iy, iw, ih = x + 0.15, y + 0.48, w - 0.3, h - 0.63
        p = SHOTS / img
        if p.exists():
            self.picture(s, p, ix, iy, iw, ih, parent=fid)
        else:
            self.placeholder(s, ix, iy, iw, ih, f"screenshots/{img}", parent=fid)

    def table(self, s, rows, x, y, w, col_w, row_h=0.5, size=14, header_fill=NAVY, highlight=None,
              align=None, name="table", parent=None):
        """rows[0] is the header. highlight: set of row indexes painted green (selected model)."""
        nr, nc = len(rows), len(rows[0])
        gf = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
        t = gf.table
        tot = sum(col_w)
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(w * cw / tot)
        for i in range(nr):
            t.rows[i].height = Inches(row_h)
            for j in range(nc):
                c = t.cell(i, j)
                c.margin_left = c.margin_right = Inches(0.08)
                c.margin_top = c.margin_bottom = Inches(0.03)
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
                hl = highlight and i in highlight
                c.fill.solid()
                c.fill.fore_color.rgb = rgb(header_fill if i == 0 else GREEN if hl else (LIGHT if i % 2 else WHITE))
                tf = c.text_frame
                tf.word_wrap = True
                tf.text = ""
                p = tf.paragraphs[0]
                p.alignment = (align[j] if align else (PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER))
                r = p.add_run()
                r.text = str(rows[i][j])
                r.font.name, r.font.size = FONT, Pt(size)
                r.font.bold = i == 0 or bool(hl)
                r.font.color.rgb = rgb(WHITE if (i == 0 or hl) else INK)
                cw_in = w * col_w[j] / tot
                self.texts.append((self.n, f"{name}[{i},{j}]", [{"text": str(rows[i][j]), "size": size,
                                   "bold": i == 0 or bool(hl), "italic": False}], cw_in - 0.12, row_h - 0.02, 1.0))
        self._register(x, y, w, row_h * nr, parent, "table", name)
        return t

    def chart(self, s, kind, cats, series, x, y, w, h, title, colors, fmt="0.0%", labels=True,
              legend=True, vmax=None, vmin=None, axis_fmt=None, label_size=12, name="chart"):
        cd = CategoryChartData()
        cd.categories = cats
        for nm, vals in series:
            cd.add_series(nm, vals)
        gf = s.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
        ch = gf.chart
        ch.font.name, ch.font.size = FONT, Pt(12)
        ch.font.color.rgb = rgb(INK)
        ch.has_title = True
        ch.chart_title.text_frame.text = title
        for p in ch.chart_title.text_frame.paragraphs:
            for r in p.runs:
                r.font.size, r.font.bold, r.font.name = Pt(15), True, FONT
                r.font.color.rgb = rgb(INK)
        ch.has_legend = legend and len(series) > 1
        if ch.has_legend:
            ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.BOTTOM, False
            ch.legend.font.size = Pt(12)
        plot = ch.plots[0]
        line_kind = kind in (XL_CHART_TYPE.LINE_MARKERS, XL_CHART_TYPE.LINE)
        for i, sr in enumerate(plot.series):
            col = rgb(colors[i % len(colors)])
            if line_kind:
                sr.format.line.color.rgb, sr.format.line.width = col, Pt(2.5)
                sr.smooth = False
                if kind == XL_CHART_TYPE.LINE_MARKERS:
                    sr.marker.style, sr.marker.size = XL_MARKER_STYLE.CIRCLE, 8
                    sr.marker.format.fill.solid(); sr.marker.format.fill.fore_color.rgb = col
                    sr.marker.format.line.color.rgb = col
            else:
                sr.format.fill.solid(); sr.format.fill.fore_color.rgb = col
        if not line_kind:
            plot.gap_width = 60
        if labels:
            plot.has_data_labels = True
            dl = plot.data_labels
            dl.number_format, dl.number_format_is_linked = fmt, False
            dl.font.size, dl.font.name = Pt(label_size), FONT
            dl.position = XL_LABEL_POSITION.ABOVE if line_kind else XL_LABEL_POSITION.OUTSIDE_END
        va = ch.value_axis
        va.has_major_gridlines = True
        va.major_gridlines.format.line.color.rgb = rgb("E3E8EE")
        va.format.line.fill.background()
        va.tick_labels.font.size = Pt(11)
        va.tick_labels.number_format, va.tick_labels.number_format_is_linked = axis_fmt or fmt, False
        if vmax is not None:
            va.maximum_scale = vmax
        if vmin is not None:
            va.minimum_scale = vmin
        ca = ch.category_axis
        ca.tick_labels.font.size = Pt(12)
        ca.format.line.color.rgb = rgb(BORDER)
        self._register(x, y, w, h, None, "chart", name)
        return ch

    def icon_badge(self, s, x, y, d, num, fill=TEAL, parent=None):
        self.box(s, x, y, d, d, fill=fill, line=None, shape=MSO_SHAPE.OVAL, parent=parent, kind="icon")
        self.text(s, str(num), x, y, d, d, size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
                  anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="badge")

    def card(self, s, x, y, w, h, title, body, accent=TEAL, title_size=18, body_size=14, fill=LIGHT,
             line=BORDER, lw=1.0, title_color=INK, body_color=MUTED, name="card"):
        _, cid = self.box(s, x, y, w, h, fill=fill, line=line, lw=lw, name=name)
        self.text(s, title, x + 0.2, y + 0.15, w - 0.4, 0.45, size=title_size, bold=True, color=title_color,
                  parent=cid, name=f"{name}-title")
        if body:
            self.text(s, body, x + 0.2, y + 0.65, w - 0.4, h - 0.8, size=body_size, color=body_color,
                      parent=cid, name=f"{name}-body")
        return cid


def skip_category_labels(ch, n):
    ax = ch.category_axis._element
    anchor = ax.find(qn("c:noMultiLvlLbl"))
    for tag in ("c:tickLblSkip", "c:tickMarkSkip"):
        el = ax.find(qn(tag))
        if el is None:
            el = etree.SubElement(ax, qn(tag))
            if anchor is not None:
                anchor.addprevious(el)
        el.set("val", str(n))


# ======================================================================= slides

def arch_slides(k, d):
    """Two diagram slides: the running system (processes, files, links) and the six modules (files, in/out)."""
    cfg, sel = d.cfg, d.sel or {}
    S = label(sel.get("name")) if sel.get("name") else "selected model"
    W, cap_lines = cfg["window"]["W"], cfg["daemon"]["max_llm_lines"]
    llm, keep = cfg["llm"]["model"], cfg["llm"]["keep_alive"]
    d.src(k.n + 1, "window size, LLM model, keep-alive", "v2/config.yaml")
    d.src(k.n + 2, "window size, LLM line cap", "v2/config.yaml")

    # ── system architecture diagram ─────────────────────────────────────────────────────────
    s = k.slide("System architecture diagram")
    _, mid = k.box(s, 0.6, 1.5, 12.2, 5.35, fill=WHITE, line=MUTED, lw=1.25, dash=True, radius=0.03, name="machine")
    k.text(s, "LOCAL MACHINE  ·  everything runs offline on one laptop", 0.85, 1.6, 8.0, 0.4, size=13, bold=True,
           color=MUTED, parent=mid, name="machine-t")

    def process(x, w, title, body, col):
        _, pid = k.box(s, x, 2.15, w, 1.85, fill=LIGHT, line=col, lw=1.75, radius=0.06, parent=mid, kind="proc")
        k.text(s, [{"text": title, "size": 17, "bold": True, "color": col},
                   {"text": body, "size": 15, "color": INK, "space": 6}], x + 0.15, 2.22, w - 0.3, 1.72,
               parent=pid, name="proc-t")

    def store(x, w, title, body, shape):
        k.box(s, x, 5.1, w, 1.35, fill=WHITE, line=TEAL, lw=1.5, shape=shape, parent=mid, kind="store")
        k.text(s, [{"text": title, "size": 14, "bold": True, "color": TEAL},
                   {"text": body, "size": 11, "color": MUTED, "space": 2}],
               x + 0.12, 5.18 if shape == MSO_SHAPE.FLOWCHART_DOCUMENT else 5.42, w - 0.24, 0.95,
               align=PP_ALIGN.CENTER, parent=-1, name="store-t")

    process(0.95, 3.3, "Streamlit dashboard", "live log feed · confidence chart · incident cards · demo launcher", TEAL)
    process(5.0, 3.3, "Fault-prediction daemon",
            f"replay → Drain3 encoder → {W}-line windows → {S} detect() → alert gate → incident builder", AMBER)
    process(9.05, 3.4, "Ollama server", f"{llm} running locally · writes JSON incident cards · kept warm {keep}", GREEN)
    store(2.2, 2.6, "live_events.jsonl", "window scores + incident cards", MSO_SHAPE.FLOWCHART_DOCUMENT)
    store(5.35, 2.6, "Held-out BGL log", "system log lines, replayed", MSO_SHAPE.FLOWCHART_DOCUMENT)
    store(9.45, 2.6, "Model store", "weights · selected.json · Drain3 state", MSO_SHAPE.CAN)

    def lbl(t, x, y, w, col=MUTED):
        k.text(s, t, x, y, w, 0.3, size=11, italic=True, color=col, kind="label", name="arch-lbl")

    k.arrow(s, 4.25, 2.75, 5.0, 2.75, color=TEAL); lbl("launches", 4.27, 2.38, 0.75, TEAL)
    k.arrow(s, 8.3, 2.7, 9.05, 2.7, color=GREEN); lbl("prompt", 8.35, 2.33, 0.7, GREEN)
    k.arrow(s, 9.05, 3.35, 8.3, 3.35, color=GREEN); lbl("card", 8.45, 3.4, 0.6, GREEN)
    k.arrow(s, 5.4, 4.0, 4.3, 5.1, color=AMBER); lbl("writes", 5.05, 4.5, 0.7, AMBER)
    k.arrow(s, 3.0, 5.1, 2.6, 4.0, color=TEAL); lbl("tails every 1 s", 1.2, 4.4, 1.35, TEAL)
    k.arrow(s, 6.65, 5.1, 6.65, 4.0, color=AMBER); lbl("replays lines", 6.75, 4.4, 1.2, AMBER)
    k.arrow(s, 9.8, 5.1, 8.35, 3.95, color=AMBER); lbl("loads model + gate settings", 9.55, 4.4, 2.6, AMBER)

    # ── module architecture diagram ─────────────────────────────────────────────────────────
    s = k.slide("Module architecture diagram")
    mods = [  # (column, row, number, name, files, input, output, colour)
        (0, 0, 1, "Data preprocessing", "build_dataset.py · common.py", "BGL system log",
         f"Drain3 templates, encoded lines, labelled {W}-line windows, 70 / 10 / 20 split", TEAL),
        (1, 0, 2, "Model training & selection", "train_transformer.py · train_classical.py · evaluate.py · select_model.py",
         "windows + labels", f"{len(d.models)} trained models; selected.json (model, τ, N, cooldown)", TEAL),
        (2, 0, 6, "Dashboard", "dashboard.py (Streamlit)", "live_events.jsonl",
         "live feed, confidence chart, incident cards; starts the daemon", GREEN),
        (0, 1, 3, "Daemon / stream loop", "daemon.py · detector.py", "held-out log lines, selected model",
         f"confidence for every {W}-line window", AMBER),
        (1, 1, 4, "Alert gate", "daemon.py (τ / N / cooldown)", "confidence stream",
         "one incident per run of flagged windows", AMBER),
        (2, 1, 5, "Incident reporting", f"daemon.py → Ollama ({llm})", f"episode lines, deduplicated (≤ {cap_lines})",
         "JSON card: summary, component, severity, action", AMBER),
    ]
    xs, ys, w, h = [0.6, 4.8, 9.0], [1.5, 4.3], 3.8, 2.05
    for c, r, num, name, files, inp, out, col in mods:
        x, y = xs[c], ys[r]
        _, cid = k.box(s, x, y, w, h, fill=LIGHT, line=col, lw=1.75, radius=0.05, name=f"mod{num}")
        k.icon_badge(s, x + 0.15, y + 0.15, 0.48, num, fill=col, parent=cid)
        k.text(s, name, x + 0.75, y + 0.12, w - 0.9, 0.52, size=16, bold=True, anchor=MSO_ANCHOR.MIDDLE,
               parent=cid, name="mod-name")
        k.text(s, files, x + 0.15, y + 0.68, w - 0.3, 0.42, size=11, italic=True, color=col, parent=cid,
               name="mod-files")
        k.text(s, [{"text": f"In:  {inp}", "size": 12}, {"text": f"Out:  {out}", "size": 12, "space": 3}],
               x + 0.15, y + 1.12, w - 0.3, 0.92, color=INK, parent=cid, name="mod-io")

    def lbl(t, x, y, w, col=MUTED):
        k.text(s, t, x, y, w, 0.3, size=11, italic=True, color=col, kind="label", name="mod-lbl")

    k.arrow(s, 4.4, 2.55, 4.8, 2.55, color=TEAL)                       # 1 -> 2
    k.arrow(s, 2.5, 3.55, 2.5, 4.3, color=TEAL); lbl("Drain3 state", 2.6, 3.8, 1.3, TEAL)        # 1 -> 3
    k.arrow(s, 5.6, 3.55, 4.2, 4.3, color=TEAL); lbl("selected model", 5.25, 3.8, 1.5, TEAL)     # 2 -> 3
    k.arrow(s, 4.4, 5.32, 4.8, 5.32, color=AMBER)                      # 3 -> 4
    k.arrow(s, 8.6, 5.32, 9.0, 5.32, color=AMBER)                      # 4 -> 5
    k.arrow(s, 10.9, 4.3, 10.9, 3.55, color=GREEN); lbl("event log: scores + cards", 8.6, 3.8, 2.2, GREEN)  # 5 -> 6
    for i, (t, c) in enumerate([("Offline", TEAL), ("Runtime", AMBER), ("Operator interface", GREEN)]):
        x = 0.6 + i * 2.4
        _, lg = k.box(s, x, 6.65, 0.28, 0.28, fill=c, line=None, radius=0.5, name="legend-dot")
        k.text(s, t, x + 0.38, 6.6, 1.9, 0.38, size=12, color=MUTED, anchor=MSO_ANCHOR.MIDDLE, parent=lg,
               name="legend")


def build(d):
    k = Deck(d)
    sel = d.sel["name"] if d.sel else None
    S = label(sel) if sel else "selected model"
    comp, meta, parse, kpi = d.comp, d.meta, d.parse, d.kpi or {}
    w = comp.loc[sel] if comp is not None and sel in comp.index else None
    W, K = (meta["W"], meta["K"]) if meta else (None, None)

    # 1 ── title ------------------------------------------------------------------------------
    s = k.slide(dark=True, notes="Product demo for Review 2. Owners: Ritvik Battula and Shaurya Kumar.")
    k.box(s, 0.6, 0.6, 1.5, 0.45, fill=TEAL, line=None, radius=0.5, name="chip")
    k.text(s, "Review 2", 0.6, 0.6, 1.5, 0.45, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
           anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="chip-text")
    k.text(s, "OS Fault-Prediction Daemon", 0.6, 2.1, 9.0, 1.1, size=54, bold=True, color=WHITE, name="product")
    k.text(s, "Predicts system faults before they happen, and explains them in plain English.",
           0.6, 3.5, 9.0, 0.9, size=22, color="B8C7D6", name="tagline")
    k.text(s, [{"text": "Ritvik Battula  (24BAI0156)", "size": 18, "bold": True, "color": WHITE},
               {"text": "Shaurya Kumar  (24BCE0876)", "size": 18, "bold": True, "color": WHITE},
               {"text": "VIT Vellore", "size": 16, "color": "B8C7D6", "space": 8}],
           0.6, 5.05, 6.0, 1.35, name="owners")
    # visual: a confidence pulse built from shapes (stylised, no data)
    _, vid = k.box(s, 10.0, 1.9, 2.8, 3.6, fill=PANEL_NAVY, line=None, radius=0.06, name="hero-visual")
    for i, hh in enumerate([0.5, 0.7, 0.6, 0.9, 1.3, 1.9, 2.6]):
        k.box(s, 10.25 + i * 0.36, 5.2 - hh, 0.22, hh, fill=RED if i == 6 else AMBER if i >= 4 else LTEAL,
              line=None, radius=0.2, parent=vid, kind="bar")
    k.text(s, "risk rising → alert", 10.15, 2.0, 2.5, 0.4, size=12, italic=True, color="B8C7D6",
           parent=vid, name="hero-cap")

    # 2 ── the problem --------------------------------------------------------------------------
    s = k.slide("The problem")
    _, wid = k.box(s, 0.6, 1.55, 6.6, 5.25, fill=NAVY, line=None, radius=0.04, name="logwall")
    widths = [5.6, 4.1, 5.0, 3.2, 5.8, 4.6, 2.9, 5.2, 4.4, 5.9, 3.6, 4.9, 5.4, 3.9]
    for i, ww in enumerate(widths):
        k.box(s, 0.85, 1.8 + i * 0.29, ww, 0.13, fill="2B4560" if i % 3 else "35587A", line=None, radius=0.5,
              parent=wid, kind="logline")
    k.box(s, 0.85, 1.8 + len(widths) * 0.29 + 0.1, 6.1, 0.42, fill=RED, line=None, radius=0.2, parent=wid,
          kind="faultbar")
    k.text(s, "SYSTEM FAILURE  -  noticed only now", 0.95, 1.8 + len(widths) * 0.29 + 0.1, 5.9, 0.42, size=14,
           bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="fault-text")
    if parse:
        d.src(2, "log lines, alert lines, templates", "data/v2/parse_meta.json")
        facts = [(f"{parse['lines']:,}", "log lines in one supercomputer system log (BGL)"),
                 (f"{parse['templates']:,}", "distinct message types in the training period alone"),
                 ("After", "the fault: that is when operators usually find out")]
    else:
        facts = None
    if facts:
        for i, (big, small) in enumerate(facts):
            cid = k.box(s, 7.6, 1.55 + i * 1.85, 5.2, 1.55, name=f"fact{i}")[1]
            k.text(s, big, 7.8, 1.65 + i * 1.85, 4.8, 0.75, size=36, bold=True, color=RED if i == 2 else TEAL,
                   parent=cid, name="fact-big")
            k.text(s, small, 7.8, 2.4 + i * 1.85, 4.8, 0.6, size=15, color=MUTED, parent=cid, name="fact-small")
    else:
        k.placeholder(s, 7.6, 1.55, 5.2, 5.25, "data/v2/parse_meta.json")

    # 3 ── meet the product ---------------------------------------------------------------------
    s = k.slide("Meet the product")
    steps = [("Watch", "reads the system log as a live stream", TEAL, MSO_SHAPE.OVAL),
             ("Predict", "spots the run-up to a fault early", AMBER, MSO_SHAPE.ISOSCELES_TRIANGLE),
             ("Explain", "writes a plain-English incident card", GREEN, MSO_SHAPE.ROUNDED_RECTANGULAR_CALLOUT)]
    for i, (t, b, c, shp) in enumerate(steps):
        y = 1.55 + i * 1.75
        cid = k.box(s, 0.6, y, 4.6, 1.45, name=f"step{i}")[1]
        k.box(s, 0.8, y + 0.37, 0.7, 0.7, fill=c, line=None, shape=shp, parent=cid, kind="icon")
        k.text(s, t, 1.75, y + 0.18, 3.3, 0.5, size=22, bold=True, parent=cid, name="step-t")
        k.text(s, b, 1.75, y + 0.72, 3.3, 0.6, size=15, color=MUTED, parent=cid, name="step-b")
        if i < 2:
            k.arrow(s, 2.9, y + 1.47, 2.9, y + 1.73, color=MUTED)
    k.terminal(s, 5.6, 1.55, 7.2, 5.25, "dashboard.png", "Live dashboard - simulated live stream from held-out BGL logs")

    # 4 ── live demonstration -------------------------------------------------------------------
    s = k.slide("Live demonstration", dark=True, notes=(
        "Before starting: make sure Ollama is running and pre-warm the model (the daemon sends a 1-token warm-up "
        "call at start; run the demo once beforehand so llama3.2:3b is loaded). Start from the dashboard sidebar: "
        f"model {S}, 'Model's showcase' slice, speed 200 lines/s."))
    demo_steps = [("Replay", "held-out log lines stream in, 20 lines per window"),
                  ("Confidence climbs", "the model scores every window; watch it cross the threshold"),
                  ("Incident fires", "the alert gate turns a run of flagged windows into one incident"),
                  ("Card appears", "the local LLM writes summary, component, severity, action")]
    for i, (t, b) in enumerate(demo_steps):
        x = 0.6 + i * 3.1
        cid = k.box(s, x, 1.7, 2.8, 3.6, fill=PANEL_NAVY, line="244563", name=f"demo{i}")[1]
        k.icon_badge(s, x + 0.25, 1.95, 0.7, i + 1, fill=[TEAL, AMBER, RED, GREEN][i], parent=cid)
        k.text(s, t, x + 0.25, 2.85, 2.3, 0.85, size=20, bold=True, color=WHITE, parent=cid, name="demo-t")
        k.text(s, b, x + 0.25, 3.75, 2.3, 1.4, size=15, color="B8C7D6", parent=cid, name="demo-b")
        if i < 3:
            k.arrow(s, x + 2.82, 3.5, x + 3.08, 3.5, color=LTEAL)
    k.box(s, 0.6, 5.75, 12.1, 0.75, fill=TEAL, line=None, radius=0.2, name="banner")
    k.text(s, "Simulated live stream from held-out BGL logs.", 0.6, 5.75, 12.1, 0.75, size=20, bold=True,
           color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="banner-t")

    # 4b ── one prediction, start to finish ---------------------------------------------------
    s = k.slide("One fault, start to finish")
    if d.demo and d.sel and d.demo_row:
        d.src(k.n, "confidence per window, incidents", f"results/demo_runs/{sel}_shared.jsonl")
        d.src(k.n, "warning lead (lines, s)", "results/model_report.md (section 5)")
        wins = [e for e in d.demo if e["type"] == "window"]
        start = next(e for e in d.demo if e["type"] == "start")
        fired = [e["window_range"][1] for e in d.demo if e["type"] == "incident_fired"]
        first_alert = next((e["window_idx"] for e in wins if e.get("truth", {}).get("alert_in_window")), None)
        n = len(wins)
        alerts = [wins[i]["confidence"] if i in fired else None for i in range(n)]
        first = [wins[i]["confidence"] if i == first_alert else None for i in range(n)]
        series = [("Model confidence", [e["confidence"] for e in wins]),
                  (f"Threshold τ = {start['tau']:.3f}", [start["tau"]] * n),
                  ("Alert fired", alerts)]
        if first_alert is not None:
            series.append(("First fault line", first))
        ch = k.chart(s, XL_CHART_TYPE.LINE, [str(e["window_idx"]) for e in wins], series,
                     0.6, 1.5, 8.4, 5.3, f"{S} on a replayed slice of held-out log ({n} windows)",
                     [TEAL, RED, RED, INK], fmt="0.00", labels=False, vmax=1.0, vmin=0.0, axis_fmt="0.0")
        sr = ch.plots[0].series
        sr[1].format.line.dash_style = MSO_LINE.DASH
        sr[1].format.line.width = Pt(1.75)
        for idx, (style, col) in [(2, (XL_MARKER_STYLE.TRIANGLE, RED)), (3, (XL_MARKER_STYLE.DIAMOND, INK))]:
            if idx < len(sr):
                sr[idx].format.line.fill.background()
                sr[idx].marker.style, sr[idx].marker.size = style, 13
                sr[idx].marker.format.fill.solid(); sr[idx].marker.format.fill.fore_color.rgb = rgb(col)
                sr[idx].marker.format.line.color.rgb = rgb(WHITE)
        skip_category_labels(ch, 10)
        ch.category_axis.has_title = True
        ch.category_axis.axis_title.text_frame.text = f"Window ({W} log lines each)"
        for p_ in ch.category_axis.axis_title.text_frame.paragraphs:
            for r_ in p_.runs:
                r_.font.size, r_.font.name, r_.font.bold = Pt(12), FONT, False
        r = d.demo_row
        cid = k.box(s, 9.4, 1.5, 3.4, 2.4, name="lead")[1]
        k.text(s, f"{r['Warning lead (lines)']} lines", 9.6, 1.65, 3.0, 0.75, size=36, bold=True, color=GREEN,
               parent=cid, name="lead-big")
        k.text(s, f"warning before the first fault line ({r['Warning lead (s)']} s of log time)", 9.6, 2.45,
               3.0, 1.3, size=15, color=MUTED, parent=cid, name="lead-small")
        cid = k.box(s, 9.4, 4.2, 3.4, 2.6, name="incidents")[1]
        k.text(s, f"{r['Incidents']} incidents", 9.6, 4.35, 3.0, 0.65, size=30, bold=True, color=INK,
               parent=cid, name="inc-big")
        k.text(s, f"{r['Before fault']} before the fault, {r['Late']} once it had started, "
                  f"{r['False alarm']} false alarms on this slice", 9.6, 5.05, 3.0, 1.55, size=15, color=MUTED,
               parent=cid, name="inc-small")
    else:
        k.placeholder(s, 0.6, 1.5, 12.2, 5.3, f"results/demo_runs/{sel}_shared.jsonl or results/model_report.md")

    # 5 ── architecture ------------------------------------------------------------------------
    s = k.slide("System architecture")
    layers = [("Dashboard", "Streamlit", ["live log feed", "confidence chart", "incident cards", "demo launcher"], TEAL),
              ("Daemon", "Python service", ["stream loop", "windowing", "alert gate", "incident builder"], LTEAL),
              ("Models", "PyTorch + Ollama", [f"{S} detect()", "Llama 3.2 3B incident writer"], AMBER),
              ("Data", "BGL + Drain3", ["BGL system log", "Drain3 templates", "20-line windows", "saved models"], GREEN)]
    for i, (nm, tech, comps, c) in enumerate(layers):
        y = 1.5 + i * 1.35
        cid = k.box(s, 0.6, y, 12.2, 1.05, name=f"layer{i}")[1]
        k.box(s, 0.6, y, 0.16, 1.05, fill=c, line=None, radius=0.0, shape=MSO_SHAPE.RECTANGLE, parent=cid, kind="tab")
        k.text(s, [{"text": nm, "size": 20, "bold": True}, {"text": tech, "size": 13, "color": MUTED, "space": 0}],
               0.95, y + 0.12, 2.6, 0.85, parent=cid, name="layer-name")
        cw = 2.0 if len(comps) == 4 else 3.4
        for j, comp_name in enumerate(comps):
            cx = 3.75 + j * (cw + 0.2)
            k.box(s, cx, y + 0.25, cw, 0.55, fill=WHITE, line=c, lw=1.25, radius=0.3, parent=cid, kind="chip")
            k.text(s, comp_name, cx, y + 0.25, cw, 0.55, size=13, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
                   parent=-1, name="layer-chip")

    # 5b ── system + module architecture diagrams ------------------------------------------------
    arch_slides(k, d)

    # 6 ── system design ----------------------------------------------------------------------
    s = k.slide("System design")
    n_models = len(d.models)
    off = ["Preprocess logs", f"Train {n_models} models", "Evaluate on validation", "Select the best", "Write to store"]
    run = ["Replay held-out log", "Cut 20-line windows", "detect()", "Alert gate", "LLM incident card", "Dashboard"]
    _, oid = k.box(s, 0.6, 1.5, 4.5, 5.3, fill=WHITE, line=TEAL, lw=1.5, dash=True, radius=0.04, name="offline")
    k.text(s, "OFFLINE", 0.8, 1.6, 4.1, 0.4, size=14, bold=True, color=TEAL, parent=oid, name="off-t")
    for i, t in enumerate(off[:4]):
        k.box(s, 0.9, 2.15 + i * 1.12, 3.9, 0.75, fill=LIGHT, line=BORDER, parent=oid, kind="step")
        k.text(s, t, 0.9, 2.15 + i * 1.12, 3.9, 0.75, size=15, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
               parent=-1, name="off-step")
        if i < 3:
            k.arrow(s, 2.85, 2.9 + i * 1.12, 2.85, 3.27 + i * 1.12)
    k.box(s, 5.75, 1.55, 1.7, 0.85, fill=LIGHT, line=BORDER, shape=MSO_SHAPE.FLOWCHART_DOCUMENT, name="source")
    k.text(s, "Held-out BGL log", 5.75, 1.55, 1.7, 0.7, size=13, bold=True, align=PP_ALIGN.CENTER,
           anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="source-t")
    k.box(s, 5.75, 3.0, 1.7, 2.0, fill=TEAL, line=None, shape=MSO_SHAPE.CAN, name="store")
    k.text(s, "Model store", 5.75, 3.55, 1.7, 1.1, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
           anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="store-t")
    _, rid = k.box(s, 8.1, 1.5, 4.7, 5.3, fill=WHITE, line=AMBER, lw=1.5, dash=True, radius=0.04, name="runtime")
    k.text(s, "RUNTIME", 8.3, 1.6, 4.3, 0.4, size=14, bold=True, color=AMBER, parent=rid, name="run-t")
    for i, t in enumerate(run):
        k.box(s, 8.4, 2.1 + i * 0.77, 4.1, 0.55, fill=LIGHT, line=BORDER, parent=rid, kind="step")
        k.text(s, t, 8.4, 2.1 + i * 0.77, 4.1, 0.55, size=14, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
               parent=-1, name="run-step")
        if i < 5:
            k.arrow(s, 10.45, 2.65 + i * 0.77, 10.45, 2.87 + i * 0.77)
    k.arrow(s, 4.8, 5.89, 5.9, 4.85, color=TEAL)                   # select -> store (writes)
    k.text(s, "writes", 5.45, 5.5, 0.75, 0.35, size=12, italic=True, color=TEAL, kind="label", name="lbl-writes")
    k.arrow(s, 7.45, 3.92, 8.4, 3.92, color=AMBER)                 # store -> detect (loads)
    k.text(s, "loads", 7.5, 3.5, 0.6, 0.35, size=12, italic=True, color=AMBER, kind="label", name="lbl-loads")
    k.arrow(s, 7.45, 1.95, 8.4, 2.37, color=AMBER)                 # source -> replay (streams)
    k.text(s, "streams", 7.45, 2.3, 0.65, 0.35, size=12, italic=True, color=AMBER, kind="label", name="lbl-streams")

    # 7 ── how it decides ---------------------------------------------------------------------
    s = k.slide("How it decides")
    tau_s = f"{d.sel['tau']:.3f}" if d.sel else "τ"

    def node(x, y, w, h, t, fill=LIGHT, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, color=INK, size=14):
        k.box(s, x, y, w, h, fill=fill, line=line, lw=1.25, shape=shape, name="node")
        k.text(s, t, x + 0.1, y, w - 0.2, h, size=size, bold=shape != MSO_SHAPE.ROUNDED_RECTANGLE,
               color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="node-t")

    node(0.6, 2.2, 1.6, 0.9, "Start", fill=NAVY, line=None, shape=MSO_SHAPE.OVAL, color=WHITE)
    node(2.6, 2.2, 2.2, 0.9, "Read next window")
    node(5.2, 2.2, 2.2, 0.9, "detect() → confidence")
    node(7.8, 2.0, 2.6, 1.3, f"conf ≥ τ ({tau_s})?", fill="FFF4E0", line=AMBER, shape=MSO_SHAPE.DIAMOND, size=13)
    node(7.8, 4.3, 2.6, 1.6, f"streak ≥ N ({d.sel['N'] if d.sel else 'N'}) and cooldown OK?",
         fill="FFF4E0", line=AMBER, shape=MSO_SHAPE.DIAMOND, size=12)
    node(5.2, 4.65, 2.2, 0.9, "Aggregate the episode")
    node(2.6, 4.65, 2.2, 0.9, "LLM incident card", fill="E6F4EC", line=GREEN)
    node(0.6, 4.55, 1.6, 1.1, "Dashboard + start cooldown", size=13)
    k.arrow(s, 2.2, 2.65, 2.6, 2.65); k.arrow(s, 4.8, 2.65, 5.2, 2.65); k.arrow(s, 7.4, 2.65, 7.8, 2.65)
    k.arrow(s, 9.1, 3.3, 9.1, 4.3, color=GREEN)
    k.text(s, "yes", 9.2, 3.55, 0.6, 0.35, size=12, bold=True, color=GREEN, kind="label", name="yes1")
    k.arrow(s, 7.8, 5.1, 7.4, 5.1, color=GREEN)
    k.text(s, "yes", 7.42, 4.7, 0.5, 0.35, size=12, bold=True, color=GREEN, kind="label", name="yes2")
    k.arrow(s, 5.2, 5.1, 4.8, 5.1); k.arrow(s, 2.6, 5.1, 2.2, 5.1)
    # "no" branches loop back to "Read next window"
    k.arrow(s, 10.4, 2.65, 11.5, 2.65, color=RED, head=False)
    k.arrow(s, 10.4, 5.1, 11.5, 5.1, color=RED, head=False)
    k.arrow(s, 11.5, 5.1, 11.5, 1.75, color=RED, head=False)
    k.arrow(s, 11.5, 1.75, 3.7, 1.75, color=RED, head=False)
    k.arrow(s, 3.7, 1.75, 3.7, 2.2, color=RED)
    k.text(s, "no", 10.55, 2.25, 0.5, 0.35, size=12, bold=True, color=RED, kind="label", name="no1")
    k.text(s, "no", 10.55, 4.7, 0.5, 0.35, size=12, bold=True, color=RED, kind="label", name="no2")
    k.box(s, 0.6, 6.25, 12.2, 0.6, fill=LIGHT, line=None, radius=0.3, name="caption-box")
    k.text(s, "One fault episode = one alert: a run of flagged windows becomes a single incident.", 0.6, 6.25,
           12.2, 0.6, size=16, bold=True, color=TEAL, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
           parent=-1, name="decide-cap")

    # 8 ── dataset pipeline -------------------------------------------------------------------
    s = k.slide("Dataset pipeline")
    if meta and parse:
        d.src(k.n, "lines, templates, alert lines", "data/v2/parse_meta.json")
        d.src(k.n, "windows, split sizes, positive rates, W, K", "data/v2/default/meta.json")
        sp = meta["splits"]
        total_w = sum(v["windows"] for v in sp.values())
        tiles = [(f"{parse['lines']:,}", "raw log lines", 4.6),
                 (f"{parse['templates']:,}", "Drain3 templates (from training lines)", 4.1),
                 (f"{total_w:,}", "labelled 20-line windows", 3.6),
                 (pct(sp["test"]["positive_rate"], 2), "pre-fault windows in held-out test", 3.1)]
        for i, (big, small, ww) in enumerate(tiles):
            x, y = 0.6 + (4.6 - ww) / 2, 1.5 + i * 1.35
            cid = k.box(s, x, y, ww, 1.05, fill=[NAVY, PANEL_NAVY, TEAL, AMBER][i], line=None, name=f"funnel{i}")[1]
            k.text(s, [{"text": big, "size": 26, "bold": True, "color": WHITE},
                       {"text": small, "size": 12, "color": WHITE, "space": 0}],
                   x + 0.1, y + 0.05, ww - 0.2, 0.95, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
                   parent=cid, name="funnel-t")
        # labelling diagram
        _, lid = k.box(s, 5.6, 1.5, 7.2, 2.55, fill=WHITE, line=BORDER, radius=0.04, name="labelling")
        k.text(s, "How a window is labelled", 5.8, 1.6, 6.8, 0.4, size=16, bold=True, parent=lid, name="lab-t")
        for x, ww, t, c in [(5.85, 1.9, f"{W}-line window", TEAL), (8.0, 2.8, f"look-ahead: next {K} lines", LIGHT),
                            (11.05, 1.5, "fault", RED)]:
            k.box(s, x, 2.15, ww, 0.75, fill=c, line=BORDER if c == LIGHT else None, radius=0.1, parent=lid,
                  kind="lab-box")
            k.text(s, t, x, 2.15, ww, 0.75, size=14, bold=True, color=INK if c == LIGHT else WHITE,
                   align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="lab-box-t")
        k.text(s, [f"Pre-fault if a fault line follows within {K} lines.",
                   "Windows that already contain a fault line are excluded."],
               5.8, 3.0, 6.8, 0.95, size=14, color=MUTED, parent=lid, name="lab-rule")
        rows = [["Split", "Windows", "Pre-fault", "Pre-fault rate"]]
        for nm, key in [("Train", "train"), ("Validation", "val"), ("Test (held out)", "test")]:
            v = sp[key]
            rows.append([nm, f"{v['windows']:,}", f"{v['positives']:,}", pct(v["positive_rate"], 2)])
        k.table(s, rows, 5.6, 4.35, 7.2, [2.2, 1.6, 1.6, 1.8], row_h=0.45, size=14, name="splits")
        k.text(s, "Chronological split 70 / 10 / 20. Train negatives sampled 10:1; validation and test keep the real rate.",
               5.6, 6.45, 7.2, 0.45, size=11, italic=True, color=MUTED, name="split-cap")
    else:
        k.placeholder(s, 0.6, 1.5, 12.2, 5.3, "data/v2/default/meta.json or data/v2/parse_meta.json")

    # 9 ── module identification --------------------------------------------------------------
    s = k.slide("Module identification")
    core = ["Data preprocessing", "Model training & selection", "Daemon / stream loop",
            "Alert gate (τ / N / cooldown)", "Incident reporting (LLM)", "Dashboard"]
    for i, t in enumerate(core):
        x, y = 0.6 + (i % 3) * 4.15, 1.5 + (i // 3) * 1.4
        cid = k.box(s, x, y, 3.85, 1.1, name=f"core{i}")[1]
        k.icon_badge(s, x + 0.2, y + 0.25, 0.6, i + 1, fill=TEAL, parent=cid)
        k.text(s, t, x + 1.0, y + 0.1, 2.7, 0.6, size=16, bold=True, anchor=MSO_ANCHOR.MIDDLE, parent=cid,
               name="core-t")
        k.box(s, x + 1.0, y + 0.7, 1.35, 0.3, fill=GREEN, line=None, radius=0.5, parent=cid, kind="status")
        k.text(s, "Operational", x + 1.0, y + 0.7, 1.35, 0.3, size=11, bold=True, color=WHITE,
               align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="status-t")
    k.text(s, "Planned upgrades", 0.6, 4.4, 6.0, 0.4, size=16, bold=True, color=MUTED, name="band")
    planned = ["Detector explainability", "Drift monitor", "Incident correlation", "Live journalctl monitoring"]
    for i, t in enumerate(planned):
        x = 0.6 + i * 3.1
        cid = k.box(s, x, 5.1, 2.8, 0.95, fill=WHITE, line=LTEAL, lw=1.25, name=f"plan{i}")[1]
        k.text(s, t, x + 0.15, 5.1, 2.5, 0.95, size=15, bold=True, color=LTEAL, align=PP_ALIGN.CENTER,
               anchor=MSO_ANCHOR.MIDDLE, parent=cid, name="plan-t")
    for i, (t, c) in enumerate([("Operational", GREEN), ("Pending", AMBER), ("Planned", LTEAL)]):
        x = 0.6 + i * 2.3
        _, lgid = k.box(s, x, 6.45, 0.3, 0.3, fill=c, line=None, radius=0.5, kind="legend-dot", name="legend-dot")
        k.text(s, t, x + 0.4, 6.37, 1.5, 0.45, size=13, color=MUTED, anchor=MSO_ANCHOR.MIDDLE, parent=lgid, name="legend")

    # 10 ── module completion -----------------------------------------------------------------
    s = k.slide("Module completion")
    panels = [("Operational", GREEN, core + [f"{n_models}-model comparison with the best deployed"]),
              ("Pending", AMBER, ["None: every core module runs end to end on the held-out stream"]),
              ("Planned", LTEAL, planned)]
    for i, (t, c, items) in enumerate(panels):
        x = 0.6 + i * 4.15
        cid = k.box(s, x, 1.5, 3.85, 5.3, fill=LIGHT, line=BORDER, name=f"panel{i}")[1]
        k.box(s, x, 1.5, 3.85, 0.7, fill=c, line=None, radius=0.12, parent=cid, kind="panel-head")
        k.text(s, f"{t}  ({len(items) if t != 'Pending' else 0})", x + 0.2, 1.5, 3.45, 0.7, size=20, bold=True,
               color=WHITE, anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="panel-t")
        k.text(s, [{"text": f"•  {it}", "size": 15, "space": 8} for it in items], x + 0.2, 2.4, 3.45, 4.25,
               color=INK, parent=cid, name="panel-items")

    # 11 ── four models tested -----------------------------------------------------------------
    s = k.slide(f"Choosing the brain: {n_models} models tested")
    if comp is not None:
        d.src(k.n, "precision, recall, F1, PR-AUC, val PR-AUC, ms/window", "results/model_comparison.csv")
        rows = [["Model", "Window precision", "Window recall", "Window F1", "Test PR-AUC", "Val PR-AUC",
                 "ms / window"]]
        for m, r in comp.iterrows():
            rows.append([f"{label(m)}   ✓ Selected" if m == sel else label(m), pct(r.test_precision),
                         pct(r.test_recall), f"{r.test_f1:.3f}", f"{r.test_pr_auc:.3f}", f"{r.val_pr_auc:.3f}",
                         f"{r.infer_ms_per_window:.1f}"])
        k.table(s, rows, 0.6, 1.6, 12.2, [3.2, 1.7, 1.6, 1.5, 1.5, 1.5, 1.4], row_h=0.62, size=15,
                highlight={1 + list(comp.index).index(sel)} if sel in comp.index else None, name="models")
        k.box(s, 0.6, 5.15, 12.2, 1.1, fill=LIGHT, line=None, radius=0.06, name="models-note")
        k.text(s, [{"text": "How to read this", "size": 15, "bold": True},
                   {"text": "Window-level metrics on unseen test data, each model at its own threshold "
                            f"(lowest threshold reaching {pct(d.cfg['eval']['precision_target'], 0)} precision on "
                            "validation). Selection metric: validation PR-AUC.", "size": 14, "color": MUTED}],
               0.8, 5.22, 11.8, 0.95, parent=-2, name="models-note-t")
    else:
        k.placeholder(s, 0.6, 1.5, 12.2, 5.3, "results/model_comparison.csv")

    # 12 ── confusion matrices -----------------------------------------------------------------
    s = k.slide("How each model performs")
    if comp is not None:
        d.src(k.n, "TP / FP / FN / TN", "results/model_comparison.csv")
        for i, (m, r) in enumerate(comp.iterrows()):
            x, y = 0.6 + (i % 2) * 6.3, 1.5 + (i // 2) * 2.55
            is_sel = m == sel
            cid = k.box(s, x, y, 5.9, 2.25, fill=WHITE, line=GREEN if is_sel else BORDER, lw=3 if is_sel else 1,
                        name=f"cm{i}")[1]
            k.text(s, f"{label(m)}{'   ✓ Selected' if is_sel else ''}", x + 0.2, y + 0.1, 5.5, 0.45, size=17,
                   bold=True, color=GREEN if is_sel else INK, parent=cid, name="cm-t")
            rows = [["", "Predicted normal", "Predicted pre-fault"],
                    ["Actually normal", f"{int(r.test_tn):,}", f"{int(r.test_fp):,} false alarms"],
                    ["Actually pre-fault", f"{int(r.test_fn):,} missed", f"{int(r.test_tp):,} caught"]]
            k.table(s, rows, x + 0.2, y + 0.65, 5.5, [1.9, 1.7, 1.9], row_h=0.48, size=13, header_fill=PANEL_NAVY,
                    parent=cid, name=f"cm{m}")
        k.text(s, "False positives = false alarms; each model's threshold was set to keep them low. "
                  "Window-level counts on held-out test data.", 0.6, 6.62, 12.2, 0.38, size=12, italic=True,
               color=MUTED, name="cm-cap")
    else:
        k.placeholder(s, 0.6, 1.5, 12.2, 5.3, "results/model_comparison.csv")

    # 12b ── incident-level comparison ---------------------------------------------------------
    s = k.slide("Alerts that matter: incident-level")
    if comp is not None:
        d.src(k.n, "incidents, before fault, late, false, incident precision, faults warned",
              "results/model_comparison.csv")
        rows = [["Model", "Gate (N, cooldown)", "Alerts fired", "Before the fault", "During a fault",
                 "False alarms", "Incident precision", "Faults warned"]]
        for m, r in comp.iterrows():
            rows.append([f"{label(m)}   ✓ Selected" if m == sel else label(m), f"{int(r.N)}, {int(r.cooldown)}",
                         f"{int(r.test_incidents)}", f"{int(r.test_predictive)}", f"{int(r.test_in_episode)}",
                         f"{int(r.test_false)}", pct(r.test_incident_precision),
                         f"{int(r.test_episodes_caught)} / {int(r.test_episodes)}"])
        k.table(s, rows, 0.6, 1.6, 12.2, [2.9, 1.5, 1.2, 1.4, 1.3, 1.2, 1.5, 1.4], row_h=0.62, size=14,
                highlight={1 + list(comp.index).index(sel)} if sel in comp.index else None, name="incident-table")
        k.box(s, 0.6, 5.15, 12.2, 1.1, fill=LIGHT, line=None, radius=0.06, name="inc-note")
        k.text(s, [{"text": "Incident-level, held-out test data", "size": 15, "bold": True},
                   {"text": "One incident = one alert to an operator. Incident precision counts only alerts that "
                            f"came before a fault started (within {K} lines); alerts during a fault are not credited.",
                    "size": 14, "color": MUTED}], 0.8, 5.22, 11.8, 0.95, parent=-2, name="inc-note-t")
    else:
        k.placeholder(s, 0.6, 1.5, 12.2, 5.3, "results/model_comparison.csv")

    # 13 ── why we chose the selected model ----------------------------------------------------
    s = k.slide(f"Why we chose {S}")
    if (RES / "pr_curves.png").exists():
        d.src(k.n, "PR curves (validation)", "results/pr_curves.png")
        k.picture(s, RES / "pr_curves.png", 0.6, 1.5, 6.6, 4.5)
    else:
        k.placeholder(s, 0.6, 1.5, 6.6, 4.6, "results/pr_curves.png")
    if comp is not None and w is not None:
        d.src(k.n, "PR-AUC, recall, ms/window", "results/model_comparison.csv")
        d.src(k.n, "peak VRAM", f"results/train_{sel}.json")
        reasons = []
        if comp.test_precision.idxmax() == sel:
            reasons.append(("Most precise", f"{pct(w.test_precision)} window-level precision on unseen test data"))
        if comp.val_pr_auc.idxmax() == sel:
            reasons.append(("Best at ranking risk", f"highest PR-AUC of the {n_models}: validation "
                                                    f"{w.val_pr_auc:.3f}, test {w.test_pr_auc:.3f}"))
        if comp.test_recall.idxmax() == sel:
            reasons.append(("Catches the most pre-fault windows",
                            f"{pct(w.test_recall)} recall at {pct(w.test_precision)} window-level precision"))
        tr = d.train.get(sel) or {}
        reasons.append(("Fast enough for live use",
                        f"{w.infer_ms_per_window:.1f} ms to score a {W}-line window"))
        if tr.get("peak_vram_mb"):
            reasons.append(("Runs locally", f"{tr['peak_vram_mb'] / 1024:.1f} GB peak VRAM while training on a "
                                            "6 GB laptop GPU; no cloud needed"))
        # three cards: the strongest true claim about accuracy, then speed, then running locally
        reasons = [r for r in reasons if r[0] not in ("Fast enough for live use", "Runs locally")][:1] +                   [r for r in reasons if r[0] in ("Fast enough for live use", "Runs locally")]
        hh = (4.5 - 0.3 * (len(reasons) - 1)) / len(reasons)
        for i, (t, b) in enumerate(reasons):
            y = 1.5 + i * (hh + 0.3)
            cid = k.box(s, 7.6, y, 5.2, hh, fill=LIGHT, line=BORDER, name=f"reason{i}")[1]
            k.box(s, 7.6, y, 0.12, hh, fill=GREEN, line=None, shape=MSO_SHAPE.RECTANGLE, parent=cid, kind="tab")
            k.text(s, [{"text": t, "size": 17, "bold": True},
                       {"text": b, "size": 13, "color": MUTED, "space": 2}],
                   7.85, y + 0.08, 4.8, hh - 0.16, anchor=MSO_ANCHOR.MIDDLE, parent=cid, name="reason-t")
        conf = comp.test_pr_auc.idxmax() == sel
        k.box(s, 0.6, 6.35, 12.2, 0.55, fill=GREEN, line=None, radius=0.3, name="pick")
        k.text(s, "Picked on the validation set" + (", confirmed on unseen test data." if conf else "."),
               0.6, 6.35, 12.2, 0.55, size=17, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
               anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="pick-t")
    else:
        k.placeholder(s, 7.6, 1.5, 5.2, 4.6, "results/model_comparison.csv")

    # 14 ── built for precision -----------------------------------------------------------------
    s = k.slide("Built for precision")
    if w is not None:
        d.src(k.n, "window / incident precision & recall, lenient incident precision", "results/model_comparison.csv")
        k.chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, ["Window-level", "Incident-level"],
                [("Precision", [float(w.test_precision), float(w.test_incident_precision)]),
                 ("Recall (incident-level: faults warned)", [float(w.test_recall), float(w.test_episode_recall)])],
                0.6, 1.5, 6.9, 4.5, f"{S}: precision and recall on held-out test data", [GREEN, LTEAL],
                vmax=1.0, vmin=0, axis_fmt="0%", label_size=13)
        cid = k.box(s, 7.9, 1.5, 4.9, 2.6, fill=WHITE, line=GREEN, lw=2, name="callout")[1]
        k.text(s, pct(w.test_incident_precision, 0), 8.1, 1.6, 4.5, 1.0, size=54, bold=True, color=GREEN,
               parent=cid, name="call-big")
        k.text(s, "of alerts arrive before a real fault starts (incident-level)", 8.1, 2.65, 4.5, 1.3, size=17,
               color=INK, parent=cid, name="call-small")
        cid = k.box(s, 7.9, 4.4, 4.9, 1.6, name="callout2")[1]
        k.text(s, [{"text": pct(w.test_incident_precision_lenient, 0), "size": 32, "bold": True, "color": TEAL},
                   {"text": "of alerts coincide with a real fault, before or during it (incident-level)",
                    "size": 14, "color": MUTED, "space": 0}], 8.1, 4.45, 4.5, 1.5, parent=cid, name="call2")
    else:
        k.placeholder(s, 0.6, 1.5, 12.2, 4.5, "results/model_comparison.csv")
    if meta and d.sel:
        d.src(k.n, "truncation %, buffer", "data/v2/default/meta.json")
        trunc = meta["tokens"].get(sel, next(iter(meta["tokens"].values())))["pct_truncated"]
        chips = [f"Whole windows kept: {trunc:.1f}% truncated at {meta['max_length']} tokens",
                 f"Ambiguity buffer: {meta['buffer']} lines",
                 f"Tuned gate: τ {d.sel['tau']:.3f}, N {d.sel['N']}, cooldown {d.sel['cooldown']}"]
        for i, t in enumerate(chips):
            x = 0.6 + i * 4.15
            k.box(s, x, 6.3, 3.85, 0.6, fill=LIGHT, line=BORDER, radius=0.3, name=f"chip{i}")
            k.text(s, t, x + 0.1, 6.3, 3.65, 0.6, size=13, bold=True, color=TEAL, align=PP_ALIGN.CENTER,
                   anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="chip-t")

    # 15 ── what drives precision (ablation) ---------------------------------------------------
    s = k.slide("What drives precision")
    if d.abl is not None:
        d.src(k.n, "ablation rows", "results/ablation.csv (same table as results/ablation.md)")
        a = d.abl
        short = {"all levers on (selected": f"{S}, all design choices (full training)",
                 "all levers on - control": "All choices on, 1-epoch retrain (reference)",
                 "no RLE": "Without whole-window text rendering",
                 "no ambiguity": "Without the ambiguity buffer",
                 "no overlapping": "Without overlapping training windows",
                 "no negative": "Without negative downsampling"}
        def nm(v):
            return next((t for key, t in short.items() if v.startswith(key)), v)
        rows = [["Configuration", "Val PR-AUC", "Window precision", "Window recall", "Train time"]]
        for _, r in a.iterrows():
            rows.append([nm(r.variant), f"{r.val_pr_auc:.3f}", pct(r.test_precision), pct(r.test_recall),
                         f"{r.train_minutes:.0f} min"])
        k.table(s, rows, 0.6, 1.5, 8.2, [4.0, 1.3, 1.6, 1.5, 1.3], row_h=0.6, size=13, highlight={1},
                name="ablation")
        ctrl = a.iloc[1]
        deltas = [(nm(r.variant), r.val_pr_auc - ctrl.val_pr_auc) for _, r in a.iloc[2:].iterrows()]
        worst = min(deltas, key=lambda t: t[1])
        cid = k.box(s, 9.2, 1.5, 3.6, 2.6, fill=WHITE, line=GREEN, lw=2, name="abl-call")[1]
        k.text(s, [{"text": "Biggest lever", "size": 16, "bold": True, "color": GREEN},
                   {"text": worst[0].replace("Without ", "").capitalize(), "size": 18, "bold": True},
                   {"text": f"switching it off drops validation PR-AUC by {abs(worst[1]):.3f}", "size": 14,
                    "color": MUTED}], 9.4, 1.6, 3.2, 2.4, parent=cid, name="abl-call-t")
        cid = k.box(s, 9.2, 4.4, 3.6, 2.4, name="abl-how")[1]
        k.text(s, [{"text": "How it was tested", "size": 16, "bold": True},
                   {"text": "Each choice switched off in turn and the model retrained for one epoch; compare "
                            "with the 1-epoch reference row.", "size": 13, "color": MUTED}],
               9.4, 4.5, 3.2, 2.2, parent=cid, name="abl-how-t")
        k.text(s, "Window-level precision and recall on held-out test data.", 0.6, 6.05, 8.2, 0.4, size=12,
               italic=True, color=MUTED, name="abl-cap")
    else:
        k.placeholder(s, 0.6, 1.5, 12.2, 5.3, "results/ablation.csv")

    # 16 ── tuning the alert gate -------------------------------------------------------------
    s = k.slide("Tuning the alert gate")
    if d.gate is not None and d.sel:
        d.src(k.n, "N / cooldown sweep, chosen setting", "results/gate_sweep.csv")
        d.src(k.n, "τ, N, cooldown", "models/selected.json")
        g = d.gate[d.gate.model == sel]
        chosen = g[g.chosen].iloc[0]
        fallback = "fallback" in str(chosen.chosen_by)
        why_gate = ("best balance of incident precision and faults caught on validation" if fallback else
                    "highest incident precision while still catching enough faults on validation")
        cards = [("Threshold τ", f"{d.sel['tau']:.3f}",
                  f"lowest score reaching {pct(d.cfg['eval']['precision_target'], 0)} window precision on validation"),
                 ("Streak N", f"{d.sel['N']} window{'s' if d.sel['N'] != 1 else ''}", why_gate),
                 ("Cooldown", f"{d.sel['cooldown']} windows",
                  "flagged runs already merge into one incident" if d.sel["cooldown"] == 0 else why_gate)]
        for i, (t, v, b) in enumerate(cards):
            x = 0.6 + i * 4.15
            cid = k.box(s, x, 1.5, 3.85, 1.7, name=f"gate{i}")[1]
            k.text(s, [{"text": t, "size": 15, "bold": True, "color": MUTED},
                       {"text": v, "size": 28, "bold": True, "color": AMBER, "space": 0},
                       {"text": b, "size": 12, "color": MUTED, "space": 2}], x + 0.2, 1.55, 3.45, 1.6,
                   parent=cid, name="gate-t")
        cd = int(chosen.cooldown_windows)
        gg = g[g.cooldown_windows == cd].sort_values("N")
        ch = k.chart(s, XL_CHART_TYPE.LINE_MARKERS, [f"N = {int(n)}" for n in gg.N],
                [("Incident precision", list(gg.incident_precision)), ("Episode recall (faults caught)",
                                                                        list(gg.episode_recall))],
                0.6, 3.5, 12.2, 2.7, f"Validation sweep for {S} at cooldown {cd} (incident-level)",
                [GREEN, AMBER], fmt="0.0%", axis_fmt="0%", vmin=0)
        try:
            dl2 = ch.plots[0].series[1].data_labels
            dl2.show_value, dl2.position = True, XL_LABEL_POSITION.RIGHT
            dl2.number_format, dl2.number_format_is_linked = "0.0%", False
            dl2.font.size, dl2.font.name = Pt(12), FONT
        except AttributeError:
            pass
        k.text(s, "Tuned on validation, never test.", 0.6, 6.5, 12.2, 0.4, size=14, bold=True, color=TEAL,
               align=PP_ALIGN.CENTER, name="gate-note")
    else:
        k.placeholder(s, 0.6, 1.5, 12.2, 5.3, "results/gate_sweep.csv or models/selected.json")

    # 17 ── what the operator gets ----------------------------------------------------------------
    s = k.slide("What the operator gets")
    if kpi and w is not None:
        d.src(k.n, "incident precision, episode recall, median lead, LLM latency", "results/RUN_REPORT.md")
        d.src(k.n, "ms/window", "results/model_comparison.csv")
        tiles = [(pct(kpi["incident_precision"], 0), "incident precision: alerts that came before a fault"),
                 (pct(kpi["episode_recall"], 0), f"of faults warned in advance ({kpi['caught']} of {kpi['episodes']})"),
                 (f"{kpi['lead_median']:.0f} s", "median warning before the fault's first line"),
                 (f"{kpi['llm_median']:.1f} s", f"to write the incident card ({kpi['llm']}, local)"),
                 (f"{w.infer_ms_per_window:.0f} ms", f"to score a {W}-line window")]
        for i, (big, small) in enumerate(tiles):
            x = 0.6 + i * 2.5
            cid = k.box(s, x, 1.5, 2.2, 1.95, fill=LIGHT, line=BORDER, name=f"kpi{i}")[1]
            k.text(s, big, x + 0.1, 1.6, 2.0, 0.75, size=34, bold=True, color=GREEN if i == 0 else TEAL,
                   align=PP_ALIGN.CENTER, parent=cid, name="kpi-big")
            k.text(s, small, x + 0.12, 2.4, 1.96, 1.0, size=12, color=MUTED, align=PP_ALIGN.CENTER, parent=cid,
                   name="kpi-small")
    else:
        k.placeholder(s, 0.6, 1.5, 12.2, 1.95, "results/RUN_REPORT.md")
    if d.eval_sel and d.eval_sel.get("test_lead_seconds"):
        d.src(k.n, "lead-time values", f"results/eval_{sel}.json (test_lead_seconds)")
        vals = d.eval_sel["test_lead_seconds"]
        edges = list(range(0, 21, 4))
        cats = [f"{a}-{b} s" for a, b in zip(edges, edges[1:])] + ["over 20 s"]
        counts = [sum(a <= v < b for v in vals) for a, b in zip(edges, edges[1:])] + [sum(v >= 20 for v in vals)]
        k.chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cats, [("Faults warned", counts)], 0.6, 3.75, 12.2, 3.1,
                f"How early the warning came: {len(vals)} faults warned in advance (held-out test)", [TEAL],
                fmt="0", axis_fmt="0", legend=False)
    else:
        k.placeholder(s, 0.6, 3.75, 12.2, 3.1, f"results/eval_{sel}.json (test_lead_seconds)")

    # 18 ── screenshots -------------------------------------------------------------------------
    base = ["preprocess.png", "model_comparison.png", "incident_card.png", "dashboard.png"]
    caps = {"preprocess.png": "Preprocessing: templates, windows, splits", "model_comparison.png": "Model comparison",
            "incident_card.png": "Incident card written by the local LLM", "dashboard.png": "Live dashboard"}
    cap_file = SHOTS / "captions.json"                       # optional {filename: caption}
    if cap_file.exists():
        caps.update(json.loads(cap_file.read_text(encoding="utf-8")))
    extras = sorted(p.name for p in SHOTS.glob("*.png") if p.name not in base) if SHOTS.exists() else []
    cap = lambda img: caps.get(img, img.rsplit(".", 1)[0].replace("_", " "))
    s = k.slide("Screenshots of all outputs")
    for i, img in enumerate(base):
        x, y = 0.6 + (i % 2) * 6.25, 1.45 + (i // 2) * 2.75
        k.terminal(s, x, y, 5.95, 2.45, img, cap(img))
    # extras: two tall frames per slide (tables and plots stay readable), or one large frame
    for i in range(0, len(extras), 2):
        page = extras[i:i + 2]
        s = k.slide("Screenshots (cont.)")
        if len(page) == 1:
            k.terminal(s, 0.6, 1.45, 12.2, 5.4, page[0], cap(page[0]))
        else:
            for j, img in enumerate(page):
                k.terminal(s, 0.6 + j * 6.25, 1.45, 5.95, 5.4, img, cap(img))

    # 19 ── what's next -------------------------------------------------------------------------
    s = k.slide("What's next")
    nxt = [("Detector explainability", "show which log lines pushed the confidence up"),
           ("Drift monitor", "flag when new, unseen message types start to dominate the stream"),
           ("Incident correlation", "link related incidents across nodes into one story"),
           ("Live journalctl", "run the daemon on a real Linux machine's system journal")]
    for i, (t, b) in enumerate(nxt):
        y = 1.5 + i * 1.35
        cid = k.box(s, 0.6, y, 12.2, 1.05, name=f"next{i}")[1]
        k.icon_badge(s, 0.85, y + 0.2, 0.65, i + 1, fill=LTEAL, parent=cid)
        k.text(s, t, 1.8, y + 0.1, 4.0, 0.85, size=20, bold=True, anchor=MSO_ANCHOR.MIDDLE, parent=cid,
               name="next-t")
        k.text(s, b, 5.9, y + 0.1, 6.7, 0.85, size=16, color=MUTED, anchor=MSO_ANCHOR.MIDDLE, parent=cid,
               name="next-b")

    # 20 ── close -------------------------------------------------------------------------------
    s = k.slide("Delivered", dark=True)
    reqs = [("Public dataset", "BGL system log from Loghub"),
            ("Fine-tuned models", f"{sum(1 for m in d.models if (d.train.get(m) or {}).get('type') == 'transformer')} "
                                  "transformers fine-tuned on a 6 GB GPU"),
            ("OS functionality", "a daemon consuming a log stream"),
            (f"{n_models} models compared", f"best one ({S}) deployed")]
    for i, (t, b) in enumerate(reqs):
        x = 0.6 + i * 3.1
        cid = k.box(s, x, 1.65, 2.8, 2.1, fill=PANEL_NAVY, line="244563", name=f"req{i}")[1]
        k.box(s, x + 0.2, 1.85, 0.55, 0.55, fill=GREEN, line=None, shape=MSO_SHAPE.OVAL, parent=cid, kind="tick")
        k.text(s, "✓", x + 0.2, 1.85, 0.55, 0.55, size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
               anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="tick-t")
        k.text(s, [{"text": t, "size": 18, "bold": True, "color": WHITE},
                   {"text": b, "size": 14, "color": "B8C7D6", "space": 2}], x + 0.2, 2.55, 2.4, 1.1,
               parent=cid, name="req-t")
    if w is not None:
        k.text(s, f"{S} warns ahead of {int(w.test_episodes_caught)} faults on unseen data; "
                  f"{pct(w.test_incident_precision, 0)} of its alerts arrive before the fault starts (incident-level).",
               0.6, 4.15, 12.2, 0.9, size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER, name="headline")
    k.text(s, "Thank you", 0.6, 5.35, 12.2, 1.1, size=48, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
           name="thanks")

    # B1 ── training setup ---------------------------------------------------------------------
    s = k.slide("Backup: training setup")
    if all(d.train.get(m) for m in d.models) and d.models:
        d.src(k.n, "hyperparameters, train time, peak VRAM", "results/train_<model>.json")
        rows = [["Model", "Setup", "Best epoch", "Train time", "Peak VRAM"]]
        for m in d.models:
            t = d.train[m]
            if t.get("type") == "transformer":
                hist = t.get("history") or []
                be = max(hist, key=lambda h: h["val_pr_auc"])["epoch"] if hist else "-"
                setup = f"batch {t['batch']}×{t['accum']}, lr {t['lr']:.0e}, max length {t['max_length']}, fp16"
                rows.append([label(m), setup, f"{be} of {len(hist)}", f"{t['train_seconds'] / 60:.0f} min",
                             f"{t['peak_vram_mb'] / 1024:.1f} GB"])
            else:
                rows.append([label(m), f"{t['max_iter']} trees, lr {t['learning_rate']}, {t['n_features']} features",
                             "-", f"{t['train_seconds'] / 60:.1f} min", "CPU only"])
        k.table(s, rows, 0.6, 1.6, 12.2, [2.6, 5.0, 1.4, 1.5, 1.5], row_h=0.62, size=14,
                highlight={1 + d.models.index(sel)} if sel in d.models else None, name="train")
        k.box(s, 0.6, 5.15, 12.2, 1.1, fill=LIGHT, line=None, radius=0.06, name="train-note")
        k.text(s, [{"text": "Same data for every model", "size": 15, "bold": True},
                   {"text": f"{d.train[d.models[0]]['train_windows']:,} training windows; early stopping on "
                            "validation PR-AUC; class weight 2 on pre-fault windows.", "size": 14, "color": MUTED}],
               0.8, 5.22, 11.8, 0.95, parent=-2, name="train-note-t")
    else:
        k.placeholder(s, 0.6, 1.5, 12.2, 5.3, "results/train_<model>.json")

    # B2 ── window vs incident precision --------------------------------------------------------
    s = k.slide("Backup: window vs incident precision")
    pattern = [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0]
    for i, f in enumerate(pattern):
        x = 0.6 + i * 1.0
        k.box(s, x, 1.7, 0.7, 0.85, fill=AMBER if f else LIGHT, line=BORDER, radius=0.12, name=f"win{i}")
        k.text(s, f"W{i + 1}", x, 1.7, 0.7, 0.85, size=13, bold=True, color=WHITE if f else MUTED,
               align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="win-t")
    k.text(s, "flagged windows (confidence ≥ τ) in amber", 0.6, 2.85, 6.0, 0.4, size=12, italic=True, color=MUTED,
           name="win-cap")
    k.arrow(s, 4.45, 3.4, 4.45, 4.15, color=AMBER, w=2.5)
    k.box(s, 2.85, 4.25, 3.2, 1.55, fill="E6F4EC", line=GREEN, lw=2, name="one-card")
    k.text(s, [{"text": "1 incident card", "size": 20, "bold": True, "color": GREEN},
               {"text": "windows 3-6 are one episode", "size": 14, "color": MUTED, "space": 2}],
           3.0, 4.35, 2.9, 1.35, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, parent=-1, name="one-card-t")
    k.arrow(s, 8.95, 3.4, 8.95, 4.15, color=AMBER, w=2.5)
    k.box(s, 7.35, 4.25, 3.2, 1.55, fill="FFF4E0", line=AMBER, lw=2, name="second-card")
    k.text(s, [{"text": "a new incident", "size": 20, "bold": True, "color": AMBER},
               {"text": "only after the run breaks and the cooldown has passed", "size": 14, "color": MUTED,
                "space": 2}], 7.5, 4.35, 2.9, 1.35, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, parent=-1,
           name="second-card-t")
    k.text(s, "Window-level precision scores every flagged window; incident-level precision scores each alert an "
              "operator actually receives. Many flagged windows collapse into one alert.", 0.6, 6.15, 12.2, 0.75,
           size=15, color=INK, align=PP_ALIGN.CENTER, name="b2-cap")
    return k


# ======================================================================= QA

def lines_needed(p, width_pt):
    f = _FONTS[(bool(p["bold"]), bool(p.get("italic")))]
    n, cur = 0, ""
    for para in p["text"].split("\n"):
        cur, cnt = "", 1
        for word in para.split(" "):
            cand = (cur + " " + word).strip()
            if f.getlength(cand) * p["size"] / 100 <= width_pt or not cur:
                if not cur and f.getlength(word) * p["size"] / 100 > width_pt:
                    return 99                                # a single word wider than the box
                cur = cand
            else:
                cnt, cur = cnt + 1, word
        n += cnt
    return n


def qa(k, d):
    issues = []
    # text fit
    for sn, name, paras, w, h, spacing in k.texts:
        need = 0.0
        for i, p in enumerate(paras):
            n = lines_needed(p, (w - 0.1) * 72)
            if n == 99:
                issues.append(f"slide {sn}: word wider than box in '{name}': {p['text'][:40]}")
                continue
            need += n * p["size"] * 1.2 * spacing + (p.get("space", 4) if i else 0)
        if need > (h - 0.06) * 72 + 0.5:
            issues.append(f"slide {sn}: text overflows '{name}' ({need:.0f}pt > {(h - 0.06) * 72:.0f}pt): "
                          f"{paras[0]['text'][:50]}")
    # bounds + spacing of top-level elements
    by_slide = defaultdict(list)
    for r in k.reg:
        by_slide[r[0]].append(r)
    for sn, items in by_slide.items():
        for (_, i, x, y, w, h, parent, kind, name) in items:
            if kind == "conn":
                continue
            if x < M - 1e-6 or y < M - 1e-6 or x + w > SW - M + 1e-6 or y + h > SH - M + 1e-6:
                issues.append(f"slide {sn}: '{name or kind}' outside the 0.5 in margin "
                              f"({x:.2f},{y:.2f},{x + w:.2f},{y + h:.2f})")
        top = [r for r in items if r[6] is None and r[7] not in ("conn", "label")]
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                _, _, x1, y1, w1, h1, _, _, n1 = top[a]
                _, _, x2, y2, w2, h2, _, _, n2 = top[b]
                gx = max(x2 - (x1 + w1), x1 - (x2 + w2))
                gy = max(y2 - (y1 + h1), y1 - (y2 + h2))
                gap = max(gx, gy)
                if gap < 0.3 - 1e-6:
                    what = "overlap" if gap < 0 else f"gap {gap:.2f} in"
                    if n1 == "title" or n2 == "title":
                        if gap < 0:
                            issues.append(f"slide {sn}: {what} between '{n1}' and '{n2}'")
                        continue
                    issues.append(f"slide {sn}: {what} between '{n1}' and '{n2}'")
    return issues


def deck_text(path):
    prs = Presentation(path)
    out = []
    for i, s in enumerate(prs.slides, 1):
        for sh in s.shapes:
            if sh.has_text_frame:
                out.append((i, sh.text_frame.text))
            if sh.has_table:
                out += [(i, c.text) for row in sh.table.rows for c in row.cells]
            if sh.has_chart:
                ch = sh.chart
                if ch.has_title:
                    out.append((i, ch.chart_title.text_frame.text))
                out += [(i, str(c)) for c in ch.plots[0].categories]
                out += [(i, sr.name) for sr in ch.plots[0].series]
        if s.has_notes_slide:
            out.append((i, s.notes_slide.notes_text_frame.text))
    return out


def render(pptx_path, pdf=None):
    pdf = pdf or pptx_path.with_suffix(".pdf")
    prev = RES / "deck_preview"
    prev.mkdir(exist_ok=True)
    for f in prev.glob("*.png"):
        f.unlink()
    ps = f"""
$pp = New-Object -ComObject PowerPoint.Application
$p = $pp.Presentations.Open('{pptx_path}', $true, $false, $false)
$p.SaveAs('{pdf}', 32)
$i = 1
foreach ($s in $p.Slides) {{ $s.Export(('{prev}\\slide_' + $i.ToString('00') + '.png'), 'PNG', 1600, 900); $i++ }}
$p.Close(); $pp.Quit()
"""
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    if r.returncode:
        print("render failed:", r.stderr[:500])
        return None
    return pdf


def slide_title(slide):
    return next((sh.text_frame.text.strip() for sh in slide.shapes
                 if sh.has_text_frame and sh.text_frame.text.strip()), "")


def insert_into(d, path, after, do_render):
    """Add the system + module architecture diagram slides to an existing deck, right after `after`."""
    import shutil
    prs = Presentation(path)
    titles = [slide_title(s) for s in prs.slides]
    if "System architecture diagram" in titles or "Module architecture diagram" in titles:
        print(f"{path} already has the architecture diagram slides - nothing to do")
        return 0
    if after not in titles:
        print(f"no slide titled '{after}' in {path}; titles are: {titles}")
        return 1
    backup = path.with_name(path.stem + "_backup" + path.suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backed up the original to {backup}")
    k = Deck(d)
    k.prs, k.n = prs, len(prs.slides)
    arch_slides(k, d)
    ids = prs.slides._sldIdLst
    new = list(ids)[-2:]
    pos = titles.index(after) + 1
    for j, el in enumerate(new):
        ids.remove(el)
        ids.insert(pos + j, el)
    prs.save(path)
    print(f"inserted 'System architecture diagram' and 'Module architecture diagram' as slides {pos + 1}-{pos + 2} "
          f"of {len(prs.slides)} in {path}")
    issues = [x.replace(f"slide {n}:", f"new slide {n - len(titles)}:") for x in qa(k, d)
              for n in [int(re.match(r"slide (\d+)", x)[1])]]
    hits = [(i, f, t[:80]) for i, t in deck_text(path) for f in FORBIDDEN
            if i in (pos + 1, pos + 2) and re.search(rf"(?<![A-Za-z0-9]){re.escape(f)}(?![0-9])", t, re.I)]
    print(f"QA on the new slides: {len(issues)} layout / text-fit issues, {len(hits)} forbidden-term hits")
    for x in issues + [str(h) for h in hits]:
        print("  -", x)
    if do_render:
        pdf = render(path, RES / "deck_preview" / "inserted_check.pdf")
        print("rendered previews to results/deck_preview/" if pdf else "render skipped")
    return 1 if (issues or hits) else 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="export PDF + PNG previews via PowerPoint")
    ap.add_argument("--insert-into", metavar="PPTX", help="add the two architecture-diagram slides to an existing deck")
    ap.add_argument("--after", default="System architecture", help="title of the slide to insert after")
    args = ap.parse_args()
    d = Data()
    d.load()
    if args.insert_into:
        return insert_into(d, Path(args.insert_into), args.after, args.render)
    k = build(d)
    k.prs.save(OUT)
    print(f"wrote {OUT} ({k.n} slides)")

    issues = qa(k, d)
    hits = [(i, f, t[:80]) for i, t in deck_text(OUT) for f in FORBIDDEN
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(f)}(?![0-9])", t, re.I)]
    fonts = {r.font.name for s in Presentation(OUT).slides for sh in s.shapes if sh.has_text_frame
             for p in sh.text_frame.paragraphs for r in p.runs}
    if args.render:
        pdf = render(OUT)
        print(f"rendered {pdf} and results/deck_preview/slide_XX.png" if pdf else "render skipped")

    print("\n================ QA ================")
    print(f"layout / text-fit issues: {len(issues)}")
    for x in issues:
        print("  -", x)
    print(f"forbidden-term hits: {len(hits)}")
    for x in hits:
        print("  -", x)
    print(f"fonts used in text boxes: {sorted(f for f in fonts if f)}")
    print("\n================ CHECKLIST ================")
    titles = []
    for i, s in enumerate(Presentation(OUT).slides, 1):
        t = next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), "")
        titles.append(t)
        print(f"slide {i:2d}: {t[:60]}")
        for src in sorted(d.sources.get(i, [])):
            print(f"           {src}")
    print("\nplaceholders left:")
    for sn, what in k.placeholders:
        print(f"  - slide {sn}: MISSING {what}")
    if not k.placeholders:
        print("  - none")
    if d.missing:
        print("missing inputs:", ", ".join(sorted(set(d.missing))))
    return 1 if (issues or hits) else 0


if __name__ == "__main__":
    sys.exit(main())
