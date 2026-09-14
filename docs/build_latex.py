import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
MD   = os.path.join(DOCS, "DriftJax_paper.md")
PRE  = "/tmp/driftjax_paper_pre.md"
TEX  = os.path.join(DOCS, "DriftJax_paper.tex")
TEMPL= os.path.join(DOCS, "paper.template.tex")
FIGDIR = os.path.join(DOCS, "figures")

TEMPLATE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{graphicx}
\usepackage{grffile}
\graphicspath{{figures/}}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\usepackage{caption}
\usepackage{float}
\usepackage{listings}
\usepackage{enumitem}
\newcommand{\passthrough}[1]{#1}
\newcommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\lstset{basicstyle=\ttfamily\small,breaklines=true,language=Python,frame=single,backgroundcolor=\color{gray!5},showstringspaces=false}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{3pt}
\sloppy
$if(title)$
\title{$title$}
$endif$
$if(author)$
\author{$author$}
$endif$
$if(date)$
\date{$date$}
$endif$
\begin{document}
$if(title)$
\maketitle
$endif$
$body$
\end{document}
"""
with open(TEMPL, "w") as template_file:
    template_file.write(TEMPLATE)

# 1. preprocess headings
with open(MD) as markdown_file:
    lines = markdown_file.read().split("\n")
start = next(i for i, line in enumerate(lines) if re.match(r'^##\s+Abstract\b', line))
body = lines[start:]
out = []
for ln in body:
    m = re.match(r'^(#{1,6})\s+(.*)$', ln)
    if not m:
        out.append(ln)
        continue
    h, c = m.group(1), m.group(2)
    if c.strip() in ("Abstract", "Program Summary", "References"):
        out.append(h + " " + c + " {-}")
        continue
    ma = re.match(r'^Appendix\s+([A-Z])\.\s*(.*)$', c)
    if ma:
        out.append(h + " " + ma.group(2).strip())
        continue
    mn = re.match(r'^(\d+(?:\.\d+)*)\.?\s+(.*)$', c)
    if mn:
        out.append(h + " " + mn.group(2).strip())
        continue
    out.append(ln)
with open(PRE, "w") as preprocessed_file:
    preprocessed_file.write("\n".join(out))

# 2. pandoc
cmd = ["pandoc", PRE, "-o", TEX, "--from", "markdown", "--to", "latex",
        "--standalone", "--template", TEMPL, "--listings",
        "--shift-heading-level-by=-1",
        "-V", "title=DriftJax: A Differentiable One-Dimensional Drift-Diffusion Solver for Photovoltaic Device Design",
        "-V", "author=The DriftJax Authors", "-V", "date=v0.1.14"]
r = subprocess.run(cmd, capture_output=True, text=True)
print("pandoc rc:", r.returncode, r.stderr[:400])

# 3. figures: copy originals from examples/validation (with underscores; grffile handles them)
#    Preserve paper/supplement figures (fig*, supp_fig*) — do NOT delete those.
os.makedirs(FIGDIR, exist_ok=True)
for f in os.listdir(FIGDIR):
    if f.endswith(".png") and not f.startswith("fig") and not f.startswith("supp_fig"):
        os.remove(os.path.join(FIGDIR, f))
cnt = 0
for src in ["examples/research/outputs", "examples/tutorial/outputs", "examples/developer/outputs", "validation"]:
    d = os.path.join(ROOT, src)
    if not os.path.isdir(d):
        continue
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".png"):
                shutil.copy(os.path.join(root, f), os.path.join(FIGDIR, f))
                cnt += 1
print("copied figures:", cnt)

# 4. post-process tex: appendix + figures + unicode
with open(TEX) as tex_file:
    tex = tex_file.read()
APX = r"\section{The Drift-Diffusion Model}"
if APX in tex and "\\appendix" not in tex:
    tex = tex.replace(APX, "\\appendix\n" + APX, 1)
    print("injected \\appendix")

CAPTIONS = {
    "research_01_device_iv.png": "Mesh-convergence study for a silicon p--n device, showing the IV curves and relative efficiency error against the finest grid.",
    "research_03_optical_model_comparison.png": "Beer--Lambert and coherent TMM optical models compared through the predicted IV response and generation profile.",
    "research_07_tandem_current_matching.png": "Ideal two-terminal tandem current matching as the bottom-cell thickness varies.",
    "research_08_statistics_regime_map.png": "Carrier-statistics approximations and their effect on the silicon device IV curve.",
    "developer_batching_and_jit.png": "JIT warm-up, batched bias solves, and wavelength-sharding agreement.",
    "research_11_transient_small_signal.png": "Backward-Euler transient response and small-signal admittance.",
    "research_12_material_thickness_design_map.png": "Bandgap/thickness efficiency map with the bounded differentiable-design result.",
    "research_13_target_iv_structure.png": "Target-IV structure recovery from an initial guess.",
}

def fig_env(name):
    cap = CAPTIONS.get(name)
    if cap is None:
        cap = "Output of example \\texttt{" + name.replace("_", "\\_") + "}."
    return (f"\\begin{{figure}}[htbp]\n\\centering\n"
            f"\\includegraphics[width=0.85\\textwidth]{{{name}}}\n"
            f"\\caption{{{cap}}}\\end{{figure}}")

def repl(m):
    names = re.findall(r'\\lstinline\!([^!]+)\!', m.group(0))
    figs = []
    for raw in names:
        orig = raw.replace("\\_", "_")
        path = os.path.join(FIGDIR, orig)
        if os.path.exists(path):
            figs.append(fig_env(orig))
    return "\n".join(figs) if figs else m.group(0)

pat = re.compile(r'Outputs?:\s*(?:\\passthrough\{\\lstinline\![^!]+?\!\}(?:\s*,\s*)?)+\s*\.?')
tex = pat.sub(repl, tex)
print("figure envs:", tex.count("\\begin{figure}"))

# raw partial symbol -> math
tex = tex.replace("\u2202", "$\\partial$")
# Pandoc emits one extra closing brace for labels produced by the local
# heading preprocessor; remove it before the LaTeX source is handed to a
# journal toolchain.
tex = re.sub(r"(\\label\{[^}]+\})\}", r"\1", tex)
# The standalone article template does not need Pandoc's hypertarget wrapper;
# removing it also avoids an unterminated argument in the generated headings.
tex = re.sub(r"\\hypertarget\{[^}]+\}\{%\n", "", tex)
# Pandoc also emits a stray closing brace on its own line between a figure's
# \label and \end{figure}; drop a lone '}' that directly follows a label line.
tex = re.sub(r"(\\label\{[^}]+\})\n\s*\}\n", r"\1\n", tex)
# Give figures inserted from Markdown an explicit journal-column width so
# their generated filenames cannot create overfull boxes.
tex = tex.replace("\\includegraphics{figures/", "\\includegraphics[width=0.85\\textwidth]{figures/")
with open(TEX, "w") as tex_file:
    tex_file.write(tex)
print("tex bytes:", os.path.getsize(TEX))
