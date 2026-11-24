"""Build docs/Brain_Tumor_Segmentation_Report.docx (two-pass: page numbers for the contents pages
are measured from a LibreOffice PDF render of the first pass).  Numbers come from docs/results.json."""
import json, re, subprocess, sys, time
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "docs" / "figures"
R = json.loads((ROOT / "docs" / "results.json").read_text())
OUT = ROOT / "docs" / "Brain_Tumor_Segmentation_Report.docx"
TITLE = "UNCERTAINTY-QUANTIFIED DEEP LEARNING FOR BRAIN TUMOR SEGMENTATION FROM MRI"

# ------------------------------------------------------------------ numbers
S, C, U = R["segmentation"], R["calibration"], R["uncertainty_quality"]
RET, REG, FD, RC, NS, CF = R["retained_dice"], R["entropy_by_region"], R["failure_detection"], R["risk_coverage"], R["noise_shift"], R["config"]
f2, f3 = (lambda x: f"{x:.2f}"), (lambda x: f"{x:.3f}")

def timing():
    from btseg.classical import ClassicalEngine
    from btseg.synthetic import make_dataset
    xs, ys, bs = make_dataset(20, seed=1)
    t = time.time(); e = ClassicalEngine(n_trees=64).fit(xs, ys, bs); tf = time.time() - t
    xt, _, bt = make_dataset(10, seed=3); t = time.time()
    for i in range(10): e.predict(xt[i], bt[i])
    return tf, (time.time() - t) / 10
T_FIT, T_INF = timing()

# --------------------------------------------------------------- doc helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Inches(1.25), Inches(1), Inches(1), Inches(1)
sec.different_first_page_header_footer = True
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 1.4; st.paragraph_format.space_after = Pt(6)
for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 12)):
    h = doc.styles[name]; h.font.name = "Times New Roman"; h.font.size = Pt(size); h.font.bold = True
    h.font.color.rgb = RGBColor(0, 0, 0); h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    h.paragraph_format.space_before = Pt(12); h.paragraph_format.space_after = Pt(6); h.paragraph_format.keep_with_next = True

# page-number footer
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run()
for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    if tag:
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag)
    else:
        e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
    r._r.append(e)

counters = {"fig": 0, "tab": 0, "ch": 0}
ENTRIES, LOT, LOF = [], [], []     # (level, text) for contents / lists

def para(text="", bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, after=None, style=None):
    p = doc.add_paragraph(style=style); p.alignment = align
    if text:
        r = p.add_run(text); r.bold, r.italic = bold, italic
        if size: r.font.size = Pt(size)
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p

def rich(text, **kw):
    """**bold** inline markup."""
    text = text.replace("`", "")
    p = para(**kw)
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part: p.add_run(part).bold = bool(i % 2)
    return p

def P(*texts):
    for t in texts: rich(t)

def bullets(items, style="List Bullet"):
    for t in items:
        p = rich(t, style=style); p.paragraph_format.line_spacing = 1.2; p.paragraph_format.space_after = Pt(3)

def numbered(items):
    for i, t in enumerate(items, 1):
        p = rich(f"{i}.\t" + t, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        p.paragraph_format.left_indent = Inches(0.35); p.paragraph_format.first_line_indent = Inches(-0.35)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.35)); p.paragraph_format.line_spacing = 1.2; p.paragraph_format.space_after = Pt(3)

def pagebreak(): doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

def front_title(text):
    p = para(text, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=15, after=14); p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.keep_with_next = True
    return p

def chapter(title):
    counters["ch"] += 1; pagebreak()
    n = counters["ch"]
    p = para(f"CHAPTER {n}", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, after=0)
    h = doc.add_heading(f"{n}. {title.upper()}", 1); h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ENTRIES.append((0, f"{n}. {title.upper()}"))

def section(num, title):
    t = f"{num} {title.upper()}"; doc.add_heading(t, 2); ENTRIES.append((1, t))

def sub(t): doc.add_heading(t, 3)

def caption(kind, text):
    counters[kind] += 1
    ch = counters["ch"]
    key = "Figure" if kind == "fig" else "Table"
    n = counters[kind]
    cap = f"{key} {n}: {text}".replace("`", "")
    (LOF if kind == "fig" else LOT).append(cap)
    p = para(cap, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=11, after=8)
    return p

def fig(file, text, width=5.6):
    p = para(align=WD_ALIGN_PARAGRAPH.CENTER, after=2); p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / file), width=Inches(width))
    caption("fig", text)

def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color); tcPr.append(sh)

def table(text, header, rows, widths=None, size=10, cap=True):
    if cap: caption("tab", text).paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(str(h)); r.bold = True; r.font.size = Pt(size); shade(c, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(v).replace("`", "")); r.font.size = Pt(size)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        for i, c in enumerate(row.cells):
            for pp in c.paragraphs: pp.paragraph_format.line_spacing = 1.0; pp.paragraph_format.space_after = Pt(2); pp.alignment = WD_ALIGN_PARAGRAPH.LEFT
            if widths: c.width = Inches(widths[i])
    para(after=4)

def code(text):
    for line in text.strip("\n").split("\n"):
        p = para(align=WD_ALIGN_PARAGRAPH.LEFT, after=0); p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.left_indent = Inches(0.2)
        r = p.add_run(line or " "); r.font.name = "Courier New"; r.font.size = Pt(9)
        r._r.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Courier New")
    para(after=4)

def toc_line(text, page, level):
    p = para(align=WD_ALIGN_PARAGRAPH.LEFT, after=2); p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.left_indent = Inches(0.0 if level == 0 else 0.35)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(5.95), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    r = p.add_run(f"{text}\t{page}"); r.bold = level == 0

PAGES = json.loads(Path("/tmp/claude-0/-home-claude/1a07952f-25d8-5a39-9e10-b1737da07fc5/scratchpad/pages.json").read_text()) if len(sys.argv) > 1 and sys.argv[1] == "pass2" else {}
pg = lambda k: PAGES.get(k, "00")

# ================================================================= COVER
for _ in range(2): para()
para("A PROJECT REPORT ON", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13)
para(TITLE, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=20, after=18)
para("Healthcare AI", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, after=24)
para("Submitted in partial fulfilment of the requirements for the degree of", align=WD_ALIGN_PARAGRAPH.CENTER)
para("BACHELOR OF TECHNOLOGY\nin\nCOMPUTER SCIENCE AND ENGINEERING", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13, after=24)
para("Submitted by", align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para("SURYA U", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, after=0)
para("USN: 25BTRCM053   |   Branch / Section: CSE-CTMA", align=WD_ALIGN_PARAGRAPH.CENTER, after=24)
para("Under the guidance of", align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para("[Guide / Faculty name, designation]", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=30)
para("SCHOOL OF COMPUTER SCIENCE AND ENGINEERING\nJAIN (DEEMED-TO-BE UNIVERSITY)\nAcademic Year 2026 – 2027", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=12)

# ============================================================ CERTIFICATE
pagebreak(); front_title("CERTIFICATE")
P("This is to certify that the project work entitled **“Uncertainty-Quantified Deep Learning for Brain Tumor Segmentation from MRI”** is a bonafide work carried out by **Surya U (USN: 25BTRCM053)**, a student of the Bachelor of Technology programme in Computer Science and Engineering (Section CSE-CTMA), School of Computer Science and Engineering, Jain (Deemed-to-be) University, in partial fulfilment of the requirements for the award of the degree during the academic year 2026–2027.",
  "It is certified that all corrections and suggestions indicated during internal assessment have been incorporated in the report. The project report has been approved as it satisfies the academic requirements in respect of the project work prescribed for the said degree.")
for _ in range(5): para()
para("Signature of the Guide\t\t\t\tSignature of the Head of the Department", align=WD_ALIGN_PARAGRAPH.LEFT)
para("[Guide name]\t\t\t\t\t[HoD name]", align=WD_ALIGN_PARAGRAPH.LEFT)
para("\nInternal Examiner: ____________________\t\tExternal Examiner: ____________________", align=WD_ALIGN_PARAGRAPH.LEFT)

# ============================================================ DECLARATION
pagebreak(); front_title("DECLARATION")
P("I, **Surya U**, bearing USN **25BTRCM053**, a student of B.Tech Computer Science and Engineering, School of Computer Science and Engineering, Jain (Deemed-to-be) University, hereby declare that the project report entitled **“Uncertainty-Quantified Deep Learning for Brain Tumor Segmentation from MRI”** is an original work carried out by me under the guidance of [Guide name].",
  "I further declare that this work has not been submitted, in part or in full, to any other university or institution for the award of any degree, diploma or certificate. All sources of information, published or unpublished, have been duly acknowledged in the references.")
for _ in range(3): para()
para("Place: Bengaluru\t\t\t\t\t\tSurya U", align=WD_ALIGN_PARAGRAPH.LEFT)
para("Date:\t\t\t\t\t\t\tUSN: 25BTRCM053", align=WD_ALIGN_PARAGRAPH.LEFT)

# ======================================================== ACKNOWLEDGEMENTS
pagebreak(); front_title("ACKNOWLEDGEMENTS")
P("I express my sincere gratitude to my project guide, [Guide name], for the constant encouragement, technical direction and patient feedback that shaped this work from the initial idea to the final report.",
  "I am thankful to the Head of the Department and the faculty of the School of Computer Science and Engineering, Jain (Deemed-to-be) University, for providing the academic environment and resources required to complete this project.",
  "I also acknowledge the authors of the open scientific work on which this project builds — in particular the U-Net, Monte-Carlo Dropout, deep-ensemble and BraTS-benchmark communities — and the maintainers of the open-source libraries used (Python, NumPy, SciPy, scikit-learn, PyTorch, OpenCV, Flask and Matplotlib).",
  "Finally, I thank my family and friends for their support throughout.")
para("Surya U", bold=True, align=WD_ALIGN_PARAGRAPH.RIGHT)

# =============================================================== ABSTRACT
pagebreak(); front_title("ABSTRACT")
P("Brain tumours such as gliomas are routinely delineated on magnetic-resonance images (MRI) for diagnosis, surgical planning and follow-up. Manual delineation is slow and varies between experts, and modern deep-learning segmenters, although accurate on average, return a single confident-looking mask even when they are wrong. In a clinical setting an over-confident error is more dangerous than an honest “I am not sure”.",
  "This project builds an uncertainty-quantified brain-tumour segmentation system. A 2-D U-Net with dropout is used with Monte-Carlo Dropout (and optionally deep ensembles) to obtain T stochastic predictions per slice. Their mean gives the segmentation; the predictive entropy gives the total uncertainty, which is decomposed into aleatoric (data) and epistemic (model) components via mutual information. An image-level review score ranks slices that should be checked by a clinician. The system is delivered as a Flask web application, a documented Python package, a reproducible experiment script and a pytest suite, and the complete project is published on GitHub.",
  f"Because registration-gated clinical data cannot be redistributed, the shipped experiments use a procedural MRI phantom with bias field, Rician noise, soft lesion edges and distractor lesions, and a CPU-only bagged-tree ensemble that exposes the same uncertainty interface as the U-Net. On {CF['n_test']} held-out slices the ensemble reached a Dice of {f3(S['ensemble']['dice'])} (single tree {f3(S['single_tree']['dice'])}, Otsu thresholding {f3(S['otsu']['dice'])}); the uncertainty map separated wrongly classified pixels from correct ones with an AUROC of {f2(U['auroc_total'])}; discarding the 20 % most uncertain pixels raised Dice to {f3(RET['0.2'])}; and the image-level review score correlated with Dice at Spearman ρ = {f2(FD['spearman_review_vs_dice'])}. The U-Net training pipeline for real BraTS data is provided and is the intended next experiment.")
para("Keywords: brain tumour segmentation, MRI, U-Net, uncertainty quantification, Monte-Carlo Dropout, epistemic and aleatoric uncertainty, calibration, healthcare AI.", italic=True)

# ============================================= CONTENTS + LISTS (placeholders)
CONTENTS_MARK = len(doc.element.body)  # remember position; we append, structure fixed in order below
pagebreak(); front_title("LIST OF CONTENTS")
FRONT = [("CERTIFICATE", "CERTIFICATE"), ("DECLARATION", "DECLARATION"), ("ACKNOWLEDGEMENTS", "ACKNOWLEDGEMENTS"), ("ABSTRACT", "ABSTRACT"),
         ("LIST OF TABLES", "LIST OF TABLES"), ("LIST OF FIGURES", "LIST OF FIGURES"), ("LIST OF ABBREVIATIONS", "LIST OF ABBREVIATIONS")]
BODY_STRUCT = [  # (level, text) - fixed structure of the body, mirrors the college template
    (0, "1. INTRODUCTION"), (1, "1.1 KEY FEATURES"), (1, "1.2 OVERVIEW AND WORKING OF MACHINE LEARNING"), (1, "1.3 DESIGN STANDARDS"),
    (1, "1.4 OBJECTIVES OF THE PROJECT"), (1, "1.5 OVERALL ORGANIZATION"),
    (0, "2. RELATED WORK"), (1, "2.1 LITERATURE SURVEY"), (1, "2.2 EXISTING SYSTEM"), (1, "2.3 SYSTEM STUDY"), (1, "2.4 LIMITATIONS"),
    (0, "3. PROPOSED METHODOLOGY"), (1, "3.1 PROPOSED WORKFLOW"), (1, "3.2 INPUT DESIGN AND OUTPUT DESIGN"), (1, "3.3 ADVANTAGES OF PROPOSED SYSTEM"),
    (1, "3.4 SYSTEM DESIGN"), (1, "3.5 SYSTEM REQUIREMENTS"),
    (0, "4. IMPLEMENTATION"), (1, "4.1 TECHNOLOGIES USED"),
    (0, "5. RESULT AND TESTING"), (1, "5.1 TESTING METHODOLOGY"), (1, "5.2 TYPES OF TESTING"), (1, "5.3 SCREENSHOTS OF WEBPAGE"),
    (0, "6. CONCLUSION"), (1, "6.1 CONCLUSION"), (1, "6.2 FUTURE WORK"), (0, "7. REFERENCES")]
for _, t in FRONT: toc_line(t, pg(t), 0)
for lv, t in BODY_STRUCT: toc_line(t, pg(t), lv)
CONTENT_PAGE_FIXED = True

pagebreak(); front_title("LIST OF TABLES")
LOT_ANCHOR = doc.add_paragraph()   # filled after body is built
pagebreak(); front_title("LIST OF FIGURES")
LOF_ANCHOR = doc.add_paragraph()

pagebreak(); front_title("LIST OF ABBREVIATIONS")
abbr = [("AI", "Artificial Intelligence"), ("API", "Application Programming Interface"), ("AUROC", "Area Under the ROC Curve"), ("BraTS", "Brain Tumor Segmentation (challenge / benchmark)"),
        ("BN", "Batch Normalisation"), ("CNN", "Convolutional Neural Network"), ("CPU / GPU", "Central / Graphics Processing Unit"), ("CSV / JSON", "Comma-Separated Values / JavaScript Object Notation"),
        ("DICOM", "Digital Imaging and Communications in Medicine"), ("ECE", "Expected Calibration Error"), ("FLAIR", "Fluid-Attenuated Inversion Recovery"), ("HD95", "95th-percentile Hausdorff Distance"),
        ("HTTP", "Hypertext Transfer Protocol"), ("IoU", "Intersection over Union"), ("MC", "Monte-Carlo"), ("MI", "Mutual Information"), ("MRI", "Magnetic Resonance Imaging"),
        ("NIfTI", "Neuroimaging Informatics Technology Initiative (file format)"), ("OOD", "Out-of-Distribution"), ("ReLU", "Rectified Linear Unit"), ("ROC", "Receiver Operating Characteristic"),
        ("T1ce", "T1-weighted contrast-enhanced MRI"), ("UQ", "Uncertainty Quantification"), ("U-Net", "U-shaped encoder-decoder segmentation network"), ("UI", "User Interface")]
table("Abbreviations used in this report", ["Abbreviation", "Expansion"], abbr, [1.5, 4.4], 11, cap=False)

# ============================================================== CHAPTER 1
chapter("Introduction")
P("Gliomas are the most common primary malignant tumours of the brain. Their treatment — surgery, radiotherapy and chemotherapy — depends on knowing exactly where the tumour is and how large it is. Magnetic-resonance imaging (MRI) is the modality of choice because different sequences (T1, T1 with contrast, T2, FLAIR) highlight different tissue properties: FLAIR shows the oedema surrounding a lesion, while contrast-enhanced T1 highlights the active tumour core. Turning these images into a precise tumour outline (segmentation) is today largely manual, takes tens of minutes per patient and differs from one radiologist to the next.",
  "Deep convolutional networks, and the U-Net family in particular, have made automatic segmentation highly accurate; the winning entries of the Multimodal Brain Tumor Segmentation (BraTS) challenge routinely reach Dice scores above 0.85 on the whole tumour. However, a standard network outputs a single hard mask and gives no sign of when it is guessing. Networks are known to be over-confident, especially on images that differ from their training data (a new scanner, more noise, an unusual anatomy). A clinician cannot tell a trustworthy mask from an unreliable one.",
  "This project therefore treats **“how sure am I?”** as a first-class output. For every pixel the system reports the probability of tumour and the uncertainty of that probability, separates uncertainty caused by ambiguous data (aleatoric) from uncertainty caused by the model's limited knowledge (epistemic), and raises a review flag for slices on which the model is unusually unsure. The goal is not to replace the radiologist, but to direct the radiologist's attention to where it is needed.")
section("1.1", "Key Features")
bullets(["**Segmentation with uncertainty** – every prediction comes with total, aleatoric and epistemic uncertainty maps (in bits, 0 = certain, 1 = coin-flip).",
         "**Two stochastic back-ends behind one interface** – a PyTorch U-Net with Monte-Carlo Dropout / deep ensembles for GPU users, and a CPU-only bagged-tree ensemble so that the whole system runs on any laptop.",
         "**Review flag** – an image-level score (mean entropy in and around the predicted tumour) ranks slices for human inspection.",
         "**Interactive web application** – upload a slice or generate a synthetic case, inspect five overlay views and eleven numeric indicators; works on desktop and phone.",
         "**Calibration and quality metrics** – Dice, IoU, HD95, expected calibration error, Brier score, uncertainty-AUROC and retained-Dice (risk–coverage) curves.",
         "**Reproducible research package** – deterministic seeds, one-command experiment script, 11 automated tests, MIT licence, public GitHub repository.",
         "**Real-data path** – conversion script for BraTS NIfTI volumes and a training script with Dice + BCE loss, augmentation and automatic calibration of the review threshold."])
section("1.2", "Overview and Working of Machine Learning")
sub("1.2.1 Supervised learning for segmentation")
P("Image segmentation is a supervised learning problem: given a set of images x with expert masks y, learn a function f_θ that predicts, for every pixel, the probability that it belongs to the tumour. The parameters θ are adjusted by gradient descent so as to minimise a loss that measures disagreement between prediction and expert mask. Here the loss is the sum of the binary cross-entropy (good for stable gradients) and the soft Dice loss (good for the severe class imbalance, because tumour pixels are a small fraction of the image).")
sub("1.2.2 Convolutional networks and the U-Net")
P("A convolutional neural network (CNN) learns small filters that detect edges, textures and eventually tumour-like patterns. The U-Net (Ronneberger et al., 2015) arranges convolutions in an encoder that repeatedly halves the resolution while doubling the number of feature channels — capturing context — and a decoder that restores the resolution. Skip connections copy encoder features directly to the decoder so that fine boundary detail is not lost. Each stage uses two 3×3 convolutions, batch normalisation, ReLU activation and, in this project, a Dropout2d layer.")
fig("fig_unet.png", "U-Net with dropout used in this project", 5.4)
sub("1.2.3 Why probabilities are not enough: aleatoric and epistemic uncertainty")
P("The sigmoid output of a network looks like a probability but is frequently mis-calibrated. Two different sources of doubt exist and call for different reactions:",)
bullets(["**Aleatoric uncertainty** is inherent in the data: image noise, partial-volume blurring at the tumour edge, or genuinely ambiguous tissue. More training data does not remove it. It tells the clinician “even an expert would hesitate here”.",
         "**Epistemic uncertainty** reflects the model's ignorance: unusual appearance, too few similar training examples, or a scanner it has never seen. It can be reduced with more or better data and is the right signal for out-of-distribution detection."])
sub("1.2.4 Monte-Carlo Dropout and deep ensembles")
P("Dropout randomly switches off a fraction of neurons during training. Gal and Ghahramani (2016) showed that keeping dropout active at test time and averaging T stochastic forward passes approximates Bayesian inference over the weights. Lakshminarayanan et al. (2017) obtained comparable or better uncertainty by training M networks independently (deep ensembles). Both produce T foreground-probability maps p_1(x) … p_T(x) per image.")
sub("1.2.5 Uncertainty decomposition")
P("With the T samples, the prediction and the uncertainties are computed per pixel as follows (entropies in bits):")
code("""p_mean(x)    = (1/T) * sum_t p_t(x)          # prediction
total(x)     = H[ p_mean(x) ]                # predictive entropy
aleatoric(x) = (1/T) * sum_t H[ p_t(x) ]     # expected entropy
epistemic(x) = total(x) - aleatoric(x)       # mutual information (>= 0)
H[p] = -p*log2(p) - (1-p)*log2(1-p)""")
P("If all passes agree on p = 0.5, the uncertainty is purely aleatoric; if half of the passes say 0 and half say 1, each pass is sure but they disagree, which is purely epistemic. Section 5.2 verifies these two limiting cases by unit tests.")
sub("1.2.6 Calibration")
P("A model is calibrated if, among pixels to which it assigns probability 0.7, about 70 % really are tumour. The expected calibration error (ECE) bins the predicted probabilities and averages the gap between confidence and observed frequency; the Brier score is the mean squared error of the probabilities. Averaging many stochastic predictions usually improves both (Guo et al., 2017; Mehrtash et al., 2020).")
section("1.3", "Design Standards")
P("The project follows established conventions so that its results are comparable and its code is maintainable. Compliance with regulatory standards is *not* claimed — the system is a research prototype.".replace("*", ""))
table("Design standards and conventions followed", ["Area", "Standard / convention", "How it is applied"],
      [["Evaluation", "BraTS challenge protocol (Menze et al., 2015; Bakas et al., 2017)", "Dice and HD95 on the whole-tumour region; per-slice reporting with mean ± SD"],
       ["Data formats", "NIfTI (neuroimaging) and DICOM awareness", "`prepare_brats.py` reads NIfTI volumes; model consumes plain 2-D arrays"],
       ["Pre-processing", "Per-channel z-score within the brain mask", "Standard MRI intensity normalisation; skull rim excluded"],
       ["Calibration", "Guo et al. (2017) ECE with 15 equal-width bins", "`metrics.ece` and reliability diagram"],
       ["Code quality", "PEP 8, type hints, docstrings, module-per-concern", "`src/btseg` package, no logic in the web layer"],
       ["Testing", "pytest unit, API and integration tests", "11 automated tests plus optional PyTorch tests"],
       ["Reproducibility", "Fixed random seeds, pinned requirements, one-command experiments", "`make experiments` regenerates every table and figure"],
       ["Web / UI", "Responsive layout, WCAG-minded colour contrast, light/dark theme", "Single-page UI usable at 390 px width"],
       ["Safety & ethics", "Research-use-only labelling; no patient data in repository", "Disclaimer in UI footer, README and report"]],
      [1.1, 2.3, 2.5], 9.5)
section("1.4", "Objectives of the Project")
numbered(["To design and implement a deep-learning pipeline that segments brain tumours from MRI slices.",
          "To quantify predictive uncertainty at pixel level using Monte-Carlo Dropout and ensembles and to decompose it into aleatoric and epistemic parts.",
          "To derive an image-level review score that identifies slices likely to be segmented poorly.",
          "To evaluate accuracy, calibration, uncertainty quality and robustness to image noise with standard metrics.",
          "To build a usable web application that presents the segmentation and the uncertainty maps to a human reader.",
          "To publish the complete, documented, tested project (source code, scripts, report) on GitHub."])
section("1.5", "Overall Organization")
P("Chapter 1 introduced the problem, features, background theory, standards and objectives. Chapter 2 reviews related work, the existing systems and their limitations, and studies the feasibility of the proposed system. Chapter 3 describes the proposed methodology: workflow, input/output design, advantages, system design and requirements. Chapter 4 explains the implementation and the technologies used. Chapter 5 reports the testing methodology, test results, model evaluation and screenshots of the web application. Chapter 6 concludes the work and lists future enhancements, and Chapter 7 lists the references.")

# ============================================================== CHAPTER 2
chapter("Related Work")
section("2.1", "Literature Survey")
P("Automatic brain-tumour segmentation has progressed from intensity-threshold and atlas-based methods to random forests and, since about 2015, deep networks. The BraTS benchmark (Menze et al., 2015; Bakas et al., 2017) provides multi-institutional multi-parametric MRI with expert labels and has driven most of this progress. The U-Net (Ronneberger et al., 2015) and its 3-D variants (Çiçek et al., 2016; Milletari et al., 2016) became the standard backbone, and nnU-Net (Isensee et al., 2021) showed that a carefully self-configuring U-Net matches or beats most specialised architectures. Transformer-based segmenters such as UNETR (Hatamizadeh et al., 2022) offer further gains at higher computational cost.",
  "Uncertainty estimation entered medical imaging through Bayesian deep learning. Gal and Ghahramani (2016) interpreted dropout as approximate Bayesian inference; Kendall and Gal (2017) separated aleatoric and epistemic uncertainty; Lakshminarayanan et al. (2017) proposed deep ensembles. Applied to medical segmentation, Nair et al. (2020) compared uncertainty measures for multiple-sclerosis lesion detection, Mehrtash et al. (2020) studied calibration of segmentation networks, Jungo and Reyes (2019) examined how reliable voxel-wise uncertainty really is, and Wang et al. (2019) obtained aleatoric uncertainty through test-time augmentation. Abdar et al. (2021) review the field.")
table("Summary of the key literature", ["Work", "Contribution", "Relevance here"],
      [["Ronneberger et al., 2015", "U-Net encoder–decoder with skip connections", "Backbone of the proposed model"],
       ["Menze et al., 2015; Bakas et al., 2017", "BraTS benchmark and labelling protocol", "Evaluation protocol, target dataset"],
       ["Isensee et al., 2021", "nnU-Net self-configuring pipeline", "Shows strong U-Net baselines; guides training recipe"],
       ["Gal & Ghahramani, 2016", "MC Dropout as approximate Bayesian inference", "Main uncertainty mechanism"],
       ["Kendall & Gal, 2017", "Aleatoric vs epistemic uncertainty", "Decomposition used in `uncertainty.py`"],
       ["Lakshminarayanan et al., 2017", "Deep ensembles", "Alternative sampler; motivates ensemble back-end"],
       ["Guo et al., 2017", "Modern networks are mis-calibrated; ECE", "Calibration metrics"],
       ["Mehrtash et al., 2020", "Calibration and uncertainty in medical segmentation", "Justifies calibration analysis"],
       ["Jungo & Reyes, 2019", "Reliability of voxel-wise uncertainty", "Motivates image-level score and caution"],
       ["Nair et al., 2020", "Uncertainty measures for lesion detection", "Uncertainty-vs-error evaluation"],
       ["Wang et al., 2019", "Test-time augmentation for aleatoric uncertainty", "Possible future sampler"],
       ["Breiman, 2001", "Random forests / bagging", "CPU fallback ensemble"]],
      [1.7, 2.4, 1.8], 9.5)
section("2.2", "Existing System")
P("In current practice tumour delineation is performed either manually by a radiologist or semi-automatically with tools that require seed points and manual correction. Fully automatic deep-learning tools exist in research software and in some commercial products; they return a mask (and sometimes a single “confidence” number) but typically no spatial uncertainty. Open-source BraTS solutions such as nnU-Net provide high accuracy but are designed for competition-style batch inference on 3-D volumes with GPUs and offer no interactive interface that explains where the prediction is doubtful.")
section("2.3", "System Study")
sub("2.3.1 Problem analysis")
P("The study of the existing systems revealed three gaps: (i) outputs carry no spatial indication of reliability; (ii) there is no distinction between uncertainty that can be fixed with more data and uncertainty that cannot; and (iii) tools are difficult for a student or small lab to run because of data access and GPU requirements. The proposed system addresses each gap.")
sub("2.3.2 Feasibility study")
table("Feasibility analysis", ["Dimension", "Assessment"],
      [["Technical", "All components (U-Net, MC Dropout, entropy decomposition, Flask) rely on mature open-source libraries; a CPU fallback removes the GPU dependency."],
       ["Economic", "No licence cost; development and the shipped experiments run on a 2-core CPU. Real BraTS training is feasible on a free cloud GPU."],
       ["Operational", "A single web page with two actions (generate / upload) and plain-language maps; no installation for the end user beyond `python app/app.py`."],
       ["Data / legal", "BraTS needs registration and cannot be redistributed, so the repository ships a synthetic phantom and a conversion script for the real data."],
       ["Schedule", "Implementation, experiments, report and GitHub release were completed within the project period."]],
      [1.2, 4.7], 10)
section("2.4", "Limitations")
bullets(["Existing deep segmenters do not communicate where they are unsure; over-confident errors go unnoticed.",
         "Uncertainty methods multiply inference cost by T (passes or members).",
         "Voxel-wise uncertainty can be unreliable under strong distribution shift (Jungo & Reyes, 2019); it supports, but does not replace, expert review.",
         "Public benchmarks are limited to specific institutions and sequences; generalisation to other scanners is unproven.",
         "Most research code lacks a user interface and automated tests, limiting reuse."])

# ============================================================== CHAPTER 3
chapter("Proposed Methodology")
section("3.1", "Proposed Workflow")
P("The workflow (Figure 2) converts one MRI slice into a segmentation, three uncertainty maps and a review decision.")
fig("fig_workflow.png", "Proposed workflow", 6.0)
numbered(["**Acquire** – a FLAIR slice (uploaded file, BraTS slice or synthetic phantom).",
          "**Pre-process** – estimate the brain mask (smoothing, thresholding, hole filling, largest component, erosion of the skull rim), resize to 128×128 and z-score normalise inside the brain.",
          "**Stochastic inference** – run T passes through the model: MC-Dropout U-Net (T = 20 by default) or the 64 trees of the ensemble back-end.",
          "**Aggregate** – average the probabilities and threshold at 0.5 to obtain the tumour mask.",
          "**Decompose uncertainty** – compute total entropy, aleatoric and epistemic maps.",
          "**Decide and present** – compute the review score, compare with the threshold calibrated on validation data, and display overlays and measurements."])
section("3.2", "Input Design and Output Design")
table("Input design", ["Input", "Format", "Handling"],
      [["MRI slice upload", ".png, .jpg, .bmp, .tif (grey-scale) or .npy (H×W or C×H×W)", "Decoded, first channel used, resized to 128×128 (area interpolation)"],
       ["Synthetic case", "seed (integer), noise level σ ∈ [0.02, 0.25]", "Procedural phantom generator; also returns ground-truth mask and brain mask"],
       ["Maximum upload", "20 MB", "Larger requests rejected by the server"],
       ["Validation", "Unsupported extension or undecodable file", "HTTP 400 with a readable message shown in the UI"]],
      [1.4, 2.4, 2.1], 9.5)
table("Output design", ["Output", "Description"],
      [["Segmentation overlay", "Predicted tumour contour (red); ground truth contour (green) for synthetic cases"],
       ["Total-uncertainty map", "Predictive entropy, magma colour map, 0 – 1 bit"],
       ["Epistemic map", "Mutual information – model disagreement"],
       ["Aleatoric map", "Expected entropy – data ambiguity"],
       ["Measurements", "Tumour area (px) and share of brain, mean total / epistemic / aleatoric uncertainty, % pixels near coin-flip (> 0.8 bit), review score and threshold, Dice (synthetic only)"],
       ["Review banner", "“Review recommended” or “within normal range”, never a diagnosis"],
       ["JSON API", "Same content as base-64 PNGs plus a statistics object, for integration"]],
      [1.8, 4.1], 10)
section("3.3", "Advantages of Proposed System")
bullets(["**Transparent** – shows where and why the model hesitates; boundaries and false-positive regions light up (Section 5.2).",
         "**Actionable** – the review score lets a reader prioritise slices; keeping the 50 % most confident slices raised mean Dice from " + f3(RC['0.0']['mean_dice']) + " to " + f3(RC['0.5']['mean_dice']) + ".",
         "**Accessible** – runs on a CPU laptop; real-data GPU training is optional.",
         "**Modular** – one `predict()` interface for any sampler; new back-ends need ~40 lines.",
         "**Verified and reproducible** – automated tests, fixed seeds, single command to regenerate results.",
         "**Safe by design** – explicit research-only disclaimers and no patient data in the repository."])
section("3.4", "System Design")
fig("fig_system.png", "System architecture", 6.0)
P("The browser talks to a thin Flask layer through three endpoints. The layer decodes the input, calls the engine selected by `load_engine()` (U-Net if weights are present, otherwise the CPU ensemble), receives the probability, mask and uncertainty arrays, renders them as PNG overlays and returns JSON. All scientific code lives in the `btseg` package and has no dependency on Flask.")
table("Modules of the `btseg` package", ["Module", "Responsibility"],
      [["synthetic.py", "Procedural MRI phantom, dataset generator, z-score normalisation"],
       ["model.py", "U-Net, MC-Dropout and ensemble sampling, Dice + BCE loss (PyTorch)"],
       ["classical.py", "Feature extraction and bagged-tree ensemble (CPU back-end), brain-mask estimation"],
       ["uncertainty.py", "Entropy, aleatoric / epistemic decomposition, review score"],
       ["metrics.py", "Dice, IoU, HD95, ECE, Brier, error-AUROC, retained-Dice, boundary bands"],
       ["engine.py", "Back-end selection and unified `predict()` result dictionary"]],
      [1.5, 4.4], 10)
table("REST API", ["Method and path", "Purpose", "Response"],
      [["GET /", "Web interface", "HTML"], ["GET /api/health", "Liveness and active back-end", "{status, backend}"],
       ["GET /api/sample?seed=&noise=", "Generate and analyse a phantom", "{images{5 PNGs}, stats}"],
       ["POST /api/segment (file)", "Analyse an uploaded slice", "{images, stats} or 400 {error}"]],
      [2.1, 2.2, 1.6], 9.5)
section("3.5", "System Requirements")
table("Hardware requirements", ["Item", "Minimum (CPU back-end)", "Recommended (U-Net training)"],
      [["Processor", "2-core x86-64 CPU", "8-core CPU"], ["Memory", "4 GB RAM", "16 GB RAM"], ["GPU", "not required", "NVIDIA GPU, ≥ 8 GB VRAM"],
       ["Storage", "1 GB free (model cache ≈ 5 MB)", "20 GB+ for BraTS slices"], ["Display", "any modern browser", "—"]], [1.4, 2.2, 2.3], 10)
table("Software requirements", ["Component", "Version / note"],
      [["Operating system", "Linux, Windows 10/11 or macOS"], ["Python", "3.10 or newer"], ["Core libraries", "NumPy, SciPy, scikit-learn, scikit-image, OpenCV, Matplotlib, joblib"],
       ["Web", "Flask ≥ 3"], ["Deep learning (optional)", "PyTorch ≥ 2.1; nibabel for NIfTI"], ["Testing", "pytest"], ["Version control", "Git / GitHub"]], [2.0, 3.9], 10)
sub("Functional requirements")
bullets(["FR1 – accept an MRI slice or generate a synthetic case.", "FR2 – return a tumour mask.", "FR3 – return total, aleatoric and epistemic uncertainty maps.",
         "FR4 – return statistics and a review flag.", "FR5 – reject invalid input with a clear message."])
sub("Non-functional requirements")
bullets([f"NFR1 – performance: inference below one second per slice on a CPU (measured {T_INF:.2f} s).", "NFR2 – portability: no GPU or network needed at run-time.",
         "NFR3 – usability: usable on a 390 px-wide phone screen, light and dark theme.", "NFR4 – maintainability: separated scientific and web layers with automated tests.",
         "NFR5 – safety: research-use-only disclaimer on every view."])

# ============================================================== CHAPTER 4
chapter("Implementation")
section("4.1", "Technologies Used")
table("Technology stack", ["Technology", "Used for"],
      [["Python 3", "All code"], ["NumPy / SciPy", "Array maths, filtering, distance transforms, statistics (Spearman)"],
       ["scikit-learn", "Random-forest ensemble, ROC-AUC"], ["scikit-image", "Otsu thresholding baseline"], ["OpenCV", "Image decoding, resizing, PNG encoding"],
       ["PyTorch", "U-Net, MC Dropout, training loop (optional dependency)"], ["Matplotlib", "Figures and colour maps"], ["Flask", "Web server and REST API"],
       ["HTML / CSS / JavaScript", "Single-page responsive interface (no external libraries)"], ["pytest", "Automated testing"], ["python-docx", "Generation of this report"], ["Git / GitHub", "Version control and publication"]],
      [1.8, 4.1], 10)
sub("4.1.1 Data: synthetic MRI phantom")
P("Real BraTS volumes require registration and may not be redistributed, so a procedural generator (`synthetic.py`) produces unlimited labelled slices. Each slice has an elliptical brain with texture and dark ventricles, a multiplicative bias field, Rician noise (σ between 0.03 and 0.07 for training), a lesion with an irregular oedema region and ring-enhancing core, soft lesion edges produced by a sigmoid (partial-volume effect), 0–3 small bright non-tumour “distractor” lesions and a variable lesion contrast. The ground truth is the binary whole-tumour mask. About 15 % of slices contain no tumour. The phantom deliberately contains the two uncertainty sources of Section 1.2.3: blurred edges (aleatoric) and look-alike distractors (epistemic).")
sub("4.1.2 Pre-processing")
P("Each channel is z-scored inside the brain mask. For uploads, the brain mask is estimated by Gaussian smoothing, thresholding at 20 % of the 99th percentile, morphological opening, hole filling, selection of the largest connected component and erosion by three pixels to remove the skull rim.")
sub("4.1.3 U-Net with MC Dropout (model.py)")
P("The network has four down-sampling stages (16-32-64-128 channels, 256 in the bottleneck), transposed-convolution up-sampling and concatenating skip connections, 3×3 convolutions with batch normalisation and ReLU, and Dropout2d (p = 0.2) in every block. At test time `enable_dropout()` keeps the dropout layers stochastic while batch-norm remains in evaluation mode.")
code("""def enable_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

@torch.no_grad()
def mc_dropout_samples(model, x, T=20):
    enable_dropout(model)
    return np.stack([torch.sigmoid(model(x))[0, 0].cpu().numpy() for _ in range(T)])""")
sub("4.1.4 Training (train_unet.py)")
P("Training uses Adam (learning rate 1e-3, cosine decay), the combined BCE + soft-Dice loss, random flips and intensity jitter, batch size 16 and model selection by validation Dice. After training, the review threshold is set to the 80th percentile of the review scores on the validation split and stored in the checkpoint. The same script accepts BraTS slices prepared by `prepare_brats.py`.")
sub("4.1.5 CPU back-end (classical.py)")
P("To make the complete pipeline executable without a GPU, a second back-end uses 15 hand-crafted features per pixel (intensity, Gaussian-smoothed intensities at four scales, difference-of-Gaussians, gradient magnitudes, Laplacian-of-Gaussian, local standard deviations and position relative to the brain centroid) and a random forest of 64 bagged trees (depth ≤ 16, ≥ 10 samples per leaf). Each tree's soft prediction is one stochastic sample p_t(x), exactly as one MC-Dropout pass would be, so the same decomposition applies. Training pixels are drawn half from the neighbourhood of the lesion and half from the whole brain. On the development machine (2 CPU cores) training on 20 slices took " + f"{T_FIT:.0f} s and inference takes {T_INF:.2f} s per slice.")
sub("4.1.6 Uncertainty and review score (uncertainty.py)")
code("""def decompose(samples):                 # samples: (T, H, W)
    mean = samples.mean(0)
    total = binary_entropy(mean)
    aleatoric = binary_entropy(samples).mean(0)
    epistemic = np.clip(total - aleatoric, 0, None)
    return dict(prob=mean, total=total, aleatoric=aleatoric, epistemic=epistemic)""")
P("The image-level review score is the mean total entropy over the predicted tumour dilated by three pixels; it is zero when nothing is predicted.")
sub("4.1.7 Web application (app/)")
P("`app.py` exposes the three endpoints of Table 7, loads the engine lazily, and builds overlays with OpenCV: a red tumour contour on the contrast-stretched slice and magma heat-maps whose opacity grows with uncertainty so that certain regions stay clean. `index.html` is a dependency-free responsive page with a seed/noise control for synthetic cases, an upload form, a review banner, five image panels and a measurements grid. The first start trains and caches the CPU model in `models/classical_rf.joblib` (about 30 s); later starts are immediate.")
sub("4.1.8 Repository structure")
code("""brain-tumor-uncertainty-segmentation/
  src/btseg/      synthetic.py  model.py  classical.py  uncertainty.py  metrics.py  engine.py
  app/            app.py  templates/index.html
  scripts/        run_experiments.py  train_unet.py  prepare_brats.py  make_diagrams.py  build_report.py
  tests/          test_core.py  test_app.py  test_model_torch.py
  docs/           report (.docx)  results.json  figures/
  README.md  requirements.txt  requirements-torch.txt  Makefile  LICENSE""")

# ============================================================== CHAPTER 5
chapter("Result and Testing")
section("5.1", "Testing Methodology")
P(f"Two kinds of verification were performed. **Software testing** checks that the code does what it claims (mathematics, API, error handling). **Model evaluation** measures how well the system segments and how trustworthy its uncertainty is. Model evaluation used {CF['n_train']} training slices, {CF['n_val']} validation slices (to calibrate the review threshold) and {CF['n_test']} held-out test slices ({CF['n_tumor_slices']} with tumour, {CF['n_tumor_free']} tumour-free) generated with different random seeds, so that no test slice was seen during training. All numbers in this chapter are produced by `scripts/run_experiments.py` and stored in `docs/results.json`.")
P("**Important scope note.** The measurements below were obtained with the CPU bagged-tree back-end on synthetic phantoms. They demonstrate that the uncertainty pipeline behaves correctly and show the kind of analysis performed; they are *not* clinical performance figures and must not be compared directly with published BraTS results. The U-Net back-end is implemented and unit-tested only where PyTorch is available; its BraTS results are to be entered in Table 18 after training.".replace("*", ""))
section("5.2", "Types of Testing")
sub("5.2.1 Unit testing")
P("Seven unit tests cover the mathematical core: entropy bounds, absence of epistemic uncertainty when all samples agree, full epistemic uncertainty when confident samples disagree, Dice / IoU / HD95 on known masks, ECE of a perfectly calibrated predictor, phantom shapes and labels, and a zero review score for an empty prediction.")
sub("5.2.2 Integration and functional (API) testing")
P("Four tests exercise the Flask application with a small trained engine: health endpoint, sample endpoint (all five images and statistics present, entropy within [0,1]), upload of a `.npy` slice, and rejection of an unsupported file type or a missing file with HTTP 400.")
sub("5.2.3 Deep-learning component tests")
P("`test_model_torch.py` checks that the U-Net output has the input resolution, that MC-Dropout passes differ from each other, and that the loss decreases on a toy batch. These tests are skipped automatically when PyTorch is not installed (it was not available in the environment where the report experiments were run).")
table("Test summary", ["Test group", "Cases", "Result"],
      [["Core mathematics and data (test_core.py)", "7", "Passed"], ["Web API (test_app.py)", "4", "Passed"], ["PyTorch model (test_model_torch.py)", "2", "Module skipped – torch not installed in test environment"],
       ["Total run", "11", "11 passed, 0 failed"]], [3.0, 0.8, 2.1], 10)
sub("5.2.4 Usability and responsive testing")
P("The interface was exercised in a headless Chromium browser at 1280 px and 390 px widths (Section 5.3). Layout reflows to a single column on the phone-sized viewport, there is no horizontal scrolling, and the review banner, maps and measurements remain readable.")
sub("5.2.5 Performance testing")
P(f"With the CPU back-end, a 128×128 slice is segmented with uncertainty in {T_INF:.2f} s on two CPU cores; the first application start (training and caching of the model) takes about {T_FIT + 10:.0f} s. MC-Dropout inference with T = 20 passes costs roughly 20 forward passes of a small U-Net, which is practical on a CPU for single slices and fast on a GPU.")
sub("5.2.6 Model evaluation on held-out phantoms")
table("Segmentation accuracy on tumour-bearing test slices (n = " + str(CF['n_tumor_slices']) + ")", ["Method", "Dice (mean ± SD)", "IoU", "HD95 (px)"],
      [["Otsu thresholding of FLAIR", f"{f3(S['otsu']['dice'])} ± {f3(S['otsu']['dice_sd'])}", "–", "–"],
       ["Single decision tree", f"{f3(S['single_tree']['dice'])} ± {f3(S['single_tree']['dice_sd'])}", "–", "–"],
       ["Bagged-tree ensemble (proposed CPU back-end)", f"{f3(S['ensemble']['dice'])} ± {f3(S['ensemble']['dice_sd'])}", f3(S['ensemble']['iou']), f"{S['ensemble']['hd95']:.1f}"]],
      [2.6, 1.5, 0.8, 1.0], 10)
P(f"Averaging the trees improves Dice by {S['ensemble']['dice']-S['single_tree']['dice']:.3f} over a single tree and far outperforms global thresholding, because bright distractors and bias field defeat a single intensity threshold. On the {CF['n_tumor_free']} tumour-free slices, {R['tumor_free_false_positive_rate']*100:.0f} % received a false-positive region of more than 20 pixels (almost always on a bright distractor lesion) — a limitation discussed in Section 6.2 and the motivation for epistemic uncertainty: these false positives are exactly the regions that light up in the epistemic map (Figure 4, bottom row).")
fig("fig_examples.png", "Qualitative results with uncertainty maps", 6.1)
P("Figure 4 shows, from top to bottom, a good, a median and a poor segmentation and a tumour-free slice. In every row the uncertainty is concentrated at the lesion boundary (aleatoric: soft edge) and on look-alike structures (epistemic), while the homogeneous background is certain. The poorest case (third row) has a large uncertain region around the spurious detection.")
table("Calibration and uncertainty quality", ["Metric", "Ensemble", "Single tree"],
      [["Expected calibration error (lower is better)", f3(C['ece_ensemble']), f3(C['ece_single'])], ["Brier score (lower is better)", f3(C['brier_ensemble']), f3(C['brier_single'])],
       ["AUROC: total entropy → misclassified pixel", f3(U['auroc_total']), "–"], ["AUROC: epistemic (MI) → misclassified pixel", f3(U['auroc_epistemic']), "–"],
       ["AUROC: aleatoric → misclassified pixel", f3(U['auroc_aleatoric']), "–"]], [3.5, 1.2, 1.2], 10)
fig("fig_reliability.png", "Reliability diagram", 3.6)
P(f"The ensemble is better calibrated than a single tree (ECE {f3(C['ece_ensemble'])} vs {f3(C['ece_single'])}) and has a markedly lower Brier score. An AUROC of {f2(U['auroc_total'])} means that a randomly chosen wrong pixel has a higher entropy than a randomly chosen correct pixel about {U['auroc_total']*100:.0f} % of the time. Caveat: most errors lie on the lesion boundary, which is intrinsically uncertain, so part of this AUROC reflects ‘boundary-ness’; the experiment below isolates this effect.")
table("Where is the uncertainty? (mean predictive entropy, bits)", ["Region (relative to ground-truth lesion)", "Mean entropy"],
      [["Exterior (≥ 2 px outside)", f3(REG['exterior'])], ["Interior (≥ 2 px inside)", f3(REG['interior'])], ["Boundary band (±2 px)", f3(REG['boundary'])]], [3.6, 1.6], 10)
P(f"Uncertainty is about {REG['boundary']/REG['exterior']:.0f}× higher on the boundary than outside and twice as high as inside the lesion, matching the radiological intuition that edges are the hard part.")
table("Retained-pixel analysis: Dice after discarding the most uncertain pixels", ["Discarded (%)"] + [f"{float(k)*100:.0f}" for k in RET], [["Dice"] + [f3(v) for v in RET.values()]], None, 9.5)
fig("fig_retention.png", "Dice versus discarded uncertain pixels", 3.8)
P(f"Dice increases monotonically from {f3(RET['0.0'])} to {f3(RET['0.2'])} when the 20 % most uncertain pixels are ignored, showing that the uncertainty map ranks errors usefully.")
table("Image-level review score (risk–coverage)", ["Slices sent for review (%)", "Slices kept", "Mean Dice of kept tumour slices", "Kept slices with Dice < 0.80"],
      [[f"{float(k)*100:.0f}", v['n_kept'], f3(v['mean_dice']), v['n_below_080']] for k, v in RC.items()], [1.6, 1.1, 1.9, 1.5], 10)
fig("fig_review_vs_dice.png", "Review score against Dice", 3.8)
P(f"The review score correlates with Dice (Spearman ρ = {f2(FD['spearman_review_vs_dice'])}) and detects poorly segmented slices (Dice < 0.80, {FD['n_failures']} slices) with an AUROC of {f2(FD['auroc_review_for_failure'])}. Referring the 30 % highest-scoring slices removes four of five poor segmentations ({RC['0.0']['n_below_080']} → {RC['0.3']['n_below_080']}). Caveat: the score is partly confounded with lesion size, because small predicted lesions have proportionally more boundary. The binary flag at the validation-derived threshold ({CF['review_threshold']:.2f}) was conservative on this test set (it flagged {R['selective_prediction']['flagged_fraction']*100:.0f} % of slices, none of them tumour-bearing), so the threshold should be re-calibrated for each new data set; the ranking itself is the reliable part.")
table("Robustness to noise (distribution shift; model trained with σ ≤ 0.07; 40 tumour slices per level)", ["Noise σ", "Dice", "Total unc. (bits)", "Epistemic (bits)", "Aleatoric (bits)", "Flagged (%)"],
      [[f"{s['noise']:.2f}", f3(s['dice']), f3(s['total']), f3(s['epistemic']), f3(s['aleatoric']), f"{s['flagged']*100:.1f}"] for s in NS], None, 9.5)
fig("fig_noise_shift.png", "Uncertainty and Dice under increasing noise", 5.6)
P(f"Moving from σ = {NS[0]['noise']:.2f} to σ = {NS[-1]['noise']:.2f} lowers Dice from {f3(NS[0]['dice'])} to {f3(NS[-1]['dice'])} while the mean total uncertainty rises from {f3(NS[0]['total'])} to {f3(NS[-1]['total'])} (+{(NS[-1]['total']/NS[0]['total']-1)*100:.0f} %) and the share of flagged slices grows from {NS[0]['flagged']*100:.0f} % to {NS[-1]['flagged']*100:.1f} %. The trend is monotone in uncertainty, but Dice is not strictly monotone (e.g. σ = 0.10 scored higher than σ = 0.06) because only 40 slices were used per level; the shift response is also modest, which is expected for hand-crafted smoothed features that average noise out.")
sub("5.2.7 U-Net evaluation on BraTS (to be completed after training)")
P("Run `scripts/prepare_brats.py` and `scripts/train_unet.py` as described in the README, then `app/app.py --backend unet`, and record the values below. This table is intentionally left blank: no U-Net results are claimed in this report.")
table("U-Net + MC Dropout on BraTS whole tumour (to be filled in)", ["Metric", "Deterministic U-Net", "MC Dropout (T = 20)", "Deep ensemble (M = 5)"],
      [["Dice", "", "", ""], ["HD95", "", "", ""], ["ECE", "", "", ""], ["Uncertainty AUROC", "", "", ""]], [1.8, 1.4, 1.4, 1.3], 10)
section("5.3", "Screenshots of Webpage")
fig("ui_home.png", "Home screen", 5.0)
fig("ui_result.png", "Analysis result for a synthetic case", 5.4)
P("Figure 10 shows a typical result: the red contour (prediction) and green contour (ground truth) agree closely (Dice 0.95); total and epistemic uncertainty form a ring on the lesion edge and mark a small distractor; the banner reports uncertainty within the normal range.")
fig("ui_noisy.png", "Noisy case flagged for review", 5.4)
P("At a high noise level (σ = 0.20) the image is degraded, the uncertainty maps fill with high values, and the banner changes to “Review recommended”.")
fig("ui_mobile.png", "Mobile layout (390 px width, top of page)", 2.3)

# ============================================================== CHAPTER 6
chapter("Conclusion")
section("6.1", "Conclusion")
P("This project designed, implemented and tested an uncertainty-aware brain-tumour segmentation system. It couples a U-Net with Monte-Carlo Dropout or ensembling to produce, besides the mask, pixel-wise total, aleatoric and epistemic uncertainty maps and an image-level review score, and delivers them through a responsive web application and a documented, tested, reproducible code base published on GitHub.",
  f"Experiments with a CPU back-end on synthetic phantoms showed that the sampled ensemble segments more accurately and is better calibrated than a single predictor, that the uncertainty is concentrated at lesion boundaries and on look-alike structures, that discarding uncertain pixels raises Dice from {f3(RET['0.0'])} to {f3(RET['0.2'])}, that the review score ranks poor segmentations well (ρ = {f2(FD['spearman_review_vs_dice'])}), and that mean uncertainty increases as image quality degrades. The same pipeline is ready to be trained on real BraTS data; those results remain to be produced and are explicitly not claimed here.",
  "The work reinforces a central principle for medical AI: a model that knows what it does not know is more useful, and safer, than one that is merely accurate on average.")
section("6.2", "Future Work")
bullets(["Train and evaluate the U-Net on BraTS and compare MC Dropout, deep ensembles and test-time augmentation with Dice, HD95, ECE and uncertainty-AUROC (Table 18).",
         "Extend to multi-modal input (T1, T1ce, T2, FLAIR) and multi-class labels (oedema, enhancing core, necrosis) and to 3-D volumes.",
         "Reduce false positives on bright non-tumour lesions through multi-sequence input and hard-negative mining.",
         "Add post-hoc calibration (temperature scaling) and a review score that is normalised for lesion size.",
         "Support NIfTI/DICOM upload and slice scrolling in the web interface; add an uncertainty-guided correction tool.",
         "Evaluate on data from other institutions to study out-of-distribution behaviour, and conduct a reader study with radiologists.",
         "Containerise the application (Docker) and add continuous-integration testing on GitHub Actions."])

# ============================================================== CHAPTER 7
chapter("References")
refs = ["Abdar, M. et al. (2021). A review of uncertainty quantification in deep learning: techniques, applications and challenges. Information Fusion, 76, 243–297.",
 "Bakas, S. et al. (2017). Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. Scientific Data, 4, 170117.",
 "Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32.",
 "Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: learning dense volumetric segmentation from sparse annotation. MICCAI 2016.",
 "Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. ICML 2016, PMLR 48, 1050–1059.",
 "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML 2017, PMLR 70, 1321–1330.",
 "Hatamizadeh, A. et al. (2022). UNETR: Transformers for 3D medical image segmentation. WACV 2022.",
 "Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18, 203–211.",
 "Jungo, A., & Reyes, M. (2019). Assessing reliability and challenges of uncertainty estimations for medical image segmentation. MICCAI 2019.",
 "Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? NeurIPS 2017.",
 "Kingma, D. P., & Ba, J. (2015). Adam: a method for stochastic optimization. ICLR 2015.",
 "Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS 2017.",
 "Mehrtash, A., Wells, W. M., Tempany, C. M., Abolmaesumi, P., & Kapur, T. (2020). Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. IEEE Transactions on Medical Imaging, 39(12), 3868–3878.",
 "Menze, B. H. et al. (2015). The multimodal brain tumor image segmentation benchmark (BRATS). IEEE Transactions on Medical Imaging, 34(10), 1993–2024.",
 "Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: fully convolutional neural networks for volumetric medical image segmentation. 3DV 2016.",
 "Nair, T., Precup, D., Arnold, D. L., & Arbel, T. (2020). Exploring uncertainty measures in deep networks for multiple sclerosis lesion detection and segmentation. Medical Image Analysis, 59, 101557.",
 "Paszke, A. et al. (2019). PyTorch: an imperative style, high-performance deep learning library. NeurIPS 2019.",
 "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
 "Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015.",
 "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. JMLR, 15, 1929–1958.",
 "Wang, G., Li, W., Aertsen, M., Deprest, J., Ourselin, S., & Vercauteren, T. (2019). Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks. Neurocomputing, 338, 34–45."]
for i, r_ in enumerate(refs, 1):
    p = para(f"[{i}]  {r_}", align=WD_ALIGN_PARAGRAPH.LEFT); p.paragraph_format.left_indent = Inches(0.4); p.paragraph_format.first_line_indent = Inches(-0.4); p.paragraph_format.line_spacing = 1.15

# ---------------------------------------------------- fill lists of tables/figures
def fill(anchor, items, key):
    for it in items:
        p = anchor.insert_paragraph_before(); p.paragraph_format.tab_stops.add_tab_stop(Inches(5.95), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        p.paragraph_format.line_spacing = 1.15; p.paragraph_format.space_after = Pt(3)
        p.add_run(f"{it}\t{pg(it)}")
    anchor._p.getparent().remove(anchor._p)
fill(LOT_ANCHOR, LOT, "tab"); fill(LOF_ANCHOR, LOF, "fig")

doc.save(OUT)
Path("/tmp/claude-0/-home-claude/1a07952f-25d8-5a39-9e10-b1737da07fc5/scratchpad/anchors.json").write_text(
    json.dumps([t for _, t in [(0, x) for x in [a for a, _b in FRONT]]] + [t for _, t in BODY_STRUCT] + LOT + LOF))
print("saved", OUT, "tables", len(LOT), "figures", len(LOF))
