# =====================================================================================
# R7  Repository package for the second revision (Reviewer 2, comment 2)
# Markerless movement-quality study: builds one folder + zip on Google Drive that holds
# Supplementary Data S1, Supplementary Data S2 (with the second-round R6 sheets added),
# the feature dictionary, the 3,010-row selection manifest, all analysis scripts and
# notebooks, the R6 outputs, the graphical-abstract script, README, licences and
# Zenodo metadata. Nothing is deleted or modified in place. Raw source videos and
# motion-capture archives are never copied (they stay on Harvard Dataverse / Zenodo).
# Run in Google Colab after mounting Drive. Prints a QA summary with hard gates.
# =====================================================================================
import os, re, json, shutil, hashlib, zipfile, fnmatch, datetime, glob, time

GA_SCRIPT = '"""Graphical abstract v3 (simple): one message, one comparison per dataset.\nValues: Table 3 (MobiPhysio MAE), Table 4 (differences with 95% CI), Table 5 (REHAB24-6 person-weighted BA).\nDrawn with Matplotlib vector shapes; outputs PNG (600 dpi), PDF, SVG and runs a layout check."""\nimport json, sys\nimport numpy as np\nimport matplotlib\nmatplotlib.use(\'Agg\')\nimport matplotlib.pyplot as plt\nfrom matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle, FancyArrowPatch, Ellipse\nfrom matplotlib import font_manager\n\nfam = \'Inter\' if any(f.name == \'Inter\' for f in font_manager.fontManager.ttflist) else \'DejaVu Sans\'\nplt.rcParams.update({\'font.family\': fam, \'svg.fonttype\': \'none\', \'pdf.fonttype\': 42})\nOUT = sys.argv[1] if len(sys.argv) > 1 else \'Graphical_Abstract_R2\'\n\nBLUE, SKY, GREEN, VERM, ORANGE = \'#0072B2\', \'#56B4E9\', \'#009E73\', \'#D55E00\', \'#E69F00\'\nINK, MUTED, LINE, BG, NAVY, GREYBAR = \'#1B2631\', \'#5D6D7E\', \'#D5DCE3\', \'#F4F7FA\', \'#123B5C\', \'#A9B6C3\'\n\n# ---- values from the manuscript ----\nREHAB_BASE, REHAB_POSE = 0.538, 0.688      # Table 5, balanced accuracy per subject\nREHAB_NOTE = \'Gain +0.15 (95% CI 0.08 to 0.24); 10 subjects\'      # Table 4: .149 [.084, .237]\nMOBI_BASE, MOBI_POSE = 9.55, 9.55          # Table 3, mean absolute error (exercise median; best pose model)\nMOBI_NOTE = \'Difference −0.02 (95% CI −0.30 to 0.24); 24 people\'  # Table 4\n\nfig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)\nT = fig.transFigure\n\nASP = 7.5 / 13.333\ndef fcirc(x, y, r, **kw):\n    e = Ellipse((x, y), 2 * r * ASP, 2 * r, transform=T, **kw); fig.patches.append(e); return e\n\ndef box(x, y, w, h, fc=\'white\', ec=LINE, lw=1.0, r=0.014):\n    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f\'round,pad=0,rounding_size={r}\', transform=T,\n                                      fc=fc, ec=ec, lw=lw))\n\ndef txt(x, y, s, size=14, color=INK, weight=\'normal\', ha=\'left\', va=\'center\'):\n    return fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va)\n\n# ---------------- headline ----------------\ntxt(0.035, 0.925, \'Markerless pose estimation detects exercise errors,\', size=26, weight=\'bold\', color=NAVY)\ntxt(0.035, 0.862, \'but does not measure how well an exercise is performed\', size=26, weight=\'bold\', color=NAVY)\ntxt(0.035, 0.805, \'Two public rehabilitation video datasets; every person was left out of model training and tested as new\',\n    size=14, color=MUTED)\n\n# ---------------- left: method in three icons ----------------\nLX, LW, LY, LH = 0.035, 0.215, 0.215, 0.540\nbox(LX, LY, LW, LH)\nmid = LX + LW / 2\nIX, LBX = LX + 0.060, LX + 0.118\ntxt(mid, LY + LH - 0.045, \'How\', size=17, weight=\'bold\', ha=\'center\')\n\n# icon 1: camera\ncy = 0.615\nfig.patches.append(FancyBboxPatch((IX - 0.035, cy - 0.030), 0.050, 0.060,\n                                  boxstyle=\'round,pad=0,rounding_size=0.007\', transform=T, fc=NAVY, ec=\'none\'))\nfig.patches.append(Polygon([[IX + 0.015, cy - 0.010], [IX + 0.035, cy - 0.026], [IX + 0.035, cy + 0.026], [IX + 0.015, cy + 0.010]],\n                           closed=True, transform=T, fc=NAVY, ec=\'none\'))\nfcirc(IX - 0.010, cy, 0.017, fc=SKY, ec=\'white\', lw=1.5)\ntxt(LBX, cy, \'Video\', size=14, va=\'center\')\n\ndef down(y0, y1):\n    fig.patches.append(FancyArrowPatch((IX, y0), (IX, y1), transform=T, arrowstyle=\'-|>\',\n                                       mutation_scale=14, color=MUTED, lw=1.8))\ndown(0.578, 0.548)\n\n# icon 2: pose skeleton\nax = fig.add_axes([IX - 0.040, 0.405, 0.080, 0.140]); ax.set_axis_off(); ax.set_zorder(5); ax.patch.set_alpha(0)\nax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.1, 3.6); ax.set_aspect(\'equal\', adjustable=\'box\')\nP = {\'head\': (0.05, 3.25), \'sh\': (0.0, 2.75), \'el\': (0.55, 2.55), \'wr\': (1.05, 2.55), \'hip\': (-0.35, 1.65),\n     \'kn\': (0.35, 1.15), \'an\': (-0.05, 0.25), \'toe\': (0.35, 0.18)}\nfor a, b in [(\'sh\', \'el\'), (\'el\', \'wr\'), (\'sh\', \'hip\'), (\'hip\', \'kn\'), (\'kn\', \'an\'), (\'an\', \'toe\')]:\n    ax.plot(*zip(P[a], P[b]), color=BLUE, lw=3.2, solid_capstyle=\'round\', zorder=2)\nax.add_patch(Circle(P[\'head\'], 0.30, fc=\'white\', ec=BLUE, lw=2.6, zorder=3))\nfor k in [\'sh\', \'el\', \'wr\', \'hip\', \'kn\', \'an\', \'toe\']:\n    ax.plot(*P[k], \'o\', ms=6.5, mfc=ORANGE, mec=\'white\', mew=1.2, zorder=4)\ntxt(LBX, 0.475, \'Pose\', size=14, va=\'center\')\ndown(0.400, 0.370)\n\n# icon 3: people, one held out\ndef person(x, y, s, color):\n    fcirc(x, y + 0.62 * s, 0.17 * s, fc=color, ec=\'none\')\n    fig.patches.append(FancyBboxPatch((x - 0.21 * s, y - 0.10 * s), 0.42 * s, 0.50 * s,\n                                      boxstyle=f\'round,pad=0,rounding_size={0.16 * s}\', transform=T, fc=color, ec=\'none\'))\nfor i in range(4):\n    person(IX - 0.033 + i * 0.022, 0.318, 0.040, ORANGE if i == 3 else \'#B7C7D6\')\ntxt(LBX, 0.330, \'New person\', size=14, va=\'center\')\ntxt(mid, 0.268, \'Benchmark: knowing\', size=13, color=MUTED, ha=\'center\')\ntxt(mid, 0.238, \'only the exercise\', size=13, color=MUTED, ha=\'center\')\n\n# ---------------- right: two findings ----------------\ndef finding(x, w, accent, good, dataset, head, sub, rows, vmax, fmt, unit, note, chance=None):\n    y, h = LY, LH\n    box(x, y, w, h)\n    fig.patches.append(Rectangle((x, y + h - 0.012), w, 0.012, transform=T, fc=accent, ec=\'none\'))\n    # tick or cross badge\n    bx, by = x + 0.040, y + h - 0.080\n    fcirc(bx, by, 0.030, fc=accent, ec=\'none\')\n    if good:\n        fig.add_artist(plt.Line2D([bx - 0.011, bx - 0.002, bx + 0.012], [by, by - 0.012, by + 0.012],\n                                  transform=T, color=\'white\', lw=3.2, solid_capstyle=\'round\'))\n    else:\n        for dx in (1, -1):\n            fig.add_artist(plt.Line2D([bx - 0.010, bx + 0.010], [by - 0.010 * dx, by + 0.010 * dx], transform=T,\n                                      color=\'white\', lw=3.2, solid_capstyle=\'round\'))\n    txt(x + 0.072, by + 0.018, dataset, size=12.5, color=MUTED)\n    txt(x + 0.072, by - 0.020, head, size=17.5, weight=\'bold\', color=accent)\n    txt(x + 0.030, y + h - 0.160, sub, size=13, color=INK)\n    # two big bars\n    bar_x0, bar_w = x + 0.030, w - 0.060\n    for i, (lab, v, c) in enumerate(rows):\n        yy = y + h - 0.235 - i * 0.115\n        txt(bar_x0, yy + 0.032, lab, size=14, color=INK)\n        fig.patches.append(FancyBboxPatch((bar_x0, yy - 0.030), bar_w * v / vmax, 0.042,\n                                          boxstyle=\'round,pad=0,rounding_size=0.006\', transform=T, fc=c, ec=\'none\'))\n        txt(bar_x0 + bar_w * v / vmax + 0.010, yy - 0.009, fmt(v), size=18, weight=\'bold\', color=c if c != GREYBAR else MUTED)\n    if chance is not None:\n        cx = bar_x0 + bar_w * chance / vmax\n        top, bot = y + h - 0.185, y + h - 0.385\n        fig.add_artist(plt.Line2D([cx, cx], [bot, top], transform=T, color=INK, lw=1.3, ls=(0, (3, 2))))\n        txt(cx + 0.004, bot - 0.012, \'chance\', size=11, color=MUTED, va=\'top\')\n    txt(x + 0.030, y + 0.040, note, size=11, color=MUTED)\n\nFW = 0.345\nfinding(0.270, FW, GREEN, True, \'REHAB24-6  (correct vs instructed error)\', \'Detects errors in new people\',\n        \'Balanced accuracy (higher is better)\',\n        [(\'Exercise only\', REHAB_BASE, GREYBAR), (\'Pose model\', REHAB_POSE, GREEN)], 0.83,\n        lambda v: f\'{v * 100:.0f}%\', \'%\', REHAB_NOTE, chance=0.50)\nfinding(0.630, FW, VERM, False, \'MobiPhysio  (expert quality score 0–100)\', \'No better than exercise alone\',\n        \'Average error vs expert score (lower is better)\',\n        [(\'Exercise only\', MOBI_BASE, GREYBAR), (\'Pose model\', MOBI_POSE, VERM)], 12.5,\n        lambda v: f\'{v:.2f}\', \'points\', MOBI_NOTE)\n\n# ---------------- bottom message ----------------\nbox(0.035, 0.045, 0.940, 0.120, fc=NAVY, ec=\'none\', r=0.016)\ntxt(0.505, 0.105, \'Useful for flagging instructed errors, not for scoring movement quality or for autonomous clinical use\',\n    size=16, weight=\'bold\', color=\'white\', ha=\'center\')\n\nfig.savefig(OUT + \'.png\', dpi=600, facecolor=BG)\nfig.savefig(OUT + \'.pdf\', facecolor=BG)\nfig.savefig(OUT + \'.svg\', facecolor=BG)\n\n# ---------------- layout check ----------------\nfig.canvas.draw(); rend = fig.canvas.get_renderer()\ntexts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]\nfw, fh = fig.bbox.width, fig.bbox.height\nbbs = [(t.get_text(), t.get_window_extent(rend), t.get_fontsize()) for t in texts]\nover = [(a[0][:28], b[0][:28]) for i, a in enumerate(bbs) for b in bbs[i + 1:]\n        if a[1].x0 < b[1].x1 - 1 and b[1].x0 < a[1].x1 - 1 and a[1].y0 < b[1].y1 - 1 and b[1].y0 < a[1].y1 - 1]\nsmall = [(s[:28], f) for s, _, f in bbs if f < 11]\npanels = [(LX, LY, LW, LH), (0.270, LY, FW, LH), (0.630, LY, FW, LH), (0.035, 0.045, 0.940, 0.120)]\nspill = []\nfor s, bb, _ in bbs:\n    cx, cy = (bb.x0 + bb.x1) / 2 / fw, (bb.y0 + bb.y1) / 2 / fh\n    for px, py, pw, ph in panels:\n        if px <= cx <= px + pw and py <= cy <= py + ph and (bb.x0 / fw < px + 0.004 or bb.x1 / fw > px + pw - 0.004):\n            spill.append(s[:28])\noutside = [s[:28] for s, bb, _ in bbs if bb.x0 < 0 or bb.x1 > fw or bb.y0 < 0 or bb.y1 > fh]\nres = {\'font\': fam, \'overlaps\': over, \'fonts_below_11pt\': small, \'spill\': spill, \'outside\': outside,\n       \'pass\': not (over or small or spill or outside)}\nprint(json.dumps(res, ensure_ascii=False))\n'

# ----------------------------- configuration -----------------------------------------
IN_COLAB = os.path.exists('/content')
if IN_COLAB and not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')
MYDRIVE = os.environ.get('R7_MYDRIVE', '/content/drive/MyDrive')
ROOT = os.path.join(MYDRIVE, 'Markerless_Movement_Quality_Study')
NOTEBOOK_DIR = os.path.join(MYDRIVE, 'Colab Notebooks')
NOTEBOOKS = ['MARKERLESS 2.ipynb', 'Rev sports 28-10-26.ipynb']      # original + revision notebooks
OUT_DIR = os.path.join(ROOT, '06_repository')
PKG_NAME = 'Markerless_R2_repository'
# Optional manual overrides (leave None for automatic search)
S1_PATH = None
S2_PATH = None
EXCLUDE_DIRS = {'01_raw', '02_pose', 'raw', 'videos', '.ipynb_checkpoints', '06_repository'}
EXCLUDE_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.npz', '.npy', '.parquet', '.h5', '.pkl', '.task'}
MAX_FILE_MB = 200
SEARCH_DEPTH = 4                                                  # depth for Drive-wide script search

t0 = time.time()
gates, notes = {}, []
def gate(name, ok, detail=''):
    gates[name] = {'pass': bool(ok), 'detail': str(detail)}

def walk(top, depth=99):
    top = os.path.abspath(top)
    base = top.rstrip(os.sep).count(os.sep)
    for d, dirs, files in os.walk(top):
        dirs[:] = [x for x in dirs if x not in EXCLUDE_DIRS and not x.startswith('.')]
        if d.count(os.sep) - base >= depth:
            dirs[:] = []
        for f in files:
            yield os.path.join(d, f)

def newest(paths):
    paths = [p for p in paths if os.path.isfile(p)]
    return max(paths, key=os.path.getmtime) if paths else None

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

gate('root_exists', os.path.isdir(ROOT), ROOT)
if not os.path.isdir(ROOT):
    raise SystemExit('Project folder not found: ' + ROOT)
root_files = list(walk(ROOT))

def find(patterns, pool=None):
    pool = root_files if pool is None else pool
    hits = [p for p in pool if any(fnmatch.fnmatch(os.path.basename(p).lower(), pat.lower()) for pat in patterns)]
    return newest(hits), hits

# ----------------------------- locate inputs -----------------------------------------
s2, s2_all = (S2_PATH, [S2_PATH]) if S2_PATH else find(['Supplementary_Data_S2*.xlsx'])
s1, s1_all = (S1_PATH, [S1_PATH]) if S1_PATH else find(['Supplementary_Data_S1*'])
if s1 is None:
    s1, s1_all = find(['*MANUSCRIPT_FIGURES_TABLES*.zip'])
    if s1: notes.append('Supplementary Data S1 taken from the original submission archive: ' + os.path.basename(s1))
fdict, _ = find(['R2a_FEATURE_DICTIONARY.csv', '*FEATURE_DICTIONARY*.csv'])
manifest, _ = find(['R2a_MOBI_SELECTION_MANIFEST.csv', '*SELECTION_MANIFEST*.csv'])
r6_dir = next((os.path.dirname(p) for p in root_files if os.path.basename(p) == 'R6_MANIFEST.json'
               and os.path.basename(os.path.dirname(p)) == 'R6_second_round'), None)
r6_files = sorted(glob.glob(os.path.join(r6_dir, 'R6_*'))) if r6_dir else []
R6_REQUIRED = ['R6_CALIBRATION_SUBJECT_BOOTSTRAP.csv', 'R6_CALIBRATION_RAW_VS_RECALIBRATED.csv',
               'R6_CALIBRATION_PER_SUBJECT.csv', 'R6_VIEW_STANDARDIZED_DIFFERENCE_CI.csv',
               'R6_BLAND_ALTMAN_CLUSTERED.csv', 'R6_ROM_AGREEMENT_SUBJECT_BOOTSTRAP.csv',
               'R6_MANIFEST.json', 'R6_QA_SUMMARY.json']
code_pool = root_files + [p for p in walk(MYDRIVE, SEARCH_DEPTH) if p.endswith('.py')]
scripts = sorted({p for p in code_pool if p.endswith('.py')
                  and re.match(r'^(R\d+[a-z]?_|section|analysis_section|S\d+_|CP\d+_|markerless)', os.path.basename(p), re.I)})
notebooks = [os.path.join(NOTEBOOK_DIR, n) for n in NOTEBOOKS if os.path.isfile(os.path.join(NOTEBOOK_DIR, n))]
qa_jsons = sorted({p for p in root_files if re.match(r'^R\d+[a-z]?_(MANIFEST|QA_SUMMARY)\.json$', os.path.basename(p))})

gate('S1_found', s1 is not None, s1 or 'none; set S1_PATH')
gate('S2_found', s2 is not None, s2 or 'none; set S2_PATH')
gate('feature_dictionary_found', fdict is not None, fdict)
gate('selection_manifest_found', manifest is not None, manifest)
gate('R6_outputs_complete', all(any(os.path.basename(f) == r for f in r6_files) for r in R6_REQUIRED),
     f'{len(r6_files)} files in {r6_dir}')
gate('scripts_found', len(scripts) >= 1, f'{len(scripts)} .py files')
gate('notebooks_found', len(notebooks) == len(NOTEBOOKS), [os.path.basename(n) for n in notebooks])

# content checks on the dictionary and manifest
import pandas as pd
if fdict:
    fd = pd.read_csv(fdict)
    gate('feature_dictionary_rows>=147', len(fd) >= 147, f'{len(fd)} rows, columns {list(fd.columns)[:6]}')
if manifest:
    mf = pd.read_csv(manifest)
    gate('selection_manifest_rows==3010', len(mf) == 3010, f'{len(mf)} rows')

hard = [k for k, v in gates.items() if not v['pass']]
if hard:
    print(json.dumps(gates, indent=1))
    raise SystemExit('R7 stopped: failed gates ' + ', '.join(hard))

# ----------------------------- assemble package --------------------------------------
stamp = datetime.datetime.utcnow().strftime('%Y%m%d')
pkg = os.path.join(OUT_DIR, f'{PKG_NAME}_{stamp}')
if os.path.exists(pkg):
    pkg = pkg + '_' + datetime.datetime.utcnow().strftime('%H%M%S')
for sub in ['data', 'data/revision2_R6', 'code/scripts', 'code/notebooks', 'code/graphical_abstract', 'qa']:
    os.makedirs(os.path.join(pkg, sub), exist_ok=True)

def put(src, rel):
    dst = os.path.join(pkg, rel)
    if os.path.getsize(src) > MAX_FILE_MB * 1e6:
        notes.append(f'skipped (>{MAX_FILE_MB} MB): {src}'); return None
    shutil.copy2(src, dst); return dst

ext1 = os.path.splitext(s1)[1]
put(s1, 'data/Supplementary_Data_S1' + ext1)
# S2 v2: original workbook plus the second-round sheets (original sheets untouched)
s2v2 = os.path.join(pkg, 'data', 'Supplementary_Data_S2_v2_second_revision.xlsx')
shutil.copy2(s2, s2v2)
from openpyxl import load_workbook
wb = load_workbook(s2v2)
old_sheets = list(wb.sheetnames)
sheet_map = {'R6_CALIBRATION_SUBJECT_BOOTSTRAP.csv': 'R6_calibration_CI',
             'R6_CALIBRATION_RAW_VS_RECALIBRATED.csv': 'R6_calibration_raw_recal',
             'R6_CALIBRATION_PER_SUBJECT.csv': 'R6_calibration_per_subject',
             'R6_VIEW_STANDARDIZED_DIFFERENCE_CI.csv': 'R6_view_standardized_CI',
             'R6_BLAND_ALTMAN_CLUSTERED.csv': 'R6_BlandAltman_clustered',
             'R6_ROM_AGREEMENT_SUBJECT_BOOTSTRAP.csv': 'R6_ROM_agreement_CI'}
wb.close()
with pd.ExcelWriter(s2v2, engine='openpyxl', mode='a', if_sheet_exists='error') as xw:
    for fname, sheet in sheet_map.items():
        pd.read_csv(os.path.join(r6_dir, fname)).to_excel(xw, sheet_name=sheet, index=False)
wb = load_workbook(s2v2, read_only=True)
gate('S2_v2_keeps_original_sheets', all(s in wb.sheetnames for s in old_sheets), f'{len(old_sheets)} original sheets')
gate('S2_v2_has_R6_sheets', all(s in wb.sheetnames for s in sheet_map.values()), list(sheet_map.values()))
wb.close()
put(fdict, 'data/feature_dictionary.csv')
put(manifest, 'data/MobiPhysio_selection_manifest_3010.csv')
for f in r6_files:
    put(f, 'data/revision2_R6/' + os.path.basename(f))
for f in scripts:
    rel = 'code/scripts/' + os.path.basename(f)
    if os.path.exists(os.path.join(pkg, rel)):
        rel = 'code/scripts/' + os.path.basename(os.path.dirname(f)) + '__' + os.path.basename(f)
    put(f, rel)
for f in notebooks:
    put(f, 'code/notebooks/' + os.path.basename(f).replace(' ', '_'))
for f in qa_jsons:
    put(f, 'qa/' + os.path.basename(f))
with open(os.path.join(pkg, 'code/graphical_abstract/make_graphical_abstract.py'), 'w') as fh:
    fh.write(GA_SCRIPT)

# ----------------------------- README, licences, metadata ----------------------------
README = f"""# Markerless pose estimation, instructed exercise errors and graded movement quality: data and code

Repository package for the manuscript "Markerless Pose Estimation Distinguishes Instructed Exercise Errors
but Not Graded Movement Quality in Two Public Rehabilitation Video Datasets" (second revision, {stamp}).

## Contents
| Path | Manuscript item | Description |
|---|---|---|
| data/Supplementary_Data_S1{ext1} | Supplementary Data S1 | Figure source tables and row-level model outputs of the original analysis |
| data/Supplementary_Data_S2_v2_second_revision.xlsx | Supplementary Data S2 | All tables of the revision analyses; sheets starting with R6_ were added in the second revision |
| data/feature_dictionary.csv | Supplementary Data S2 (feature dictionary) | Landmarks, coordinate space, units, formula and normalization for every pose descriptor |
| data/MobiPhysio_selection_manifest_3010.csv | Supplementary Data S2 (selection manifest) | Selection key, within-stratum rank and selection flag for all 3,010 scored MobiPhysio videos |
| data/revision2_R6/ | Tables 6, 8, 10 | Second-revision outputs: subject-cluster intervals for calibration and the standardized view difference, repetition-clustered Bland-Altman limits, QA gates |
| code/scripts/ | Supplementary Software S1 | Restartable Colab scripts, one per analysis section |
| code/notebooks/ | Supplementary Software S1 | Colab notebooks of the original analysis and of the revisions, with saved outputs |
| code/graphical_abstract/ | Graphical abstract | Script that draws the graphical abstract from the tabulated values |
| qa/ | Reproducibility | Manifests with input and output SHA-256 checksums and QA summaries of each analysis section |
| MANIFEST.csv | | SHA-256 checksum and size of every file in this package |

## Source data (not redistributed here)
- MobiPhysio: Harvard Dataverse, https://doi.org/10.7910/DVN/XSI0QN (CC BY 4.0)
- REHAB24-6: Zenodo, https://doi.org/10.5281/zenodo.13305826 (academic/non-profit, noncommercial use)
Raw videos and motion-capture archives must be downloaded from these records.

## Reproducing the analysis
1. Download the source datasets into Markerless_Movement_Quality_Study/01_raw/ on Google Drive.
2. Open the notebooks in code/notebooks/ in Google Colab and run the sections in order; each section
   checks the SHA-256 checksums of its inputs and stops if a check fails.
3. Software versions, random seeds and search grids are listed in the manuscript (Section 2.10) and in
   the section manifests in qa/.

## Licences
Code (code/): MIT licence (LICENSE). Derived data (data/): CC BY-NC 4.0 (LICENSE-DATA), because part of
the data is derived from REHAB24-6, which is released for noncommercial research only.
"""
open(os.path.join(pkg, 'README.md'), 'w').write(README)
open(os.path.join(pkg, 'LICENSE'), 'w').write(f"""MIT License

Copyright (c) {datetime.datetime.utcnow().year} The authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")
open(os.path.join(pkg, 'LICENSE-DATA'), 'w').write(
    'The files in data/ are licensed under the Creative Commons Attribution-NonCommercial 4.0 International '
    'licence (CC BY-NC 4.0): https://creativecommons.org/licenses/by-nc/4.0/legalcode\n')
zen = {'upload_type': 'dataset',
       'title': 'Data and code for: Markerless Pose Estimation Distinguishes Instructed Exercise Errors but Not '
                'Graded Movement Quality in Two Public Rehabilitation Video Datasets',
       'description': 'Derived feature tables, split assignments, selection manifest, out-of-fold predictions, '
                      'revision analyses and Google Colab code for a participant-independent evaluation of '
                      'markerless pose descriptors in the public MobiPhysio and REHAB24-6 datasets.',
       'creators': [{'name': 'AUTHOR, Surname Given'}],
       'license': 'cc-by-nc-4.0', 'access_right': 'open',
       'keywords': ['markerless motion capture', 'pose estimation', 'rehabilitation', 'MediaPipe', 'REHAB24-6',
                    'MobiPhysio', 'participant-independent validation'],
       'related_identifiers': [{'identifier': '10.7910/DVN/XSI0QN', 'relation': 'isDerivedFrom', 'scheme': 'doi'},
                               {'identifier': '10.5281/zenodo.13305826', 'relation': 'isDerivedFrom', 'scheme': 'doi'}]}
json.dump(zen, open(os.path.join(pkg, '.zenodo.json'), 'w'), indent=2)

# ----------------------------- manifest, zip, final gates ----------------------------
rows = []
for d, _, files in os.walk(pkg):
    for f in files:
        p = os.path.join(d, f)
        rows.append({'path': os.path.relpath(p, pkg), 'bytes': os.path.getsize(p), 'sha256': sha256(p)})
pd.DataFrame(rows).sort_values('path').to_csv(os.path.join(pkg, 'MANIFEST.csv'), index=False)
bad_ext = [r['path'] for r in rows if os.path.splitext(r['path'])[1].lower() in EXCLUDE_EXT]
gate('no_raw_media_or_binary_arrays', not bad_ext, bad_ext[:5])
zpath = pkg + '.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for d, _, files in os.walk(pkg):
        for f in files:
            p = os.path.join(d, f)
            z.write(p, os.path.join(os.path.basename(pkg), os.path.relpath(p, pkg)))
with zipfile.ZipFile(zpath) as z:
    gate('zip_integrity', z.testzip() is None, f'{len(z.namelist())} members')
total_mb = sum(r['bytes'] for r in rows) / 1e6
gate('package_under_50GB', total_mb < 50000, f'{total_mb:.1f} MB')

qa = {'section': 'R7_repository_package', 'utc': datetime.datetime.utcnow().isoformat(),
      'package_dir': pkg, 'zip': zpath, 'n_files': len(rows), 'total_mb': round(total_mb, 2),
      'inputs': {'S1': s1, 'S2': s2, 'feature_dictionary': fdict, 'selection_manifest': manifest,
                 'R6_dir': r6_dir, 'n_scripts': len(scripts), 'notebooks': notebooks},
      'gates': gates, 'notes': notes, 'all_gates_pass': all(v['pass'] for v in gates.values()),
      'runtime_sec': round(time.time() - t0, 1)}
json.dump(qa, open(os.path.join(OUT_DIR, f'R7_QA_SUMMARY_{stamp}.json'), 'w'), indent=1)
print(json.dumps(qa, indent=1))
if not qa['all_gates_pass']:
    raise SystemExit('R7 finished with failed gates; see above')
print('\nR7 complete. Upload', zpath, 'to Zenodo (see README in the package).')
