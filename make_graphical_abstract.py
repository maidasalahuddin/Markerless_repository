"""Graphical abstract v3 (simple): one message, one comparison per dataset.
Values: Table 3 (MobiPhysio MAE), Table 4 (differences with 95% CI), Table 5 (REHAB24-6 person-weighted BA).
Drawn with Matplotlib vector shapes; outputs PNG (600 dpi), PDF, SVG and runs a layout check."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle, FancyArrowPatch, Ellipse
from matplotlib import font_manager

fam = 'Inter' if any(f.name == 'Inter' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': fam, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
OUT = sys.argv[1] if len(sys.argv) > 1 else 'Graphical_Abstract_R2'

BLUE, SKY, GREEN, VERM, ORANGE = '#0072B2', '#56B4E9', '#009E73', '#D55E00', '#E69F00'
INK, MUTED, LINE, BG, NAVY, GREYBAR = '#1B2631', '#5D6D7E', '#D5DCE3', '#F4F7FA', '#123B5C', '#A9B6C3'

# ---- values from the manuscript ----
REHAB_BASE, REHAB_POSE = 0.538, 0.688      # Table 5, balanced accuracy per subject
REHAB_NOTE = 'Gain +0.15 (95% CI 0.08 to 0.24); 10 subjects'      # Table 4: .149 [.084, .237]
MOBI_BASE, MOBI_POSE = 9.55, 9.55          # Table 3, mean absolute error (exercise median; best pose model)
MOBI_NOTE = 'Difference −0.02 (95% CI −0.30 to 0.24); 24 people'  # Table 4

fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
T = fig.transFigure

ASP = 7.5 / 13.333
def fcirc(x, y, r, **kw):
    e = Ellipse((x, y), 2 * r * ASP, 2 * r, transform=T, **kw); fig.patches.append(e); return e

def box(x, y, w, h, fc='white', ec=LINE, lw=1.0, r=0.014):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', transform=T,
                                      fc=fc, ec=ec, lw=lw))

def txt(x, y, s, size=14, color=INK, weight='normal', ha='left', va='center'):
    return fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va)

# ---------------- headline ----------------
txt(0.035, 0.925, 'Markerless pose estimation detects exercise errors,', size=26, weight='bold', color=NAVY)
txt(0.035, 0.862, 'but does not measure how well an exercise is performed', size=26, weight='bold', color=NAVY)
txt(0.035, 0.805, 'Two public rehabilitation video datasets; every person was left out of model training and tested as new',
    size=14, color=MUTED)

# ---------------- left: method in three icons ----------------
LX, LW, LY, LH = 0.035, 0.215, 0.215, 0.540
box(LX, LY, LW, LH)
mid = LX + LW / 2
IX, LBX = LX + 0.060, LX + 0.118
txt(mid, LY + LH - 0.045, 'How', size=17, weight='bold', ha='center')

# icon 1: camera
cy = 0.615
fig.patches.append(FancyBboxPatch((IX - 0.035, cy - 0.030), 0.050, 0.060,
                                  boxstyle='round,pad=0,rounding_size=0.007', transform=T, fc=NAVY, ec='none'))
fig.patches.append(Polygon([[IX + 0.015, cy - 0.010], [IX + 0.035, cy - 0.026], [IX + 0.035, cy + 0.026], [IX + 0.015, cy + 0.010]],
                           closed=True, transform=T, fc=NAVY, ec='none'))
fcirc(IX - 0.010, cy, 0.017, fc=SKY, ec='white', lw=1.5)
txt(LBX, cy, 'Video', size=14, va='center')

def down(y0, y1):
    fig.patches.append(FancyArrowPatch((IX, y0), (IX, y1), transform=T, arrowstyle='-|>',
                                       mutation_scale=14, color=MUTED, lw=1.8))
down(0.578, 0.548)

# icon 2: pose skeleton
ax = fig.add_axes([IX - 0.040, 0.405, 0.080, 0.140]); ax.set_axis_off(); ax.set_zorder(5); ax.patch.set_alpha(0)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.1, 3.6); ax.set_aspect('equal', adjustable='box')
P = {'head': (0.05, 3.25), 'sh': (0.0, 2.75), 'el': (0.55, 2.55), 'wr': (1.05, 2.55), 'hip': (-0.35, 1.65),
     'kn': (0.35, 1.15), 'an': (-0.05, 0.25), 'toe': (0.35, 0.18)}
for a, b in [('sh', 'el'), ('el', 'wr'), ('sh', 'hip'), ('hip', 'kn'), ('kn', 'an'), ('an', 'toe')]:
    ax.plot(*zip(P[a], P[b]), color=BLUE, lw=3.2, solid_capstyle='round', zorder=2)
ax.add_patch(Circle(P['head'], 0.30, fc='white', ec=BLUE, lw=2.6, zorder=3))
for k in ['sh', 'el', 'wr', 'hip', 'kn', 'an', 'toe']:
    ax.plot(*P[k], 'o', ms=6.5, mfc=ORANGE, mec='white', mew=1.2, zorder=4)
txt(LBX, 0.475, 'Pose', size=14, va='center')
down(0.400, 0.370)

# icon 3: people, one held out
def person(x, y, s, color):
    fcirc(x, y + 0.62 * s, 0.17 * s, fc=color, ec='none')
    fig.patches.append(FancyBboxPatch((x - 0.21 * s, y - 0.10 * s), 0.42 * s, 0.50 * s,
                                      boxstyle=f'round,pad=0,rounding_size={0.16 * s}', transform=T, fc=color, ec='none'))
for i in range(4):
    person(IX - 0.033 + i * 0.022, 0.318, 0.040, ORANGE if i == 3 else '#B7C7D6')
txt(LBX, 0.330, 'New person', size=14, va='center')
txt(mid, 0.268, 'Benchmark: knowing', size=13, color=MUTED, ha='center')
txt(mid, 0.238, 'only the exercise', size=13, color=MUTED, ha='center')

# ---------------- right: two findings ----------------
def finding(x, w, accent, good, dataset, head, sub, rows, vmax, fmt, unit, note, chance=None):
    y, h = LY, LH
    box(x, y, w, h)
    fig.patches.append(Rectangle((x, y + h - 0.012), w, 0.012, transform=T, fc=accent, ec='none'))
    # tick or cross badge
    bx, by = x + 0.040, y + h - 0.080
    fcirc(bx, by, 0.030, fc=accent, ec='none')
    if good:
        fig.add_artist(plt.Line2D([bx - 0.011, bx - 0.002, bx + 0.012], [by, by - 0.012, by + 0.012],
                                  transform=T, color='white', lw=3.2, solid_capstyle='round'))
    else:
        for dx in (1, -1):
            fig.add_artist(plt.Line2D([bx - 0.010, bx + 0.010], [by - 0.010 * dx, by + 0.010 * dx], transform=T,
                                      color='white', lw=3.2, solid_capstyle='round'))
    txt(x + 0.072, by + 0.018, dataset, size=12.5, color=MUTED)
    txt(x + 0.072, by - 0.020, head, size=17.5, weight='bold', color=accent)
    txt(x + 0.030, y + h - 0.160, sub, size=13, color=INK)
    # two big bars
    bar_x0, bar_w = x + 0.030, w - 0.060
    for i, (lab, v, c) in enumerate(rows):
        yy = y + h - 0.235 - i * 0.115
        txt(bar_x0, yy + 0.032, lab, size=14, color=INK)
        fig.patches.append(FancyBboxPatch((bar_x0, yy - 0.030), bar_w * v / vmax, 0.042,
                                          boxstyle='round,pad=0,rounding_size=0.006', transform=T, fc=c, ec='none'))
        txt(bar_x0 + bar_w * v / vmax + 0.010, yy - 0.009, fmt(v), size=18, weight='bold', color=c if c != GREYBAR else MUTED)
    if chance is not None:
        cx = bar_x0 + bar_w * chance / vmax
        top, bot = y + h - 0.185, y + h - 0.385
        fig.add_artist(plt.Line2D([cx, cx], [bot, top], transform=T, color=INK, lw=1.3, ls=(0, (3, 2))))
        txt(cx + 0.004, bot - 0.012, 'chance', size=11, color=MUTED, va='top')
    txt(x + 0.030, y + 0.040, note, size=11, color=MUTED)

FW = 0.345
finding(0.270, FW, GREEN, True, 'REHAB24-6  (correct vs instructed error)', 'Detects errors in new people',
        'Balanced accuracy (higher is better)',
        [('Exercise only', REHAB_BASE, GREYBAR), ('Pose model', REHAB_POSE, GREEN)], 0.83,
        lambda v: f'{v * 100:.0f}%', '%', REHAB_NOTE, chance=0.50)
finding(0.630, FW, VERM, False, 'MobiPhysio  (expert quality score 0–100)', 'No better than exercise alone',
        'Average error vs expert score (lower is better)',
        [('Exercise only', MOBI_BASE, GREYBAR), ('Pose model', MOBI_POSE, VERM)], 12.5,
        lambda v: f'{v:.2f}', 'points', MOBI_NOTE)

# ---------------- bottom message ----------------
box(0.035, 0.045, 0.940, 0.120, fc=NAVY, ec='none', r=0.016)
txt(0.505, 0.105, 'Useful for flagging instructed errors, not for scoring movement quality or for autonomous clinical use',
    size=16, weight='bold', color='white', ha='center')

fig.savefig(OUT + '.png', dpi=600, facecolor=BG)
fig.savefig(OUT + '.pdf', facecolor=BG)
fig.savefig(OUT + '.svg', facecolor=BG)

# ---------------- layout check ----------------
fig.canvas.draw(); rend = fig.canvas.get_renderer()
texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
fw, fh = fig.bbox.width, fig.bbox.height
bbs = [(t.get_text(), t.get_window_extent(rend), t.get_fontsize()) for t in texts]
over = [(a[0][:28], b[0][:28]) for i, a in enumerate(bbs) for b in bbs[i + 1:]
        if a[1].x0 < b[1].x1 - 1 and b[1].x0 < a[1].x1 - 1 and a[1].y0 < b[1].y1 - 1 and b[1].y0 < a[1].y1 - 1]
small = [(s[:28], f) for s, _, f in bbs if f < 11]
panels = [(LX, LY, LW, LH), (0.270, LY, FW, LH), (0.630, LY, FW, LH), (0.035, 0.045, 0.940, 0.120)]
spill = []
for s, bb, _ in bbs:
    cx, cy = (bb.x0 + bb.x1) / 2 / fw, (bb.y0 + bb.y1) / 2 / fh
    for px, py, pw, ph in panels:
        if px <= cx <= px + pw and py <= cy <= py + ph and (bb.x0 / fw < px + 0.004 or bb.x1 / fw > px + pw - 0.004):
            spill.append(s[:28])
outside = [s[:28] for s, bb, _ in bbs if bb.x0 < 0 or bb.x1 > fw or bb.y0 < 0 or bb.y1 > fh]
res = {'font': fam, 'overlaps': over, 'fonts_below_11pt': small, 'spill': spill, 'outside': outside,
       'pass': not (over or small or spill or outside)}
print(json.dumps(res, ensure_ascii=False))
