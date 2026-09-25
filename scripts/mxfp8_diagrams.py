"""Generate the MXFP8 post diagrams (public/blog/mxfp8/*.svg) in the house style.

Run: python3 scripts/mxfp8_diagrams.py  (needs fonttools + brotli). Embeds a subset of
public/fonts/Excalifont-Regular.ttf as woff2, since SVGs in <img> cannot fetch fonts.
"""
import base64, io, os, re
from html import escape
from fontTools.ttLib import TTFont
from fontTools import subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f'{REPO}/public/blog/mxfp8'

INK, SUB, FAINT = '#1e1e1e', '#495057', '#868e96'
RED, BLUE_T, GREEN_T = '#c92a2a', '#1971c2', '#2b8a3e'
SIGN, EXP, MANT, SCALE = '#ffd8a8', '#a5d8ff', '#b2f2bb', '#d0bfff'
GRAY, YEL, PINK = '#e9ecef', '#ffec99', '#ffc9c9'

def rich(s):
    """'2^{-127}' -> raised exponent tspans. Everything else escaped."""
    out, i = [], 0
    for m in re.finditer(r'\^\{([^}]*)\}', s):
        out.append(escape(s[i:m.start()]))
        out.append(f'<tspan baseline-shift="super" font-size="68%">{escape(m.group(1))}</tspan>')
        i = m.end()
    out.append(escape(s[i:]))
    return ''.join(out)

class Fig:
    def __init__(s, w, h):
        s.w, s.h, s.el, s.chars = w, h, [], set()
    def text(s, x, y, t, size=16, fill=INK, anchor='start', weight=None, italic=False):
        s.chars |= set(re.sub(r'[\^{}]', '', t))
        a = f' text-anchor="{anchor}"' if anchor != 'start' else ''
        wt = f' font-weight="{weight}"' if weight else ''
        it = ' font-style="italic"' if italic else ''
        s.el.append(f'<text x="{x:g}" y="{y:g}" font-size="{size}" fill="{fill}"{a}{wt}{it}>{rich(t)}</text>')
    def rect(s, x, y, w, h, fill='#ffffff', stroke=INK, sw=2, rx=8, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        o = f' fill-opacity="{opacity}"' if opacity else ''
        s.el.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}" fill="{fill}"{o} stroke="{stroke}" stroke-width="{sw}"{d}/>')
    def line(s, x1, y1, x2, y2, stroke=INK, sw=2, arrow=False, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        m = ' marker-end="url(#ah)"' if arrow else ''
        s.el.append(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>')
    def path(s, d, stroke=INK, sw=2, fill='none', arrow=False):
        m = ' marker-end="url(#ah)"' if arrow else ''
        s.el.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{m}/>')
    def title(s, t):
        s.text(40, 50, t, size=28)
    def brace(s, x1, x2, y, label, size=15, fill=SUB):
        """Horizontal bracket under a span, with a centred label."""
        s.path(f'M{x1},{y} L{x1},{y+8} L{x2},{y+8} L{x2},{y}', stroke=FAINT, sw=1.5)
        s.text((x1+x2)/2, y+30, label, size=size, fill=fill, anchor='middle')
    def svg(s, font_css):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.w}" height="{s.h}" viewBox="0 0 {s.w} {s.h}" '
                f'font-family="Excalifont, \'Segoe UI Symbol\', \'DejaVu Sans\', sans-serif">\n'
                f'<style>{font_css}</style>\n'
                '<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" '
                'markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#1e1e1e"/></marker></defs>\n'
                f'<rect width="{s.w}" height="{s.h}" fill="#ffffff"/>\n')
        return head + '\n'.join(s.el) + '\n</svg>\n'

figs = {}

# ---------------------------------------------------------------- 01 bit layouts
f = Fig(1000, 500)
f.title('Eight bits, split three ways')
lx = [(SIGN, 'sign'), (EXP, 'exponent'), (MANT, 'mantissa')]
x = 170
for col, name in lx:
    f.rect(x, 80, 18, 18, fill=col, sw=1.5, rx=3); f.text(x+26, 95, name, size=15, fill=SUB); x += 130
rows = [
    ('E5M2', 'FP8 element', '0 11110 11', 'largest: 1.75 × 2^{15} = 57,344',
     'neighbours 25% apart · same range as FP16'),
    ('E4M3', 'FP8 element', '0 1111 110', 'largest: 1.75 × 2^{8} = 448',
     'neighbours 12.5% apart · 0 1111 111 is NaN, no infinity'),
    ('E8M0', 'block scale', '01111111', 'byte E means 2^{E − 127}; here 127, so scale = 1',
     'range 2^{−127} to 2^{127} · all bits exponent · 0xFF is NaN'),
]
y = 130
for name, role, bits, l1, l2 in rows:
    f.text(40, y+26, name, size=22)
    f.text(40, y+48, role, size=14, fill=FAINT)
    fields = bits.split(' ')
    if len(fields) == 1: kinds = ['e']*8
    else: kinds = ['s']*len(fields[0]) + ['e']*len(fields[1]) + ['m']*len(fields[2])
    flat = bits.replace(' ', '')
    for i, (k, b) in enumerate(zip(kinds, flat)):
        col = {'s': SIGN, 'e': EXP if name != 'E8M0' else SCALE, 'm': MANT}[k]
        f.rect(170 + i*46, y, 46, 50, fill=col, sw=1.5, rx=0)
        f.text(170 + i*46 + 23, y+33, b, size=22, anchor='middle')
    f.rect(170, y, 368, 50, fill='none', sw=2, rx=4)
    f.text(575, y+22, l1, size=18)
    f.text(575, y+46, l2, size=15, fill=SUB)
    y += 100
f.rect(40, 430, 920, 48, fill=GRAY, sw=1.5, rx=10)
f.text(500, 461, 'an MXFP8 block = one E8M0 scale + 32 FP8 elements (E4M3 here; the spec also allows E5M2)',
       size=17, anchor='middle')
figs['mxfp8_01_bit_layouts.svg'] = f

# ---------------------------------------------------------------- 02 block layout
f = Fig(1000, 590)
f.title('Thirty-two values share one scale byte')
f.text(40, 96, 'one row of a tensor, along K (the GEMM reduction dimension)', size=16, fill=SUB)
bw, gap, x0 = 222, 10, 40
for b in range(4):
    x = x0 + b*(bw+gap)
    f.rect(x+bw/2-46, 112, 92, 30, fill=SCALE, sw=1.5, rx=6)
    f.text(x+bw/2, 133, f'scale {b}', size=15, anchor='middle')
    f.line(x+bw/2, 142, x+bw/2, 156, sw=1.5)
    f.rect(x, 158, bw, 40, fill=EXP if b != 1 else '#74c0fc', sw=1.5, rx=4)
    for t in range(1, 32):
        tx = x + t*bw/32
        f.line(tx, 160, tx, 196, stroke='#ffffff', sw=1)
    f.text(x+bw/2, 222, f'block {b} · 32 values', size=15, fill=SUB, anchor='middle')
# zoom from block 1
bx = x0 + 1*(bw+gap)
f.path(f'M{bx},{230} L{40},{290}', stroke=FAINT, sw=1.5)
f.path(f'M{bx+bw},{230} L{960},{290}', stroke=FAINT, sw=1.5)
f.text(500, 272, 'block 1 in memory', size=16, fill=SUB, anchor='middle')
cw = 920/33
f.rect(40, 296, cw, 44, fill=SCALE, sw=1.5, rx=0)
for i in range(32):
    f.rect(40 + (i+1)*cw, 296, cw, 44, fill=EXP, sw=1.2, rx=0)
f.path(f'M40,348 L40,356 L{40+cw},356 L{40+cw},348', stroke=FAINT, sw=1.5)
f.text(40, 378, '1 byte: E8M0 scale', size=14, fill=SUB)
f.brace(40+cw+4, 960, 348, '32 bytes: one E4M3 element per value', size=15)
# byte comparison
f.text(40, 434, 'bytes to store 32 values', size=16, fill=SUB)
px, bx0 = 11.5, 190
rowsb = [('BF16', [(64, EXP)], '64 B'),
         ('MXFP8', [(1, SCALE), (32, EXP)], '33 B  (1.03 B per value)'),
         ('FP8, no scale', [(32, EXP)], '32 B, but only ±448 of range')]
y = 452
for name, segs, lab in rowsb:
    f.text(40, y+22, name, size=17)
    x = bx0
    for n, col in segs:
        f.rect(x, y, n*px, 30, fill=col, sw=1.5, rx=3); x += n*px
    f.text(x+12, y+21, lab, size=15, fill=SUB if name != 'MXFP8' else GREEN_T)
    y += 44
figs['mxfp8_02_block_scaling.svg'] = f

# ---------------------------------------------------------------- 03 dynamic range
f = Fig(1000, 540)
f.title('A block sees five decades; its scale picks which five')
X = lambda v: 170 + (v + 45) * 9          # log10|x| -> px, -45..+45
f.line(X(-45), 100, X(45)+10, 100, sw=2, arrow=True)
for t in (-40, -20, 0, 20, 40):
    f.line(X(t), 94, X(t), 106, sw=1.5)
    f.line(X(t), 110, X(t), 440, stroke='#dee2e6', sw=1, dash='4,4')
    f.text(X(t), 84, f'10^{{{str(t).replace("-", "−")}}}', size=15, fill=SUB, anchor='middle')
bars = [('FP32', -44.85, 38.53, GRAY, None), ('BF16', -40.04, 38.53, GRAY, None),
        ('E5M2', -4.82, 4.76, GRAY, '±57,344'), ('E4M3', -2.71, 2.65, PINK, '±448')]
y = 126
for name, lo, hi, col, lab in bars:
    f.text(40, y+19, name, size=18)
    f.rect(X(lo), y, X(hi)-X(lo), 24, fill=col, sw=1.5, rx=5)
    if lab: f.text(X(hi)+10, y+18, lab, size=15, fill=SUB)
    y += 44
# MXFP8 row(s)
y = 330
f.text(40, y+19, 'MXFP8', size=18, fill=GREEN_T)
f.rect(X(-40.94), y, X(40.88)-X(-40.94), 24, fill='none', stroke=GREEN_T, sw=1.5, rx=5, dash='6,5')
f.text(X(-40.94), y+50, 'all possible scales together: about 10^{−41} to 10^{41}', size=15, fill=GREEN_T)
for sc, lab in ((-100, 'block A · scale 2^{−100}'), (0, 'block B · scale 2^{0}'), (100, 'block C · scale 2^{100}')):
    sh = sc * 0.30103
    lo, hi = -2.71 + sh, 2.65 + sh
    f.rect(X(lo), y, X(hi)-X(lo), 24, fill=MANT, stroke=INK, sw=2, rx=5)
    f.text((X(lo)+X(hi))/2, y-12, lab, size=15, anchor='middle')
f.rect(40, 452, 920, 66, fill=GRAY, sw=1.5, rx=10)
f.text(60, 480, 'Each solid window is E4M3’s ~5.4 decades, shifted by that block’s scale. Across a tensor the',
       size=16)
f.text(60, 503, 'windows cover FP32-like range, but values inside one block still have to fit in one window.',
       size=16)
figs['mxfp8_03_dynamic_range.svg'] = f

# ---------------------------------------------------------------- 04 GEMM dataflow
f = Fig(1060, 600)
f.title('The scales go into the instruction, not into a dequant pass')
# HBM
f.rect(40, 100, 230, 250, fill=GRAY, sw=2, rx=12)
f.text(155, 130, 'HBM', size=20, anchor='middle')
for i, nm in enumerate(('A', 'B')):
    yy = 150 + i*100
    f.text(60, yy+20, f'{nm}:', size=17)
    for k in range(6):
        f.rect(92 + k*22, yy+2, 22, 28, fill=EXP, sw=1.2, rx=0)
    f.rect(92 + 6*22 + 6, yy+2, 22, 28, fill=SCALE, sw=1.2, rx=0)
    f.text(60, yy+56, 'FP8 data + E8M0 scales', size=14, fill=SUB)
f.line(272, 225, 330, 225, arrow=True)
f.text(301, 212, 'load', size=14, fill=SUB, anchor='middle')
# registers
f.rect(335, 100, 300, 250, fill='#ffffff', sw=2, rx=12)
f.text(485, 130, 'registers, per lane', size=20, anchor='middle')
for i, nm in enumerate(('A', 'B')):
    yy = 150 + i*100
    f.text(355, yy+20, f'{nm}:', size=17)
    for k in range(8):
        f.rect(387 + k*22, yy+2, 22, 28, fill=EXP, sw=1.2, rx=0)
    f.rect(387 + 8*22 + 10, yy+2, 28, 28, fill=SCALE, sw=1.2, rx=0)
    f.text(355, yy+56, '8 VGPRs: 32 FP8 values + 1 scale byte', size=14, fill=SUB)
f.line(637, 225, 695, 225, arrow=True)
# MFMA
f.rect(700, 100, 320, 250, fill=MANT, sw=2, rx=12)
f.text(860, 130, 'v_mfma_scale_f32_16x16x128_f8f6f4', size=15, anchor='middle')
f.text(860, 172, 'for each of the 4 K-blocks b:', size=17, anchor='middle')
f.text(860, 204, 'acc += 2^{sA(b) + sB(b)}', size=18, anchor='middle')
f.text(860, 230, '× (sum of 32 FP8 products)', size=17, anchor='middle')
f.text(860, 262, 'multiply FP8 × FP8 natively', size=15, fill=SUB, anchor='middle')
f.text(860, 286, 'accumulate in FP32', size=15, fill=SUB, anchor='middle')
f.line(860, 352, 860, 400, arrow=True)
f.rect(740, 405, 240, 56, fill=GRAY, sw=2, rx=10)
f.text(860, 440, 'C: 16 × 16 tile, FP32', size=17, anchor='middle')
# what is absent
f.rect(40, 390, 595, 72, fill='#ffffff', stroke=RED, sw=1.5, rx=10, dash='7,5')
f.text(60, 420, 'not in this picture: a dequantize step', size=17, fill=RED)
f.text(60, 446, 'A and B never exist as FP32 or BF16, in HBM or in registers', size=15, fill=SUB)
# counts
f.rect(40, 490, 980, 88, fill=GRAY, sw=1.5, rx=10)
f.text(60, 520, 'per instruction:  A = 16 × 128 FP8 = 2,048 bytes = 64 lanes × 8 VGPRs', size=16)
f.text(60, 546, 'scales:  16 rows × 4 blocks = 64 bytes, exactly one byte per lane (same for B)', size=16)
f.text(60, 570, 'the 32 × 32 × 64 sibling covers 2 K-blocks per instruction instead of 4', size=15, fill=SUB)
figs['mxfp8_04_gemm_dataflow.svg'] = f

# ---------------------------------------------------------------- 05 scale locality
f = Fig(1000, 540)
f.title('How far one outlier reaches')
R, C, cs = 4, 64, 13.5
gx = 60
orow, ocol = 1, 41
panels = [('per-tensor scale', 'all 256 values share the outlier’s scale', lambda r, c: True),
          ('per-row scale', 'the outlier’s row: 64 values', lambda r, c: r == orow),
          ('MXFP8: one scale per 32', 'the outlier’s block: 32 values', lambda r, c: r == orow and c // 32 == ocol // 32)]
y = 96
for name, what, hit in panels:
    f.text(gx, y, name, size=19)
    f.text(gx + 260, y, what, size=16, fill=RED)
    gy = y + 14
    for r in range(R):
        for c in range(C):
            if r == orow and c == ocol: col = '#e03131'
            elif hit(r, c): col = PINK
            else: col = '#ffffff'
            if col != '#ffffff':
                f.el.append(f'<rect x="{gx+c*cs:g}" y="{gy+r*cs:g}" width="{cs:g}" height="{cs:g}" fill="{col}"/>')
    for r in range(R+1):
        f.line(gx, gy+r*cs, gx+C*cs, gy+r*cs, stroke='#ced4da', sw=1)
    for c in range(C+1):
        f.line(gx+c*cs, gy, gx+c*cs, gy+R*cs, stroke='#ced4da', sw=1)
    if name.startswith('MXFP8'):
        f.line(gx+32*cs, gy-4, gx+32*cs, gy+R*cs+4, sw=2.5)
    f.rect(gx, gy, C*cs, R*cs, fill='none', sw=2, rx=0)
    y += 124
f.el.append(f'<rect x="{gx}" y="470" width="16" height="16" fill="#e03131"/>')
f.text(gx+26, 484, 'the outlier', size=15, fill=SUB)
f.rect(gx+150, 470, 16, 16, fill=PINK, sw=1, rx=0, stroke='#ced4da')
f.text(gx+176, 484, 'scaled by the outlier: its small values get pushed toward subnormals or zero', size=15, fill=SUB)
f.rect(gx, 500, 16, 16, fill='#ffffff', sw=1, rx=0, stroke='#ced4da')
f.text(gx+26, 514, 'keeps a scale that fits its own values', size=15, fill=SUB)
f.text(gx+C*cs, 514, '4 rows × 64 values shown', size=14, fill=FAINT, anchor='end')
figs['mxfp8_05_scale_locality.svg'] = f

# ---------------------------------------------------------------- font + write
chars = set()
for fg in figs.values(): chars |= fg.chars
font = TTFont(f'{REPO}/public/fonts/Excalifont-Regular.ttf')
cmap = font.getBestCmap()
have = ''.join(sorted(c for c in chars if ord(c) in cmap))
missing = ''.join(sorted(c for c in chars if ord(c) not in cmap and c.strip()))
opts = subset.Options(); opts.flavor = 'woff2'; opts.layout_features = ['*']
sub = subset.Subsetter(opts); sub.populate(text=have); sub.subset(font)
buf = io.BytesIO(); font.flavor = 'woff2'; font.save(buf)
b64 = base64.b64encode(buf.getvalue()).decode()
css = ('@font-face{font-family:"Excalifont";src:url(data:font/woff2;base64,' + b64 +
       ') format("woff2");font-style:normal;font-weight:100 900;font-display:block}'
       'text{font-family:"Excalifont",\'Segoe UI Symbol\',\'DejaVu Sans\',sans-serif}')
for name, fg in figs.items():
    open(f'{OUT}/{name}', 'w', encoding='utf-8').write(fg.svg(css))
print('font', len(buf.getvalue()), 'bytes; glyphs', len(have), '; fallback for:', repr(missing))
