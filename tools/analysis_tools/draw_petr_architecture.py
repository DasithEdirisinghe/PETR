"""Render the code-aligned PETR diagram as editable SVG (stdlib only)."""

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COLORS = dict(blue='#2563eb', green='#148268', purple='#7952c7',
              orange='#b96715', gray='#526174')
FILLS = dict(blue='#eff6ff', green='#edf9f4', purple='#f5f0ff',
             orange='#fff6e9', gray='#f5f7fa')


def render():
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1740" viewBox="0 0 1600 1740">
<defs><style>
text{font-family:Arial,sans-serif;fill:#203047}
.title{font-size:30px;font-weight:700}.section{font-size:21px;font-weight:700}
.label{font-size:18px;font-weight:700}.formula{font-size:18px}
.note{font-size:16px;fill:#536176}.badge{font-size:14px;font-weight:700}
</style>''']
    for name, color in COLORS.items():
        parts.append('<marker id="{}" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="{}"/></marker>'.format(name, color))
    parts.append('</defs><rect width="1600" height="1740" fill="white"/>')

    def text(x, y, value, cls='formula', anchor='start'):
        # Arial/librsvg lacks the Unicode subscript-t glyph on some systems.
        value = value.replace('ₜ', '_t')
        parts.append('<text x="{}" y="{}" class="{}" text-anchor="{}">{}</text>'.format(x, y, cls, anchor, escape(value)))

    def panel(y, h, title, color):
        parts.append('<rect x="35" y="{}" width="1530" height="{}" rx="14" fill="{}" stroke="#d4dce7"/>'.format(y, h, FILLS[color]))
        text(55, y + 30, title, 'section')

    def box(x, y, w, h, title, lines, color='gray', badge=None):
        parts.append('<rect x="{}" y="{}" width="{}" height="{}" rx="10" fill="white" stroke="{}" stroke-width="1.7"/>'.format(x, y, w, h, COLORS[color]))
        text(x + w / 2, y + 30, title, 'label', 'middle')
        for i, line in enumerate(lines):
            text(x + w / 2, y + 59 + i * 25, line, 'formula', 'middle')
        if badge:
            text(x + w - 10, y + h - 12, badge, 'badge', 'end')

    def arrow(points, color='gray', dashed=False):
        path = 'M' + ' L'.join('{},{}'.format(x, y) for x, y in points)
        parts.append('<path d="{}" fill="none" stroke="{}" stroke-width="2" {} marker-end="url(#{})"/>'.format(path, COLORS[color], 'stroke-dasharray="6 5"' if dashed else '', color))

    text(45, 45, 'PETR · architecture and ablation map', 'title')
    text(45, 75, 'Current PCCR implementation · matching symbols connect panels · detailed explanations in the companion legend', 'note')
    text(45, 102, '[L] trainable parameters    [F] parameter-free operation    dashed border / arrow: optional path', 'note')

    panel(120, 185, 'A   Image content → X', 'blue')
    box(60, 175, 260, 104, 'Images I', ['B × N × 3 × Hi × Wi'], 'blue')
    box(365, 175, 320, 104, 'Backbone + CPFPN', ['ResNet-50 / DCN → level 0'], 'blue', '[L]')
    box(730, 175, 310, 104, '1×1 projection', ['Xgrid : B × N × C × Hf × Wf'], 'blue', '[L]')
    box(1085, 175, 450, 104, 'Flatten camera, row, column', ['X : B × L × C     L = N Hf Wf'], 'blue', '[F]')
    for a, b in [(320, 365), (685, 730), (1040, 1085)]:
        arrow([(a, 227), (b, 227)], 'blue')

    panel(325, 325, 'B   Image position → P = G + H', 'green')
    text(60, 395, 'G', 'section')
    text(60, 421, 'grid PE', 'note')
    box(185, 380, 475, 106, '2D PE  OR  MPE', ['2D: S(y) ⊕ S(x) → 256', 'MPE: S(n) ⊕ S(y) ⊕ S(x) → 384'], 'green')
    box(730, 380, 420, 106, 'Grid adapter', ['2D*: 256 → 1024 → C', 'MPE: 384 → 1024 → C'], 'green', '[L]')
    arrow([(660, 433), (730, 433)], 'green')
    text(1180, 438, 'G : B × L × C', 'label')
    text(60, 550, 'H', 'section')
    text(60, 576, 'geometry PE', 'note')
    box(185, 510, 475, 106, 'PETR 3DPE  OR  LiDAR Oracle PE', ['PETR: Pcam⁻¹ [ud, vd, d, 1]ᵀ;  d₁…dD', 'Oracle: projected LiDAR + depth fill → XYZ'], 'green')
    box(730, 510, 420, 106, 'Geometry encoder', ['PETR: 3D → 1024 → C  [L]', 'Oracle: S₈₄(y) ⊕ S₈₄(x) ⊕ S₈₈(z)  [F]'], 'green')
    arrow([(660, 563), (730, 563)], 'green')
    text(1180, 568, 'H : B × L × C', 'label')
    text(60, 638, 'Disabled branch = 0.  *2D adapter is used with PETR 3DPE; 2D-only bypasses it.  Coordinate preprocessing: see legend B.', 'note')

    panel(670, 270, 'C   Query position → Eₜ = Mθₜ(S(Rₜ))', 'purple')
    box(60, 725, 300, 120, 'Reference points Rₜ', ['T × 3', 'current learned (x, y, z)'], 'purple', '[L]')
    box(415, 725, 360, 120, 'Sine / cosine transform S', ['T × 3 → T × 384', 'S₁₂₈(y) ⊕ S₁₂₈(x) ⊕ S₁₂₈(z)'], 'purple', '[F]')
    box(830, 725, 350, 120, 'Query MLP Mθₜ', ['384 → 256 → ReLU → 256', 'shared across all T queries'], 'purple', '[L]')
    box(1235, 725, 300, 120, 'Query position Eₜ', ['T × C → B × T × C', 'reused in layers 1…6'], 'purple')
    for a, b in [(360, 415), (775, 830), (1180, 1235)]:
        arrow([(a, 780), (b, 780)], 'purple')
    text(65, 879, 'Fixed frequencies; changing coordinates: R_t changes → S(R_t) changes → E_t changes.', 'label')
    text(65, 912, 'Training: loss → backward → optimizer updates (Rₜ, θₜ) → next forward recomputes Eₜ₊₁ = Mθₜ₊₁(S(Rₜ₊₁)).', 'note')

    panel(960, 385, 'D   Decoder layer ℓ = 1…6     ·     h₀ = 0     ·     X, P and E shared across layers', 'orange')
    box(60, 1020, 330, 135, 'Query self-attention', ['Qsa = Wq(hℓ₋₁ + E)', 'Ksa = Wk(hℓ₋₁ + E)', 'Vsa = Wv(hℓ₋₁)'], 'purple')
    box(440, 1040, 190, 90, 'Add + Norm', ['→ uℓ'], 'orange')
    arrow([(390, 1085), (440, 1085)])
    parts.append('<rect x="680" y="1010" width="620" height="245" rx="12" fill="white" stroke="#b96715" stroke-width="2"/>')
    text(700, 1038, 'Cross-attention', 'label')
    text(700, 1070, 'Q = Wq(uℓ + E)     K = Wk(X + P)     V = Wv(X)', 'formula')
    parts.append('<rect x="710" y="1093" width="560" height="65" rx="8" fill="#f5f0ff" stroke="#7952c7" stroke-dasharray="6 5"/>')
    text(990, 1119, 'Optional URoPE: Q → ℛq Q,   K → ℛk K', 'formula', 'middle')
    text(990, 1143, 'rotation occurs before QKᵀ; V is unchanged', 'note', 'middle')
    arrow([(990, 1158), (990, 1180)])
    text(990, 1204, 'A = softmax(QKᵀ / √dh + padding mask)', 'formula', 'middle')
    text(990, 1235, 'output = Wo concat_heads(A V)', 'formula', 'middle')
    arrow([(630, 1085), (680, 1085)])
    box(1340, 1040, 195, 90, 'Add + Norm', ['→ vℓ'], 'orange')
    arrow([(1300, 1085), (1340, 1085)])
    box(60, 1190, 535, 105, 'Attention dimensions', ['Q: B × a × T × dh;   K,V: B × a × L × dh', 'A: B × a × T × L'], 'gray')
    box(1340, 1170, 195, 125, 'FFN', ['C → 2048 → C', 'Add + Norm → hℓ'], 'orange', '[L]')
    arrow([(1437, 1130), (1437, 1170)])
    text(700, 1315, 'hℓ → next layer + heads; E is reused in each attention call.', 'note')

    panel(1365, 230, 'E   Per-layer predictions; final-layer decoding', 'gray')
    box(60, 1420, 235, 130, 'Decoder output', ['hℓ : B × T × C', 'ℓ = 1…6'], 'orange')
    box(350, 1420, 390, 130, 'Class + box heads', ['class logits: B × T × 9', 'box code: B × T × 8'], 'orange', '[L]')
    box(795, 1420, 370, 130, 'Centre anchor reuse', ['ĉℓ = σ(logitε(R) + Δℓ)', 'ĉℓ → metric XYZ via pc_range'], 'purple')
    box(1220, 1420, 315, 130, 'Final layer → coder', ['top-K scores + range filter', 'boxes, scores, labels'], 'gray', '[F]')
    for a, b in [(295, 350), (740, 795), (1165, 1220)]:
        arrow([(a, 1485), (b, 1485)])
    text(60, 1579, 'R is reused from C. Predicted centres do not replace R inside the decoder. Training supervises predictions from every layer.', 'note')
    text(45, 1640, 'C = 256   D = 64   T = 900   a = 8   dh = C/a = 32   L = N Hf Wf', 'label')
    text(45, 1672, 'For 6 cameras and a 20×50 feature grid: L = 6000. Shapes use batch-first notation for readability.', 'note')
    text(45, 1704, 'Symbols, ablation matrix, normalization details and code links: PETR_ARCHITECTURE_ABLATION_MAP.md', 'note')
    parts.append('</svg>')
    output = ROOT / 'PETR_ARCHITECTURE_ABLATION_MAP.svg'
    output.write_text('\n'.join(parts))
    print(output)


if __name__ == '__main__':
    render()
