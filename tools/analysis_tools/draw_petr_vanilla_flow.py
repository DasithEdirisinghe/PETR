"""Generate a code-aligned vanilla PETR flow diagram (stdlib; render with librsvg)."""

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def render():
    colors = {'content': '#2563eb', 'geometry': '#168069',
              'grid': '#a16b12', 'query': '#7952c7', 'neutral': '#526174'}
    out = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="2380" viewBox="0 0 1800 2380">
<defs><style>
text{font-family:Arial,sans-serif;fill:#203047}
.title{font-size:32px;font-weight:bold}.heading{font-size:23px;font-weight:bold}
.label{font-size:20px;font-weight:bold}.body{font-size:19px}.note{font-size:17px;fill:#526174}
</style>''']
    for name, color in colors.items():
        out.append(f'<marker id="{name}" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="{color}"/></marker>')
    out.append('</defs><rect width="1800" height="2380" fill="white"/>')

    def text(x, y, s, cls='body', anchor='start'):
        out.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>')

    def panel(y, h, title):
        out.append(f'<rect x="30" y="{y}" width="1740" height="{h}" rx="14" fill="#f7f9fc" stroke="#d4dce7"/>')
        text(55, y + 34, title, 'heading')

    def box(x, y, w, h, title, lines, kind='neutral'):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="white" stroke="{colors[kind]}" stroke-width="2"/>')
        text(x + 18, y + 30, title, 'label')
        for i, s in enumerate(lines):
            text(x + 18, y + 62 + i * 27, s)

    def arrow(points, kind='neutral'):
        path = 'M' + ' L'.join(f'{x},{y}' for x, y in points)
        out.append(f'<path d="{path}" fill="none" stroke="{colors[kind]}" stroke-width="2.3" marker-end="url(#{kind})"/>')

    text(40, 48, 'VANILLA PETR · one image cell → all-camera attention → 3D boxes', 'title')
    text(40, 81, 'Code path: PCCR R50-DCN / CPFPN baseline · image 3DPE ON · multiview PE ON · URoPE OFF · LiDAR input OFF', 'note')
    for x, label, kind in [(40, 'Image content', 'content'), (310, 'Calibrated geometry', 'geometry'),
                           (650, 'Camera / row / column', 'grid'), (1020, 'Object-query position', 'query')]:
        out.append(f'<rect x="{x}" y="103" width="18" height="18" fill="{colors[kind]}"/>')
        text(x + 28, 119, label, 'note')
    text(1400, 119, '[L] learned   [F] formula', 'note')

    panel(145, 230, '1   Shared image backbone: follow cell (camera c, row r, column s)')
    box(55, 205, 340, 120, 'Augmented / padded images', ['I: B × N × 3 × Hi × Wi', 'Same CNN for every camera'], 'content')
    box(440, 205, 370, 120, 'ResNet-50 DCN → CPFPN [L]', ['Process B·N images; select level 0', 'F: B × N × 256 × H × W'], 'content')
    box(855, 205, 350, 120, 'input_proj: 1×1 conv [L]', ['256 → C = 256', 'Xgrid: B × N × C × H × W'], 'content')
    box(1250, 205, 490, 120, 'One content vector X[c,r,s]', ['256 numbers summarizing a receptive field', 'Keep content separate from PE until attention'], 'content')
    for a, b in [(395, 440), (810, 855), (1205, 1250)]:
        arrow([(a, 265), (b, 265)], 'content')
    text(55, 354, 'Example: 320×800 images → H=20, W=50.  Each cell is a CNN feature, not one original pixel or a known 3D point.', 'note')

    panel(395, 530, '2   Geometry branch: use calibration to describe the possible 3D locations of that cell')
    box(55, 455, 510, 175, 'Camera c: calibrated projection [F]',
        ['E_c = T(common LiDAR frame → camera)', 'K_c: intrinsics (focal lengths, principal point)',
         'P_c = Kbar_c E_c = lidar2img[c]', 'Use augmentation-updated P_c, not raw K / E'], 'geometry')
    box(630, 455, 520, 175, 'Feature grid → image ray samples [F]',
        ['u = s · Wi/W;     v = r · Hi/H', 'D=64 candidate camera-z depths: d_k',
         'b_k = [u·d_k, v·d_k, d_k, 1]ᵀ', 'Same depth schedule at every feature cell'], 'geometry')
    box(1215, 455, 525, 175, 'Inverse projection [F]',
        ['p_k = (P_c)⁻¹ b_k', 'p_k[:3] = (x_k, y_k, z_k) in common frame',
         '64 points on this camera ray', 'Different cameras → different rays'], 'geometry')
    arrow([(1150, 540), (1215, 540)], 'geometry')
    arrow([(310, 630), (310, 657), (1475, 657), (1475, 630)], 'geometry')
    text(590, 649, 'P_c carries camera orientation, location and intrinsics', 'note')
    box(1215, 700, 525, 150, 'Range normalization + channel packing [F]',
        ['Normalize XYZ by position_range; logit + clamp', 'Pack [x0,y0,z0, x1,y1,z1, …] → 192',
         'Tensor: (B·N) × (3D) × H × W'], 'geometry')
    box(630, 700, 520, 150, 'position_encoder: shared 1×1 convs [L]',
        ['192 → 1024 → ReLU → 256', 'Mix coordinates / depths within each cell',
         'No sine transform in this image-3DPE branch'], 'geometry')
    box(55, 700, 510, 150, 'Geometry PE: G[c,r,s] ∈ R²⁵⁶',
        ['Ggrid: B × N × 256 × H × W', 'One vector for the entire sampled ray',
         'No predicted depth; no LiDAR returns'], 'geometry')
    arrow([(1700, 630), (1700, 700)], 'geometry')
    arrow([(1215, 775), (1150, 775)], 'geometry')
    arrow([(630, 775), (565, 775)], 'geometry')
    text(55, 889, 'Calibration aligns rays in one 3D frame.  It does not tell PETR which sampled depth contains the visible surface.', 'label')

    panel(945, 345, '3   Multiview branch + aligned token assembly: repeat for every camera cell')
    box(55, 1005, 510, 120, 'Camera / row / column sine PE [F]',
        ['Mask-aware cumulative indices → sine / cosine', 'S128(camera) ⊕ S128(row) ⊕ S128(col) = 384'], 'grid')
    box(630, 1005, 520, 120, 'adapt_pos3d: shared 1×1 convs [L]',
        ['384 → 1024 → ReLU → 256', 'Multiview PE M[c,r,s]; no calibration input'], 'grid')
    box(1215, 1005, 525, 120, 'Add position branches, not content',
        ['P[c,r,s] = G[c,r,s] + M[c,r,s]', 'Both 256D; sum remains 256D'], 'geometry')
    arrow([(565, 1065), (630, 1065)], 'grid')
    arrow([(1150, 1065), (1215, 1065)], 'grid')
    box(55, 1160, 1685, 100, 'Matched flattening: j = (c·H + r)·W + s;  L = N·H·W',
        ['Xgrid → X: [L,B,256]       Pgrid → P: [L,B,256]       image padding mask → [B,L]       6 × 20 × 50 = 6000 tokens'])
    arrow([(1475, 1125), (1475, 1160)], 'geometry')

    panel(1310, 220, '4   Object queries: independent of image-cell positions')
    box(55, 1370, 380, 120, '900 reference points R [L]',
        ['R: T × 3; T=900', 'Learned XYZ anchors; initialize in [0,1]'], 'query')
    box(480, 1370, 430, 120, 'pos2posemb3d(R) [F]',
        ['S128(y) ⊕ S128(x) ⊕ S128(z)', 'Fixed frequencies; coordinate-dependent values'], 'query')
    box(955, 1370, 400, 120, 'query_embedding MLP [L]',
        ['384 → 256 → ReLU → 256', 'E: [T,B,256] after batch broadcast'], 'query')
    box(1400, 1370, 340, 120, 'Decoder input',
        ['Content h0 = zeros_like(E)', 'Same E used in all 6 layers'], 'query')
    for a, b in [(435, 480), (910, 955), (1355, 1400)]:
        arrow([(a, 1430), (b, 1430)], 'query')

    panel(1550, 510, '5   Decoder layer ℓ = 1…6: how query i selects image-cell j across every camera')
    box(55, 1610, 450, 160, 'Self-attention → residual + norm',
        ['Qsa = Wq(h_prev + E)', 'Ksa = Wk(h_prev + E);  Vsa = Wv(h_prev)',
         'Output u: queries exchange information'], 'query')
    box(570, 1610, 540, 160, 'Cross-attention inputs [L projections]',
        ['Q = Wq(u + E)     ← query content + query PE', 'K = Wk(X + P)    ← image content + image PE',
         'V = Wv(X)           ← image content only'], 'content')
    box(1175, 1610, 565, 160, 'Split into 8 heads; 32 channels / head',
        ['Q: B × 8 × 900 × 32', 'K, V: B × 8 × L × 32',
         'All-camera memory X, P stays fixed across layers'])
    arrow([(505, 1690), (570, 1690)], 'query')
    arrow([(1110, 1690), (1175, 1690)])
    box(55, 1830, 650, 160, 'Attention weights: one row per query and head',
        ['A[i,j] = softmax_j(Q[i] · K[j] / √32 + mask[j])',
         'A: B × 8 × 900 × L;  padded cells excluded',
         'ONE softmax across all cameras and spatial cells'])
    box(770, 1830, 440, 160, 'Read content using attention',
        ['o_i = Σ_j A[i,j] · V[j]', 'Concatenate 8 heads → 256 channels',
         'Output projection Wo [L]'], 'content')
    box(1275, 1830, 465, 160, 'Update query content → h_next',
        ['Cross-attention residual + LayerNorm', 'FFN 256 → 2048 → 256 [L]',
         'FFN residual + LayerNorm → next layer'])
    arrow([(1460, 1770), (1460, 1800), (380, 1800), (380, 1830)])
    arrow([(705, 1910), (770, 1910)], 'content')
    arrow([(1210, 1910), (1275, 1910)], 'content')
    text(55, 2030, 'Position affects which cells are read through Q and K.  The values carry image content.  There is no separate PE-only attention map.', 'note')

    panel(2080, 220, '6   Detection heads: query content becomes scored 3D boxes')
    box(55, 2140, 480, 120, 'Per-layer class + regression heads [L]',
        ['Class logits + box residuals for each query', 'Training supervises all 6 decoder outputs'])
    box(590, 2140, 565, 120, 'Use the original reference R for centers',
        ['center_norm = sigmoid(logit(R) + delta_xyz)', 'Rescale by pc_range → metric XYZ; decode size/yaw'])
    box(1210, 2140, 530, 120, 'Inference: final layer → NMSFreeCoder',
        ['Sigmoid class scores → top-K query/class pairs', 'Decode boxes + range filtering → 3D detections'])
    arrow([(535, 2200), (590, 2200)])
    arrow([(1155, 2200), (1210, 2200)])
    text(40, 2332, 'No layer-to-layer reference-point refinement in this vanilla path. Optimizer updates R between training steps; next forward recomputes E.', 'note')
    text(40, 2361, 'Symbols: B batch · N cameras · H,W feature grid · C=256 · D=64 · T=900 · L=NHW.  Details, calibration conventions and code links: companion .md.', 'note')
    out.append('</svg>')
    return '\n'.join(out)


if __name__ == '__main__':
    path = ROOT / 'PETR_VANILLA_FEATURE_TO_DECODER.svg'
    path.write_text(render(), encoding='utf-8')
    print(path)
