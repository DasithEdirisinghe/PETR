#!/usr/bin/env python3
"""Build an interactive report for the projection-correspondence audit."""

import json
from pathlib import Path


MODEL_SPECS = (
    ('petr_r50', 'PETR R50'),
    ('streampetr_r50', 'StreamPETR R50'),
    ('repdetr3d_eva02_l', 'RepDETR3D EVA02-L'),
    ('petr_eva02_l_epoch16', 'PETR EVA02-L epoch 16'),
)
CLASSES = ('pedestrian', 'car', 'bicycle')


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Projection correspondence audit</title><style>
:root{font-family:Arial,sans-serif;color:#17233a;background:#f3f6fa}*{box-sizing:border-box}body{margin:0;padding:16px;max-width:1500px;margin:auto}h1{margin:0 0 6px;font-size:24px}h2{font-size:19px;margin:8px 0}p{color:#52627a;line-height:1.45}.controls,.card{background:white;border:1px solid #ccd7e3;border-radius:9px;padding:12px}.controls{display:flex;gap:12px;align-items:center;margin:14px 0;position:sticky;top:0;z-index:3}.controls label{font-weight:bold}.controls select{padding:7px;border:1px solid #afbed0;border-radius:5px}.summary{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}.pill{background:white;border:1px solid #ccd7e3;border-radius:8px;padding:9px 13px}.pill b{display:block;font-size:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}.chart{overflow-x:auto}.bar-row{display:grid;grid-template-columns:210px minmax(420px,1fr);gap:9px;align-items:center;margin:8px 0}.bar-label{font-size:12px;text-align:right}.stack{display:flex;height:29px;border-radius:5px;overflow:hidden;background:#e5e7eb}.segment{display:flex;align-items:center;justify-content:center;color:white;font-size:11px;white-space:nowrap;min-width:0}.target{background:#2563eb}.nearby{background:#d97706}.other{background:#64748b}.confidence-row{display:grid;grid-template-columns:210px repeat(3,1fr);gap:6px;align-items:end;margin:10px 0}.confidence-label{font-size:12px;text-align:right;align-self:center}.confidence-track{height:64px;background:#edf1f5;display:flex;align-items:end;border-radius:4px;position:relative}.confidence-bar{width:100%;border-radius:4px 4px 0 0;min-height:1px}.confidence-value{position:absolute;left:0;right:0;top:2px;text-align:center;font-size:11px;color:#17233a}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:#52627a;margin:7px 0}.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:4px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:7px 8px;border-bottom:1px solid #dbe3ec;text-align:right;white-space:nowrap}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}thead th{background:#eef3f8}.note{font-size:12px}@media(max-width:950px){.grid{grid-template-columns:1fr}.bar-row,.confidence-row{grid-template-columns:150px minmax(380px,1fr)}.confidence-row{grid-template-columns:150px repeat(3,1fr)}}
</style></head><body><h1>Projection correspondence audit</h1><p>For each officially matched GT at 4 m, predictions are classified by the annotation with the highest projected 2D IoU: the intended target, another same-class object, or another class/background. Better alternatives are officially unmatched and selected using the image/geometric preference rule.</p>
<div class="controls"><label>Class <select id="class"><option value="pedestrian">Pedestrian</option><option value="car">Car</option><option value="bicycle">Bicycle</option></select></label><span class="note">Bar labels show percentages; hover for count and mean confidence.</span></div><div id="summary" class="summary"></div>
<div class="legend"><span><i class="swatch target"></i>Target GT</span><span><i class="swatch nearby"></i>Nearby same-class GT</span><span><i class="swatch other"></i>Other class/background</span></div>
<div class="grid"><section class="card chart"><h2>Better unmatched alternative exists</h2><div id="alternative-distribution"></div></section><section class="card chart"><h2>No better unmatched alternative</h2><div id="best-distribution"></div></section></div>
<div class="grid"><section class="card chart"><h2>Mean confidence: alternative cases</h2><div id="alternative-confidence"></div></section><section class="card chart"><h2>Mean confidence: no-alternative cases</h2><div id="best-confidence"></div></section></div>
<section class="card"><h2>Exact values</h2><div class="table-wrap"><table><thead><tr><th>Model</th><th>Group / prediction</th><th>Cases</th><th>Target</th><th>Nearby same class</th><th>Other/background</th></tr></thead><tbody id="values"></tbody></table></div><p class="note">Cell format: count (percentage; mean confidence). The two case groups partition all officially matched GTs.</p></section>
<script>const reportData=__DATA__;const $=id=>document.getElementById(id),cats=['target_gt','other_same_class_gt','other_class_or_background'],styles=['target','nearby','other'];function fmt(x){return x==null?'—':Number(x).toFixed(3)}function cell(x){return `${x.count.toLocaleString()} (${x.percent.toFixed(1)}%; ${fmt(x.mean_confidence)})`}function index(rows){const out={};for(const x of rows)out[`${x.population}|${x.prediction_type}|${x.alignment_category}`]=x;return out}function rowData(d,pop,pred){const ix=index(d.three_category_summary);return cats.map(c=>ix[`${pop}|${pred}|${c}`])}function stacked(host,label,rows){const row=document.createElement('div');row.className='bar-row';const text=document.createElement('div');text.className='bar-label';text.textContent=label;const bar=document.createElement('div');bar.className='stack';rows.forEach((x,i)=>{const s=document.createElement('div');s.className=`segment ${styles[i]}`;s.style.width=`${x.percent}%`;s.textContent=x.percent>=8?`${x.percent.toFixed(1)}%`:'';s.title=`${x.count.toLocaleString()} cases · ${x.percent.toFixed(1)}% · mean confidence ${fmt(x.mean_confidence)}`;bar.appendChild(s)});row.append(text,bar);host.appendChild(row)}function confidence(host,label,rows){const row=document.createElement('div');row.className='confidence-row';const text=document.createElement('div');text.className='confidence-label';text.textContent=label;row.appendChild(text);rows.forEach((x,i)=>{const track=document.createElement('div'),bar=document.createElement('div'),value=document.createElement('div');track.className='confidence-track';bar.className=`confidence-bar ${styles[i]}`;bar.style.height=`${100*(x.mean_confidence||0)}%`;value.className='confidence-value';value.textContent=fmt(x.mean_confidence);track.title=`${styles[i]} · ${x.count.toLocaleString()} cases`;track.append(bar,value);row.appendChild(track)});host.appendChild(row)}function render(){const cls=$('class').value,data=reportData[cls];for(const id of ['alternative-distribution','best-distribution','alternative-confidence','best-confidence','values'])$(id).innerHTML='';let matched=0,better=0,best=0;for(const d of data){matched+=d.matched_gt_count;better+=d.better_alternative_gt_count;best+=d.official_tp_best_gt_count;const officialAlt=rowData(d,'better_alternative','official_tp'),alternative=rowData(d,'better_alternative','alternative'),officialBest=rowData(d,'official_tp_best','official_tp');stacked($('alternative-distribution'),`${d.model} · TP`,officialAlt);stacked($('alternative-distribution'),`${d.model} · ALT`,alternative);stacked($('best-distribution'),d.model,officialBest);confidence($('alternative-confidence'),`${d.model} · TP`,officialAlt);confidence($('alternative-confidence'),`${d.model} · ALT`,alternative);confidence($('best-confidence'),d.model,officialBest);for(const [name,n,rows] of [['Alternative cases · TP',d.better_alternative_gt_count,officialAlt],['Alternative cases · ALT',d.better_alternative_gt_count,alternative],['No better alternative · TP',d.official_tp_best_gt_count,officialBest]]){$('values').insertAdjacentHTML('beforeend',`<tr><td>${d.model}</td><td>${name}</td><td>${n.toLocaleString()}</td>${rows.map(x=>`<td>${cell(x)}</td>`).join('')}</tr>`)}}$('summary').innerHTML=`<div class="pill"><b>${matched.toLocaleString()}</b>matched GT across models</div><div class="pill"><b>${better.toLocaleString()}</b>better-alternative cases</div><div class="pill"><b>${best.toLocaleString()}</b>no-better-alternative cases</div>`}$('class').addEventListener('change',render);render();</script></body></html>'''


def patch_main_index(index_path):
    page = index_path.read_text()
    button = '<button id="correspondence-tab-button" type="button" aria-selected="false">Correspondence audit</button>'
    if button not in page:
        page = page.replace(
            '<button id="plots-tab-button" type="button" aria-selected="false">Visualizations</button>',
            '<button id="plots-tab-button" type="button" aria-selected="false">Visualizations</button>'+button)
    section = ('<section id="correspondence-tab" hidden><iframe id="correspondence-frame" '
               'class="pair-frame" title="Projection correspondence audit" '
               'data-src="projection_correspondence/index.html?v=1"></iframe></section>')
    if section not in page:
        page = page.replace('<dialog id="sam-lightbox"', section+'\n<dialog id="sam-lightbox"')
    page = page.replace("['detector','pairs','sam','plots']",
                        "['detector','pairs','sam','plots','correspondence']")
    page = page.replace(
        "if(tab==='pairs'&&!$('pairs-frame').src)$('pairs-frame').src=$('pairs-frame').dataset.src;",
        "if(tab==='pairs'&&!$('pairs-frame').src)$('pairs-frame').src=$('pairs-frame').dataset.src;if(tab==='correspondence'&&!$('correspondence-frame').src)$('correspondence-frame').src=$('correspondence-frame').dataset.src;")
    listener = "$('correspondence-tab-button').addEventListener('click',()=>selectTab('correspondence'));"
    if listener not in page:
        page = page.replace(
            "$('plots-tab-button').addEventListener('click',()=>selectTab('plots'));",
            "$('plots-tab-button').addEventListener('click',()=>selectTab('plots'));\n"+listener)
    page = page.replace(
        "if(location.hash==='#pairs')selectTab('pairs');else if(location.hash==='#sam')selectTab('sam');else if(location.hash==='#plots')selectTab('plots');",
        "if(location.hash==='#pairs')selectTab('pairs');else if(location.hash==='#sam')selectTab('sam');else if(location.hash==='#plots')selectTab('plots');else if(location.hash==='#correspondence')selectTab('correspondence');")
    index_path.write_text(page)


def main():
    base = Path(__file__).resolve().parent
    audit_root = base/'outputs/projection_correspondence_audit_4m'
    data = {}
    for class_name in CLASSES:
        data[class_name] = []
        for slug, expected_name in MODEL_SPECS:
            path = audit_root/slug/class_name/'summary.json'
            if not path.is_file():
                raise SystemExit('Missing audit summary: {}'.format(path))
            record = json.loads(path.read_text())
            record['model'] = expected_name
            data[class_name].append(record)
    gallery = base/'outputs/visualizations'
    report_dir = gallery/'projection_correspondence'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir/'index.html').write_text(
        HTML.replace('__DATA__', json.dumps(data, separators=(',', ':'))))
    main_index = gallery/'index.html'
    if main_index.is_file():
        patch_main_index(main_index)
    print('Wrote {}'.format(report_dir/'index.html'))
    print('Updated {}'.format(main_index))


if __name__ == '__main__':
    main()
