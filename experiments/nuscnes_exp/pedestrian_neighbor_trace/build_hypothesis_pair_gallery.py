#!/usr/bin/env python3
"""Build a lazy camera/BEV review page for PETR pedestrian query pairs."""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

from visualize import iter_json_array

CLASS_CATEGORIES = {
    'car': {'vehicle.car'},
    'truck': {'vehicle.truck'},
    'bus': {'vehicle.bus.bendy', 'vehicle.bus.rigid'},
    'trailer': {'vehicle.trailer'},
    'construction_vehicle': {'vehicle.construction'},
    'pedestrian': {'human.pedestrian.adult', 'human.pedestrian.child',
                   'human.pedestrian.police_officer',
                   'human.pedestrian.construction_worker'},
    'motorcycle': {'vehicle.motorcycle'},
    'bicycle': {'vehicle.bicycle'},
    'traffic_cone': {'movable_object.trafficcone'},
    'barrier': {'movable_object.barrier'},
}
CLASS_RANGES = {
    'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50,
    'construction_vehicle': 50, 'pedestrian': 40, 'motorcycle': 40,
    'bicycle': 40, 'traffic_cone': 30, 'barrier': 30,
}
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
         (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def inverse_rotate(q, vector):
    w, x, y, z = q
    x, y, z = -x, -y, -z
    vx, vy, vz = vector
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return (vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz,
            vz+w*tz+x*ty-y*tx)


def rotate(q, vector):
    return inverse_rotate((q[0], -q[1], -q[2], -q[3]), vector)


def corners(box):
    width, length, height = box['size']
    result = []
    for z in (-height/2, height/2):
        for x, y in ((-length/2, -width/2), (length/2, -width/2),
                     (length/2, width/2), (-length/2, width/2)):
            delta = rotate(box['rotation'], (x, y, z))
            result.append([box['translation'][i]+delta[i] for i in range(3)])
    return result


def camera_point(point, pose, calibration):
    local = inverse_rotate(pose['rotation'], [a-b for a, b in
                           zip(point, pose['translation'])])
    return inverse_rotate(calibration['rotation'], [a-b for a, b in
                          zip(local, calibration['translation'])])


def projected_segments(box, camera):
    points = [camera_point(p, camera['pose'], camera['calibration'])
              for p in corners(box)]
    intrinsic = camera['calibration']['camera_intrinsic']
    width, height = camera['record']['width'], camera['record']['height']

    def uv(p):
        return [(intrinsic[0][0]*p[0]+intrinsic[0][1]*p[1]+
                 intrinsic[0][2]*p[2])/p[2],
                (intrinsic[1][0]*p[0]+intrinsic[1][1]*p[1]+
                 intrinsic[1][2]*p[2])/p[2]]

    segments = []
    for ia, ib in EDGES:
        a, b = points[ia], points[ib]
        if a[2] <= .1 and b[2] <= .1:
            continue
        if a[2] <= .1:
            t = (.1-a[2])/(b[2]-a[2]); a = [a[i]+t*(b[i]-a[i]) for i in range(3)]
        if b[2] <= .1:
            t = (.1-b[2])/(a[2]-b[2]); b = [b[i]+t*(a[i]-b[i]) for i in range(3)]
        a, b = uv(a), uv(b)
        if max(a[0], b[0]) < 0 or min(a[0], b[0]) >= width or \
                max(a[1], b[1]) < 0 or min(a[1], b[1]) >= height:
            continue
        segments.append([round(a[0], 1), round(a[1], 1),
                         round(b[0], 1), round(b[1], 1)])
    return segments


def load_metadata(data_root, rows, class_name):
    table = data_root/'v1.0-trainval'
    tokens = {row['sample_token'] for row in rows}
    cameras_by_token = {}
    for row in rows:
        cameras_by_token.setdefault(row['sample_token'], set()).add(row['camera'])
    needed_channels = {(t, channel) for t in tokens
                       for channel in {'LIDAR_TOP', *cameras_by_token[t]}}
    sample_data = {}
    for record in iter_json_array(table/'sample_data.json'):
        channel = Path(record['filename']).parent.name
        key = (record['sample_token'], channel)
        if key in needed_channels and record['is_key_frame']:
            sample_data[key] = record
            if len(sample_data) == len(needed_channels):
                break
    if set(sample_data) != needed_channels:
        raise ValueError('Missing keyframe camera or LIDAR_TOP sample data')
    pose_tokens = {r['ego_pose_token'] for r in sample_data.values()}
    calibration_tokens = {sample_data[t, channel]
                          ['calibrated_sensor_token'] for t in tokens
                          for channel in cameras_by_token[t]}
    poses = {r['token']: r for r in iter_json_array(table/'ego_pose.json')
             if r['token'] in pose_tokens}
    calibrations = {r['token']: r for r in iter_json_array(table/'calibrated_sensor.json')
                    if r['token'] in calibration_tokens}
    categories = {r['token']: r['name'] for r in iter_json_array(table/'category.json')}
    class_instances = {r['token'] for r in iter_json_array(table/'instance.json')
                       if categories[r['category_token']] in
                       CLASS_CATEGORIES[class_name]}
    annotations = {token: [] for token in tokens}
    for record in iter_json_array(table/'sample_annotation.json'):
        if record['sample_token'] in tokens and \
                record['instance_token'] in class_instances:
            annotations[record['sample_token']].append(record)
    gt, cameras = {}, {}
    for token in tokens:
        lidar = sample_data[token, 'LIDAR_TOP']
        ego = poses[lidar['ego_pose_token']]['translation']
        filtered = []
        for ann in annotations[token]:
            if ann['num_lidar_pts']+ann['num_radar_pts'] <= 0:
                continue
            if math.hypot(ann['translation'][0]-ego[0],
                          ann['translation'][1]-ego[1]) > CLASS_RANGES[class_name]:
                continue
            filtered.append({'translation': ann['translation'], 'size': ann['size'],
                             'rotation': ann['rotation'],
                             'visibility': int(ann['visibility_token'])})
        gt[token] = filtered
        for channel in cameras_by_token[token]:
            camera_sd = sample_data[token, channel]
            cameras[token, channel] = {
                'record': camera_sd, 'pose': poses[camera_sd['ego_pose_token']],
                'calibration': calibrations[camera_sd['calibrated_sensor_token']]}
    return gt, cameras


def number(row, key):
    return float(row[key]) if row[key] != '' else None


def bev_context(camera, gt):
    pose, calibration = camera['pose'], camera['calibration']
    ego = pose['translation']
    sensor_offset = rotate(pose['rotation'], calibration['translation'])
    origin = [ego[i] + sensor_offset[i] for i in range(3)]
    camera_forward_ego = rotate(calibration['rotation'], (0, 0, 1))
    camera_forward = rotate(pose['rotation'], camera_forward_ego)
    ego_forward = rotate(pose['rotation'], (1, 0, 0))
    fx = calibration['camera_intrinsic'][0][0]
    horizontal_fov = 2*math.atan(camera['record']['width']/(2*fx))
    return {
        'ego': ego[:2], 'ego_forward': ego_forward[:2],
        'camera_origin': origin[:2], 'camera_forward': camera_forward[:2],
        'camera_fov_rad': horizontal_fov,
        'gt_distance_m': math.hypot(gt['translation'][0]-ego[0],
                                    gt['translation'][1]-ego[1]),
    }


def build_cases(rows, metadata, data_root, gallery_dir):
    gt_by_token, cameras = metadata
    cases = []
    for index, row in enumerate(rows, 1):
        token, gi = row['sample_token'], int(row['gt_index'])
        gt_boxes, camera = gt_by_token[token], cameras[token, row['camera']]
        if gi >= len(gt_boxes):
            raise ValueError(f'GT index {gi} missing for {token}; filtering mismatch')
        gt = gt_boxes[gi]
        selected = json.loads(row['selected_box_json'])
        alternative = json.loads(row['best_3d_box_json'])
        boxes = {'gt': gt, 'selected': selected, 'alternative': alternative}
        for box in boxes.values():
            box['segments'] = projected_segments(box, camera)
            box['bev_corners'] = [p[:2] for p in corners(box)[:4]]
        image = os.path.relpath(data_root/camera['record']['filename'], gallery_dir)
        cases.append({
            'index': index, 'token': token, 'gt_index': gi,
            'camera': row['camera'], 'image': image,
            'visibility': int(row['visibility_token']),
            'visibility_percent': row['visibility_percent'],
            'width': camera['record']['width'], 'height': camera['record']['height'],
            'bev_context': bev_context(camera, gt),
            'boxes': boxes,
            'selected': {k: number(row, 'selected_'+k) for k in
                         ('score','center_m','iou3d','projected_iou2d','radial_error_m','transverse_error_m')},
            'alternative': {k: number(row, 'best_3d_'+k) for k in
                            ('score','center_m','iou3d','projected_iou2d','radial_error_m','transverse_error_m')},
            'delta_iou3d': number(row, 'delta_iou3d'),
            'delta_iou2d': number(row, 'delta_projected_iou2d'),
        })
    return cases


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PETR hypothesis pairs</title><style>
:root{font-family:Arial,sans-serif;color:#17233a;background:#f3f6fa}*{box-sizing:border-box}body{margin:0;padding:14px}h1{font-size:22px;margin:0 0 5px}p{color:#52627a}.controls,.card{background:white;border:1px solid #ccd7e3;border-radius:8px;padding:10px}.controls{display:flex;gap:8px;align-items:end;flex-wrap:wrap;position:sticky;top:0;z-index:5}.controls label{font-size:12px;font-weight:bold;display:flex;flex-direction:column;gap:3px}.controls select{max-width:440px}.controls button,.controls select{padding:6px;border:1px solid #afbed0;border-radius:5px;background:white}.box-checks{display:flex;gap:9px;align-items:center;padding:5px 8px;border-left:1px solid #ccd7e3}.box-checks label{display:inline-flex;flex-direction:row;align-items:center;gap:4px;font-size:13px}.overall{margin:10px 0}.overall h2{margin:0 0 8px}.overall-grid{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:7px}.overall-stat{background:#f5f8fc;border-radius:6px;padding:8px}.overall-stat b{display:block;font-size:19px;margin-bottom:3px}.overall-stat span{font-size:12px;color:#52627a}.overall table{width:100%;border-collapse:collapse;font-size:12px;margin-top:9px}.overall th,.overall td{padding:5px 7px;border-top:1px solid #dde5ee;text-align:right}.overall th:first-child,.overall td:first-child{text-align:left}.metrics{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:10px;margin:10px 0}.metric{padding:9px}.metric h3{margin:0 0 6px}.metric div{line-height:1.55}.layout{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(360px,1fr);gap:10px}.view{position:relative}.view img{width:100%;display:block}.view svg{position:absolute;inset:0;width:100%;height:100%}.bev svg{width:100%;display:block;background:#f8fafc}.legend{font-size:13px;color:#52627a;margin-bottom:7px}.selected{color:#d97706}.alternative{color:#7c3aed}.gt{color:#2563eb}.comparison{margin-top:10px;overflow-x:auto}.comparison h2{margin:0 0 8px}.comparison table{width:100%;border-collapse:collapse;font-size:13px}.comparison th,.comparison td{padding:7px 9px;border-bottom:1px solid #dde5ee;text-align:right}.comparison th:first-child,.comparison td:first-child{text-align:left}.comparison thead th{background:#f5f8fc}.case-note{font-size:13px;color:#52627a;margin-top:8px}@media(max-width:1100px){.overall-grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:900px){.layout,.metrics{grid-template-columns:1fr}}@media(max-width:600px){.overall-grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><h1>Alternative __MODEL_NAME__ __CLASS_NAME__ hypotheses</h1><p>Each displayed case has a different unmatched prediction selected as the GT's best-3D candidate. Its 3D IoU is higher than or tied with the official TP; ties are resolved by center error and then confidence. The browser uses a deterministic sample of the eligible unmatched cases. Projected overlap and radial error are reported as measurements; they are not required to improve.</p>
<section class="card overall"><h2>Validation-wide summary at 4 m</h2><div id="overall-grid" class="overall-grid"></div></section>
<div class="controls"><label>Class<select id="class-select"><option value="pedestrian">Pedestrian</option><option value="car">Car</option><option value="bicycle">Bicycle</option></select></label><button id="prev">◀</button><label>Case<select id="case"></select></label><button id="next">▶</button><label>Sort<select id="sort"><option value="combined">Combined IoU gain</option><option value="iou3d">3D IoU gain</option><option value="iou2d">Projected 2D IoU gain</option><option value="score">Alternative confidence</option></select></label><span class="box-checks"><strong>Boxes:</strong><label class="gt"><input id="show-gt-box" type="checkbox" checked>GT</label><label class="selected"><input id="show-selected-box" type="checkbox" checked>Official TP</label><label class="alternative"><input id="show-alternative-box" type="checkbox" checked>Alternative</label></span><span id="status"></span></div>
<div class="metrics"><div class="card metric gt"><h3>GT</h3><div id="gt-metrics"></div></div><div class="card metric selected"><h3>Official selected TP</h3><div id="selected-metrics"></div></div><div class="card metric alternative"><h3>Better alternative</h3><div id="alternative-metrics"></div></div></div>
<div class="layout"><section class="card"><h2>Camera projection</h2><div class="legend"><b class="gt">Blue</b> GT · <b class="selected">amber</b> official TP · <b class="alternative">purple</b> alternative</div><div class="view"><img id="image"><svg id="overlay"></svg></div></section><section class="card bev"><h2>Camera-aligned BEV</h2><div class="legend">Selected camera always points up · ego is fixed near bottom centre · cyan wedge = camera FOV · grid = 5 m.</div><svg id="bev" viewBox="0 0 800 800"></svg></section></div>
<section class="card comparison"><h2>Case statistics</h2><table><thead><tr><th>Measurement</th><th>Official TP</th><th>Alternative</th><th>Alternative − TP</th></tr></thead><tbody id="comparison-body"></tbody></table><div id="case-note" class="case-note"></div></section>
<script src="overall.js?v=4"></script><script src="cases.js?v=4"></script><script>
const colors={gt:'#2563eb',selected:'#d97706',alternative:'#7c3aed'},$=x=>document.getElementById(x);let order=[];function f(x,n=3){return x==null?'—':Number(x).toFixed(n)}
function visible(name){return $(name==='gt'?'show-gt-box':name==='selected'?'show-selected-box':'show-alternative-box').checked}
function renderOverall(){const c=overall.counts,matched=overall.official_matched_gt,different=overall.matched_gt_with_different_best_3d,assignedOther=c.best_3d_already_tp_other_gt,unmatched=different-assignedOther,stats=[['GTs',overall.gt_count,'validation GTs'],['Official matches',matched,`${f(100*matched/overall.gt_count,1)}% of all GTs`],['Different best-3D',different,`${f(100*different/matched,1)}% of matched GTs`],['Best-3D alternative unmatched',unmatched,`${f(100*unmatched/different,1)}% of different-best cases`],['Best-3D alternative assigned to another GT',assignedOther,`${f(100*assignedOther/different,1)}% of different-best cases`]];$('overall-grid').innerHTML=stats.map(([label,value,note])=>`<div class="overall-stat"><b>${Number(value).toLocaleString()}</b><span>${label}<br>${note}</span></div>`).join('')}
function sortCases(){const mode=$('sort').value;order=[...cases].sort((a,b)=>{if(mode==='iou3d')return b.delta_iou3d-a.delta_iou3d;if(mode==='iou2d')return b.delta_iou2d-a.delta_iou2d;if(mode==='score')return b.alternative.score-a.alternative.score;return (b.delta_iou3d+b.delta_iou2d)-(a.delta_iou3d+a.delta_iou2d)});$('case').innerHTML=order.map((c,i)=>`<option value="${i}">#${c.index} · ${c.token.slice(0,8)} · G${c.gt_index} · Δ3D ${f(c.delta_iou3d,2)} · Δ2D ${f(c.delta_iou2d,2)}</option>`).join('');render()}
function metric(m){return `confidence <b>${f(m.score)}</b><br>center error <b>${f(m.center_m)} m</b><br>3D IoU <b>${f(m.iou3d)}</b><br>projected 2D IoU <b>${f(m.projected_iou2d)}</b><br>radial / transverse <b>${f(m.radial_error_m)} / ${f(m.transverse_error_m)} m</b>`}
function line(s,color){return `<line x1="${s[0]}" y1="${s[1]}" x2="${s[2]}" y2="${s[3]}" stroke="${color}" stroke-width="4"/>`}
function render(){if(!order.length)return;const c=order[Number($('case').value)||0];$('status').textContent=`${c.camera} · ${order.length} sampled unmatched-alternative cases`;$('image').src=c.image;$('overlay').setAttribute('viewBox',`0 0 ${c.width} ${c.height}`);let overlay='';for(const name of ['gt','selected','alternative'])if(visible(name))for(const s of c.boxes[name].segments)overlay+=line(s,colors[name]);$('overlay').innerHTML=overlay;$('gt-metrics').innerHTML=`${c.token.slice(0,8)} · G${c.gt_index}<br>visibility <b>V${c.visibility} (${c.visibility_percent})</b><br>Δ3D / Δ2D: <b>${f(c.delta_iou3d)} / ${f(c.delta_iou2d)}</b>`;$('selected-metrics').innerHTML=metric(c.selected);$('alternative-metrics').innerHTML=metric(c.alternative);
const ctx=c.bev_context,cam=ctx.camera_origin,ego=ctx.ego,flen=Math.hypot(...ctx.camera_forward),forward=ctx.camera_forward.map(v=>v/flen),side=[-forward[1],forward[0]],range=Math.max(20,ctx.gt_distance_m+10),turn=(v,a)=>[v[0]*Math.cos(a)-v[1]*Math.sin(a),v[0]*Math.sin(a)+v[1]*Math.cos(a)],left=turn(forward,ctx.camera_fov_rad/2),right=turn(forward,-ctx.camera_fov_rad/2),rayL=[cam[0]+left[0]*range,cam[1]+left[1]*range],rayR=[cam[0]+right[0]*range,cam[1]+right[1]*range],world=[ego,cam,rayL,rayR,...Object.values(c.boxes).flatMap(b=>b.bev_corners)],local=p=>{const dx=p[0]-ego[0],dy=p[1]-ego[1];return[dx*side[0]+dy*side[1],dx*forward[0]+dy*forward[1]]},locals=world.map(local),maxAbsLat=Math.max(...locals.map(p=>Math.abs(p[0]))),maxLong=Math.max(...locals.map(p=>p[1])),minLong=Math.min(...locals.map(p=>p[1])),scale=Math.min(45,340/Math.max(6,maxAbsLat),620/Math.max(12,maxLong),90/Math.max(2,-minLong)),xy=p=>{const q=local(p);return[400-q[0]*scale,700-q[1]*scale]};let bev='<rect width="800" height="800" fill="#f8fafc"/>';const step=5*scale;for(let x=400%step;x<=800;x+=step)bev+=`<line x1="${x}" y1="0" x2="${x}" y2="800" stroke="#dde6ef"/>`;for(let y=700%step;y<=800;y+=step)bev+=`<line x1="0" y1="${y}" x2="800" y2="${y}" stroke="#dde6ef"/>`;const wedge=[cam,rayL,rayR].map(xy).map(p=>p.join(',')).join(' ');bev+=`<polygon points="${wedge}" fill="#06b6d4" fill-opacity=".10" stroke="#0891b2" stroke-width="2" stroke-dasharray="8 6"/>`;const efLen=Math.hypot(...ctx.ego_forward),ef=ctx.ego_forward.map(v=>v/efLen),ep=[-ef[1],ef[0]],vehicle=[[ego[0]+ef[0]*2.4,ego[1]+ef[1]*2.4],[ego[0]-ef[0]*2.1+ep[0],ego[1]-ef[1]*2.1+ep[1]],[ego[0]-ef[0]*2.1-ep[0],ego[1]-ef[1]*2.1-ep[1]]].map(xy).map(p=>p.join(',')).join(' '),cameraPx=xy(cam);bev+=`<polygon points="${vehicle}" fill="#16a34a" stroke="#166534" stroke-width="2"/><text x="408" y="692" fill="#166534" font-size="19" font-weight="bold">EGO</text><circle cx="${cameraPx[0]}" cy="${cameraPx[1]}" r="5" fill="#0891b2"/><text x="${cameraPx[0]+8}" y="${cameraPx[1]+18}" fill="#0e7490" font-size="16">${c.camera}</text>`;for(const name of ['gt','selected','alternative'])if(visible(name)){const pts=c.boxes[name].bev_corners.map(xy).map(p=>p.join(',')).join(' '),center=xy(c.boxes[name].translation);bev+=`<polygon points="${pts}" fill="none" stroke="${colors[name]}" stroke-width="4"/><text x="${center[0]+7}" y="${center[1]-7}" fill="${colors[name]}" font-size="22" font-weight="bold">${name==='gt'?'GT':name==='selected'?'TP':'ALT'}</text>`}$('bev').innerHTML=bev;
const rows=[['Confidence','score'],['Ground-plane center error (m)','center_m'],['Absolute radial error (m)','radial_error_m',true],['Absolute transverse error (m)','transverse_error_m',true],['3D IoU','iou3d'],['Projected 2D IoU','projected_iou2d']];$('comparison-body').innerHTML=rows.map(([label,key,absolute])=>{const a=absolute?Math.abs(c.selected[key]):c.selected[key],b=absolute?Math.abs(c.alternative[key]):c.alternative[key];return `<tr><td>${label}</td><td>${f(a)}</td><td>${f(b)}</td><td>${a==null||b==null?'—':f(b-a)}</td></tr>`}).join('');$('case-note').textContent=`V${c.visibility} (${c.visibility_percent}) · Alternative is officially unmatched and is best or tied-best by 3D IoU. Projected 2D overlap and radial error may improve or worsen.`}
$('class-select').value='__CLASS_NAME_SLUG__';$('class-select').onchange=()=>{const name=$('class-select').value;location.href=`../hypothesis_pairs${name==='pedestrian'?'':'_'+name}/index.html`};$('sort').onchange=sortCases;$('case').onchange=render;for(const id of ['show-gt-box','show-selected-box','show-alternative-box'])$(id).onchange=render;$('prev').onclick=()=>{$('case').selectedIndex=($('case').selectedIndex-1+order.length)%order.length;render()};$('next').onclick=()=>{$('case').selectedIndex=($('case').selectedIndex+1)%order.length;render()};renderOverall();sortCases();
</script></body></html>'''


def main():
    base = Path(__file__).resolve().parent
    workspace = base.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-csv', type=Path,
                        default=base/'outputs/pedestrian_image_alignment_4m/all_gt.csv')
    parser.add_argument('--summary-json', type=Path,
                        default=base/'outputs/pedestrian_image_alignment_4m/summary.json')
    parser.add_argument('--data-root', type=Path,
                        default=workspace/'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--gallery-dir', type=Path,
                        default=base/'outputs/visualizations/hypothesis_pairs')
    parser.add_argument('--include-assigned-other-gt', action='store_true')
    parser.add_argument('--max-cases', type=int, default=1000,
                        help='Maximum cases to visualize; 0 keeps all selected cases')
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--class-name', default='pedestrian',
                        choices=tuple(CLASS_CATEGORIES))
    parser.add_argument('--model-name', default='PETR')
    args = parser.parse_args()
    with args.cases_csv.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows = [r for r in rows if r.get('different_3d_candidate', '1') == '1']
    if not args.include_assigned_other_gt:
        rows = [r for r in rows if r['best_3d_status'] == 'unmatched']
    if not rows:
        parser.error('No hypothesis-pair rows selected')
    selected_count = len(rows)
    if args.max_cases and len(rows) > args.max_cases:
        rows = random.Random(args.sample_seed).sample(rows, args.max_cases)
        rows.sort(key=lambda row: (row['sample_token'], int(row['gt_index'])))
    metadata = load_metadata(args.data_root, rows, args.class_name)
    args.gallery_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(rows, metadata, args.data_root, args.gallery_dir)
    (args.gallery_dir/'cases.js').write_text(
        'const cases='+json.dumps(cases, separators=(',', ':'))+';\n')
    summary = json.loads(args.summary_json.read_text())
    (args.gallery_dir/'overall.js').write_text(
        'const overall='+json.dumps(summary, separators=(',', ':'))+';\n')
    (args.gallery_dir/'index.html').write_text(
        HTML.replace('__MODEL_NAME__', args.model_name)
        .replace('__CLASS_NAME__', args.class_name.replace('_', ' '))
        .replace('__CLASS_NAME_SLUG__', args.class_name))
    print('Wrote', len(cases), 'of', selected_count, 'eligible cases to', args.gallery_dir)


if __name__ == '__main__':
    main()
