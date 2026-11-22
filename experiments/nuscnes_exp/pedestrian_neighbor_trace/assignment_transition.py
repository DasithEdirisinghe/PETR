#!/usr/bin/env python3
"""Compare final greedy GT assignments under PETR and the existing 3D-IoU oracle."""
import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from all_class_reranking_oracle import class_boxes
from analyze import overlaps
from localization_capacity import THRESHOLDS
from reranking_oracle import greedy_match_records, iou3d_oracle_order, tp_measurements

IOU_TOL = 1e-9
SCORE_TOL = 1e-12
METRIC_TOL = 1e-6


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values, prefix):
    return {prefix + '_n': len(values), prefix + '_mean': statistics.mean(values) if values else math.nan,
            prefix + '_median': statistics.median(values) if values else math.nan,
            prefix + '_p90': sorted(values)[math.ceil(.9 * len(values)) - 1] if values else math.nan}


def rate(n, denominator):
    return 100 * n / denominator if denominator else math.nan


def check_metric(actual, expected, label):
    if not math.isclose(actual, float(expected), abs_tol=METRIC_TOL, rel_tol=0):
        raise ValueError('{}: {:.10g} != saved {:.10g}'.format(label, actual, float(expected)))


def gt_tokens(nusc, gt_by_sample, class_name):
    """Map filtered eval GTs back to stable nuScenes annotation tokens."""
    result = {}
    for token, boxes in gt_by_sample.items():
        if not boxes:
            continue
        annotations = [nusc.get('sample_annotation', ann)
                       for ann in nusc.get('sample', token)['anns']]
        for gi, box in enumerate(boxes):
            # nuScenes annotation categories are names such as "vehicle.car";
            # detection boxes use mapped names such as "car". The existing
            # gt_identity() helper resolves annotations by position alone.
            hits = [a['token'] for a in annotations
                    if max(abs(x-y) for x, y in zip(a['translation'], box.translation)) < 1e-4]
            if len(hits) != 1:
                raise ValueError('Ambiguous GT annotation: {} {} {}'.format(class_name, token, gi))
            result[token, gi] = hits[0]
    if len(set(result.values())) != len(result):
        raise ValueError('GT annotation tokens are not unique')
    return result


def make_plot(path, title, labels, series, y_label, ymax=None):
    width, height = 1100, 460
    left, top, right, bottom = 95, 65, 30, 100
    maximum = ymax or max((v for _, values in series for v in values if math.isfinite(v)), default=1)
    maximum = max(maximum, 1e-9)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width,height,width,height),
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="35" y="35" font-size="21" font-family="Arial">{}</text>'.format(escape(title)),
             '<text x="12" y="65" font-size="13" font-family="Arial">{}</text>'.format(escape(y_label))]
    colors = ['#2563eb','#ea580c','#059669','#7c3aed','#db2777','#0891b2','#65a30d','#b45309','#4f46e5','#475569']
    xstep = (width-left-right)/max(1,len(labels)-1)
    for level in range(6):
        y = top + (height-top-bottom)*(1-level/5)
        parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#ddd"/>'.format(left,y,width-right,y))
        parts.append('<text x="{}" y="{}" font-size="11" font-family="Arial">{:.1f}</text>'.format(42,y+4,maximum*level/5))
    for k,label in enumerate(labels):
        x=left+k*xstep
        parts.append('<text x="{:.1f}" y="{}" text-anchor="middle" font-size="11" font-family="Arial">{}</text>'.format(x,height-bottom+20,escape(str(label))))
    for si,(name,values) in enumerate(series):
        color=colors[si%len(colors)]
        points=['{:.1f},{:.1f}'.format(left+k*xstep,top+(height-top-bottom)*(1-v/maximum)) for k,v in enumerate(values) if math.isfinite(v)]
        if len(points)>1:
            parts.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="2"/>'.format(' '.join(points),color))
        for point in points:
            x,y=point.split(',')
            parts.append('<circle cx="{}" cy="{}" r="3" fill="{}"/>'.format(x,y,color))
        if len(series)>1:
            lx=left+(si%5)*195; ly=height-55+(si//5)*20
            parts.append('<text x="{}" y="{}" font-size="11" font-family="Arial" fill="{}">{}</text>'.format(lx,ly,color,escape(name)))
    parts.append('</svg>')
    path.write_text('\n'.join(parts)+'\n')


def analyze_class(evaluator, nusc, class_name, reference):
    gt, _, predictions = class_boxes(evaluator, class_name)
    gt_ids = gt_tokens(nusc, gt, class_name)
    n_gt = len(gt_ids)
    if n_gt != int(reference[class_name, 2.0]['gt_count']):
        raise ValueError('Filtered GT count differs from saved evaluation for '+class_name)
    # Stable IDs are indices in this class's filtered PETR-score-sorted list.
    original = [(p.sample_token, tuple(p.translation), tuple(p.size), tuple(p.rotation), p.detection_score) for p in predictions]
    frame_indices = {}
    for pi,p in enumerate(predictions):
        frame_indices.setdefault(p.sample_token, []).append(pi)
    all_rows, summaries = [], []
    common = set(gt_ids)
    b1_at_threshold = {}
    for threshold in THRESHOLDS:
        oracle_order, _ = iou3d_oracle_order(gt, predictions, threshold)
        if len(oracle_order) != len(predictions) or len(set(oracle_order)) != len(predictions):
            raise ValueError('Oracle order does not contain each original prediction once')
        if original != [(p.sample_token, tuple(p.translation), tuple(p.size), tuple(p.rotation), p.detection_score) for p in predictions]:
            raise ValueError('Oracle changed the prediction set')
        baseline = greedy_match_records(gt, predictions, range(len(predictions)), threshold)
        oracle = greedy_match_records(gt, predictions, oracle_order, threshold)
        ref = reference[class_name, threshold]
        for name,records,prefix in [('baseline',baseline,'baseline'),('oracle',oracle,'max_iou3d_rank')]:
            if len({pi for pi,_ in records.values()}) != len(records):
                raise ValueError(name+' reused a prediction')
            if len(records) != int(ref[prefix+'_full_set_tp']):
                raise ValueError('{} {}m {} TP count differs from saved evaluator'.format(class_name,threshold,name))
            m=tp_measurements(records,gt,predictions)
            check_metric(m['mean_xy_center_error_m'],ref[prefix+'_tp_mean_xy_center_error_m'],name+' XY error')
            check_metric(m['mean_3d_iou'],ref[prefix+'_tp_mean_3d_iou'],name+' 3D IoU')
            check_metric(m['mean_original_score'],ref[prefix+'_tp_mean_original_score'],name+' original score')
        both_keys=set(baseline)&set(oracle)
        common &= both_keys
        b1_keys=set()
        counts=Counter(); b1_ambiguous=0; b_ambiguous=0
        improvements=[]; errors=[]; scores=[]; b1_improvements=[]
        for token,boxes in gt.items():
            for gi,g in enumerate(boxes):
                key=(token,gi)
                bp=baseline.get(key); op=oracle.get(key)
                bi=bp[0] if bp else None; oi=op[0] if op else None
                if bp and not bp[1]<threshold or op and not op[1]<threshold:
                    raise ValueError('Matched pair violates strict XY threshold')
                bs=predictions[bi].detection_score if bp else math.nan
                os=predictions[oi].detection_score if op else math.nan
                bd=bp[1] if bp else math.nan; od=op[1] if op else math.nan
                b_iou=overlaps(predictions[bi],g)[1] if bp else math.nan
                o_iou=overlaps(predictions[oi],g)[1] if op else math.nan
                di=o_iou-b_iou if bp and op else math.nan
                ds=os-bs if bp and op else math.nan
                dd=od-bd if bp and op else math.nan
                bk=sum(math.hypot(predictions[bi].translation[0]-x.translation[0],predictions[bi].translation[1]-x.translation[1])<threshold for x in gt[token]) if bp else None
                ok=sum(math.hypot(predictions[oi].translation[0]-x.translation[0],predictions[oi].translation[1]-x.translation[1])<threshold for x in gt[token]) if op else None
                ambiguity=('multi-GT eligible' if max(bk,ok)>1 else 'single-GT eligible') if bp and op else ''
                if bp and op:
                    if bi==oi: cat='A'
                    elif di>IOU_TOL: cat='B'
                    elif di < -IOU_TOL: cat='C'
                    else: cat='D'
                elif op: cat='E'
                elif bp: cat='F'
                else: cat='G'
                counts[cat]+=1
                subtype=''
                if cat=='B':
                    subtype='B1' if ds < -SCORE_TOL else ('B2' if ds > SCORE_TOL else 'B3')
                    counts[subtype]+=1
                    improvements.append(di); errors.append(dd); scores.append(ds)
                    b_ambiguous+=ambiguity=='multi-GT eligible'
                    if subtype=='B1':
                        b1_keys.add(key); b1_ambiguous+=ambiguity=='multi-GT eligible'; b1_improvements.append(di)
                all_rows.append({'class':class_name,'threshold_m':threshold,'sample_token':token,'gt_id':gt_ids[key],
                    'gt_box_json':json.dumps({'translation':g.translation,'size':g.size,'rotation':g.rotation}),
                    'baseline_prediction_id':('{}:{}'.format(class_name,bi) if bp else ''),
                    'oracle_prediction_id':('{}:{}'.format(class_name,oi) if op else ''),
                    'baseline_prediction_box_json':json.dumps(predictions[bi].serialize()) if bp else '',
                    'oracle_prediction_box_json':json.dumps(predictions[oi].serialize()) if op else '',
                    'category':cat,'subtype':subtype,'baseline_original_score':bs,'oracle_original_score':os,
                    'baseline_3d_iou':b_iou,'oracle_3d_iou':o_iou,'baseline_xy_error_m':bd,'oracle_xy_error_m':od,
                    'delta_3d_iou':di,'delta_original_score':ds,'delta_xy_error_m':dd,
                    'baseline_eligible_gt_count':bk,'oracle_eligible_gt_count':ok,'ambiguity':ambiguity})
        if sum(counts[c] for c in 'ABCDEFG')!=n_gt or sum(counts[c] for c in ('B1','B2','B3'))!=counts['B']:
            raise ValueError('Transition categories do not partition '+class_name)
        if counts['A']+counts['B']+counts['C']+counts['D']+counts['F']!=len(baseline) or counts['A']+counts['B']+counts['C']+counts['D']+counts['E']!=len(oracle):
            raise ValueError('Transition counts do not reproduce TP totals')
        both=counts['A']+counts['B']+counts['C']+counts['D']
        summaries.append({'class':class_name,'threshold_m':threshold,'total_gt':n_gt,'baseline_matched':len(baseline),
            'oracle_matched':len(oracle),'both_matched':both,'changed_assignments':sum(counts[c] for c in 'BCD'),
            **{'category_'+c:counts[c] for c in 'ABCDEFG'},**{'subtype_'+c:counts[c] for c in ('B1','B2','B3')},
            'changed_percent':rate(sum(counts[c] for c in 'BCD'),both),'improved_iou_percent':rate(counts['B'],both),
            'improved_iou_lower_score_percent':rate(counts['B1'],both),'recovered_percent':rate(counts['E'],n_gt),
            'lost_percent':rate(counts['F'],n_gt),'B_multi_gt_count':b_ambiguous,'B1_multi_gt_count':b1_ambiguous,
            'B_multi_gt_percent':rate(b_ambiguous,counts['B']),'B1_multi_gt_percent':rate(b1_ambiguous,counts['B1']),
            **stats(improvements,'B_iou_delta'),**stats(errors,'B_xy_delta_m'),**stats(scores,'B_score_delta'),
            **stats(b1_improvements,'B1_iou_delta')})
        b1_at_threshold[threshold]=b1_keys
        print('{} @ {:g}m: B1={} / both={} ({:.2f}%)'.format(class_name,threshold,counts['B1'],both,rate(counts['B1'],both)),flush=True)
    common_count=len(common)
    for row in summaries:
        row['all_threshold_both_matched_gt']=common_count
        row['B1_on_all_threshold_common_gt']=len(b1_at_threshold[row['threshold_m']] & common)
        row['B1_on_all_threshold_common_percent']=rate(row['B1_on_all_threshold_common_gt'],common_count)
    return all_rows,summaries


def main():
    base=Path(__file__).resolve().parent; root=base.parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-json',type=Path,default=root/'experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/results_nusc.json')
    parser.add_argument('--data-root',type=Path,default=root.parent/'RoboBEV/data/sets/nuscenes')
    parser.add_argument('--reference-csv',type=Path,default=base/'outputs/all_class_reranking_oracle/class_threshold_comparison.csv')
    parser.add_argument('--output-dir',type=Path,default=base/'outputs/assignment_transition')
    args=parser.parse_args()
    for file in (args.results_json,args.reference_csv):
        if not file.is_file(): parser.error('Missing '+str(file))
    if not (args.data_root/'v1.0-trainval').is_dir(): parser.error('Missing nuScenes metadata')
    with args.reference_csv.open(newline='') as f:
        reference={(r['class'],float(r['threshold_m'])):r for r in csv.DictReader(f)}
    from nuscenes import NuScenes
    try: from nuscenes.eval.detection.config import config_factory
    except ImportError: from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    cfg=config_factory('detection_cvpr_2019')
    classes=list(cfg.class_names)
    if set(reference)!={(c,t) for c in classes for t in THRESHOLDS}:
        raise ValueError('Reference CSV must contain exactly ten classes and four thresholds')
    nusc=NuScenes(version='v1.0-trainval',dataroot=str(args.data_root),verbose=False)
    evaluator=NuScenesEval(nusc,config=cfg,result_path=str(args.results_json),eval_set='val',output_dir=str(args.output_dir/'source_eval'),verbose=False)
    records=[]; summary=[]
    for c in classes:
        r,s=analyze_class(evaluator,nusc,c,reference); records.extend(r); summary.extend(s)
    # All validation above precedes creation of research outputs.
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/'gt_transitions.csv',records)
    write_csv(args.output_dir/'all_threshold_summary.csv',summary)
    primary=[r for r in summary if r['threshold_m']==2.0]
    write_csv(args.output_dir/'primary_2m_summary.csv',primary)
    make_plot(args.output_dir/'b1_by_class_2m.svg','B1 frequency by class at 2 m',
              [r['class'] for r in primary],[('B1 / both matched %',[r['improved_iou_lower_score_percent'] for r in primary])],
              '% of GTs matched by both')
    make_plot(args.output_dir/'b1_by_threshold.svg','B1 frequency by threshold',
              [str(t)+' m' for t in THRESHOLDS],[(c,[next(r['improved_iou_lower_score_percent'] for r in summary if r['class']==c and r['threshold_m']==t) for t in THRESHOLDS]) for c in classes],
              '% of GTs matched by both')
    # A 10-class distribution is shown as per-class empirical quantiles of B1 gains.
    quantiles=[]
    for c in classes:
        values=sorted(float(r['delta_3d_iou']) for r in records if r['class']==c and r['threshold_m']==2.0 and r['subtype']=='B1')
        quantiles.append((c,[values[min(len(values)-1,math.ceil(q*len(values))-1)] if values else math.nan for q in (.1,.25,.5,.75,.9)]))
    make_plot(args.output_dir/'b1_iou_gain_distribution_2m.svg','B1 IoU gain quantiles at 2 m',
              ['10th','25th','Median','75th','90th'],quantiles,'positive 3D IoU improvement')
    (args.output_dir/'method.json').write_text(json.dumps({'source_results_json':str(args.results_json.resolve()),
       'reference_csv':str(args.reference_csv.resolve()),'prediction_id':'class:zero-based index in filtered PETR-score-sorted predictions',
       'final_assignment':'greedy nearest unmatched GT, strict XY center distance',
       'iou_tolerance':IOU_TOL,'score_tolerance':SCORE_TOL,'saved_geometry_tolerance':METRIC_TOL,
       'b1_denominator':'GTs matched under both evaluations',
       'distribution_plot':'2 m B1 positive IoU gain empirical quantiles'},indent=2)+'\n')
    print('Validated and wrote {} GT-threshold records to {}'.format(len(records),args.output_dir))

if __name__=='__main__': main()
