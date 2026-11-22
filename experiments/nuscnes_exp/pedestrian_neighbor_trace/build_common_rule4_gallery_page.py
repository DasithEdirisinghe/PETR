#!/usr/bin/env python3
"""Write synchronized four-model Rule-4 hypothesis comparison pages."""

import csv
import json
from pathlib import Path

from build_common_multimodel_gallery_page import HTML
from prepare_common_multimodel_gallery import MODEL_SPECS


def main():
    base = Path(__file__).resolve().parent
    outputs = base/'outputs'
    gallery_root = outputs/'visualizations'
    analysis_root = outputs/'multimodel_rule4_analysis_4m_margin_0p05'
    page_template = HTML.replace('__RULE_VERSION__', '4').replace(
        'Common multi-model hypothesis comparison',
        'Common multi-model Rule-4 hypothesis comparison').replace(
        'Each case is the same nuScenes GT across all four models.',
        'Rule 4 requires an officially unmatched, target-corresponding, '
        'geometrically better alternative. Each case is the same nuScenes GT '
        'across all four models.')
    old_table = ('<table><thead><tr><th>Model</th><th>Matched GT</th>'
                 '<th>Any better</th><th>Unmatched better</th>'
                 '<th>Better assigned elsewhere</th></tr></thead>'
                 '<tbody id="model-counts"></tbody></table>')
    new_table = (
        '<h3>Rule-4 alternative exists</h3>'
        '<table><thead><tr><th>Model</th><th>Rule-4 cases (% matched)</th>'
        '<th>Matches active projection filter</th>'
        '<th>Alternative mean confidence</th><th>Related TP → target GT</th>'
        '<th>Related TP → nearby same-class GT</th>'
        '<th>Related TP → other class/background</th></tr></thead>'
        '<tbody id="rule4-model-counts"></tbody></table>'
        '<h3>No Rule-4 alternative</h3>'
        '<table><thead><tr><th>Model</th><th>No-alternative cases (% matched)</th>'
        '<th>Matches active projection filter</th>'
        '<th>TP → target GT</th><th>TP → nearby same-class GT</th>'
        '<th>TP → other class/background</th></tr></thead>'
        '<tbody id="no-rule4-model-counts"></tbody></table>')
    page_template = page_template.replace(old_table, new_table).replace(
        'Better uses projected 2D IoU first, then 3D IoU, then center distance '
        'when both 3D IoUs are zero. Any better = unmatched better + better '
        'assigned elsewhere.',
        'Rule 4 uses only officially unmatched candidates whose target projected '
        'IoU exceeds every competing annotation by at least 0.05. Confidence is '
        'reported but is not a selection condition.')
    old_script = ("$('model-counts').innerHTML=modelCounts.map(r=>`<tr><td>"
                  "${r.model}</td><td>${r.matched.toLocaleString()}</td><td>"
                  "${r.any_better.toLocaleString()}</td><td>${r.unmatched_better.toLocaleString()}"
                  "</td><td>${r.assigned_other_better.toLocaleString()}</td></tr>`).join('');")
    new_script = (
        "const alignmentCell=x=>`${x.count.toLocaleString()} "
        "(${x.percent.toFixed(1)}%; ${x.mean_confidence.toFixed(3)})`;"
        "function activeCount(r,group,filter){const prefix=group==='rule4'?'rule4':'no_rule4',"
        "total=r[prefix],target=r[prefix+'_target'].count,nearby=r[prefix+'_nearby'].count,"
        "other=r[prefix+'_other'].count;if(filter==='all')return total;"
        "if(filter==='tp_target')return target;if(filter==='tp_nearby')return nearby;"
        "if(filter==='tp_other')return other;if(group==='no_rule4')return 0;"
        "if(filter==='alt_target')return total;if(filter==='alt_nearby'||filter==='alt_other')return 0;"
        "if(filter==='tp_nearby_alt_target')return nearby;"
        "if(filter==='tp_target_alt_target')return target;"
        "if(filter==='tp_nearby_alt_nearby')return 0;return total}"
        "function renderRule4Counts(){const filter=$('correspondence').value;"
        "$('rule4-model-counts').innerHTML=modelCounts.map(r=>`<tr><td>"
        "${r.model}</td><td>${r.rule4.toLocaleString()} (${r.percent.toFixed(1)}%)"
        "</td><td><b>${activeCount(r,'rule4',filter).toLocaleString()}</b></td>"
        "<td>${r.alt_score.toFixed(3)}</td><td>${alignmentCell(r.rule4_target)}"
        "</td><td>${alignmentCell(r.rule4_nearby)}</td><td>"
        "${alignmentCell(r.rule4_other)}</td></tr>`).join('');"
        "$('no-rule4-model-counts').innerHTML=modelCounts.map(r=>`<tr><td>"
        "${r.model}</td><td>${r.no_rule4.toLocaleString()} "
        "(${r.no_rule4_percent.toFixed(1)}%)</td><td><b>"
        "${activeCount(r,'no_rule4',filter).toLocaleString()}</b></td><td>"
        "${alignmentCell(r.no_rule4_target)}</td><td>"
        "${alignmentCell(r.no_rule4_nearby)}</td><td>"
        "${alignmentCell(r.no_rule4_other)}</td></tr>`).join('')}renderRule4Counts();")
    page_template = page_template.replace(old_script, new_script)
    found = 0
    for class_name in ('pedestrian', 'car', 'bicycle'):
        directory = gallery_root/('hypothesis_pairs_rule4_common_'+class_name)
        if not (directory/'comparison_data.js').is_file():
            continue
        counts = []
        for slug, model_name in MODEL_SPECS:
            summary = json.loads((analysis_root/slug/class_name/'summary.json').read_text())
            metrics = summary['mean_metrics_rule4_cases']
            rule4_alignment = {
                row['alignment_category']: row
                for row in summary['alignment_rule4_cases']
                if row['prediction_type'] == 'official_tp'}
            no_rule4_alignment = {
                row['alignment_category']: row
                for row in summary['alignment_no_rule4_cases']
                if row['prediction_type'] == 'official_tp'}
            counts.append({
                'model': model_name, 'matched': summary['matched_gt_count'],
                'rule4': summary['rule4_alternative_count'],
                'percent': summary['rule4_alternative_percent_of_matched'],
                'no_rule4': summary['no_rule4_alternative_count'],
                'no_rule4_percent': 100*summary['no_rule4_alternative_count'] /
                summary['matched_gt_count'],
                'alt_score': metrics['alternative']['score'],
                'rule4_target': rule4_alignment['target_gt'],
                'rule4_nearby': rule4_alignment['other_same_class_gt'],
                'rule4_other': rule4_alignment['other_class_or_background'],
                'no_rule4_target': no_rule4_alignment['target_gt'],
                'no_rule4_nearby': no_rule4_alignment['other_same_class_gt'],
                'no_rule4_other':
                no_rule4_alignment['other_class_or_background']})
        (directory/'model_counts.js').write_text(
            'const modelCounts='+json.dumps(counts, separators=(',', ':'))+';\n')
        coverage = []
        if class_name == 'pedestrian':
            comparison = outputs/'gt_pointing_coverage/comparison.csv'
            if comparison.is_file():
                model_names = {slug: model_name
                               for slug, model_name in MODEL_SPECS}
                with comparison.open(newline='') as handle:
                    rows = {row['model_slug']: row
                            for row in csv.DictReader(handle)
                            if row['visibility'] == 'all'}
                for slug, _ in MODEL_SPECS:
                    row = rows[slug]
                    coverage.append({
                            'model': model_names[slug],
                            'covered_gt': int(row['covered_gt']),
                            'coverage_percent': float(row['coverage_percent']),
                            'mean_max_pointing_score': float(
                                row['mean_max_pointing_score']),
                            'median_max_pointing_score': float(
                                row['median_max_pointing_score']),
                            })
        with (directory/'model_counts.js').open('a') as handle:
            handle.write('const coverageSummary='+json.dumps(
                coverage, separators=(',', ':'))+';\n')
        (directory/'index.html').write_text(page_template)
        print(directory/'index.html')
        found += 1
    if not found:
        raise SystemExit('No prepared Rule-4 comparison_data.js files found')
    main_index = gallery_root/'index.html'
    if main_index.is_file():
        page = main_index.read_text().replace(
            'data-src="hypothesis_pairs_common_pedestrian/index.html?v=3"',
            'data-src="hypothesis_pairs_rule4_common_pedestrian/index.html?v=5"')
        page = page.replace(
            'data-src="hypothesis_pairs_rule4_common_pedestrian/index.html?v=4"',
            'data-src="hypothesis_pairs_rule4_common_pedestrian/index.html?v=6"')
        page = page.replace(
            'data-src="hypothesis_pairs_rule4_common_pedestrian/index.html?v=5"',
            'data-src="hypothesis_pairs_rule4_common_pedestrian/index.html?v=7"')
        page = page.replace(
            'data-src="hypothesis_pairs_rule4_common_pedestrian/index.html?v=6"',
            'data-src="hypothesis_pairs_rule4_common_pedestrian/index.html?v=7"')
        main_index.write_text(page)


if __name__ == '__main__':
    main()
