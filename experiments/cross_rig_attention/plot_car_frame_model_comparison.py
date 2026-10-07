#!/usr/bin/env python3
"""One canvas comparing R1 and R1-f BEV and all R1-f camera projections."""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import plot_car_frame_bev as bev
import plot_car_multiview_projection as cameras_plot


SVG = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')


def tag(name):
    return '{{{}}}{}'.format(SVG, name)


def read_top(input_dir, token, model, ledger):
    with (input_dir / model / 'formatted/results_pccr.json').open() as handle:
        boxes = json.load(handle)['results'][token]
    selected = sorted(((index, box) for index, box in enumerate(boxes)
                       if box['detection_name'] == 'car'),
                      key=lambda pair: -float(pair[1]['detection_score']))[:10]
    return ([box for _, box in selected],
            [ledger[(model, index)] for index, _ in selected])


def shared_bounds(gt_boxes, ego, predictions):
    points = [corner for box in list(gt_boxes.values())+
              [box for boxes in predictions.values() for box in boxes]
              for corner in bev.box_corners(box)]
    points.append(tuple(float(value) for value in ego['translation'][:2]))
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    span = max(max(xs)-min(xs), max(ys)-min(ys), 20.0)+16
    cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    return cx-span/2, cx+span/2, cy-span/2, cy+span/2


def add_text(root, x, y, value, size, color='#183047', weight='normal'):
    node = ET.SubElement(root, tag('text'), {
        'x': str(x), 'y': str(y), 'font-family': 'Arial',
        'font-size': str(size), 'font-weight': weight, 'fill': color})
    node.text = value


def add_panel(root, path, x, y, width, height):
    panel = ET.parse(path).getroot()
    panel.set('x', str(x))
    panel.set('y', str(y))
    panel.set('width', str(width))
    panel.set('height', str(height))
    root.append(panel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=bev.DEFAULT_INPUT)
    parser.add_argument('--sample-token', default=bev.DEFAULT_TOKEN)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    token = args.sample_token
    output_dir = args.output_dir or args.input_dir / 'car_ap_rank_audit'
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_boxes = bev.car_gt(args.input_dir, token)
    camera_rows, calibration, ego = cameras_plot.records(token)
    ledger = {}
    with (args.input_dir / 'car_ap_rank_audit/ranked_car_predictions.csv').open(
            newline='') as handle:
        for row in csv.DictReader(handle):
            if row['sample_token'] == token:
                ledger[(row['model'], int(row['prediction_index']))] = row
    top = {model: read_top(args.input_dir, token, model, ledger)
           for model in ('R1', 'R1-f')}
    bounds = shared_bounds(gt_boxes, ego,
                           {model: values[0] for model, values in top.items()})

    products = {}
    for model in ('R1', 'R1-f'):
        suffix = model.replace('-', '')
        bev_path = output_dir / 'frame_{}_{}_top10_car_bev.svg'.format(token, suffix)
        camera_path = output_dir / 'frame_{}_{}_top10_car_cameras.svg'.format(token, suffix)
        boxes, statuses = top[model]
        bev.plot(bev_path, token, model, boxes, statuses, gt_boxes, ego, bounds)
        cameras_plot.plot(camera_path, token, model, camera_rows,
                          calibration, ego, gt_boxes, boxes, statuses)
        products[model] = (bev_path, camera_path)

    width, height = 2640, 2590
    root = ET.Element(tag('svg'), {'width': str(width), 'height': str(height),
                                  'viewBox': '0 0 {} {}'.format(width, height)})
    ET.SubElement(root, tag('rect'), {'width': str(width), 'height': str(height),
                                     'fill': '#eaf0f5'})
    add_text(root, 24, 47, 'Same R1-f frame: R1 cross-rig vs R1-f native', 31,
             weight='bold')
    add_text(root, 24, 78, 'Original camera images · shared BEV scale · '
             'green GT · blue TP · red FP · top 10 car outputs per model', 18,
             color='#5b7185')
    for model, x in (('R1', 10), ('R1-f', 1350)):
        bev_path, camera_path = products[model]
        add_panel(root, bev_path, x, 100, 1280, 853)
        add_panel(root, camera_path, x, 960, 1280, 1610)
    combined = output_dir / 'frame_{}_R1_vs_R1f_bev_and_cameras.svg'.format(token)
    ET.ElementTree(root).write(combined, encoding='unicode', xml_declaration=True)
    print('Saved', combined)
    for model in ('R1', 'R1-f'):
        print('Saved', products[model][0])
        print('Saved', products[model][1])


if __name__ == '__main__':
    main()
