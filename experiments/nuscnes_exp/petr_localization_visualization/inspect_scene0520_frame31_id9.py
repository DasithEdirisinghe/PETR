#!/usr/bin/env python3
"""Render the six nuScenes keyframe cameras with one GT box (stdlib only)."""

import base64
import json
import math
import mmap
import shutil
import subprocess
from pathlib import Path


DATA = Path(__file__).resolve().parents[4] / "RoboBEV/data/sets/nuscenes"
TABLES = DATA / "v1.0-trainval"
SAMPLE = "41b1bdacae994c26af6dd734e21339ea"  # scene-0520, zero-based frame 31
INSTANCE = "13e6db4bc14545f1be4e65ad1950b74c"  # displayed car ID 9
CAMERAS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
           "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def records_containing(table, needle):
    """Find flat JSON records without loading multi-GB nuScenes tables."""
    with (TABLES / table).open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            pos = 0
            while True:
                pos = data.find(needle.encode(), pos)
                if pos < 0:
                    break
                start = data.rfind(b"\n{", 0, pos) + 1
                end = data.find(b"\n}", pos) + 2
                if start <= 0 or end < 2:
                    raise ValueError("Could not isolate JSON record")
                yield json.loads(data[start:end])
                pos = end


def record(table, token):
    for row in records_containing(table, '"token": "{}"'.format(token)):
        if row["token"] == token:
            return row
    raise KeyError((table, token))


def rotate(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    uv = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    uuv = (y * uv[2] - z * uv[1], z * uv[0] - x * uv[2],
           x * uv[1] - y * uv[0])
    return tuple(v[i] + 2 * (w * uv[i] + uuv[i]) for i in range(3))


def inverse_transform(point, translation, q):
    return rotate((q[0], -q[1], -q[2], -q[3]),
                  [point[i] - translation[i] for i in range(3)])


def corners(ann):
    width, length, height = ann["size"]
    result = []
    for z in (-height / 2, height / 2):
        for x, y in ((length / 2, width / 2), (length / 2, -width / 2),
                     (-length / 2, -width / 2), (-length / 2, width / 2)):
            delta = rotate(ann["rotation"], (x, y, z))
            result.append(tuple(ann["translation"][i] + delta[i]
                                for i in range(3)))
    return result


def project(point, intrinsic):
    x, y, z = point
    return ((intrinsic[0][0] * x + intrinsic[0][1] * y + intrinsic[0][2] * z) / z,
            (intrinsic[1][0] * x + intrinsic[1][1] * y + intrinsic[1][2] * z) / z)


def main():
    ann = next(a for a in records_containing("sample_annotation.json",
                                            '"sample_token": "{}"'.format(SAMPLE))
               if a["instance_token"] == INSTANCE)
    sensors = {s["token"]: s["channel"] for s in json.load(open(TABLES / "sensor.json"))}
    data = {}
    for row in records_containing("sample_data.json",
                                  '"sample_token": "{}"'.format(SAMPLE)):
        if row["is_key_frame"] and row["fileformat"] == "jpg":
            calibration = record("calibrated_sensor.json", row["calibrated_sensor_token"])
            data[sensors[calibration["sensor_token"]]] = (row, calibration)
    if set(CAMERAS) != set(data):
        raise ValueError("Missing camera keyframes: {}".format(set(CAMERAS) - set(data)))

    scale = 0.5
    panel_w, panel_h = 1600 * scale, 900 * scale
    header = 65
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           'width="2400" height="{}" viewBox="0 0 2400 {}">'.format(
               int(2 * (panel_h + 30) + header), int(2 * (panel_h + 30) + header)),
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="20" y="35" font-size="25" font-family="sans-serif">'
           'scene-0520 / frame 31 / car ID 9 / GT visibility token 4</text>']
    summary = []
    for index, camera in enumerate(CAMERAS):
        row, calibration = data[camera]
        pose = record("ego_pose.json", row["ego_pose_token"])
        points = [inverse_transform(inverse_transform(p, pose["translation"], pose["rotation"]),
                                    calibration["translation"], calibration["rotation"])
                  for p in corners(ann)]
        left = (index % 3) * panel_w
        top = header + (index // 3) * (panel_h + 30)
        image_path = DATA / row["filename"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        jpeg = base64.b64encode(image_path.read_bytes()).decode("ascii")
        svg.append('<image x="{:.0f}" y="{:.0f}" width="800" height="450" '
                   'xlink:href="data:image/jpeg;base64,{}"/>'.format(left, top, jpeg))
        svg.append('<clipPath id="panel{}"><rect x="{:.0f}" y="{:.0f}" '
                   'width="800" height="450"/></clipPath>'.format(index, left, top))
        svg.append('<g clip-path="url(#panel{})">'.format(index))
        visible_segments = 0
        projected = []
        for i, j in EDGES:
            a, b = list(points[i]), list(points[j])
            if a[2] <= 0.1 and b[2] <= 0.1:
                continue
            if a[2] <= 0.1:
                t = (0.1 - a[2]) / (b[2] - a[2])
                a = [a[k] + t * (b[k] - a[k]) for k in range(3)]
            if b[2] <= 0.1:
                t = (0.1 - b[2]) / (a[2] - b[2])
                b = [b[k] + t * (a[k] - b[k]) for k in range(3)]
            u, v = project(a, calibration["camera_intrinsic"]), project(b, calibration["camera_intrinsic"])
            projected.extend((u, v))
            if (min(u[0], v[0]) < row["width"] and max(u[0], v[0]) >= 0 and
                    min(u[1], v[1]) < row["height"] and max(u[1], v[1]) >= 0):
                visible_segments += 1
            svg.append('<line x1="{:.2f}" y1="{:.2f}" x2="{:.2f}" y2="{:.2f}" '
                       'stroke="#00ff40" stroke-width="2"/>'.format(
                           left + scale * u[0], top + scale * u[1],
                           left + scale * v[0], top + scale * v[1]))
        svg.append('</g>')
        svg.append('<text x="{:.0f}" y="{:.0f}" font-size="20" '
                   'font-family="sans-serif">{}: {} projected edges intersect image</text>'.format(
                       left + 8, top + panel_h + 23, camera, visible_segments))
        summary.append({"camera": camera, "filename": row["filename"],
                        "segments_intersecting_image": visible_segments,
                        "projected_bounds_px": [round(min(p[k] for p in projected), 1) for k in (0, 1)] +
                                               [round(max(p[k] for p in projected), 1) for k in (0, 1)]
                                               if projected else None})
    svg.append("</svg>")
    output = Path(__file__).parent / "outputs/scene-0520_frame-31_id-9_six_cameras.svg"
    output.write_text("\n".join(svg))
    output.with_suffix(".json").write_text(json.dumps({"annotation": ann, "cameras": summary}, indent=2))
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-o", str(output.with_suffix(".png")), str(output)],
                       check=True)
    print(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
