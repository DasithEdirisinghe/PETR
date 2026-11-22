# PETR versus StreamPETR precision–recall curves

These are the **official nuScenes validation PR arrays** saved in `metrics_details.json`; no inference or new matching was run. The AP values were checked against both saved `metrics_summary.json` files.

![AP gain by class and center-distance threshold](ap_gain_heatmap.svg)

PETR mAP: 32.78%. StreamPETR mAP: 44.99%. Gain: +12.21 percentage points.

Each class figure overlays PETR and StreamPETR at 0.5, 1, 2, and 4 m. AP integrates the curves beyond 10% recall after subtracting the 10% precision floor.

- [car](pr_car.svg)
- [truck](pr_truck.svg)
- [bus](pr_bus.svg)
- [trailer](pr_trailer.svg)
- [construction vehicle](pr_construction_vehicle.svg)
- [pedestrian](pr_pedestrian.svg)
- [motorcycle](pr_motorcycle.svg)
- [bicycle](pr_bicycle.svg)
- [traffic cone](pr_traffic_cone.svg)
- [barrier](pr_barrier.svg)

See `ap_by_class_threshold.csv` for exact values. These curves cover the official class-wide evaluation; distance or visibility slices require separate custom evaluation definitions.
