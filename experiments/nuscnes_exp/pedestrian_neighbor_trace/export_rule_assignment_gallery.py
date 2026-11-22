"""Export changed pedestrian HM/rule pairs from original PETR train predictions.

Run in the PETR CUDA environment from the repository root. Uses training
augmentation with a fixed seed, no optimizer, and the last decoder layer.
"""
import argparse
import importlib
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def corners(boxes):
    signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                      [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
    out = signs[None] * boxes[:, None, 3:6].clip(0) / 2
    c, s = np.cos(boxes[:, 6])[:, None], np.sin(boxes[:, 6])[:, None]
    x, y = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0], out[..., 1] = c*x-s*y, s*x+c*y
    return out + boxes[:, None, :3]


def projected(boxes, matrix):
    xyz = corners(boxes)
    points = np.concatenate([xyz, np.ones(xyz.shape[:2]+(1,))], axis=-1)
    p = points @ np.asarray(matrix).T
    # Match the rule's conservative near-plane validity convention.
    return [(row[:, :2]/row[:, 2:3]).tolist() if (row[:, 2] > .1).all()
            and np.isfinite(row).all() else None for row in p]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='projects/configs/petr/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v6.py')
    parser.add_argument('--checkpoint', default='results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8/epoch_24.pth')
    parser.add_argument('--output-dir', default='experiments/nuscnes_exp/pedestrian_neighbor_trace/outputs/visualizations/rule_assignments')
    parser.add_argument('--max-batches', type=int, default=20)
    parser.add_argument('--max-cases-per-category', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.max_batches < 1 or args.max_cases_per_category < 1:
        raise ValueError('Batch and case limits must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    importlib.import_module('projects.mmdet3d_plugin')
    from projects.mmdet3d_plugin.models.dense_heads.ray_rule_v2 import prepare_rule_v2_layer
    from projects.mmdet3d_plugin.models.dense_heads.ray_rule_loss import _camera_geometry, _project_rectangles
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset, samples_per_gpu=cfg.data.samples_per_gpu,
                             workers_per_gpu=cfg.data.workers_per_gpu,
                             num_gpus=1, dist=False, shuffle=False, seed=args.seed)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights(); load_checkpoint(model, args.checkpoint, map_location='cpu')
    wrapper = MMDataParallel(model.cuda(), device_ids=[0]); wrapper.train()
    head = model.pts_bbox_head
    rule_cfg = dict(head.rule_v2_cfg)
    out = Path(args.output_dir); (out/'images').mkdir(parents=True, exist_ok=True)
    cases, kept, counts = [], Counter(), Counter()
    frames = batches = pedestrian_gt = changed = 0
    for batch_id, data in enumerate(loader):
        if batch_id >= args.max_batches: break
        _, scattered = wrapper.scatter((), data, [0])
        batch = scattered[0]
        metas, images = batch['img_metas'], batch['img']
        # Preserve the augmented image before extract_feat reshapes it or
        # applies GridMask. These images share the rule's augmented calibration.
        display_images = images.detach().cpu().numpy().copy()
        gt = [torch.cat((b.gravity_center, b.tensor[:, 3:]), dim=1).to(images.device)
              for b in batch['gt_bboxes_3d']]
        labels = batch['gt_labels_3d']
        with torch.no_grad():
            features = model.extract_feat(images, metas)
            predictions = head(features, metas)
            logits = predictions['all_cls_scores'][-1].float()
            encoded = predictions['all_bbox_preds'][-1].float()
            prepared = prepare_rule_v2_layer(logits, encoded, gt, labels, metas,
                                              head.assigner, rule_cfg)
        rules, originals = prepared[:2]
        for b, meta in enumerate(metas):
            frames += 1
            pedestrian_gt += int((labels[b] == 8).sum())
            if not gt[b].numel(): continue
            boxes = denormalize_bbox(encoded[b], None)
            overlaps, radial, tangent = _camera_geometry(boxes, gt[b], meta)
            distance = torch.linalg.vector_norm(boxes[:, None, :2]-gt[b][None, :, :2], dim=-1)
            best, owner = overlaps.max(1)
            second = overlaps.topk(2, dim=1).values[:, 1] if gt[b].size(0)>1 else torch.zeros_like(best)
            clear = (best >= rule_cfg['min_projected_iou']) & (best-second >= rule_cfg['owner_iou_margin'])
            probs = logits[b].sigmoid()
            def info(q, g):
                c = int(labels[b][g]); z = logits[b][q]
                compatible = bool(probs[q,c] >= rule_cfg['min_class_score'] and
                                  z[c] >= z.max()-rule_cfg['class_logit_margin'])
                has_view = float(overlaps[q,g]) > 0
                reasons = []
                if int(owner[q]) != g: reasons.append('projects more strongly to another GT')
                if not bool(clear[q]): reasons.append('insufficient overlap or ownership margin')
                if not compatible: reasons.append('class compatibility failed')
                if float(distance[q,g]) > rule_cfg['max_certified_center']: reasons.append('center distance exceeds limit')
                if has_view and abs(float(tangent[q,g])) > rule_cfg['max_certified_tangent']: reasons.append('tangential error exceeds limit')
                return dict(query=q, box=boxes[q].cpu().tolist(), score=float(probs[q,c]),
                            top_class=int(z.argmax()), projected_iou=float(overlaps[q,g]),
                            center=float(distance[q,g]), radial=float(radial[q,g]) if has_view else None,
                            tangent=float(tangent[q,g]) if has_view else None,
                            owner=int(owner[q]), owner_margin=float(best[q]-second[q]),
                            eligibility_failures=reasons, class_compatible=compatible)
            certified_pairs={(bb,q,c) for bb,q,c in prepared[2]}
            old_by_gt={int(g)-1:q for q,g in enumerate(originals[b].gt_inds.tolist()) if g>0}
            for q,g1 in enumerate(rules[b].gt_inds.tolist()):
                g=g1-1
                if g<0 or int(labels[b][g])!=8 or g not in old_by_gt or old_by_gt[g]==q: continue
                changed += 1
                old=old_by_gt[g]
                pair_type='certified' if (b,q,8) in certified_pairs else 'hm_fallback'
                overlap_type='overlap' if float(overlaps[q,g])>0 else 'zero_overlap'
                delta=float(distance[q,g]-distance[old,g])
                quality='better' if delta < -1e-4 else 'worse' if delta > 1e-4 else 'unchanged'
                category=pair_type+'_'+overlap_type
                counts[category]+=1; counts['localization_'+quality]+=1
                if kept[(category,quality)]>=args.max_cases_per_category: continue
                kept[(category,quality)]+=1
                token=str(meta.get('sample_idx', '{}_{}'.format(batch_id,b)))
                frame_key='b{}_f{}'.format(batch_id,b)
                cameras=[]
                all_boxes=torch.cat((gt[b],boxes[old:old+1],boxes[q:q+1]),0).cpu().numpy()
                for v,(matrix,shape) in enumerate(zip(meta['lidar2img'],meta['img_shape'])):
                    filename='images/{}_v{}.jpg'.format(frame_key,v)
                    if not (out/filename).exists():
                        norm=meta.get('img_norm_cfg',cfg.get('img_norm_cfg'))
                        im=display_images[b,v].transpose(1,2,0)
                        im=im*np.asarray(norm['std'])+np.asarray(norm['mean'])
                        im=np.uint8(np.clip(im,0,255))
                        if norm.get('to_rgb',True): im=im[...,::-1]
                        if not cv2.imwrite(str(out/filename),im): raise IOError(filename)
                    rect,valid=_project_rectangles(gt[b][g:g+1],matrix,shape)
                    area=float((rect[0,2:]-rect[0,:2]).clamp(min=0).prod()) if bool(valid[0]) else 0
                    gt_proj=projected(all_boxes,matrix)
                    origin=np.linalg.inv(np.asarray(matrix))[:3,3].tolist()
                    cameras.append(dict(image=filename, projection=gt_proj, origin=origin,
                                        gt_area=area, name=Path(meta['filename'][v]).parent.name,
                                        width=int(display_images.shape[-1]),height=int(display_images.shape[-2])))
                # Match the rule's best-overlap camera for this pair. If no
                # overlap exists, select the view with largest GT footprint.
                ious=[]
                from projects.mmdet3d_plugin.models.dense_heads.ray_rule_loss import _pairwise_iou
                for matrix,shape in zip(meta['lidar2img'],meta['img_shape']):
                    pr,pv=_project_rectangles(boxes[q:q+1],matrix,shape)
                    gr,gv=_project_rectangles(gt[b][g:g+1],matrix,shape)
                    ious.append(float(_pairwise_iou(pr,gr)[0,0]) if bool(pv[0] and gv[0]) else 0.)
                view=int(np.argmax(ious)) if max(ious)>0 else int(np.argmax([c['gt_area'] for c in cameras]))
                cases.append(dict(id=len(cases),sample=token,gt_index=g,category=category,
                                  quality=quality,decoder_layer=int(predictions['all_cls_scores'].size(0)-1),
                                  original=info(old,g),rule=info(q,g),gt_boxes=gt[b].cpu().tolist(),
                                  gt_labels=labels[b].cpu().tolist(),cameras=cameras,default_view=view))
        batches+=1
        payload=dict(checkpoint=args.checkpoint,config=args.config,seed=args.seed,
                     mode='train; augmented images shown before GridMask',decoder_layer='last',
                     batches=batches,frames=frames,pedestrian_gt=pedestrian_gt,changed=changed,
                     changed_fraction=changed/max(1,pedestrian_gt),counts=dict(counts),cases=cases,
                     case_selection='first cases per assignment/localization category; not representative')
        (out/'data.json').write_text(json.dumps(payload,allow_nan=False)+'\n')
        print('batches={} frames={} pedestrian_GT={} changed={} saved={}'.format(
            batches,frames,pedestrian_gt,changed,len(cases)),flush=True)
    print('Gallery data:',out/'data.json',flush=True)


if __name__=='__main__': main()
