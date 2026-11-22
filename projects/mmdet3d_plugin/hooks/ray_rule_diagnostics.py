"""First-epoch diagnostics for the opt-in PETR ray-rule loss."""

import json
import os

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook, get_dist_info


@HOOKS.register_module()
class RayRuleDiagnosticsHook(Hook):
    """Aggregate loss, role, and sampled decoder-gradient statistics.

    Gradient norms are measured at the common decoder output before the
    optimizer step. They are diagnostic signals, not parameter update norms.
    """

    def __init__(self, max_epochs=1, grad_interval=50, filename='ray_rule_diagnostics_epoch_1.json'):
        if max_epochs < 1 or grad_interval < 1:
            raise ValueError('max_epochs and grad_interval must be positive')
        self.max_epochs = max_epochs
        self.grad_interval = grad_interval
        self.filename = filename
        self.sums = {}
        self.iterations = 0
        self.grad_samples = 0

    @staticmethod
    def _model(runner):
        return runner.model.module if hasattr(runner.model, 'module') else runner.model

    @classmethod
    def _head(cls, runner):
        return cls._model(runner).pts_bbox_head

    def before_train_epoch(self, runner):
        active = runner.epoch < self.max_epochs
        head = self._head(runner)
        if getattr(head, 'ray_rule_cfg', None) is None:
            raise RuntimeError('RayRuleDiagnosticsHook requires ray_rule_cfg')
        head.ray_rule_diagnostics_active = active
        head.ray_rule_diagnostics_grad_sample = False
        if active:
            self.sums = {}
            self.iterations = 0
            self.grad_samples = 0

    def before_train_iter(self, runner):
        head = self._head(runner)
        head.ray_rule_diagnostics_grad_sample = (
            head.ray_rule_diagnostics_active and
            self.iterations % self.grad_interval == 0)

    def after_train_iter(self, runner):
        if not self._head(runner).ray_rule_diagnostics_active:
            return
        log_vars = runner.outputs.get('log_vars', {})
        metrics = {k[len('ray_diag_'):]: float(v)
                   for k, v in log_vars.items() if k.startswith('ray_diag_')}
        if not metrics:
            raise RuntimeError('Ray-rule diagnostics were not returned by the model')
        for name, value in metrics.items():
            self.sums[name] = self.sums.get(name, 0.0) + value
        self.iterations += 1
        self.grad_samples += int('grad_cls' in metrics)

    def after_train_epoch(self, runner):
        if runner.epoch >= self.max_epochs or not self.iterations:
            return
        metrics = self.sums
        rank, world_size = get_dist_info()
        report = {
            'epoch': runner.epoch + 1,
            'iterations': self.iterations,
            'gradient_samples': self.grad_samples,
            'distributed_world_size': world_size,
            'loss_mean_per_iteration': {
                key: metrics.get(key + '_value', 0.0) / self.iterations
                for key in ('cls', 'bbox', 'rule')},
            'decoder_gradient_norm_mean_on_sampled_iterations': {
                key: metrics.get('grad_' + key, 0.0) / max(1, self.grad_samples)
                for key in ('cls', 'bbox', 'rule')},
            'sampled_gradient_alignment': {
                key: metrics.get(key, 0.0) / max(1, self.grad_samples)
                for key in ('grad_cos_rule_cls', 'grad_cos_rule_bbox',
                            'logit_opposition_fraction',
                            'logit_rule_wins_fraction')},
            'cases': {},
            'per_class': {},
        }
        model = self._model(runner)
        report['parameter_freeze_state'] = getattr(
            model, 'parameter_freeze_state', None)
        report['actual_trainable_parameter_count'] = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad)
        outer_weight = metrics.get('rule_outer_weight', 0.0) / self.iterations
        report['rule_outer_weight'] = outer_weight
        for role in ('preferred', 'inferior'):
            count = metrics.get(role, 0.0)
            conflict_count = metrics.get(role + '_hm_conflict', 0.0)
            report['cases'][role] = {
                'count': count * world_size,
                'raw_bce_mean': metrics.get(role + '_bce_sum', 0.0) / max(1.0, count),
                'conflict_weighted_bce_mean': metrics.get(
                    role + '_weighted_bce_sum', 0.0) / max(1.0, count),
                'weighted_bce_mean_after_outer_weight': outer_weight * metrics.get(
                    role + '_weighted_bce_sum', 0.0) / max(1.0, count),
                'hm_conflict_count': conflict_count * world_size,
                'hm_conflict_raw_bce_mean': metrics.get(
                    role + '_conflict_bce_sum', 0.0) / max(1.0, conflict_count),
            }
        for role in ('radial_inferior', 'tangential_inferior'):
            count = metrics.get(role, 0.0)
            report['cases'][role] = {
                'count': count * world_size,
                'raw_bce_mean': metrics.get(role + '_bce_sum', 0.0) / max(1.0, count),
            }
        report['cases']['ranking_pairs'] = {
            'count': metrics.get('pair_count', 0.0) * world_size,
            'pair_penalty_mean_before_weight': metrics.get('pair_penalty_sum', 0.0) /
            max(1.0, metrics.get('pair_count', 0.0)),
            'inverted_pair_count': metrics.get('rank_violations', 0.0) * world_size,
        }
        for name in metrics:
            if not name.startswith('class_') or not name.endswith('_count'):
                continue
            stem = name[:-6]
            count = metrics[name]
            report['per_class'][stem] = {
                'count': count * world_size,
                'raw_bce_mean': metrics.get(stem + '_bce_sum', 0.0) /
                max(1.0, count),
            }
        if rank == 0:
            os.makedirs(runner.work_dir, exist_ok=True)
            path = os.path.join(
                runner.work_dir, self.filename.format(epoch=runner.epoch + 1))
            with open(path, 'w') as file:
                json.dump(report, file, indent=2)
            runner.logger.info('Ray-rule diagnostics saved to %s', path)
            runner.logger.info('Ray-rule mean losses: %s',
                               report['loss_mean_per_iteration'])
            runner.logger.info('Ray-rule decoder gradient norms: %s',
                               report['decoder_gradient_norm_mean_on_sampled_iterations'])


@HOOKS.register_module()
class RuleV2DiagnosticsHook(Hook):
    """Write per-epoch assignment comparison and ranking statistics."""

    def __init__(self, filename='rule_v2_audit_epoch_{epoch}.json',
                 grad_interval=50, max_examples=300):
        self.filename = filename
        self.grad_interval = grad_interval
        self.max_examples = max_examples
        if grad_interval < 1:
            raise ValueError('grad_interval must be positive')
        self.sums = {}
        self.iterations = 0
        self.examples = []

    def before_train_epoch(self, runner):
        self.sums = {}
        self.iterations = 0
        self.examples = []

    def before_train_iter(self, runner):
        head = RayRuleDiagnosticsHook._head(runner)
        head.rule_v2_diagnostics_grad_sample = (
            self.iterations % self.grad_interval == 0)

    def after_train_iter(self, runner):
        metrics = {k[len('rule_v2_diag_'):]: float(v)
                   for k, v in runner.outputs.get('log_vars', {}).items()
                   if k.startswith('rule_v2_diag_')}
        if not metrics:
            raise RuntimeError('RuleV2DiagnosticsHook requires rule_v2_cfg')
        for name, value in metrics.items():
            self.sums[name] = self.sums.get(name, 0.0) + value
        if len(self.examples) < self.max_examples:
            head = RayRuleDiagnosticsHook._head(runner)
            remaining = self.max_examples - len(self.examples)
            self.examples.extend(head.rule_v2_last_examples[:remaining])
        self.iterations += 1

    def after_train_epoch(self, runner):
        rank, world_size = get_dist_info()
        # Configured keys are identical on all ranks except dynamic class keys.
        # Create all class keys before all_reduce to keep collective shapes equal.
        head = RayRuleDiagnosticsHook._head(runner)
        for c in range(head.num_classes):
            self.sums.setdefault('class_{}_changed'.format(c), 0.0)
        names = sorted(self.sums)
        device = next(head.parameters()).device
        values = torch.tensor([self.iterations] + [self.sums[k] for k in names],
                              device=device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values)
        iterations = max(1.0, float(values[0]))
        totals = {key: float(values[index + 1]) for index, key in enumerate(names)}
        count = max(1.0, totals.get('changed', 0.0))
        report = {
            'epoch': runner.epoch + 1,
            'matching': head.rule_v2_cfg.get('matching', 'original'),
            'world_size': world_size,
            'optimizer_iterations_across_ranks': int(values[0]),
            'totals': totals,
            'changed_fraction_of_gt': totals.get('changed', 0.0) /
                max(1.0, totals.get('gt', 0.0)),
            'changed_mean_old_center_m': totals.get('changed_old_center_sum', 0.0) / count,
            'changed_mean_new_center_m': totals.get('changed_new_center_sum', 0.0) / count,
            'mean_losses_per_iteration': {
                key: totals.get(key + '_value', 0.0) / iterations
                for key in ('cls', 'bbox', 'rank')},
            'sampled_gradient_wrong_fraction': {
                name: totals.get(name + '_combined_wrong', 0.0) /
                    max(1.0, totals.get(name + '_gradient_count', 0.0))
                for name in ('positive', 'negative')},
        }
        if rank == 0:
            os.makedirs(runner.work_dir, exist_ok=True)
            path = os.path.join(
                runner.work_dir, self.filename.format(epoch=runner.epoch + 1))
            with open(path, 'w') as file:
                json.dump(report, file, indent=2)
            example_path = os.path.join(
                runner.work_dir,
                'rule_v2_changed_pairs_epoch_{}.jsonl'.format(runner.epoch + 1))
            with open(example_path, 'w') as file:
                for example in self.examples:
                    file.write(json.dumps(example) + '\n')
            runner.logger.info('Rule V2 assignment audit saved to %s', path)
            runner.logger.info('Rule V2 switched-pair examples saved to %s', example_path)
            runner.logger.info('Rule V2 assignment changes: %.0f / %.0f GTs',
                               totals.get('changed', 0.0), totals.get('gt', 0.0))
