"""Opt-in module freezing for detector fine-tuning.

Module names are exact names from ``model.named_modules()``. A frozen module
includes all descendants. Unfreezing restores each parameter's original
``requires_grad`` value, preserving layers that the architecture already froze.
"""


def _module_names(model, names, field):
    if not isinstance(names, (list, tuple)) or any(
            not isinstance(name, str) or not name for name in names):
        raise ValueError('{} must be a list of nonempty module names'.format(field))
    available = dict(model.named_modules())
    for name in names:
        if name not in available:
            raise ValueError('Unknown module in {}: {}'.format(field, name))
    return tuple(names)


def _inside(parameter_name, module_name):
    return parameter_name.startswith(module_name + '.')


def configure_parameter_freeze(model, config):
    """Apply a validated freeze policy and return state for train() calls."""
    config = dict(config)
    unknown = set(config) - {'freeze_modules', 'unfreeze_modules'}
    if unknown:
        raise ValueError('Unknown parameter_freeze_cfg fields: {}'.format(
            sorted(unknown)))
    frozen = _module_names(model, config.get('freeze_modules', []), 'freeze_modules')
    unfrozen = _module_names(model, config.get('unfreeze_modules', []), 'unfreeze_modules')
    if not frozen and unfrozen:
        raise ValueError('unfreeze_modules requires at least one freeze_modules entry')
    for name in unfrozen:
        if not any(name == parent or name.startswith(parent + '.') for parent in frozen):
            raise ValueError('unfreeze_modules entry is outside frozen modules: ' + name)

    original = {name: parameter.requires_grad
                for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        if any(_inside(name, module) for module in frozen):
            parameter.requires_grad_(False)
        if any(_inside(name, module) for module in unfrozen):
            parameter.requires_grad_(original[name])
    frozen_count = sum(parameter.numel() for name, parameter in model.named_parameters()
                       if original[name] and not parameter.requires_grad)
    trainable_count = sum(parameter.numel() for parameter in model.parameters()
                          if parameter.requires_grad)
    if trainable_count == 0:
        raise ValueError('Freeze policy leaves no trainable parameters')
    return dict(frozen_modules=frozen, unfrozen_modules=unfrozen,
                frozen_parameter_count=frozen_count,
                trainable_parameter_count=trainable_count)


def enforce_frozen_module_eval(model, state):
    """Keep frozen modules in eval mode after a runner calls model.train()."""
    modules = dict(model.named_modules())
    for name in state['frozen_modules']:
        modules[name].eval()
    for name in state['unfrozen_modules']:
        modules[name].train()
