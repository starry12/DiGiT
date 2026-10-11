"""Predeclared epoch schedules, independent of validation/test outcomes."""
import math


def lr_for_epoch(p, arm, epoch):
    if type(epoch) is not int or not 0 <= epoch < p['epochs']:
        raise ValueError('epoch must be zero based within the declared run')
    rule = p['lr_schedules'][p['arms'][arm]['lr_schedule']]
    return p['optimizer']['kwargs']['lr'] * rule['gamma'] ** sum(epoch >= m for m in rule['milestones'])


def apply_epoch_lr(optimizer, p, arm, epoch):
    value = lr_for_epoch(p, arm, epoch)
    for group in optimizer.param_groups:
        group['lr'] = value
    return value


def validate_schedule(r, arm, p, smoke):
    from .common import require
    n = p['smoke_epochs'] if smoke else p['epochs']
    expected = dict(p['optimizer']['kwargs'], lr=lr_for_epoch(p, arm, n - 1))
    require(r['optimizer_initial'] == p['optimizer']['kwargs'], 'Initial optimizer differs')
    require(r['optimizer_effective'] == expected, 'Final optimizer differs')
    name = p['arms'][arm]['lr_schedule']
    require(r['lr_schedule'] == name and len(r['epochs']) == n, 'Wrong schedule extent')
    for i, e in enumerate(r['epochs']):
        require(e['epoch'] == i + 1 and e['lr_schedule'] == name and
                math.isclose(e['learning_rate'], lr_for_epoch(p, arm, i), rel_tol=1e-12),
                'Incorrect epoch learning rate')
