"""Four real warmup updates must pass before measured model performance."""
from .models import state_hash


def validate(report):
    return (isinstance(report, dict) and report.get('passed') is True
            and report.get('updates') == 4 and report.get('finite') is True
            and report.get('optimizer_steps_verified') is True
            and report.get('feature_checks') == 4
            and isinstance(report.get('initial_model_sha256'), str)
            and isinstance(report.get('final_model_sha256'), str)
            and len(report['initial_model_sha256']) == len(report['final_model_sha256']) == 64
            and report['initial_model_sha256'] != report['final_model_sha256'])


class WarmupGate:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.initial = state_hash(model)
        self.losses = []
        self.result = None

    def observe(self, loss, updates, feature_checks):
        import torch
        if updates != len(self.losses) + 1 or not 1 <= updates <= 4:
            raise RuntimeError('Warmup gate requires four ordered updates')
        self.losses.append(loss.detach())
        if updates != 4:
            return
        params = list(self.model.parameters())
        finite = (bool(torch.isfinite(torch.stack(self.losses)).all().item())
                  and all(p.grad is not None and torch.isfinite(p.grad).all().item()
                          and torch.isfinite(p).all().item() for p in params))
        steps = all(p in self.optimizer.state and int(self.optimizer.state[p]['step']) == 4 for p in params)
        result = dict(passed=True, updates=4, finite=finite, optimizer_steps_verified=steps,
                      feature_checks=feature_checks, initial_model_sha256=self.initial,
                      final_model_sha256=state_hash(self.model), native_acceptance=False,
                      scope='first four real warmup updates; worker cleanup still required')
        if not validate(result):
            raise RuntimeError('New-model native warmup gate failed')
        self.result = result

    def require(self):
        if not validate(self.result):
            raise RuntimeError('Four accepted warmup updates required before measurement')
        return dict(self.result)
