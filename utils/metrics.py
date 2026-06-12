from typing import Dict

import torch


class AverageMeter:
    """Tracks average of a scalar metric"""

    def __init__(self):
        self.sum: float = 0.0
        self.count: int = 0
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    r"""
    Computes top-k accuracy

    Definition: Top-k accuracy measures whether the correct label appears among the model’s top k predicted classes,
    rather than requiring it to be the single highest-probability prediction.


    \text{Top-k Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\{y_i \in \text{Top-k predictions}_i\}
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / batch_size).item())
        return res


class MetricTracker:
    """
    Generic metric container
    """

    def __init__(self, *metric_names):
        self.meters = {name: AverageMeter() for name in metric_names}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                raise KeyError(f"Metric '{k}' not initialized")
            self.meters[k].update(v)

    def compute(self) -> Dict[str, float]:
        return {k: meter.avg for k, meter in self.meters.items()}

    def reset(self):
        for meter in self.meters.values():
            meter.reset()
