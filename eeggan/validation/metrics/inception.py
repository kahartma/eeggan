#  Author: Kay Hartmann <kg.hartma@gmail.com>
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def calculate_inception_score(preds: Tensor, splits: int, repititions: int = 1) -> Tuple[float, float]:
    with torch.no_grad():
        stepsize = int(np.ceil(preds.size(0) / splits))
        steps = np.arange(0, preds.size(0), stepsize)
        scores = []
        for rep in np.arange(repititions):
            preds_tmp = preds[torch.randperm(preds.size(0), device=preds.device)]
            for i in np.arange(len(steps)):
                pred = preds_tmp[steps[i]:steps[i] + stepsize]
                kl = pred * (torch.log(pred) - torch.log(torch.mean(pred, 0, keepdim=True)))
                kl = torch.mean(torch.sum(kl, 1))
                scores.append(torch.exp(kl).item())
        return np.mean(scores).item(), np.std(scores).item()
