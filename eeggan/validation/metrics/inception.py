#  Author: Kay Hartmann <kg.hartma@gmail.com>
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def calculate_inception_score(preds: Tensor, splits: int = 1, repititions: int = 1) -> Tuple[float, float]:
    with torch.no_grad():
        stepsize = np.max((int(np.ceil(preds.size(0) / splits)), 2))
        steps = np.arange(0, preds.size(0), stepsize)
        scores = []
        for rep in np.arange(repititions):
            preds_tmp = preds[torch.randperm(preds.size(0), device=preds.device)]
            if len(preds_tmp) < 2:
                continue
            for i in np.arange(len(steps)):
                preds_step = preds_tmp[steps[i]:steps[i] + stepsize]
                step_mean = torch.mean(preds_step, 0, keepdim=True)
                kl = preds_step * (torch.log(preds_step) - torch.log(step_mean))
                kl = torch.mean(torch.sum(kl, 1))
                scores.append(torch.exp(kl).item())
        return np.mean(scores).item(), np.std(scores).item()
