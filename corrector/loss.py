import torch
import torch.nn.functional as F


def objective(pred, target):
    """Sign-invariant rotation loss and explicitly supervised gate/contact heads.

    gate targets must be independently supplied correction reliability labels.
    No invented joint-limit loss without a calibrated anatomical joint mapping.
    """
    q = F.normalize(target["dq"], dim=-1)
    rotation = (1 - (pred["dq"]*q).sum(-1).square()).mean()
    contact = F.binary_cross_entropy(pred["contact"], target["contact"])
    gate = F.binary_cross_entropy(pred["gate"], target["gate"])
    return rotation + .5*contact + .1*gate
