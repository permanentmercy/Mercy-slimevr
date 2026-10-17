import torch
import torch.nn.functional as F


def sinc(x):
    """sin(x)/x with a finite zero branch and ONNX opset-14 operators."""
    small = x.abs() < 1e-4
    denominator = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1-x.square()/6+x.pow(4)/120, torch.sin(x)/denominator)


def rotvec_to_quat(v):
    angle = v.norm(dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle/2), .5*sinc(angle/2)*v), dim=-1)


def multiply(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av, bv, dim=-1)), dim=-1)


def residual_target(raw, clean):
    raw, clean = F.normalize(raw, dim=-1), F.normalize(clean, dim=-1)
    inverse = torch.cat((raw[..., :1], -raw[..., 1:]), dim=-1)
    return multiply(inverse, clean)  # clean = raw * dq


def blend(raw, dq, gate, threshold=.3):
    """Shortest-arc SLERP from identity residual, then right-multiply raw.

    No accumulation, foot lock, bias estimation or absolute heading recovery.
    """
    raw = F.normalize(raw, dim=-1)
    dq = F.normalize(dq, dim=-1)
    dq = torch.where(dq[..., :1] < 0, -dq, dq)
    w = torch.where(gate >= threshold, gate.clamp(0, 1), torch.zeros_like(gate))[..., None]
    a = torch.atan2(dq[..., 1:].norm(dim=-1, keepdim=True), dq[..., :1])
    scale = w*sinc(w*a)/sinc(a).clamp_min(1e-7)
    correction = torch.cat((torch.cos(w*a), scale*dq[..., 1:]), dim=-1)
    return F.normalize(multiply(raw, correction), dim=-1)
