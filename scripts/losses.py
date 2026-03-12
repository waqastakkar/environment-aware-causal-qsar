from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class IRMDiagnostic:
    env_id: int
    risk_env: float
    grad_norm_sq: float


def _weighted_mean(loss_vec: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    if sample_weight is None:
        return loss_vec.mean()
    w = sample_weight.float().clamp_min(0.0)
    denom = w.sum().clamp_min(1e-12)
    return (loss_vec * w).sum() / denom


def prediction_loss(
    yhat: torch.Tensor,
    y: torch.Tensor,
    *,
    task: str,
    loss_pred: str = "huber",
    loss_cls: str = "bce",
    sample_weight: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> torch.Tensor:
    if task == "regression":
        if loss_pred == "mse":
            lv = F.mse_loss(yhat, y, reduction="none")
        elif loss_pred == "huber":
            lv = F.huber_loss(yhat, y, reduction="none")
        else:
            raise ValueError(f"Unsupported regression loss: {loss_pred}")
        return _weighted_mean(lv, sample_weight)

    if loss_cls == "bce":
        lv = F.binary_cross_entropy_with_logits(yhat, y, reduction="none")
    elif loss_cls == "focal":
        bce = F.binary_cross_entropy_with_logits(yhat, y, reduction="none")
        p = torch.sigmoid(yhat)
        pt = p * y + (1.0 - p) * (1.0 - y)
        alpha_t = focal_alpha * y + (1.0 - focal_alpha) * (1.0 - y)
        lv = alpha_t * torch.pow((1.0 - pt).clamp_min(1e-8), focal_gamma) * bce
    else:
        raise ValueError(f"Unsupported classification loss: {loss_cls}")
    return _weighted_mean(lv, sample_weight)


def compute_env_class_weights(env: torch.Tensor, n_envs: int) -> torch.Tensor:
    counts = torch.bincount(env, minlength=n_envs).float()
    inv = torch.where(counts > 0, 1.0 / counts, torch.zeros_like(counts))
    if inv.sum() > 0:
        inv = inv * (n_envs / inv.sum())
    return inv


def adversary_env_loss(
    env_logits: torch.Tensor,
    env: torch.Tensor,
    *,
    loss_env: str = "ce",
    env_class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = env_class_weights if loss_env == "weighted_ce" else None
    return F.cross_entropy(env_logits, env, weight=weight)


def irmv1_penalty(
    yhat: torch.Tensor,
    y: torch.Tensor,
    env: torch.Tensor,
    *,
    task: str,
    loss_pred: str,
    loss_cls: str,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[IRMDiagnostic]]:
    scale = torch.tensor(1.0, device=yhat.device, requires_grad=True)
    penalty = torch.zeros((), device=yhat.device)
    diagnostics: list[IRMDiagnostic] = []

    for env_id in env.unique(sorted=True):
        m = env == env_id
        if int(m.sum().item()) < 2:
            continue
        sw = sample_weight[m] if sample_weight is not None else None
        risk_env = prediction_loss(
            yhat[m] * scale,
            y[m],
            task=task,
            loss_pred=loss_pred,
            loss_cls=loss_cls,
            sample_weight=sw,
        )
        grad = torch.autograd.grad(risk_env, [scale], create_graph=True)[0]
        grad_sq = grad.pow(2).sum()
        penalty = penalty + grad_sq
        diagnostics.append(IRMDiagnostic(int(env_id.item()), float(risk_env.detach().item()), float(grad_sq.detach().item())))

    return penalty, diagnostics


def orthogonality_penalty(z_inv: torch.Tensor, z_spu: torch.Tensor) -> tuple[torch.Tensor, float]:
    zi = z_inv - z_inv.mean(0, keepdim=True)
    zs = z_spu - z_spu.mean(0, keepdim=True)
    cov = (zi.T @ zs) / max(1, z_inv.shape[0] - 1)
    penalty = cov.pow(2).mean()
    cosine = F.cosine_similarity(z_inv, z_spu, dim=1).abs().mean().item()
    return penalty, float(cosine)


def hsic_rbf(z1: torch.Tensor, z2: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    n = z1.shape[0]
    if n < 2:
        return torch.zeros((), device=z1.device)
    d1 = torch.cdist(z1, z1, p=2).pow(2)
    d2 = torch.cdist(z2, z2, p=2).pow(2)
    if sigma is None:
        sigma = float(torch.sqrt(torch.median(d1[d1 > 0]).clamp_min(1e-12)).detach().item()) if (d1 > 0).any() else 1.0
    g = max(2.0 * sigma * sigma, 1e-12)
    k = torch.exp(-d1 / g)
    l = torch.exp(-d2 / g)
    h = torch.eye(n, device=z1.device) - (1.0 / n) * torch.ones((n, n), device=z1.device)
    return (h @ k @ h * (h @ l @ h)).sum() / ((n - 1) ** 2)


def disentanglement_loss(
    z_inv: torch.Tensor,
    z_spu: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    ortho, cosine = orthogonality_penalty(z_inv, z_spu)
    hsic = float(hsic_rbf(z_inv, z_spu).detach().item())

    if mode == "none":
        loss = torch.zeros((), device=z_inv.device)
    elif mode == "orthogonality":
        loss = ortho
    elif mode == "hsic":
        loss = hsic_rbf(z_inv, z_spu)
    else:
        raise ValueError(f"Unsupported disentanglement mode: {mode}")

    return loss, {"cosine_sim": cosine, "hsic": hsic}
