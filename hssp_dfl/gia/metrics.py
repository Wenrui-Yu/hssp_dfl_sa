"""Permutation-aware image-batch metrics for GIA experiments."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment


def denormalize_images(
    images: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Map channel-normalized images back to the [0, 1] pixel range."""

    mean_tensor = torch.as_tensor(mean, dtype=images.dtype, device=images.device)[None, :, None, None]
    std_tensor = torch.as_tensor(std, dtype=images.dtype, device=images.device)[None, :, None, None]
    return (images * std_tensor + mean_tensor).clamp(0, 1)


def _ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = gaussian_filter(x, sigma=1.5)
    mu_y = gaussian_filter(y, sigma=1.5)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = np.maximum(gaussian_filter(x * x, sigma=1.5) - mu_x_sq, 0)
    sigma_y_sq = np.maximum(gaussian_filter(y * y, sigma=1.5) - mu_y_sq, 0)
    sigma_xy = gaussian_filter(x * y, sigma=1.5) - mu_xy
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float(np.mean(numerator / (denominator + 1e-12)))


def image_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute mean channel-wise SSIM for two [0, 1] CHW images."""

    x_np = x.detach().cpu().numpy().astype(np.float64)
    y_np = y.detach().cpu().numpy().astype(np.float64)
    return float(np.mean([_ssim_channel(a, b) for a, b in zip(x_np, y_np)]))


def align_and_score_batch(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Align an unordered reconstruction batch and compute MSE/PSNR/SSIM.

    Averaged batch gradients are invariant to example order.  Reporting metrics
    by raw tensor index can therefore turn a correct reconstruction into a false
    failure.  We use minimum-cost bipartite matching on pixel MSE first.
    """

    if reconstructed.ndim != 4 or target.ndim != 4:
        raise ValueError("reconstructed and target must be NCHW tensors")
    if reconstructed.shape != target.shape:
        raise ValueError(
            f"batch shape mismatch: reconstructed={tuple(reconstructed.shape)}, "
            f"target={tuple(target.shape)}"
        )

    reconstructed_pixels = denormalize_images(reconstructed, mean, std)
    target_pixels = denormalize_images(target, mean, std)
    cost = (
        reconstructed_pixels[:, None]
        .sub(target_pixels[None, :])
        .square()
        .flatten(start_dim=2)
        .mean(dim=2)
    )
    reconstructed_indices, target_indices = linear_sum_assignment(cost.detach().cpu().numpy())

    aligned = torch.empty_like(target_pixels)
    source_for_target = {}
    for reconstructed_index, target_index in zip(reconstructed_indices, target_indices):
        aligned[target_index] = reconstructed_pixels[reconstructed_index]
        source_for_target[int(target_index)] = int(reconstructed_index)

    metrics = []
    for target_index in range(target_pixels.shape[0]):
        mse = float(torch.mean((aligned[target_index] - target_pixels[target_index]) ** 2).item())
        psnr = float("inf") if mse == 0 else float(10 * math.log10(1.0 / mse))
        metrics.append(
            {
                "target_index": target_index,
                "reconstructed_index": source_for_target[target_index],
                "mse": mse,
                "psnr": psnr,
                "ssim": image_ssim(aligned[target_index], target_pixels[target_index]),
            }
        )
    return aligned, metrics
