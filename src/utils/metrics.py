"""Image quality metrics and diagnostic tools.

Provides:
    - PSNR: Peak Signal-to-Noise Ratio
    - SSIM: Structural Similarity Index
    - Coverage: Paint coverage ratio
    - Edge preservation: Gradient similarity

Used by:
    - CI golden tests: PSNR/SSIM thresholds
    - Adversarial tests: Coverage and edge checks for reward hacking detection
    - HPO objective: Multi-metric evaluation

All metrics operate on torch tensors (3, H, W) or (B, 3, H, W).
Input images are expected to be in [0, 1] range (linear RGB) unless noted.
"""

import torch
import torch.nn.functional as F


def psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    Parameters
    ----------
    img1 : torch.Tensor
        First image, shape (3, H, W) or (B, 3, H, W), range [0, max_val]
    img2 : torch.Tensor
        Second image, same shape as img1
    max_val : float
        Maximum possible pixel value, default 1.0
    eps : float
        Small epsilon to avoid log(0), default 1e-8

    Returns
    -------
    torch.Tensor
        PSNR in dB, scalar (mean over batch if batched)

    Notes
    -----
    PSNR = 10 * log10(max_val^2 / MSE)
    Higher is better. Typical values:
        - PSNR > 40 dB: Excellent quality
        - PSNR 30-40 dB: Good quality
        - PSNR < 30 dB: Poor quality

    Examples
    --------
    >>> psnr_val = psnr(target, canvas)
    >>> assert psnr_val >= 26.0  # Golden test threshold
    """
    mse = F.mse_loss(img1, img2, reduction='mean')
    psnr_val = 10.0 * torch.log10((max_val ** 2) / (mse + eps))
    return psnr_val


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    max_val: float = 1.0
) -> torch.Tensor:
    """Compute Structural Similarity Index (SSIM).

    Parameters
    ----------
    img1 : torch.Tensor
        First image, shape (3, H, W) or (B, 3, H, W), range [0, max_val]
    img2 : torch.Tensor
        Second image, same shape as img1
    window_size : int
        Gaussian window size, default 11
    sigma : float
        Gaussian standard deviation, default 1.5
    k1, k2 : float
        Stability constants, defaults 0.01, 0.03
    max_val : float
        Maximum pixel value, default 1.0

    Returns
    -------
    torch.Tensor
        SSIM value in [-1, 1], scalar (mean over batch)
        1.0 = perfect similarity, 0.0 = no similarity

    Notes
    -----
    Computes SSIM per-channel and averages.
    Uses Gaussian weighting window for local comparison.
    More perceptually accurate than PSNR.

    References
    ----------
    Wang et al., "Image Quality Assessment: From Error Visibility to
    Structural Similarity", IEEE TIP 2004.
    """
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    B, C, H, W = img1.shape
    
    # Create Gaussian window
    def create_window(window_size, sigma, channels):
        # 1D Gaussian kernel
        gauss = torch.exp(
            -torch.arange(window_size).float() ** 2 / (2 * sigma ** 2)
        )
        gauss = gauss / gauss.sum()
        
        # 2D kernel (outer product)
        kernel_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
        return kernel.to(img1.device).type(img1.dtype)
    
    window = create_window(window_size, sigma, C)
    
    # Constants
    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2
    
    # Compute local statistics via convolution
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def paint_coverage(alpha_or_canvas: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """Compute paint coverage ratio (fraction of painted pixels).

    Parameters
    ----------
    alpha_or_canvas : torch.Tensor
        Alpha mask or canvas image, shape (H, W) or (3, H, W) or (B, 3, H, W)
        Range [0, 1], where 0 = white/unpainted, 1 = fully painted
    threshold : float
        Threshold for considering pixel "painted", default 0.01

    Returns
    -------
    torch.Tensor
        Coverage ratio in [0, 1], scalar

    Notes
    -----
    Used for adversarial tests to detect reward hacking:
        - All-black target: coverage should be near 0
        - Checkerboard: coverage should not be > 0.5 (no background wash)

    For RGB canvas, uses luminance-based coverage (deviation from white).

    Examples
    --------
    >>> coverage = paint_coverage(canvas)
    >>> assert coverage < 0.05  # For all-black target (should paint very little)
    """
    if alpha_or_canvas.ndim == 4:
        # (B, 3, H, W) → average over batch and channels
        canvas = alpha_or_canvas.mean(dim=(0, 1))  # (H, W)
    elif alpha_or_canvas.ndim == 3:
        # (3, H, W) → average over channels
        canvas = alpha_or_canvas.mean(dim=0)  # (H, W)
    else:
        # (H, W) → use as-is
        canvas = alpha_or_canvas
    
    # Coverage = fraction of pixels significantly painted (deviation from white)
    # White (1.0) = unpainted, Black (0.0) = fully painted
    # Deviation from white: |1.0 - value|
    deviation = torch.abs(1.0 - canvas)
    painted_pixels = (deviation > threshold).float()
    
    coverage_ratio = painted_pixels.mean()
    return coverage_ratio


def edge_preservation_score(
    target: torch.Tensor,
    canvas: torch.Tensor,
    threshold: float = 0.1
) -> torch.Tensor:
    """Compute edge preservation score (gradient similarity).

    Parameters
    ----------
    target : torch.Tensor
        Target image, shape (3, H, W) or (B, 3, H, W), range [0, 1]
    canvas : torch.Tensor
        Canvas image, same shape as target
    threshold : float
        Edge detection threshold, default 0.1

    Returns
    -------
    torch.Tensor
        Edge preservation score in [0, 1], scalar
        1.0 = perfect edge preservation, 0.0 = no edges preserved

    Notes
    -----
    Detects reward hacking where policy paints uniform background
    to minimize LPIPS without preserving structure.

    Computes:
        1. Sobel gradients on target and canvas
        2. Binary edge maps (gradients > threshold)
        3. Intersection-over-union (IoU) of edge maps

    Examples
    --------
    >>> edge_score = edge_preservation_score(target, canvas)
    >>> assert edge_score > 0.3  # Checkerboard test: must preserve edges
    """
    if target.ndim == 3:
        target = target.unsqueeze(0)
        canvas = canvas.unsqueeze(0)
    
    # Convert to grayscale (luminance)
    target_gray = 0.2126 * target[:, 0] + 0.7152 * target[:, 1] + 0.0722 * target[:, 2]
    canvas_gray = 0.2126 * canvas[:, 0] + 0.7152 * canvas[:, 1] + 0.0722 * canvas[:, 2]
    
    # Add channel dimension for conv2d
    target_gray = target_gray.unsqueeze(1)  # (B, 1, H, W)
    canvas_gray = canvas_gray.unsqueeze(1)
    
    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=target.dtype, device=target.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=target.dtype, device=target.device)
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    # Compute gradients
    def compute_gradient_magnitude(img_gray):
        gx = F.conv2d(img_gray, sobel_x, padding=1)
        gy = F.conv2d(img_gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return grad_mag
    
    target_grad = compute_gradient_magnitude(target_gray)
    canvas_grad = compute_gradient_magnitude(canvas_gray)
    
    # Binary edge maps
    target_edges = (target_grad > threshold).float()
    canvas_edges = (canvas_grad > threshold).float()
    
    # IoU of edge maps
    intersection = (target_edges * canvas_edges).sum()
    union = (target_edges + canvas_edges).clamp(0, 1).sum()
    
    iou = intersection / (union + 1e-8)
    return iou


def mean_absolute_error(
    img1: torch.Tensor,
    img2: torch.Tensor
) -> torch.Tensor:
    """Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    img1 : torch.Tensor
        First image, shape (3, H, W) or (B, 3, H, W)
    img2 : torch.Tensor
        Second image, same shape as img1

    Returns
    -------
    torch.Tensor
        MAE, scalar

    Notes
    -----
    Simple pixel-wise L1 distance.
    Less sensitive to outliers than MSE.
    """
    return F.l1_loss(img1, img2, reduction='mean')


def histogram_similarity(
    img1: torch.Tensor,
    img2: torch.Tensor,
    bins: int = 256
) -> torch.Tensor:
    """Compute histogram similarity (correlation).

    Parameters
    ----------
    img1 : torch.Tensor
        First image, shape (3, H, W) or (B, 3, H, W), range [0, 1]
    img2 : torch.Tensor
        Second image, same shape as img1
    bins : int
        Number of histogram bins, default 256

    Returns
    -------
    torch.Tensor
        Histogram correlation in [-1, 1], scalar
        1.0 = identical distributions

    Notes
    -----
    Compares color distributions (useful for style matching).
    Insensitive to spatial structure (complements SSIM).
    """
    # Flatten spatial dimensions
    if img1.ndim == 4:
        img1 = img1.view(img1.shape[0] * img1.shape[1], -1)
        img2 = img2.view(img2.shape[0] * img2.shape[1], -1)
    elif img1.ndim == 3:
        img1 = img1.view(img1.shape[0], -1)
        img2 = img2.view(img2.shape[0], -1)
    
    # Compute histograms
    hist1 = torch.histc(img1, bins=bins, min=0.0, max=1.0)
    hist2 = torch.histc(img2, bins=bins, min=0.0, max=1.0)
    
    # Normalize to probabilities
    hist1 = hist1 / (hist1.sum() + 1e-8)
    hist2 = hist2 / (hist2.sum() + 1e-8)
    
    # Pearson correlation
    mean1 = hist1.mean()
    mean2 = hist2.mean()
    
    numerator = ((hist1 - mean1) * (hist2 - mean2)).sum()
    denominator = torch.sqrt(((hist1 - mean1) ** 2).sum() * ((hist2 - mean2) ** 2).sum())
    
    correlation = numerator / (denominator + 1e-8)
    return correlation


def compute_all_metrics(
    target: torch.Tensor,
    canvas: torch.Tensor,
    max_val: float = 1.0
) -> dict:
    """Compute comprehensive metric suite.

    Parameters
    ----------
    target : torch.Tensor
        Target image, shape (3, H, W) or (B, 3, H, W), range [0, max_val]
    canvas : torch.Tensor
        Canvas image, same shape as target
    max_val : float
        Maximum pixel value, default 1.0

    Returns
    -------
    dict
        Dictionary with keys:
        - psnr: Peak Signal-to-Noise Ratio (dB)
        - ssim: Structural Similarity Index [0, 1]
        - mae: Mean Absolute Error
        - coverage: Paint coverage ratio [0, 1]
        - edge_preservation: Edge similarity [0, 1]

    Notes
    -----
    Used for comprehensive evaluation in CI and HPO.
    All metrics computed in single pass for efficiency.

    Examples
    --------
    >>> metrics = compute_all_metrics(target, canvas)
    >>> mlflow.log_metrics(metrics)
    """
    return {
        'psnr': psnr(target, canvas, max_val=max_val).item(),
        'ssim': ssim(target, canvas, max_val=max_val).item(),
        'mae': mean_absolute_error(target, canvas).item(),
        'coverage': paint_coverage(canvas).item(),
        'edge_preservation': edge_preservation_score(target, canvas).item()
    }
