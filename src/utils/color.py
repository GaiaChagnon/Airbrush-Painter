"""Color space conversions and perceptual metrics.

Provides:
    - sRGB ↔ linear RGB conversions (gamma 2.2 approximation)
    - RGB → Lab color space (CIE L*a*b*, D65 illuminant)
    - ΔE2000: Perceptual color difference (CIEDE2000 formula)
    - Luminance calculation from linear RGB

Used by:
    - Calibration pipeline: ΔE validation of layering model
    - Preprocessing: LAB thresholding for pen mask extraction
    - Metrics: Perceptual color accuracy assessment

All conversions operate on torch tensors (3, H, W) or (B, 3, H, W).
Input/output ranges are documented per function.

Invariants:
    - Internal pipeline uses linear RGB [0,1]
    - sRGB [0,1] only at I/O boundaries (load/save)
    - Lab coordinates: L[0,100], a,b[-128,127]
"""

import torch
import torch.nn.functional as F


def srgb_to_linear(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert sRGB [0,1] to linear RGB [0,1].

    Parameters
    ----------
    img : torch.Tensor
        sRGB image, shape (3, H, W) or (B, 3, H, W), range [0, 1]
    eps : float
        Small epsilon to avoid division by zero, default 1e-6

    Returns
    -------
    torch.Tensor
        Linear RGB image, same shape, range [0, 1]

    Notes
    -----
    Uses exact sRGB transfer function (not gamma 2.2 approximation):
        - Linear region for small values: x / 12.92
        - Power region: ((x + 0.055) / 1.055)^2.4
    """
    # Clamp to [0, 1] for safety
    img = torch.clamp(img, 0.0, 1.0)
    
    # Threshold for linear region
    linear_mask = img <= 0.04045
    
    # Linear region
    linear = img / 12.92
    
    # Power region
    power = torch.pow((img + 0.055) / 1.055, 2.4)
    
    return torch.where(linear_mask, linear, power)


def linear_to_srgb(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert linear RGB [0,1] to sRGB [0,1].

    Parameters
    ----------
    img : torch.Tensor
        Linear RGB image, shape (3, H, W) or (B, 3, H, W), range [0, 1]
    eps : float
        Small epsilon for numerical stability, default 1e-6

    Returns
    -------
    torch.Tensor
        sRGB image, same shape, range [0, 1]

    Notes
    -----
    Inverse of srgb_to_linear, uses exact sRGB transfer function.
    """
    # Clamp to [0, 1] for safety
    img = torch.clamp(img, 0.0, 1.0)
    
    # Threshold for linear region
    linear_mask = img <= 0.0031308
    
    # Linear region
    linear = img * 12.92
    
    # Power region
    power = 1.055 * torch.pow(img, 1.0 / 2.4) - 0.055
    
    return torch.where(linear_mask, linear, power)


def luminance_linear(img: torch.Tensor) -> torch.Tensor:
    """Calculate luminance from linear RGB.

    Parameters
    ----------
    img : torch.Tensor
        Linear RGB image, shape (3, H, W) or (B, 3, H, W), range [0, 1]

    Returns
    -------
    torch.Tensor
        Luminance image, shape (H, W) or (B, H, W), range [0, 1]

    Notes
    -----
    Uses Rec. 709 coefficients: Y = 0.2126*R + 0.7152*G + 0.0722*B
    """
    if img.ndim == 3:
        # (3, H, W)
        r, g, b = img[0], img[1], img[2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    elif img.ndim == 4:
        # (B, 3, H, W)
        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    else:
        raise ValueError(f"Expected shape (3, H, W) or (B, 3, H, W), got {img.shape}")


def rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB to CIE XYZ (D65 illuminant).

    Parameters
    ----------
    rgb : torch.Tensor
        Linear RGB, shape (3, H, W) or (B, 3, H, W), range [0, 1]

    Returns
    -------
    torch.Tensor
        XYZ coordinates, same shape, D65 white point

    Notes
    -----
    Uses sRGB → XYZ matrix (D65):
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]]
    """
    # sRGB to XYZ matrix (D65)
    mat = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=rgb.dtype, device=rgb.device)
    
    if rgb.ndim == 3:
        # (3, H, W) → (H, W, 3) for matmul
        rgb_hw3 = rgb.permute(1, 2, 0)
        xyz_hw3 = torch.matmul(rgb_hw3, mat.T)
        return xyz_hw3.permute(2, 0, 1)  # Back to (3, H, W)
    elif rgb.ndim == 4:
        # (B, 3, H, W) → (B, H, W, 3)
        rgb_bhw3 = rgb.permute(0, 2, 3, 1)
        xyz_bhw3 = torch.matmul(rgb_bhw3, mat.T)
        return xyz_bhw3.permute(0, 3, 1, 2)  # Back to (B, 3, H, W)
    else:
        raise ValueError(f"Expected shape (3, H, W) or (B, 3, H, W), got {rgb.shape}")


def xyz_to_lab(xyz: torch.Tensor, white_point: str = "D65") -> torch.Tensor:
    """Convert XYZ to CIE L*a*b*.

    Parameters
    ----------
    xyz : torch.Tensor
        XYZ coordinates, shape (3, H, W) or (B, 3, H, W)
    white_point : str
        Reference white point, "D65" (default) or "D50"

    Returns
    -------
    torch.Tensor
        Lab coordinates, same shape
        L: [0, 100], a,b: approximately [-128, 127]

    Notes
    -----
    D65 white point: X=0.95047, Y=1.0, Z=1.08883
    Uses CIE standard transform with 6/29 threshold.
    """
    # Reference white points
    if white_point == "D65":
        ref = torch.tensor([0.95047, 1.0, 1.08883], dtype=xyz.dtype, device=xyz.device)
    elif white_point == "D50":
        ref = torch.tensor([0.96422, 1.0, 0.82521], dtype=xyz.dtype, device=xyz.device)
    else:
        raise ValueError(f"Unknown white_point: {white_point}. Use 'D65' or 'D50'.")
    
    if xyz.ndim == 3:
        ref = ref.view(3, 1, 1)
    elif xyz.ndim == 4:
        ref = ref.view(1, 3, 1, 1)
    else:
        raise ValueError(f"Expected shape (3, H, W) or (B, 3, H, W), got {xyz.shape}")
    
    # Normalize by reference white
    xyz_norm = xyz / ref
    
    # f(t) function with threshold
    delta = 6.0 / 29.0
    delta_sq = delta * delta
    delta_cube = delta_sq * delta
    
    linear_mask = xyz_norm <= delta_cube
    linear = xyz_norm / (3.0 * delta_sq) + (4.0 / 29.0)
    power = torch.pow(xyz_norm, 1.0 / 3.0)
    f = torch.where(linear_mask, linear, power)
    
    # Extract channels
    if xyz.ndim == 3:
        fx, fy, fz = f[0], f[1], f[2]
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return torch.stack([L, a, b], dim=0)
    else:  # ndim == 4
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return torch.stack([L, a, b], dim=1)


def rgb_to_lab(img_linear_rgb: torch.Tensor, white_point: str = "D65") -> torch.Tensor:
    """Convert linear RGB to CIE L*a*b*.

    Parameters
    ----------
    img_linear_rgb : torch.Tensor
        Linear RGB image, shape (3, H, W) or (B, 3, H, W), range [0, 1]
    white_point : str
        Reference white point, "D65" (default) or "D50"

    Returns
    -------
    torch.Tensor
        Lab image, same shape
        L: [0, 100], a,b: approximately [-128, 127]

    Notes
    -----
    Composite of rgb_to_xyz and xyz_to_lab.
    Used for pen mask extraction (threshold on L channel).
    """
    xyz = rgb_to_xyz(img_linear_rgb)
    return xyz_to_lab(xyz, white_point=white_point)


def delta_e2000(
    lab1: torch.Tensor,
    lab2: torch.Tensor,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0
) -> torch.Tensor:
    """Compute CIEDE2000 color difference (ΔE2000).

    Parameters
    ----------
    lab1 : torch.Tensor
        First Lab image, shape (3, H, W) or (B, 3, H, W)
    lab2 : torch.Tensor
        Second Lab image, same shape as lab1
    kL, kC, kH : float
        Weighting factors for lightness, chroma, hue (default 1.0)

    Returns
    -------
    torch.Tensor
        ΔE2000 values, shape (H, W) or (B, H, W)
        Typical perceptual threshold: ΔE < 2.3 (just noticeable difference)

    Notes
    -----
    Implements full CIEDE2000 formula (Sharma et al. 2005).
    Used for calibration validation and quality metrics.
    Mean ΔE over patches indicates color accuracy.
    """
    eps = 1e-10  # Avoid division by zero
    
    # Extract L, a, b channels
    if lab1.ndim == 3:
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]
    else:  # ndim == 4
        L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
        L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    
    # Calculate C (chroma)
    C1 = torch.sqrt(a1**2 + b1**2 + eps)
    C2 = torch.sqrt(a2**2 + b2**2 + eps)
    C_bar = (C1 + C2) / 2.0
    
    # Calculate G (chroma adjustment)
    C_bar_7 = C_bar**7
    G = 0.5 * (1.0 - torch.sqrt(C_bar_7 / (C_bar_7 + 25.0**7)))
    
    # Adjusted a values
    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2
    
    # Recalculate C and h with adjusted a
    C1_prime = torch.sqrt(a1_prime**2 + b1**2 + eps)
    C2_prime = torch.sqrt(a2_prime**2 + b2**2 + eps)
    
    h1_prime = torch.atan2(b1, a1_prime) * 180.0 / torch.pi
    h1_prime = torch.where(h1_prime < 0, h1_prime + 360.0, h1_prime)
    
    h2_prime = torch.atan2(b2, a2_prime) * 180.0 / torch.pi
    h2_prime = torch.where(h2_prime < 0, h2_prime + 360.0, h2_prime)
    
    # Differences
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime
    
    # Hue difference (account for circularity)
    abs_diff = torch.abs(h2_prime - h1_prime)
    dh_prime = torch.where(
        abs_diff <= 180.0,
        h2_prime - h1_prime,
        torch.where(
            h2_prime <= h1_prime,
            h2_prime - h1_prime + 360.0,
            h2_prime - h1_prime - 360.0
        )
    )
    
    dH_prime = 2.0 * torch.sqrt(C1_prime * C2_prime + eps) * torch.sin(dh_prime * torch.pi / 360.0)
    
    # Averages
    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0
    
    # Average hue (account for circularity)
    sum_h = h1_prime + h2_prime
    h_bar_prime = torch.where(
        abs_diff <= 180.0,
        sum_h / 2.0,
        torch.where(
            sum_h < 360.0,
            (sum_h + 360.0) / 2.0,
            (sum_h - 360.0) / 2.0
        )
    )
    
    # Weighting functions
    T = (1.0
         - 0.17 * torch.cos((h_bar_prime - 30.0) * torch.pi / 180.0)
         + 0.24 * torch.cos(2.0 * h_bar_prime * torch.pi / 180.0)
         + 0.32 * torch.cos((3.0 * h_bar_prime + 6.0) * torch.pi / 180.0)
         - 0.20 * torch.cos((4.0 * h_bar_prime - 63.0) * torch.pi / 180.0))
    
    dTheta = 30.0 * torch.exp(-((h_bar_prime - 275.0) / 25.0)**2)
    
    C_bar_prime_7 = C_bar_prime**7
    RC = 2.0 * torch.sqrt(C_bar_prime_7 / (C_bar_prime_7 + 25.0**7))
    
    L_bar_prime_minus_50_sq = (L_bar_prime - 50.0)**2
    SL = 1.0 + (0.015 * L_bar_prime_minus_50_sq) / torch.sqrt(20.0 + L_bar_prime_minus_50_sq)
    SC = 1.0 + 0.045 * C_bar_prime
    SH = 1.0 + 0.015 * C_bar_prime * T
    
    RT = -torch.sin(2.0 * dTheta * torch.pi / 180.0) * RC
    
    # Final ΔE2000
    dE00 = torch.sqrt(
        (dL_prime / (kL * SL))**2
        + (dC_prime / (kC * SC))**2
        + (dH_prime / (kH * SH))**2
        + RT * (dC_prime / (kC * SC)) * (dH_prime / (kH * SH))
    )
    
    return dE00
