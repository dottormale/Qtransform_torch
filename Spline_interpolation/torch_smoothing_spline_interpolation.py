import torch

def bspline_interpolation(x, z, kx, p, max_iter=200, tol=1e-6, s=0.001):
    """
    Performs B-spline interpolation with knot refinement and smoothing.

    Args:
        x: Tensor of data point positions.
        z: Tensor of data point values.
        kx: Degree of the spline.
        p: Smoothing parameter.
        max_iter: Maximum number of iterations for knot refinement (default 200).
        tol: Tolerance for improvement in fit (default 1e-6).
        s: Additional smoothing term (optional, defaults to 0.001).

    Returns:
        A tuple containing the final knot sequence and the corresponding coefficients.
    """
    # Initial fit and knot refinement
    tx, coef = refine_knots(x, z, kx, p, max_iter, tol,s)
    
    return tx, coef

# Necessary helper functions
def compute_penalty_matrix(t, k):
    """
    Computes the penalty matrix for the B-spline smoothing.
    
    Args:
        t: Knot sequence tensor.
        k: Degree of the spline.
    
    Returns:
        Penalty matrix for smoothing.
    """
    n = t.shape[0] - k - 1
    D2 = torch.diag(torch.ones(n - 2, device=t.device) * 2, 0) + torch.diag(torch.ones(n - 3, device=t.device) * -1, 1) + torch.diag(torch.ones(n - 3, device=t.device) * -1, -1)
    P = torch.zeros((n, n), device=t.device)
    P[1:-1, 1:-1] = D2
    return P


def bspline_basis_smooth_torch(x, k, t, p, s):
    """
    Computes the B-spline basis functions with smoothing.

    Args:
        x: Tensor of data point positions.
        k: Degree of the spline.
        t: Knot sequence tensor.
        p: Smoothing parameter.
        s: Additional smoothing term.

    Returns:
        Tuple containing the smoothed basis matrix and the reshaped basis matrix.
    """
    n = x.shape[0]
    m = t.shape[0] - k - 1
    bx = zeroth_order(x, k, t, n, m)
    for d in range(1, k + 1):
        L, R = compute_L_R(x, t, d, m, k)
        left = L * bx[:, :, d - 1]
        right = R * torch.cat([bx[:, 1:, d - 1], torch.zeros(bx.shape[0], 1, device=x.device)], dim=1)
        bx[:, :, d] = left + right
    

    I = torch.eye(m, device=x.device)
    bx_reshaped = bx[:, :, -1]
    B_T_B = bx_reshaped.T @ bx_reshaped + p * compute_penalty_matrix(t, k) + s * I

    return B_T_B, bx_reshaped

def spline_fit_smooth_torch(x, z, kx, p, s=0.01):
    """
    Fits a smoothed B-spline to the data.

    Args:
        x: Tensor of data point positions.
        z: Tensor of data point values.
        kx: Degree of the spline.
        p: Smoothing parameter.
        s: Additional smoothing term (default 0.01).

    Returns:
        Tuple containing the spline coefficients and the knot sequence.
    """
    tx = generate_natural_knots(x, kx)
    B_T_B, bx_reshaped = bspline_basis_smooth_torch(x, kx, tx, p, s)
    B_T_z = bx_reshaped.T @ z



    #use linalg.solve
    coef= torch.linalg.solve(B_T_B, B_T_z)

    '''
    #ALTERNATIVE SOLVERS
    
    #use linalg.lstsq
    coef2,*_ = torch.linalg.lstsq(B_T_B, B_T_z)

    # Use QR decomposition
    Q, R = torch.linalg.qr(B_T_B)
    c = torch.linalg.solve(R, Q.T @ B_T_z)
    '''
    
    return coef.to(z.device), tx

def refine_knots(x, z, kx, p, max_iter=200, tol=1e-10,s=0.001):
    """
    Refines the knot sequence to improve the fit of the B-spline.

    Args:
        x: Tensor of data point positions.
        z: Tensor of data point values.
        kx: Degree of the spline.
        p: Smoothing parameter.
        max_iter: Maximum number of iterations for knot refinement (default 200).
        tol: Tolerance for improvement in fit (default 1e-10).
        s: Additional smoothing term (default 0.001).

    Returns:
        Tuple containing the refined knot sequence and the corresponding coefficients.
    """
    
    tx = generate_natural_knots(x, kx)
    coef, _ = spline_fit_smooth_torch(x, z, kx, p, s)

    for _ in range(max_iter):
        residuals = z - evaluate_spline_torch(x, coef, tx, kx)
        high_residual_intervals = identify_high_residual_intervals(residuals)
        new_tx = update_knots(tx, high_residual_intervals)
        new_coef, _ = spline_fit_smooth_torch(x, z, kx, p, s)

        if torch.abs(new_coef - coef).mean() < tol:
            break

        tx = new_tx
        coef = new_coef

    return tx, coef

def identify_high_residual_intervals(residuals, num_intervals=None):
    """
    Identifies intervals with high residuals for knot refinement.

    Args:
        residuals: Tensor of residuals between the data and the spline fit.
        num_intervals: Number of intervals to consider (default: same as the number of data points).

    Returns:
        List of indices of the intervals with the highest residuals.
    """
    
    if num_intervals==None:
        num_intervals=residuals.shape[0]
    bin_width = (residuals.max() - residuals.min()) / num_intervals
    bins = torch.arange(0, num_intervals + 1, device=residuals.device) * bin_width + residuals.min()

    interval_starts = bins[:-1].unsqueeze(0)
    interval_ends = bins[1:].unsqueeze(0)
    interval_masks = (residuals.unsqueeze(1) >= interval_starts) & (residuals.unsqueeze(1) < interval_ends)

    abs_residuals = torch.abs(residuals).unsqueeze(1)
    interval_sums = torch.sum(abs_residuals * interval_masks, dim=0)
    interval_counts = torch.sum(interval_masks, dim=0)
    interval_means = interval_sums / interval_counts
    interval_means[interval_counts == 0] = 0

    k = -1
    top_k_indices = torch.argsort(interval_means, descending=True)[:k]

    return top_k_indices.tolist()

def identify_high_residual_intervals_masked(residuals, num_intervals=20, exclude_edge_width=5):
  """
  Identifies intervals with the highest absolute residuals, excluding a specified region at the edges.

  Args:
      residuals: Tensor of absolute residuals for data points.
      num_intervals: Number of intervals to discretize the residual range.
      exclude_edge_width: Number of intervals to exclude at each edge (default: 2).

  Returns:
      List of indices corresponding to the top k intervals with the highest average absolute residuals,
      excluding the specified edge regions.
  """

  bin_width = (residuals.max() - residuals.min()) / num_intervals
  bins = torch.arange(0, num_intervals + 1, device=residuals.device) * bin_width + residuals.min()

  interval_starts = bins[:-1].unsqueeze(0)
  interval_ends = bins[1:].unsqueeze(0)

  # Create interval masks, excluding edges
  edge_mask = torch.ones(residuals.shape[0], device=residuals.device)
  edge_mask[:exclude_edge_width] = 0
  edge_mask[-exclude_edge_width:] = 0
  interval_masks = torch.logical_and(
    residuals.unsqueeze(1) >= interval_starts,
    torch.logical_and(residuals.unsqueeze(1) < interval_ends, edge_mask.unsqueeze(1))
)

  abs_residuals = torch.abs(residuals).unsqueeze(1)
  interval_sums = torch.sum(abs_residuals * interval_masks, dim=0)
  interval_counts = torch.sum(interval_masks, dim=0)
  interval_means = interval_sums / interval_counts
  interval_means[interval_counts == 0] = 0

  k = 10
  top_k_indices = torch.argsort(interval_means, descending=True)[:k]

  return top_k_indices.tolist()

def generate_natural_knots(x, k):
    """
    Generates a natural knot sequence for B-spline interpolation.

    Args:
        x: Tensor of data point positions.
        k: Degree of the spline.

    Returns:
        Tensor of knot positions.
    """
    
    n = x.shape[0]
    t = torch.zeros(n + 2 * k, device=x.device)
    t[:k] = x[0]
    t[k:-k] = x
    t[-k:] = x[-1]
    return t

def compute_L_R(x, t, d, m, k):
    """
    Compute the L and R values for B-spline basis functions.

    Args:
        x: Tensor of data point positions.
        t: Tensor of knot positions.
        d: Current degree of the basis function.
        m: Number of intervals (n - k - 1, where n is the number of knots and k is the degree).
        k: Degree of the spline.

    Returns:
        L: Tensor containing left values for the B-spline basis functions.
        R: Tensor containing right values for the B-spline basis functions.
    """
    
    left_num = x.unsqueeze(1) - t[:m].unsqueeze(0)
    left_den = t[d:m+d] - t[:m]
    L = left_num / left_den.unsqueeze(0)

    right_num = t[d+1:m+d+1] - x.unsqueeze(1)
    right_den = t[d+1:m+d+1] - t[1:m+1]
    R = right_num / right_den.unsqueeze(0)

    zero_left = left_den == 0
    zero_right = right_den == 0
    zero_left_stacked = zero_left.tile(x.shape[0], 1)
    zero_right_stacked = zero_right.tile(x.shape[0], 1)

    L[zero_left_stacked] = 0
    R[zero_right_stacked] = 0

    return L, R

def zeroth_order(x, k, t, n, m):
    """
    Compute the zeroth-order B-spline basis functions.

    Args:
        x: Tensor of data point positions.
        k: Degree of the spline.
        t: Tensor of knot positions.
        n: Number of data points.
        m: Number of intervals (n - k - 1, where n is the number of knots and k is the degree).

    Returns:
        b: Tensor containing the zeroth-order B-spline basis functions.
    """
    
    b = torch.zeros((n, m, k + 1), device=x.device)

    mask_lower = t[:m+1].unsqueeze(0)[:, :-1] <= x.unsqueeze(1)
    mask_upper = x.unsqueeze(1) < t[:m+1].unsqueeze(0)[:, 1:]

    b[:, :, 0] = mask_lower & mask_upper

    b[:, 0, 0] = torch.where(x < t[1], torch.ones_like(x), b[:, 0, 0])
    b[:, -1, 0] = torch.where(x >= t[-2], torch.ones_like(x), b[:, -1, 0])
    return b

def update_knots(tx, high_residual_intervals):
    """
    Update the knot sequence by adding new knots at the midpoints of high residual intervals.

    Args:
        tx: Tensor of current knot positions.
        high_residual_intervals: List of indices corresponding to intervals with high residuals.

    Returns:
        new_tx: Tensor of updated knot positions.
    """
    
    mid_points = (tx[high_residual_intervals] + tx[[a+1 for a in high_residual_intervals]]) / 2
    new_tx = torch.cat((tx, mid_points)).sort().values
    return new_tx

def evaluate_spline_torch(x, coef, tx, kx):
    """
    Evaluate the B-spline at given data points using the computed coefficients and knots.

    Args:
        x: Tensor of data point positions.
        coef: Tensor of spline coefficients.
        tx: Tensor of knot positions.
        kx: Degree of the spline.

    Returns:
        Evaluated spline values at data points x.
    """
    _, bx_reshaped = bspline_basis_smooth_torch(x, kx, tx, 0, 0)  # No penalty for evaluation
    return bx_reshaped @ coef
