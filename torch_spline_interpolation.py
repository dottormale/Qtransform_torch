import torch
import numpy as np
import time 
import time
import torch


#---------------------------------------------------
# MAIN FUNCTION 2D
def spline_interpolate_2d(Z, num_t_bins, num_f_bins, logf=False, kx=3, ky=3, sx=0.001,sy=0.001):
    # Empty cache
    try:
        torch.cuda.empty_cache()
        #print('Cache emptied')
    except:
        pass
    
    nx_points, ny_points = Z.shape
    print(f'{nx_points=}')
    print(f'{ny_points=}')
    
    x = torch.linspace(-1, 1, nx_points)
    y = torch.linspace(-1, 1, ny_points)
        
    coef, tx, ty = bivariate_spline_fit_natural_torch(x, y, Z, kx, ky, sx,sy)
    
    x_eval = torch.linspace(-1, 1, num_t_bins)
    y_eval = torch.linspace(-1, 1, num_f_bins)
    Z_interp = evaluate_bivariate_spline_torch(x_eval, y_eval, coef, tx, ty, kx, ky)
    return Z_interp

# MAIN FUNCTION 1D
def spline_interpolate(Z, num_x_bins, kx=3, s=0.001):
    # Empty cache
    try:
        torch.cuda.empty_cache()
        #print('Cache emptied')
    except:
        pass
    
    nx_points = Z.shape[0]
    x = torch.linspace(-1, 1, nx_points)
        
    coef, tx = spline_fit_natural_torch(x, Z, kx, s)
    
    x_eval = torch.linspace(-1, 1, num_x_bins)
    Z_interp = evaluate_spline_torch(x_eval, coef, tx, kx).view(num_x_bins)
    return Z_interp

#----------------------------------------------------------------------
#Funcitons for spline computations
def generate_natural_knots(x, k):
    n = x.shape[0]
    t = torch.zeros(n + 2 * k)
    t[:k] = x[0]
    t[k:-k] = x
    t[-k:] = x[-1]
    return t

def compute_L_R(x, t, d, m, k):
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
    b = torch.zeros((n, m, k + 1))
    
    mask_lower = t[:m+1].unsqueeze(0)[:, :-1] <= x.unsqueeze(1)
    mask_upper = x.unsqueeze(1) < t[:m+1].unsqueeze(0)[:, 1:]

    b[:, :, 0] = mask_lower & mask_upper
    
    b[:, 0, 0] = torch.where(x < t[1], torch.ones_like(x), b[:, 0, 0])
    b[:, -1, 0] = torch.where(x >= t[-2], torch.ones_like(x), b[:, -1, 0])
    return b

def bspline_basis_natural_torch(x, k, t):
    n = x.shape[0]
    m = t.shape[0] - k - 1
    
    b = zeroth_order(x, k, t, n, m)
    
    for d in range(1, k + 1):
        L, R = compute_L_R(x, t, d, m, k)
        left = L * b[:, :, d-1]

        zeros_tensor = torch.zeros(b.shape[0], 1)
        temp_b = torch.cat([b[:, 1:, d-1], zeros_tensor], dim=1)
    
        right = R * temp_b
        b[:, :, d] = left + right
    
    return b[:, :, -1]

def spline_fit_natural_torch(x, z, kx, s):
    tx = generate_natural_knots(x, kx)

    bx = bspline_basis_natural_torch(x, kx, tx).to(z.device)

    z_flat = z.view(-1)

    m = bx.size(1)
    I = torch.eye(m, device=z.device)

    B_T_B = bx.T @ bx + s * I 
    B_T_z = bx.T @ z_flat
    
    coef = torch.linalg.solve(B_T_B, B_T_z)
    
    return coef.to(z.device), tx

def bivariate_spline_fit_natural_torch(x, y, z, kx,ky, sx=0.001,sy=0.001):
    tx = generate_natural_knots(x, kx)
    ty = generate_natural_knots(y, ky)

    Bx = bspline_basis_natural_torch(x, kx, tx)
    By = bspline_basis_natural_torch(y, ky, ty)

    print(f'{Bx.shape=}')
    print(f'{x.shape=}')
    print(f'{tx.shape=}')
    print('--------------------------------\n')
    print(f'{By.shape=}')
    print(f'{y.shape=}')
    print(f'{ty.shape=}')
    print('--------------------------------\n')
    print(f'{z.shape=}')

    # Adding regularization 
    mx = Bx.size(1)
    my = By.size(1)
    Ix = torch.eye(mx, device=Bx.device)
    Iy = torch.eye(my, device=By.device)


    # First step: solve for E 
    ByT_By = By.T @ By + sy * Iy
    ByT_Z_Bx = (By.T @ z.T) @ Bx
    E = torch.linalg.solve(ByT_By, ByT_Z_Bx)

    # Second step: solve for final control points C 
    BxT_Bx = Bx.T @ Bx + sx * Ix
    C = torch.linalg.solve(BxT_Bx, E.T).T

    return C, tx, ty

def evaluate_bivariate_spline_torch(x, y, C, tx, ty, kx,ky):
    
    
    Bx = bspline_basis_natural_torch(x, kx, tx)
    By = bspline_basis_natural_torch(y, ky, ty)
    print('EVALUATE!')
    print(f'{Bx.shape=}')
    print(f'{x.shape=}')
    print(f'{tx.shape=}')
    print('--------------------------------\n')
    print(f'{By.shape=}')
    print(f'{y.shape=}')
    print(f'{ty.shape=}')
    print('--------------------------------\n')
    print(f'{C.shape=}')
    return (Bx@C.T)@By.T #torch.einsum('ij,jk,lk->il', Bx, C, By)

def evaluate_spline_torch(x, coef, tx, kx):
    bx = bspline_basis_natural_torch(x, kx, tx).to(coef.device)
    z_eval = bx @ coef
    return z_eval
        
#---------------------------------------------------
#Functions for debugging
def check_second_derivatives(x, k, knots, coef):
    basis = bspline_basis_natural(x, k, knots)
    second_derivative = torch.zeros_like(x)
    
    for i in range(basis.size(1)):
        second_derivative += coef[i] * basis[:, i]
    
    return second_derivative

def visualize_knots_and_basis(x, k, knots, basis, title):
    plt.figure(figsize=(12, 6))
    
    # Plot knots
    plt.subplot(1, 2, 1)
    plt.plot(knots.cpu(), 'o-', label='Knots')
    plt.title(f'{title} Knots')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    
    # Plot basis functions
    plt.subplot(1, 2, 2)
    for i in range(basis.size(1)):
        plt.plot(x.cpu(), basis[:, i].cpu(), label=f'B{i}')
    plt.title(f'{title} Basis Functions')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    
    plt.show()

def analyze_coefficients(coef):
    plt.figure(figsize=(12, 6))
    plt.plot(coef.cpu(), 'o-')
    plt.title('Spline Coefficients')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

def print_basis_functions(x, k, knots, basis):
    for i in range(basis.size(1)):
        print(f'Basis function B{i}:')
        for j in range(basis.size(0)):
            print(f'    x={x[j].item():.2f}: {basis[j, i].item():.6f}')
        print('')
#------------------------------------------------------------------------------------

#---------------------------------------
#EXAMPLE USAGE
'''
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import torch.nn.functional as F

# Generate data
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
x_grid, y_grid = np.meshgrid(x, y)
z = (x_grid + y_grid) * np.exp(-6.0 * (x_grid ** 2 + y_grid ** 2))

# Convert numpy arrays to torch tensors
z_torch = torch.from_numpy(z).float()

print(z_torch.shape)

# Fit and evaluate the custom Torch spline interpolation
kx, ky = 3, 3
s = 0.001
start=time.time()
coef, tx, ty = bivariate_spline_fit_natural_torch(torch.tensor(x).float(), torch.tensor(y).float(), z_torch, kx, ky,s)
end=time.time()
print(f'TIME FOR SPLINE FIT: {end-start}s')
# Evaluate the spline on a grid

x_eval = torch.linspace(-1, 1, 400)
y_eval = torch.linspace(-1, 1, 400)
start=time.time()
znew_torch = evaluate_bivariate_spline_torch(x_eval, y_eval, coef, tx, ty, kx, ky).view(400, 400)
end=time.time()
print(f'TIME FOR SPLINE EVALUATION: {end-start}s')
      
# Plotting the results
plt.figure(figsize=(24, 18))

# Plot original data
plt.subplot(3, 2, 1)
im = plt.imshow(z, extent=(-1, 1, -1, 1))
plt.colorbar(im)
plt.title('Original Data')

# Interpolate onto a new grid using SciPy for comparison
f = RectBivariateSpline(x, y, z)
znew_scipy = f(x_eval.numpy(), y_eval.numpy())

# Plot SciPy interpolated data
plt.subplot(3, 2, 2)
im = plt.imshow(znew_scipy, extent=(-1, 1, -1, 1))
plt.colorbar(im)
plt.title('SciPy Interpolated Data')

# Plot Torch spline interpolated data
plt.subplot(3, 2, 3)
im = plt.imshow(znew_torch.detach().numpy(), extent=(-1, 1, -1, 1))
plt.colorbar(im)
plt.title('Torch Spline Interpolated Data')


# Calculate the absolute differences
abs_diff_torch = np.abs(znew_scipy - znew_torch.detach().numpy())

# Plot the absolute difference between SciPy and Torch spline interpolations
plt.subplot(3, 2, 4)
im = plt.imshow(abs_diff_torch, extent=(-1, 1, -1, 1), vmin=0, vmax=0.02)
plt.colorbar(im)
plt.title('Abs Difference (SciPy vs Torch Spline)')

# Perform bicubic interpolation using PyTorch

z_torch_unsqueeze = z_torch.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
znew_bicubic_torch = F.interpolate(z_torch_unsqueeze, size=(400, 400), mode='bicubic', align_corners=True)
znew_bicubic_torch = znew_bicubic_torch.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

# Plot PyTorch bicubic interpolated data
plt.subplot(3, 2, 5)
im = plt.imshow(znew_bicubic_torch.detach().numpy(), extent=(-1, 1, -1, 1))
plt.colorbar(im)
plt.title('PyTorch Bicubic Interpolated Data')

# Calculate the absolute differences
abs_diff_bicubic = np.abs(znew_scipy - znew_bicubic_torch.detach().numpy())

# Plot the absolute difference between SciPy and PyTorch bicubic interpolations
plt.subplot(3, 2, 6)

im = plt.imshow(abs_diff_bicubic, extent=(-1, 1, -1, 1), vmin=0, vmax=0.02)
plt.colorbar(im)
plt.title('Abs Difference (SciPy vs PyTorch Bicubic)')

plt.suptitle('2-D Grid Data Interpolation Comparison')
plt.tight_layout()
plt.show()
'''
