import torch
import numpy as np
import time 
import time
import torch
from matplotlib import pyplot as plt
from matplotlib import gridspec

#---------------------------------------------------
# MAIN FUNCTION 2D
def spline_interpolate_2d(Z, num_t_bins, num_f_bins, logf=True, kx=3, ky=3, sx=0.001,sy=0.001,frange=(10,100),freqs=None,xin=None,xout=None,yin=None,yout=None):
    """
      This function performs 2D natural spline interpolation on a given data set.
    
      Args:
          Z: A 2D tensor representing the data to be interpolated.
          num_t_bins: The number of bins in the target (t) dimension for the interpolated output.
          num_f_bins: The number of bins in the frequency (f) dimension for the interpolated output.
          logf: (Optional) Boolean flag indicating if the frequency axis should be in log scale (default: False).
              This functionality is not currently implemented (ToDo).
          kx: (Optional) The degree of the B-spline in the target dimension (default: 3).
          ky: (Optional) The degree of the B-spline in the frequency dimension (default: 3).
          sx: (Optional) The smoothing parameter in the target dimension (default: 0.001).
          sy: (Optional) The smoothing parameter in the frequency dimension (default: 0.001).
    
      Returns:
          Z_interp: A 2D tensor containing the interpolated data in the specified bins.
    """
    #ToDo: check that everything is consistent with the use of GPU (i.e. add .to(Z.device) where needed)
    
    # Empty cache
    try:
        torch.cuda.empty_cache()
        #print('Cache emptied')
    except:
        pass

    # Construct datapoint grid
    nx_points, ny_points = Z.shape
    
    if xin==None:
        x = torch.linspace(-1, 1, nx_points)
    else:
        x=xin
    
    #ToDo: implement logf=True case
    print(f'{freqs=}')
    print(f'{frange=}')
    
    if freqs!=None:
        #y=map_to_range(freqs, new_min=-1, new_max=1)
        y=freqs
        print('freqs passed')
        print(f'{y=}')
    elif yin == None:
        if logf:
            #y=map_to_range(torch.tensor(np.geomspace(frange[0],frange[1],ny_points)))
            y=torch.tensor(np.geomspace(frange[0],frange[1],num=ny_points))
        else:
            y = torch.linspace(-1, 1, ny_points)
    else:
        print('y set to yin')
        y=yin

    #compute bspline coefficients
    coef, tx, ty = bivariate_spline_fit_natural_torch(x, y, Z, kx, ky, sx,sy)

    #inerpolate
    if xout==None:
        x_eval = torch.linspace(-1, 1, num_t_bins)
    else:
        x_eval=xout

    if yout!=None:
        print('y_eval set to yout')
        y_eval=yout
        
    else:
        if logf:
            #y_eval = map_to_range(torch.tensor(np.geomspace(frange[0],frange[1],num_f_bins)))
    
            y_eval = torch.tensor(np.geomspace(frange[0],frange[1],num=num_f_bins))
            print(f'{y_eval=}')
        else:
            y_eval = torch.linspace(-1, 1, num_f_bins)
    
    print(f'{y=}') 
    print(f'{y_eval=}') 
    Z_interp = evaluate_bivariate_spline_torch(x_eval, y_eval, coef, tx, ty, kx, ky)
    return Z_interp

# MAIN FUNCTION 1D
def spline_interpolate(Z, num_x_bins, kx=3, s=0.1,xin=None,xout=None):
    """
  This function performs 1D spline interpolation on a given data set.

  Args:
      Z: A 1D tensor or a batch of 1D tensors representing the data to be interpolated.
      num_x_bins: The number of bins in the target (x) dimension for the interpolated output.
      kx: (Optional) The order of the B-spline (default: 3).
      s: (Optional) The smoothing parameter (default: 0.1).

  Returns:
      Z_interp: A 1D tensor (if Z is 1D) or a tensor with the same batch dimension 
                and number of bins as the input for interpolated data.
  """
    
    #ToDo: add batch dimension to Z in order to substitute the for loop in qptransform linear for 1d interpolation with a single tensor operation thus exploiting GPU

    
    # Empty cache
    try:
        torch.cuda.empty_cache()
        #print('Cache emptied')
    except:
        pass
    nx_points = Z.shape[0]
    
    if xin==None:
        x = torch.linspace(-1, 1, nx_points)
    else:
        x= xin

    #compute bspline coefficients
    coef, tx = spline_fit_natural_torch(x, Z, kx, s)

    #inerpolate
    if xout==None:
        x_eval = torch.linspace(-1, 1, num_x_bins)
    else:
        x_eval=xout
        
    Z_interp = evaluate_spline_torch(x_eval, coef, tx, kx).view(num_x_bins)
    return Z_interp


#----------------------------------------------------------------------
def map_to_range(tensor, new_min=-1, new_max=1):
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    # Normalize to [0, 1]
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    # Scale to [new_min, new_max]
    scaled_tensor = normalized_tensor * (new_max - new_min) + new_min

    return scaled_tensor

#Funcitons for spline computations
def generate_natural_knots(x, k):
    """
    Generates a natural knot sequence for B-spline interpolation.
    Natural knot sequence means that 2*k knots are added to the beginning and end of datapoints as replicas of first and last datapoint respectively in order to enforce natural boundary conditions, i.e. second derivative =0.
    the other n nodes are placed in correspondece of thedata points.

    Args:
        x: Tensor of data point positions.
        k: Degree of the spline.

    Returns:
        Tensor of knot positions.
    """
    n = x.shape[0]
    t = torch.zeros(n + 2 * k)
    t[:k] = x[0]
    t[k:-k] = x
    t[-k:] = x[-1]
    return t

def compute_L_R(x, t, d, m, k):
    
    '''
    Compute the L and R values for B-spline basis functions.
    L and R are respectively the firs and second coefficient multiplying B_{i,p-1}(x) and B_{i+1,p-1}(x) in De Boor's recursive formula for Bspline basis funciton computation:
    #{\displaystyle B_{i,p}(x):={\frac {x-t_{i}}{t_{i+p}-t_{i}}}B_{i,p-1}(x)+{\frac {t_{i+p+1}-x}{t_{i+p+1}-t_{i+1}}}B_{i+1,p-1}(x).}
    See https://en.wikipedia.org/wiki/De_Boor%27s_algorithm for further details

    Args:
        x: Tensor of data point positions.
        t: Tensor of knot positions.
        d: Current degree of the basis function.
        m: Number of intervals (n - k - 1, where n is the number of knots and k is the degree).
        k: Degree of the spline.

    Returns:
        L: Tensor containing left values for the B-spline basis functions.
        R: Tensor containing right values for the B-spline basis functions.
    '''
    left_num = x.unsqueeze(1) - t[:m].unsqueeze(0)
    left_den = t[d:m+d] - t[:m]
    L = left_num / left_den.unsqueeze(0)
    
    right_num = t[d+1:m+d+1] - x.unsqueeze(1)
    right_den = t[d+1:m+d+1] - t[1:m+1]
    R = right_num / right_den.unsqueeze(0)

    #handle zero denominator case
    zero_left = left_den == 0
    zero_right = right_den == 0
    zero_left_stacked = zero_left.tile(x.shape[0], 1)
    zero_right_stacked = zero_right.tile(x.shape[0], 1)
    
    L[zero_left_stacked] = 0
    R[zero_right_stacked] = 0
    
    return L, R

def zeroth_order(x, k, t, n, m):
    
    """
    Compute the zeroth-order B-spline basis functions. Accoring to de Boors recursive formula:
    {\displaystyle B_{i,0}(x):={\begin{cases}1&{\text{if }}\quad t_{i}\leq x<t_{i+1}\\0&{\text{otherwise}}\end{cases}}}
    See https://en.wikipedia.org/wiki/De_Boor%27s_algorithm for reference

    Args:
        x: Tensor of data point positions.
        k: Degree of the spline.
        t: Tensor of knot positions.
        n: Number of data points.
        m: Number of intervals (n - k - 1, where n is the number of knots and k is the degree).

    Returns:
        b: Tensor containing the zeroth-order B-spline basis functions.
    """
    b = torch.zeros((n, m, k + 1))
    
    mask_lower = t[:m+1].unsqueeze(0)[:, :-1] <= x.unsqueeze(1)
    mask_upper = x.unsqueeze(1) < t[:m+1].unsqueeze(0)[:, 1:]

    b[:, :, 0] = mask_lower & mask_upper
    b[:, 0, 0] = torch.where(x < t[1], torch.ones_like(x), b[:, 0, 0])
    b[:, -1, 0] = torch.where(x >= t[-2], torch.ones_like(x), b[:, -1, 0])
    return b

def bspline_basis_natural_torch(x, k, t):
    
    '''
    Compute bspline basis function using de Boor's recursive formula (See https://en.wikipedia.org/wiki/De_Boor%27s_algorithm for reference)
    Args:
        x: Tensor of data point positions.
        k: Degree of the spline.
        t: Tensor of knot positions.

    Returns:
        b[:,:,-1]: Tensor containing the kth-order B-spline basis functions.
    '''
    
    n = x.shape[0]
    m = t.shape[0] - k - 1

    #calculate seroth order basis funciton
    b = zeroth_order(x, k, t, n, m)

    #recursive de Boors formula for bspline basis functions
    for d in range(1, k + 1):
        L, R = compute_L_R(x, t, d, m, k)
        left = L * b[:, :, d-1]

        zeros_tensor = torch.zeros(b.shape[0], 1)
        temp_b = torch.cat([b[:, 1:, d-1], zeros_tensor], dim=1)
    
        right = R * temp_b
        b[:, :, d] = left + right

    return b[:, :, -1]

def spline_fit_natural_torch(x, z, kx, s):
    """
  This function computes the B-spline coefficients for natural spline fitting of 1D data.

  Args:
      x: A 1D tensor representing the positions of the data points.
      z: A 1D tensor representing the data values.
      kx: The degree of the B-spline (integer).
      s: The smoothing parameter (positive float).

  Returns:
      coef: A 1D tensor representing the B-spline coefficients.
      tx: A 1D tensor representing the knot positions for the B-spline.
  """

    #generate natural knots
    tx = generate_natural_knots(x, kx)

    #compute basis functions
    bx = bspline_basis_natural_torch(x, kx, tx).to(z.device)

    #add regularizing term
    m = bx.size(1)
    I = torch.eye(m, device=z.device)

    #convert to float in case double
    z = z.float()
    bx = bx.float()

    #linear system for control points
    B_T_B = bx.T @ bx + s * I 
    B_T_z = bx.T @ z

    #solve linear system
    coef = torch.linalg.solve(B_T_B, B_T_z)
    
    return coef.to(z.device), tx


def bivariate_spline_fit_natural_torch(x, y, z, kx,ky, sx=0.001,sy=0.001):
    """
  This function computes the B-spline coefficients for natural bivariate spline fitting of 2D data.

  Args:
      x: A 1D tensor representing the positions of the data points in the target (t) dimension.
      y: A 1D tensor representing the positions of the data points in the frequency (f) dimension.
      z: A 2D tensor representing the data values.
      kx: The order of the B-spline in the target (t) dimension.
      ky: The order of the B-spline in the frequency (f) dimension.
      sx: (Optional) The smoothing parameter in the target dimension (default: 0.001).
      sy: (Optional) The smoothing parameter in the frequency dimension (default: 0.001).

  Returns:
      C: A 2D tensor representing the B-spline coefficients.
      tx: A 1D tensor representing the knot positions for the B-spline in the target dimension.
      ty: A 1D tensor representing the knot positions for the B-spline in the frequency dimension.
  """
    # Generate natural knots in both dimensions
    tx = generate_natural_knots(x, kx)
    ty = generate_natural_knots(y, ky)

    # Compute B-spline basis functions in both dimensions
    Bx = bspline_basis_natural_torch(x, kx, tx)
    By = bspline_basis_natural_torch(y, ky, ty)

    # Adding regularization 
    mx = Bx.size(1)
    my = By.size(1)
    Ix = torch.eye(mx, device=Bx.device)
    Iy = torch.eye(my, device=By.device)


    # Step 1: Solve for intermediate matrix E (representing coefficients for y-direction splines)
    ByT_By = By.T @ By + sy * Iy
    ByT_Z_Bx = (By.T @ z.T) @ Bx
    E = torch.linalg.solve(ByT_By, ByT_Z_Bx)

    # Step 2: Solve for final control points C (representing coefficients for entire bivariate spline)
    BxT_Bx = Bx.T @ Bx + sx * Ix
    C = torch.linalg.solve(BxT_Bx, E.T).T

    return C, tx, ty

def evaluate_bivariate_spline_torch(x, y, C, tx, ty, kx,ky):
    """
  This function evaluates a pre-computed bivariate spline at given points.

  Args:
      x: A 1D tensor representing the target (t) dimension positions for evaluation.
      y: A 1D tensor representing the frequency (f) dimension positions for evaluation.
      C: A 2D tensor representing the pre-computed B-spline coefficients.
      tx: A 1D tensor representing the knot positions for the B-spline in the target dimension.
      ty: A 1D tensor representing the knot positions for the B-spline in the frequency dimension.
      kx: The order of the B-spline in the target (t) dimension.
      ky: The order of the B-spline in the frequency (f) dimension.

  Returns:
      Z_eval: A 2D tensor containing the interpolated values at the specified points (x, y).
    """

    # Compute B-spline basis functions in both dimensions
    Bx = bspline_basis_natural_torch(x, kx, tx)
    By = bspline_basis_natural_torch(y, ky, ty)

    return (Bx@C.T)@By.T 

def evaluate_spline_torch(x, coef, tx, kx):
    """
  This function evaluates a pre-computed 1D spline at given points.

  Args:
      x: A 1D tensor representing the positions for evaluation.
      coef: A 1D tensor representing the pre-computed B-spline coefficients.
      tx: A 1D tensor representing the knot positions for the B-spline.
      kx: The order of the B-spline.

  Returns:
      z_eval: A 1D tensor containing the interpolated values at the specified points (x).
  """

    # Compute B-spline basis functions
    bx = bspline_basis_natural_torch(x, kx, tx).to(coef.device)

    # Evaluate the dot product of B-spline basis functions and coefficients
    z_eval = bx @ coef
    
    return z_eval

        
#---------------------------------------------------
#Functions for debugging
def check_second_derivatives(x, k, knots, coef):
    """
  This function computes the second derivative of a B-spline curve at specified points.

  Args:
      x: A 1D tensor representing the positions for evaluation.
      k: The order of the B-spline.
      knots: A 1D tensor representing the knot positions for the B-spline.
      coef: A 1D tensor representing the B-spline coefficients.

  Returns:
      second_derivative: A 1D tensor containing the second derivative values at the specified points (x).
    """
    # Compute B-spline basis functions
    basis = bspline_basis_natural(x, k, knots)

    # Initialize tensor for second derivatives (same size and device as x)
    second_derivative = torch.zeros_like(x)

    # Loop through each B-spline basis function and accumulate weighted second derivatives
    for i in range(basis.size(1)):
        second_derivative += coef[i] * basis[:, i]
    
    return second_derivative

def visualize_knots_and_basis(x, k, knots, basis, title):
    """
    This function visualizes the knots and basis functions of a B-spline.
    
    Args:
      x: A 1D tensor representing the domain of the B-spline (typically positions for evaluation).
      k: The order of the B-spline.
      knots: A 1D tensor representing the knot positions for the B-spline.
      basis: A 2D tensor containing the B-spline basis functions (one function per column).
      title: (Optional) String title for the plot.
    """
    
    plt.figure(figsize=(12, 6))  # Create a figure with specified size
    
    # Plot knots
    plt.subplot(1, 2, 1)  # Create subplot at position 1, row 1, column 2
    plt.plot(knots.cpu(), 'o-', label='Knots')  # Plot knots as blue circles with line
    plt.title(f'{title} Knots')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    
    # Plot basis functions
    plt.subplot(1, 2, 2)  # Create subplot at position 1, row 1, column 1
    for i in range(basis.size(1)):
        plt.plot(x.cpu(), basis[:, i].cpu(), label=f'B{i}')  # Plot each basis function with label Bi
        plt.title(f'{title} Basis Functions')
        plt.xlabel('x')
        plt.ylabel('Value')
        plt.legend()
        
    plt.show()  # Display the plot

def analyze_coefficients(coef):
    """
    This function visualizes the B-spline coefficients.
    
    Args:
      coef: A 1D tensor representing the B-spline coefficients.
    """
    
    plt.figure(figsize=(12, 6))  # Create a figure with specified size
    
    plt.plot(coef.cpu(), 'o-')  # Plot coefficients as blue circles connected by a line
    plt.title('Spline Coefficients')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()  # Display the plot

def print_basis_functions(x, k, knots, basis):
    """
    This function prints the values of the B-spline basis functions at each point in the domain.
    
    Args:
      x: A 1D tensor representing the domain of the B-spline (typically positions for evaluation).
      k: The order of the B-spline.
      knots: A 1D tensor representing the knot positions for the B-spline.
      basis: A 2D tensor containing the B-spline basis functions (one function per column).
    """
    
    for i in range(basis.size(1)):
        print(f'Basis function B{i}:')
        for j in range(basis.size(0)):
            # Print x value with 2 decimal places and basis function value with 6 decimal places
            print(f'  x = {x[j].item():.2f}: {basis[j, i].item():.6f}')
        print('')  # Print an empty line after each basis function
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