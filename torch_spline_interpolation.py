import torch
import numpy as np
import time 
import time
import torch
from matplotlib import pyplot as plt
from matplotlib import gridspec


import torch
import torch.nn as nn
import numpy as np

# Spline interpolation for 1D
class SplineInterpolate1D(nn.Module):
    def __init__(self, num_x_bins, kx=3, s=0.0):
        super().__init__()
        self.num_x_bins = num_x_bins
        self.kx = kx
        self.s = s

    def forward(self, Z, xin=None, xout=None):
        
        self.device=Z.device
        
        while len(Z.shape)<4:
           print('Adding batch and/or channel dimension dimension...')
           Z=Z.unsqueeze(0) 

        nx_points = Z.shape[-1]

        
        if xin is None:
            x = torch.linspace(-1, 1, nx_points, device=Z.device)
        else:
            x = xin.to(self.device)

        coef, tx = self.spline_fit_natural_torch(x, Z, self.kx, self.s)

        if xout is None:
            x_eval = torch.linspace(-1, 1, self.num_x_bins, device=Z.device)
        else:
            x_eval = xout.to(self.device)

        Z_interp = self.evaluate_spline_torch(x_eval, coef, tx, self.kx)#.view(-1, self.num_x_bins)
        
        print(f'{Z_interp.shape=}')
        
        return Z_interp
        
    
    def generate_fitpack_knots(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute knots placement like FITPACK for spline degree k, given data points x.
        x: 1D tensor of increasing data points, length n
        Returns knot vector of length n + k + 1
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)  # batch dim
        x=x.double()
        B, n = x.shape
        if n < k + 1:
            raise ValueError(f"Need at least {k + 1} points for degree k={k}.")

        # Number of knots
        m = n + k + 1

        # Boundary knots: repeat boundary points k+1 times
        left = x[:, 0:1].expand(B, k+1)
        right = x[:, -1:].expand(B, k+1)

        # Interior knots: average of k consecutive points **starting at i+1**
        # Number of interior knots = m - 2*(k+1) = n - k - 1
        # i ranges from 0 to n-k-2 (total n-k-1 knots)
        interior_knots = torch.stack(
            [x[:, i+1:i + k + 1].mean(dim=1) for i in range(n - k - 1)],
            dim=1
        ) if (n - k - 1) > 0 else torch.empty(B, 0, device=x.device, dtype=x.dtype)

        # Concatenate all knots
        knots = torch.cat([left, interior_knots, right], dim=1)

        # Remove batch dim if originally 1D
        return knots.squeeze(0) if knots.shape[0] == 1 else knots
    

    
    def compute_L_R(self, x, t, d, m, k):
        
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
    
    def zeroth_order(self, x, k, t, n, m):
        
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
        b = torch.zeros((n, m, k + 1),device=self.device)
        
        mask_lower = t[:m+1].unsqueeze(0)[:, :-1] <= x.unsqueeze(1)
        mask_upper = x.unsqueeze(1) < t[:m+1].unsqueeze(0)[:, 1:]
    
        b[:, :, 0] = mask_lower & mask_upper
        b[:, 0, 0] = torch.where(x < t[1], torch.ones_like(x,device=self.device), b[:, 0, 0])
        b[:, -1, 0] = torch.where(x >= t[-2], torch.ones_like(x,device=self.device), b[:, -1, 0])
        return b
    
    def bspline_basis_natural_torch(self, x, k, t):
        
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
        b = self.zeroth_order(x, k, t, n, m)
    
        #recursive de Boors formula for bspline basis functions
        for d in range(1, k + 1):
            L, R = self.compute_L_R(x, t, d, m, k)
            left = L * b[:, :, d-1]
    
            zeros_tensor = torch.zeros(b.shape[0], 1,device=self.device)
            temp_b = torch.cat([b[:, 1:, d-1], zeros_tensor], dim=1)
        
            right = R * temp_b
            b[:, :, d] = left + right
    
        return b[:, :, -1]
    
    def spline_fit_natural_torch(self, x, z, kx, s):
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
        tx= self.generate_fitpack_knots(x,kx)    #new generation matching scipy (fitpack)
    
        #compute basis functions
        bx = self.bspline_basis_natural_torch(x, kx, tx).to(self.device)
    
        #add regularizing term
        m = bx.size(1)
        I = torch.eye(m, device=self.device)
    
        #convert to float in case double
        z = z.float()
        bx = bx.float()
    
        #print(f'{bx.shape=}')
        #print(f'{z.shape=}')
    
        # Compute BTB matrix
        B_T_B = torch.einsum("mt,tn->mn", bx.mT, bx)  # [M, M]

        # Optional regularization
        if self.s > 0:
            print(f'adding regularization to solver: {s=}')
            B_T_B += self.s * torch.eye(B_T_B.shape[0], device=z.device)

        # Compute B^T @ z for each group/batch/channel:
        # Result: [G, B, C, M]
        B_T_z = torch.einsum("...mt,...t->...m", bx.mT, z)
        #print(f'{B_T_z.shape=}')
        #print(f'{B_T_B.shape=}')
        # Solve: (B_T_B @ coef) = B_T_z  → for each (g,b,c)
        # Output shape: [G, B, C, m]
        coef = torch.linalg.solve(B_T_B, B_T_z.unsqueeze(-1))
        #print(f'{coef.shape=}')
        

        return coef.to(z.device), tx

    

    def evaluate_spline_torch(self, x, coef, tx, kx):
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
        bx = self.bspline_basis_natural_torch(x, kx, tx).to(self.device)  
        
        
        z_eval = torch.einsum("tm,...ml->...tl", bx, coef).squeeze(-1)


        return z_eval
        
#---------------------------------------------------------------------

#2D

class SplineInterpolate2D(nn.Module):
    def __init__(self, num_t_bins, num_f_bins, kx=3, ky=3, sx=0.0, sy=0.0, logf=False, frange=(8, 500)):
        super().__init__()
        self.num_t_bins = num_t_bins
        self.num_f_bins = num_f_bins
        self.kx = kx
        self.ky = ky
        self.sx = sx
        self.sy = sy
        self.logf = logf
        self.frange = frange

    def forward(self, Z, freqs=None, xin=None, xout=None, yin=None, yout=None):

        self.device=Z.device
        
        while len(Z.shape)<4:
           print('Adding batch or channel dimension...')
           Z=Z.unsqueeze(0) 
        
        batch_size,channel_size, nx_points, ny_points = Z.shape[0],  Z.shape[1], Z.shape[-2], Z.shape[-1]

        if xin is None:
            x = torch.linspace(-1, 1, nx_points, device=self.device)
        else:
            x = xin.to(self.device)

        if freqs is not None:
            y = freqs
        elif yin is None:
            if self.logf:
                y = torch.tensor(np.geomspace(self.frange[0], self.frange[1], ny_points), device=self.device)
            else:
                y = torch.linspace(-1, 1, ny_points, device=self.device)
        else:
            y = yin.to(self.device)

        coef, tx, ty = self.bivariate_spline_fit_natural_torch(x.double(), y.double(), Z.double(), self.kx, self.ky, self.sx, self.sy)

        if xout is None:
            x_eval = torch.linspace(-1, 1, self.num_t_bins, device=self.device)
        else:
            x_eval = xout.to(self.device)

        if yout is None:
            if self.logf:
                y_eval = torch.tensor(np.geomspace(self.frange[0], self.frange[1], self.num_f_bins), device=self.device)
            else:
                y_eval = torch.linspace(-1, 1, self.num_f_bins, device=self.device)
        else:
            y_eval = yout.to(self.device)

        Z_interp = self.evaluate_bivariate_spline_torch_clamp(x_eval.double(), y_eval.double(), coef, tx, ty, self.kx, self.ky)
        return Z_interp.permute(0,1,3,2)

    def generate_natural_knots(self, x, k):
        n = x.shape[0]
        t = torch.zeros(n + 2 * k, device=self.device)
        t[:k] = x[0]
        t[k:-k] = x
        t[-k:] = x[-1]
        return t
    
    def generate_fitpack_knots(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute knots placement like FITPACK for spline degree k, given data points x.
        x: 1D tensor of increasing data points, length n
        Returns knot vector of length n + k + 1
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)  # batch dim
        
        x=x.double()

        B, n = x.shape
        if n < k + 1:
            raise ValueError(f"Need at least {k + 1} points for degree k={k}.")

        # Number of knots
        m = n + k + 1

        # Boundary knots: repeat boundary points k+1 times
        left = x[:, 0:1].expand(B, k+1)
        right = x[:, -1:].expand(B, k+1)

        # Interior knots: average of k consecutive points **starting at i+1**
        # Number of interior knots = m - 2*(k+1) = n - k - 1
        # i ranges from 0 to n-k-2 (total n-k-1 knots)
        interior_knots = torch.stack(
            [x[:, i+1:i + k + 1].mean(dim=1) for i in range(n - k - 1)],
            dim=1
        ) if (n - k - 1) > 0 else torch.empty(B, 0, device=x.device, dtype=x.dtype)

        # Concatenate all knots
        knots = torch.cat([left, interior_knots, right], dim=1)

        # Remove batch dim if originally 1D
        return knots.squeeze(0) if knots.shape[0] == 1 else knots
    
    def generate_fitpack_knots_refined(self, x: torch.Tensor, k: int, max_iter: int = 200, tol: float = 1e-30):
        """
        Native PyTorch emulation of FITPACK knot placement logic for interpolation (s=0).
        This does not exactly replicate FITPACK but mimics its iterative refinement behavior.
        Use it when logf=True, not needed otherwise

        Args:
            x (torch.Tensor): 1D tensor of increasing sorted values (len n)
            k (int): Degree of B-spline
            max_iter (int): Max number of refinement iterations
            tol (float): Stop if max change in interior knots < tol

        Returns:
            torch.Tensor: Knot vector of length n + k + 1
        """
        x = x.double()
        n = x.numel()
        assert n > k + 1, "Need at least k+2 points for interpolation"

        left = x[0].repeat(k + 1)
        right = x[-1].repeat(k + 1)

        m = n + k + 1
        num_internal = m - 2*(k + 1)

        if num_internal <= 0:
            return torch.cat([left, right])

        # Initial guess: Greville-like average of k points
        interior = torch.stack([
            x[i + 1:i + k + 1].mean() for i in range(n - k - 1)
        ])

        # Iterative refinement loop
        for _ in range(max_iter):
            # Pseudo residuals (distance to target "uniformized" locations)
            dx = x[1:] - x[:-1]
            weight = dx / dx.sum()
            cumw = torch.cat([torch.zeros(1, dtype=x.dtype, device=x.device), weight.cumsum(0)])

            target_interp = x[0] + (x[-1] - x[0]) * cumw[1:-1]
            new_interior = torch.stack([
                target_interp[i + k//2] for i in range(num_internal)
            ])

            diff = (new_interior - interior).abs().max()
            interior = 0.5 * interior + 0.5 * new_interior

            if diff < tol:
                break

        knots=torch.cat([left, interior, right])#.float()
        # Remove batch dim if originally 1D
        return knots.squeeze(0) if knots.shape[0] == 1 else knots

    def bspline_basis_natural_torch(self, x, k, t):
        n = x.shape[0]
        m = t.shape[0] - k - 1

        b = self.zeroth_order(x, k, t, n, m)

        for d in range(1, k + 1):
            L, R = self.compute_L_R(x, t, d, m, k)
            left = L * b[:, :, d-1]

            zeros_tensor = torch.zeros(b.shape[0], 1, device=self.device)
            temp_b = torch.cat([b[:, 1:, d-1], zeros_tensor], dim=1)

            right = R * temp_b
            b[:, :, d] = left + right

        return b[:, :, -1]

    def zeroth_order(self, x, k, t, n, m):
        b = torch.zeros((n, m, k + 1), device=self.device,dtype=torch.float64)


        mask_lower = t[:m+1].unsqueeze(0)[:, :-1] <= x.unsqueeze(1)
        mask_upper = x.unsqueeze(1) < t[:m+1].unsqueeze(0)[:, 1:]

        b[:, :, 0] = mask_lower & mask_upper
        b[:, 0, 0] = torch.where(x < t[1], torch.ones_like(x,device=self.device), b[:, 0, 0])
        b[:, -1, 0] = torch.where(x >= t[-2], torch.ones_like(x,device=self.device), b[:, -1, 0])
        return b

    def compute_L_R(self, x, t, d, m, k):
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

    def bivariate_spline_fit_natural_torch(self, x, y, z, kx, ky, sx, sy):
        
        #tx = self.generate_natural_knots(x, kx) #old unifrom generation
        tx = self.generate_fitpack_knots(x,kx)
        
        #ty = self.generate_natural_knots(y, ky) #old uniform generation
        if self.logf:
            ty = self.generate_fitpack_knots_refined(y,ky)
        else:
            ty = self.generate_fitpack_knots(y,ky)
        
        print(f'{tx.dtype=}')
        print(f'{ty.dtype=}')

        Bx = self.bspline_basis_natural_torch(x, kx, tx).to(self.device)
        By = self.bspline_basis_natural_torch(y, ky, ty).to(self.device)
        
        print(f'{Bx.dtype=}')
        print(f'{By.dtype=}')
              
        print(f'{z.dtype=}')
        print(f'{x.dtype=}')
        print(f'{y.dtype=}')
        


        mx = Bx.size(1)
        my = By.size(1)
        Ix = torch.eye(mx, device=self.device)
        Iy = torch.eye(my, device=self.device)
        
        '''
        # Adding batch dimension handling
        ByT_By = By.T.unsqueeze(0).unsqueeze(0) @ By.unsqueeze(0).unsqueeze(0) + (sy * Iy).unsqueeze(0).unsqueeze(0) 
        ByT_Z_Bx =  By.T.unsqueeze(0).unsqueeze(0)@ z.transpose(2,3) @ Bx.unsqueeze(0).unsqueeze(0)  
        E = torch.linalg.solve(ByT_By, ByT_Z_Bx) 

        BxT_Bx = Bx.T.unsqueeze(0).unsqueeze(0) @ Bx.unsqueeze(0).unsqueeze(0) + (sx * Ix).unsqueeze(0).unsqueeze(0)  
        C = torch.linalg.solve(BxT_Bx, E.transpose(2,3))

        return C.to(self.device), tx, ty
        '''
    
    
        # BxT_Bx: [mx, mx], ByT_By: [my, my]
        BxT_Bx = Bx.T @ Bx 
        if sx>0:
            BxT_Bx += sx * Ix
        ByT_By = By.T @ By 
        if sy>0:
            ByT_By += sy * Iy

        # Step 1: Apply both basis matrices
        '''
        ByT_Z_Bx = torch.einsum("ij,bcik,kl->bcjl", By, z.transpose(-1,-2), Bx)
        print(f'{ByT_Z_Bx.shape=}')

        # Step 2: Solve along frequency axis
        E = torch.linalg.solve(ByT_By, ByT_Z_Bx)   # shape: [B, C, my, mx]

        # Step 3: Solve along time axis (after transpose)
        C = torch.linalg.solve(BxT_Bx, E.mT).mT    # shape: [B, C, my, mx]

        print(f'{C.shape=}')
        '''
        BxT_Z_By = torch.einsum("ij,bcik,kl->bcjl", Bx, z, By)
        #BxT_Z_By = torch.einsum("tm,bcnf,fy->bcmy", Bx, z, By.T)
        
        # Step 2: solve along time axis
        E = torch.linalg.solve(BxT_Bx.double(), BxT_Z_By.double())  # now shape [B, C, mx, my]

        # Step 3: solve along freq axis
        C = torch.linalg.solve(ByT_By.double(), E.mT).mT   # final shape: [B, C, mx, my]

        return C, tx, ty
        
    def evaluate_bivariate_spline_torch(self, x, y, C, tx, ty, kx, ky):
        """
        Evaluate a bivariate spline on a grid of x and y points.
        
        Args:
            x: Tensor of x positions to evaluate the spline.
            y: Tensor of y positions to evaluate the spline.
            C: Coefficient tensor of shape (batch_size, mx, my).
            tx: Knot positions for x.
            ty: Knot positions for y.
            kx: Degree of spline in x.
            ky: Degree of spline in y.
            
        Returns:
            Z_interp: Interpolated values at the grid points.
        """
        Bx = self.bspline_basis_natural_torch(x, kx, tx).to(self.device)  
        By = self.bspline_basis_natural_torch(y, ky, ty).to(self.device)  
            
        # Perform matrix multiplication using einsum to get Z_interp
        '''
        print(f'{By.shape=}')
        print(f'{C.shape=}')
        print(f'{Bx.mT.shape=}')
        '''
        Z_interp = torch.einsum("xk,bckm,my->bcxy", Bx, C, By.T)
        return Z_interp
    
    def evaluate_bivariate_spline_torch_clamp(self, x, y, C, tx, ty, kx, ky):
        """
        Evaluate a bivariate spline on a grid of x and y points using clamped extrapolation,
        mimicking SciPy's RectBivariateSpline behavior.

        Args:
            x: Tensor of x positions to evaluate the spline.
            y: Tensor of y positions to evaluate the spline.
            C: Coefficient tensor of shape (batch_size, mx, my).
            tx: Knot positions for x (1D tensor).
            ty: Knot positions for y (1D tensor).
            kx: Degree of spline in x.
            ky: Degree of spline in y.

        Returns:
            Z_interp: Interpolated values at the grid points.
        """
        # ✨ Clamp x and y to valid knot ranges (including boundaries)
        x_min = tx[kx]
        x_max = tx[-kx - 1]
        y_min = ty[ky]
        y_max = ty[-ky - 1]

        x_clamped = torch.clamp(x, min=x_min, max=x_max)
        y_clamped = torch.clamp(y, min=y_min, max=y_max)

        # ✨ Compute basis functions on the clamped coordinates
        Bx = self.bspline_basis_natural_torch(x_clamped, kx, tx).to(self.device)  # [x, kx]
        By = self.bspline_basis_natural_torch(y_clamped, ky, ty).to(self.device)  # [y, ky]

        # ✨ Evaluate spline via tensor contraction
        Z_interp = torch.einsum("xk,bckm,my->bcxy", Bx, C, By.T)

        return Z_interp

        
    '''def evaluate_bivariate_spline_torch_old(self, x, y, C, tx, ty, kx, ky):
        Bx = self.bspline_basis_natural_torch(x, kx, tx)  # (num_x_eval_points, mx)
        By = self.bspline_basis_natural_torch(y, ky, ty)  # (num_y_eval_points, my)
        
        # Bx: (num_x_eval_points, mx)
        # By: (num_y_eval_points, my)
        # C: (batch_size, mx, my)
        
        # Need to expand dimensions to handle batch correctly
        Bx = Bx.unsqueeze(0)  # (1, num_x_eval_points, mx)
        By = By.unsqueeze(0)  # (1, num_y_eval_points, my)
        
        # Performing batch-wise evaluation
        Z_interp = torch.einsum('bnm,bmp,bpq->bnq', Bx, C, By.transpose(1, 2))
        return Z_interp
    '''


