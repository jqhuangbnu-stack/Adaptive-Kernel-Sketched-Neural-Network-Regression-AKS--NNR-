import torch
import torch.nn as nn

class KernelSketcher(nn.Module):
    def __init__(self, m, n, p, h=1.0, device='cuda', seed=42):
        """
        Implementation of Step 1 (Initialization) for Kernel Sketching.
        
        Parameters:
        -----------
        m : int
            Sketch dimension (number of rows in S).
        n : int
            Sample size (number of columns in S).
        p : int
            Input feature dimension (dimension of vector x).
        h : float
            Bandwidth of the Gaussian kernel (sigma).
        device : str
            Target device ('cuda' or 'cpu').
        seed : int
            Random seed for reproducibility.
        """
        super().__init__()
        self.m = m
        self.n = n
        self.p = p
        self.h = h
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Set seed for reproducibility (critical for academic simulations)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Initialize the Random Sketch Matrix S from Step 1
        # register_buffer ensures S moves with the model to GPU but is not a trainable parameter
        S = torch.randn(m, n, device=self.device)
        self.register_buffer('S', S)
        
        # Precompute constants to save cycles during training/inference
        self.register_buffer('sqrt_n', torch.sqrt(torch.tensor(n, dtype=torch.float32)))

    def compute_kn_vector(self, x, X):
        """
        Compute the kernel vector k_n(x) as defined in the mathematical notes.
        k_n(x) = (1/sqrt(n)) * [K(x, x_1), ..., K(x, x_n)]^T
        
        Parameters:
        -----------
        x : torch.Tensor
            Input query vector of shape (p,) or (1, p).
        X : torch.Tensor
            Reference sample matrix of shape (n, p).
        
        Returns:
        --------
        kn_x : torch.Tensor
            Normalized kernel values of shape (1, n).
        """
        # Ensure inputs are on the correct device
        x = x.to(self.device)
        X = X.to(self.device)
        
        if x.dim() == 1:
            x = x.unsqueeze(0) # Shape: (1, p)

        # Vectorized Euclidean distance calculation: ||x - z||^2 = ||x||^2 + ||z||^2 - 2x^Tz
        x_norm = torch.sum(x**2, dim=1, keepdim=True)      # Shape: (1, 1)
        X_norm = torch.sum(X**2, dim=1, keepdim=True).T    # Shape: (1, n)
        x_dot_X = x @ X.T                                  # Shape: (1, n)
        
        # Use clamp to prevent tiny negative values due to floating point precision
        sq_dists = (x_norm + X_norm - 2 * x_dot_X).clamp(min=0)
        
        # Gaussian Kernel calculation with correct sign: exp(-d^2 / (2h^2))
        k_values = torch.exp(-sq_dists / (2 * self.h**2))  # Shape: (1, n)
        
        # Return k_n(x) normalized by sqrt(n)
        return k_values / self.sqrt_n

    def compute_initial_loss(self, Y):
        """
        Compute initial loss L_n(f_0) = (1/n) * sum((y_i - y_mean)^2).
        Equivalent to the variance of Y.
        """
        Y = Y.to(self.device)
        y_mean = torch.mean(Y)
        loss = torch.mean((Y - y_mean) ** 2)
        return loss

    def get_sketched_features(self, k_n_matrix):
        """
        Compute the sketched representation: S @ k_n_matrix.
        Useful for dimensionality reduction of the full kernel matrix.
        
        Parameters:
        -----------
        k_n_matrix : torch.Tensor
            Kernel matrix of shape (n, num_queries).
            
        Returns:
        --------
        Sketched features of shape (m, num_queries).
        """
        return self.S @ k_n_matrix
