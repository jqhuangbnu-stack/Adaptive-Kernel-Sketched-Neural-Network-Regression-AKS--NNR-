import torch
import torch.nn as nn
import torch.optim as optim
class SketchedModel(nn.Module):
    def __init__(self, d, m, L=3, hidden_dim=64, output_dim=1):
        """
        Implementation of f_{(d)}(x) = h_{(d)}(C * S * k_n(x))
        
        Parameters:
        -----------
        d : int - The current intrinsic dimension
        m : int - The sketch dimension
        L : int - Number of layers in h_{(d)}
        hidden_dim : int - Neurons in each hidden layer
        """
        super().__init__()
        self.d = d
        self.m = m
        self.output_dim = output_dim
        
        # 1. Parameterize Matrix C (d x m)
        self.C_raw = nn.Parameter(torch.randn(d, m) * 0.01)
        
        # 2. Define h_{(d)} based on Eq. (f01)
        # Using nn.Sequential to represent the composition of Linear and ReLU
        layers = []
        current_dim = d
        for i in range(L - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # Final layer W_L * z + b_L (no ReLU)
        layers.append(nn.Linear(current_dim, output_dim))
        self.h_d = nn.Sequential(*layers)

    def get_C(self):
        """Enforces C_ij = 0 if i+1 <= j <= d (Lower Triangular Constraint)."""
        C = self.C_raw.clone()
        if self.d > 1:
            # Create a mask for the upper triangular part of the first (d x d) block
            # diagonal=1 ensures we keep the diagonal and only zero out the part above it
            mask = torch.triu(torch.ones(self.d, self.d, device=C.device), diagonal=1)
            # Masking only the first d columns
            C[:, :self.d] = C[:, :self.d] * (1 - mask)
        return C

    def transform(self, sketched_kn):
        """
        sketched_kn: (m, batch_size) - result of S @ k_n(x)
        Returns: (batch_size, d), the estimated sufficient predictor Ghat.
        """
        z = self.get_C() @ sketched_kn
        return z.T

    def forward(self, sketched_kn):
        """
        sketched_kn: (m, batch_size) - result of S @ k_n(x)
        Returns: (batch_size, output_dim)
        """
        return self.h_d(self.transform(sketched_kn))



class KernelModelTrainer:
    def __init__(self, sketcher, lambda_c=0.1, lr=1e-3, L=3, hidden_dim=64,
                 patience=20, val_split=0.2, stop_threshold=0.1,
                 max_epochs=2000):
        """
        Trainer for the Sketched Kernel Model with Early Stopping.
        
        Parameters:
        -----------
        sketcher : KernelSketcher
            The sketcher instance from Step 1.
        lambda_c : float
            Penalty weight for the orthogonality constraint ||CC^T - I||.
        lr : float
            Learning rate for the Adam optimizer.
        L : int
            Number of layers in h_{(d)}.
        hidden_dim : int
            Number of hidden units in each layer of h_{(d)}.
        patience : int
            Number of epochs to wait for improvement before early stopping.
        val_split : float
            Fraction of the training data to be used as a validation set.
        """
        self.sketcher = sketcher
        self.lambda_c = lambda_c
        self.lr = lr
        self.L = L
        self.hidden_dim = hidden_dim
        self.patience = patience
        self.val_split = val_split
        self.stop_threshold = stop_threshold
        self.max_epochs = max_epochs
        self.history_losses = []
        self.output_dim = 1

    def train_fixed_d(self, d, X_train, Y_train, max_epochs=None):
        """
        Step 2(1): Solve the optimization for a fixed dimension d using Early Stopping.
        """
        device = self.sketcher.device
        if max_epochs is None:
            max_epochs = self.max_epochs
        
        # 1. Split data into training and validation sets
        n = X_train.shape[0]
        n_val = int(n * self.val_split)
        indices = torch.randperm(n)
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        # 2. Precompute Sketched Features (Full calculation to save memory management overhead)
        with torch.no_grad():
            kn_full = self._compute_full_kn(X_train) 
            # S is (m, n), kn_full is (n, n) -> all_sketched is (m, n)
            all_sketched = self.sketcher.S @ kn_full 
            
        train_feat = all_sketched[:, train_idx]
        val_feat = all_sketched[:, val_idx]
        if Y_train.dim() == 1:
            train_y = Y_train[train_idx].to(device).view(-1, 1)
            val_y = Y_train[val_idx].to(device).view(-1, 1)
            self.output_dim = 1
        else:
            train_y = Y_train[train_idx].to(device)
            val_y = Y_train[val_idx].to(device)
            self.output_dim = train_y.shape[1]

        # 3. Initialize model and optimization components
        model = SketchedModel(
            d,
            self.sketcher.m,
            L=self.L,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Training Step
            preds = model(train_feat)
            mse_loss = criterion(preds, train_y)
            
            # Orthogonality constraint: ||CC^T - I||_2
            C = model.get_C()
            identity = torch.eye(d, device=device)
            ortho_loss = torch.norm(C @ C.T - identity, p=2)
            
            total_loss = mse_loss + self.lambda_c * ortho_loss
            
            total_loss.backward()
            optimizer.step()

            # Validation Step
            model.eval()
            with torch.no_grad():
                val_preds = model(val_feat)
                current_val_loss = criterion(val_preds, val_y).item()

            # Early Stopping Logic
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_no_improve = 0
                # Store the best model state (on CPU to save GPU memory)
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                # print(f"  Early stopping at epoch {epoch} for d={d}")
                break
        
        # Load the best performing model parameters
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        return model, best_val_loss

    def _compute_full_kn(self, X):
        """Vectorized Kernel Matrix computation on GPU."""
        X = X.to(self.sketcher.device)
        X_norm = torch.sum(X**2, dim=1, keepdim=True)
        # ||x-z||^2 = ||x||^2 + ||z||^2 - 2x^Tz
        sq_dists = X_norm + X_norm.T - 2 * (X @ X.T)
        sq_dists = sq_dists.clamp(min=0)
        kn = torch.exp(-sq_dists / (2 * self.sketcher.h**2)) / self.sketcher.sqrt_n
        return kn

    def run_iterative_training(self, X_train, Y_train, d_max=10):
        """
        Iteratively train models including f_0 as the baseline.
        """
        # Step 0: Initialize history with L_n(f_0)
        initial_loss = self.sketcher.compute_initial_loss(Y_train).item()
        self.history_losses = [initial_loss]
        var_y = initial_loss

        # 必须先初始化这个字典
        trained_models = {}

        final_d = d_max

        for d in range(1, d_max + 1):
            print(f"\n--- Training for d = {d} ---", flush=True)

            # Step 1: Train model f_d
            model, final_loss = self.train_fixed_d(d, X_train, Y_train)
            self.history_losses.append(final_loss)
            trained_models[d] = model

            # Step 2: Check Stopping Criterion for d >= 3
            if d >= 3:
                ratios = []
                for k in range(3, 0, -1):
                    L_prev = self.history_losses[d - k]
                    L_curr = self.history_losses[d - k + 1]
                    ratio = (L_prev - L_curr) / var_y
                    ratios.append(ratio)

                delta_d = max(ratios)
                print(f"Computed Improvement Delta_{d}: {delta_d:.4f} "
                    f"(threshold={self.stop_threshold})", flush=True)

                if delta_d <= self.stop_threshold:
                    print(f"Stopping criterion met: Delta_{d} <= {self.stop_threshold}. "
                        f"Setting d_max = {d}", flush=True)
                    final_d = d
                    break

        return trained_models, final_d
