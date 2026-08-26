import torch


class DimensionEstimator:
    def __init__(self, history_losses, n, d_max, lambda_n=None):
        """
        Implementation of Step 3: Dimension Estimation.

        Parameters
        ----------
        history_losses : list or torch.Tensor
            Losses [L(f_0), L(f_1), ..., L(f_d)].
        n : int
            Sample size.
        d_max : int
            Maximum dimension reached during training.
        lambda_n : float, optional
            Custom penalty factor. If None, uses:
            n^(-2 / (2 + d_max)) * log(log(n)).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not isinstance(history_losses, torch.Tensor):
            self.history_losses = torch.tensor(
                history_losses, dtype=torch.float32, device=self.device
            )
        else:
            self.history_losses = history_losses.to(self.device)

        self.n = n
        self.d_max = d_max
        self.r_hats = []
        self.dimension_fallback_used = False

        if lambda_n is None:
            n_tensor = torch.tensor(n, dtype=torch.float32, device=self.device)
            self.lambda_n = (n ** (-2 / (2 + d_max))) * torch.log(torch.log(n_tensor))
        else:
            self.lambda_n = torch.tensor(
                lambda_n, dtype=torch.float32, device=self.device
            )

    def estimate_dimension(self):
        """
        Compute r_d and estimate the structural dimension.

        Original rule:
            If any r_d < 0.5, choose max{d : r_d < 0.5} + 1.

        Fallback rule:
            If no r_d < 0.5, choose argmin(r_d) + 1 instead of returning 0.
        """
        r_hats = []

        for d in range(self.d_max + 1):
            if d + 2 >= len(self.history_losses):
                break

            L_d = self.history_losses[d]
            L_d1 = self.history_losses[d + 1]
            L_d2 = self.history_losses[d + 2]

            numerator = (L_d1 - L_d2) + self.lambda_n
            denominator = (L_d - L_d1) + self.lambda_n

            r_hat = numerator / denominator
            r_hats.append(r_hat.item())

        self.r_hats = r_hats

        if len(r_hats) == 0:
            self.dimension_fallback_used = True
            return 1

        r_hats_tensor = torch.tensor(r_hats, dtype=torch.float32)
        valid_indices = torch.nonzero(r_hats_tensor < 0.5).flatten()

        if len(valid_indices) > 0:
            self.dimension_fallback_used = False
            return int(valid_indices.max().item() + 1)

        self.dimension_fallback_used = True
        return int(torch.argmin(r_hats_tensor).item() + 1)
