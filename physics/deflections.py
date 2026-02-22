import torch

class DeflectionCalculator:
    """
    Computes deflection angles α = ∇ψ.
    """
    def __init__(self, coordinates):
        self.coords = coordinates

    def angles(self, psi):
        """
        Computes gradient of potential.
        Args:
            psi (Tensor): Lensing potential [Batch, 1, N, N]
        Returns:
            alpha_x, alpha_y (Tensor): Deflection angles [Batch, 1, N, N]
        """
        # torch.gradient computes central differences
        # spacing needs to be pixel_scale
        
        # psi shape is [B, C, H, W]
        # We want gradients along H (y) and W (x) dimensions (dim 2 and 3)
        
        scale = self.coords.pixel_scale
        
        # dy, dx corresponding to axis -2 (row) and -1 (col)
        # Note: torch.gradient returns gradients in order of dimensions specified
        grads = torch.gradient(psi, spacing=scale, dim=(-2, -1))
        
        grad_y = grads[0] # Variation along rows (H)
        grad_x = grads[1] # Variation along columns (W)
        
        return grad_x, grad_y
