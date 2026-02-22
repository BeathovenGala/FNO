import torch

class SIS:
    """
    Singular Isothermal Sphere mass profile.
    κ(x, y) = θ_E / (2 * sqrt(x^2 + y^2))
    """
    def __init__(self, coordinates):
        self.coords = coordinates

    def kappa(self, theta_E, center_x=0.0, center_y=0.0):
        """
        Generates kappa map for SIS.
        Args:
            theta_E (float): Einstein radius (arcsec).
            center_x, center_y (float): Centroid position.
            
        Returns:
            kappa_map (Tensor): [1, 1, N, N]
        """
        x, y = self.coords.get_grids() # [N, N]
        
        # Shift
        x_s = x - center_x
        y_s = y - center_y
        
        r = torch.sqrt(x_s**2 + y_s**2)
        
        # Handle r=0
        r[r < 1e-4] = 1e-4
        
        kappa = theta_E / (2.0 * r)
        
        # Add batch dim
        return kappa.unsqueeze(0).unsqueeze(0)
