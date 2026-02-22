import torch
import torch.nn.functional as F

class RayTracer:
    """
    Performs ray-tracing (warping) from source plane to image plane.
    """
    def __init__(self, coordinates):
        self.coords = coordinates

    def ray_trace(self, source_image, alpha_x, alpha_y):
        """
        Warps the source image using deflection angles.
        Image Plane -> Source Plane mapping: β = θ - α
        
        Args:
            source_image (Tensor): Source light distribution [B, C, N, N].
            alpha_x, alpha_y (Tensor): Deflection angles [B, 1, N, N].
            
        Returns:
            lensed_image (Tensor): [B, C, N, N]
        """
        # 1. Get Image Plane Coordinates (θ)
        # We need these repeated for the batch
        batch_size = source_image.shape[0]
        
        # Grid [N, N]
        theta_x_grid, theta_y_grid = self.coords.get_grids()
        
        # Expand to batch: [B, N, N]
        theta_x = theta_x_grid.expand(batch_size, -1, -1)
        theta_y = theta_y_grid.expand(batch_size, -1, -1)
        
        # 2. Ray Equation: β = θ - α
        beta_x = theta_x - alpha_x.squeeze(1)
        beta_y = theta_y - alpha_y.squeeze(1)
        
        # 3. Normalize coordinates for grid_sample [-1, 1]
        # FOV half-width
        limit = self.coords.fov / 2.0
        
        # Normalize
        # x_norm = x / limit
        norm_beta_x = beta_x / limit
        norm_beta_y = beta_y / limit
        
        # Stack into [B, N, N, 2] for grid_sample (x, y)
        # grid[..., 0] is x, grid[..., 1] is y
        grid = torch.stack((norm_beta_x, norm_beta_y), dim=-1)
        
        # 4. Sample
        # Padding mode 'zeros' means source is 0 outside FOV (darkness)
        lensed_image = F.grid_sample(source_image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return lensed_image
