import torch

class CoordinateGrid:
    """
    Manages the coordinate system for the lensing field.
    """
    def __init__(self, pixel_count=64, pixel_scale=0.08, device='cpu'):
        """
        Args:
            pixel_count (int): Number of pixels per side (N x N).
            pixel_scale (float): Arcseconds per pixel.
            device (str): torch device.
        """
        self.pixel_count = pixel_count
        self.pixel_scale = pixel_scale
        self.device = device
        
        # Calculate field of view size
        self.fov = pixel_count * pixel_scale
        
        # Create meshgrid
        # Range is from -fov/2 to +fov/2
        range_min = -self.fov / 2 + self.pixel_scale / 2
        range_max = self.fov / 2 - self.pixel_scale / 2
        
        x = torch.linspace(range_min, range_max, steps=pixel_count, device=device)
        y = torch.linspace(range_min, range_max, steps=pixel_count, device=device)
        
        # Create 2D grids [N, N]
        # X varies along columns (dim 1), Y varies along rows (dim 0)
        # We use 'ij' indexing so X corresponds to columns and Y to rows in image convention? 
        # Standard meshgrid: 
        # grid_y[i, j] = y[i]
        # grid_x[i, j] = x[j]
        # This corresponds to cartesian coordinates where x is horizontal, y is vertical
        self.y_grid, self.x_grid = torch.meshgrid(y, x, indexing='ij')

    def get_grids(self):
        """Returns (x_grid, y_grid) in arcseconds."""
        return self.x_grid, self.y_grid

    def to(self, device):
        """Moves grids to new device."""
        self.device = device
        self.x_grid = self.x_grid.to(device)
        self.y_grid = self.y_grid.to(device)
        return self
