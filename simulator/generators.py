import torch
from torch.utils.data import Dataset
import numpy as np

# Import Physics Modules
from physics.coordinates import CoordinateGrid
from physics.poisson import PoissonSolver
from physics.deflections import DeflectionCalculator
from physics.lensing import RayTracer

# Import Profiles
from simulator.profiles import SIS

class LensingDataset(Dataset):
    """
    On-the-fly generator for Strong Lensing systems.
    Generates (Lensed Image, True Kappa) pairs.
    """
    def __init__(self, epoch_size=1000, pixel_count=64, pixel_scale=0.08):
        """
        Args:
            epoch_size (int): Number of samples per epoch.
            pixel_count (int): Image size.
            pixel_scale (float): Pixel scale.
        """
        self.length = epoch_size
        
        # Initialize Physics on CPU for data generation
        self.coords = CoordinateGrid(pixel_count, pixel_scale, device='cpu')
        self.poisson = PoissonSolver(self.coords)
        self.deflector = DeflectionCalculator(self.coords)
        self.tracer = RayTracer(self.coords)
        self.sis = SIS(self.coords)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Sample Parameters
        # Independent of idx, pure random
        theta_E = np.random.uniform(0.5, 1.5)
        center_x = np.random.uniform(-0.1, 0.1)
        center_y = np.random.uniform(-0.1, 0.1)
        
        # 2. Generate Kappa (Ground Truth)
        # Returns [1, 1, N, N]
        kappa = self.sis.kappa(theta_E, center_x, center_y)
        
        # 3. Forward Physics to get Image
        # Solve Potential
        with torch.no_grad():
            psi = self.poisson.solve(kappa)
            ax, ay = self.deflector.angles(psi)
            
            # Generate Source (Simple Gaussian)
            x, y = self.coords.get_grids()
            sx = np.random.uniform(-0.1, 0.1)
            sy = np.random.uniform(-0.1, 0.1)
            source_sigma = 0.1
            source = torch.exp(-((x-sx)**2 + (y-sy)**2) / (2 * source_sigma**2))
            source = source.unsqueeze(0).unsqueeze(0)
            
            # Ray Trace
            lensed_image = self.tracer.ray_trace(source, ax, ay)
        
        # 4. Return
        # Strip batch dimension [1, N, N] -> [1, N, N] (Already correct, but verify)
        # We want [C, H, W]
        return lensed_image.squeeze(0), kappa.squeeze(0)
