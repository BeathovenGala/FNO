import torch
import torch.fft

class PoissonSolver:
    """
    Solves the Poisson equation ∇²ψ = 2κ using FFT.
    Assumes periodic boundary conditions (MVC limitation).
    """
    def __init__(self, coordinates):
        """
        Args:
            coordinates (CoordinateGrid): The spatial grid.
        """
        self.coords = coordinates
        self._precompute_k_grid()
        
    def _precompute_k_grid(self):
        """Precomputes the frequency grid for FFT."""
        N = self.coords.pixel_count
        pixel_scale = self.coords.pixel_scale
        
        # Frequency frequencies
        # fftfreq returns frequencies in cycles/unit_distance
        # We need angular frequencies (?) or just wave numbers?
        # Standard FFT derivative: d/dx -> i*kx
        # kx = 2*pi * freq
        
        # sampling_rate (d) = pixel_scale
        freqs = torch.fft.fftfreq(N, d=pixel_scale, device=self.coords.device)
        
        # 2 * pi * freq
        k = 2 * torch.pi * freqs
        
        # Create 2D k-grid (ky, kx)
        ky, kx = torch.meshgrid(k, k, indexing='ij')
        
        # k^2 = kx^2 + ky^2
        self.k_squared = kx**2 + ky**2
        
        # Avoid division by zero at k=0 (the DC component)
        # We can set the DC component of potential to 0 (arbitrary constant)
        self.k_squared[0, 0] = 1.0 # arbitrary non-zero to avoid NaN, we will mask it later

    def solve(self, kappa):
        """
        Solves ∇²ψ = 2κ.
        Args:
            kappa (Tensor): Convergence map [Batch, 1, N, N]
        Returns:
            psi (Tensor): Lensing potential [Batch, 1, N, N]
        """
        # 1. FFT of kappa
        kappa_hat = torch.fft.fft2(kappa)
        
        # 2. Solve in Fourier space: ψ_hat = -2 * κ_hat / k^2
        # Broadcast k_squared to batch dimensions
        # k_squared is [N, N], kappa is [B, C, N, N]
        
        psi_hat = -2.0 * kappa_hat / self.k_squared
        
        # 3. Handle DC component (k=0)
        # The mean of the potential is arbitrary. We set it to 0.
        # This corresponds to psi_hat[..., 0, 0] = 0
        psi_hat[..., 0, 0] = 0.0
        
        # 4. Inverse FFT
        psi = torch.fft.ifft2(psi_hat)
        
        # Return real part (imaginary part should be numerical noise)
        return psi.real
