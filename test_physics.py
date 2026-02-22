import torch
import matplotlib.pyplot as plt
from physics.coordinates import CoordinateGrid
from physics.poisson import PoissonSolver
from physics.deflections import DeflectionCalculator
from physics.lensing import RayTracer

def test_physics_forward_pass():
    print("--- Testing Physics Engine Forward Pass ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Setup
    N = 64
    coords = CoordinateGrid(pixel_count=N, pixel_scale=0.08, device=device)
    poisson = PoissonSolver(coords)
    deflector = DeflectionCalculator(coords)
    tracer = RayTracer(coords)
    
    # 2. Create a Mock Kappa (Gaussian)
    x, y = coords.get_grids()
    sigma = 0.5
    kappa = 0.5 * torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kappa = kappa.unsqueeze(0).unsqueeze(0) # [1, 1, 64, 64]
    
    # 3. Solve Potential
    psi = poisson.solve(kappa)
    print(f"Psi shape: {psi.shape}")
    
    # 4. Deflections
    ax, ay = deflector.angles(psi)
    print(f"Alpha sizes: {ax.shape}, {ay.shape}")
    
    # 5. Source (Simple blob)
    source_sigma = 0.2
    source_flux = torch.exp(-((x)**2 + (y)**2) / (2 * source_sigma**2))
    source_flux = source_flux.unsqueeze(0).unsqueeze(0)
    
    # 6. Ray Trace
    image = tracer.ray_trace(source_flux, ax, ay)
    print(f"Lensed Image shape: {image.shape}")
    
    # Visualize
    # Save a plot
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 4, 1); plt.imshow(kappa[0,0].cpu(), origin='lower'); plt.title("Kappa")
    plt.subplot(1, 4, 2); plt.imshow(psi[0,0].cpu(), origin='lower'); plt.title("Psi")
    plt.subplot(1, 4, 3); plt.imshow((ax**2 + ay**2)[0,0].cpu().sqrt(), origin='lower'); plt.title("|Alpha|")
    plt.subplot(1, 4, 4); plt.imshow(image[0,0].cpu(), origin='lower'); plt.title("Lensed Image")
    plt.savefig('test_physics_output.png')
    print("Saved test_physics_output.png")

def test_differentiability():
    print("\n--- Testing Differentiability ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    coords = CoordinateGrid(pixel_count=32, pixel_scale=0.1, device=device)
    poisson = PoissonSolver(coords)
    deflector = DeflectionCalculator(coords)
    tracer = RayTracer(coords)
    
    # Target: A shifted Gaussian
    x, y = coords.get_grids()
    target = torch.exp(-((x-0.5)**2 + (y)**2) / (2 * 0.2**2)).unsqueeze(0).unsqueeze(0)
    
    # Parameter to learn: kappa
    # Initialize with zeros
    kappa_param = torch.zeros(1, 1, 32, 32, device=device, requires_grad=True)
    
    # Source (Fixed)
    source = torch.exp(-(x**2 + y**2) / (2 * 0.2**2)).unsqueeze(0).unsqueeze(0)
    
    # Optimizer
    optimizer = torch.optim.Adam([kappa_param], lr=0.1)
    
    print("Optimizing kappa to lens source into target...")
    for i in range(10):
        optimizer.zero_grad()
        
        # Forward
        psi = poisson.solve(kappa_param)
        ax, ay = deflector.angles(psi)
        predicted_image = tracer.ray_trace(source, ax, ay)
        
        # Loss
        loss = torch.sum((predicted_image - target)**2)
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.step()
        
        grad_norm = kappa_param.grad.norm().item()
        print(f"Iter {i}: Loss = {loss.item():.4f}, Grad Norm = {grad_norm:.4f}")
        
    if grad_norm > 0:
        print("SUCCESS: Gradients are flowing!")
    else:
        print("FAILURE: No gradients!")

if __name__ == "__main__":
    test_physics_forward_pass()
    test_differentiability()
