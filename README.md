# DeepLense PINO (Physics-Informed Neural Operator)

**Goal**: Reconstruct Dark Matter mass distributions ($\kappa$) from strong lensing images using a differentiable physics engine.

## Project Structure
- `physics/`: Differentiable lensing physics (Poisson solver, Ray-tracing).
- `simulator/`: Synthetic data generation (SIS profiles).
- `models/`: Neural Operators (U-Net for MVC).
- `training/`: Training loops and physics-informed losses.

## MVC status
- [ ] Physics Engine
- [ ] Simulator
- [ ] Model
- [ ] Training Loop
