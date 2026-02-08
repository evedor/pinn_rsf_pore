# Pinn_rsf_pore
Physics-Informed Neural Networks for coupled rate-and-state friction and pore-pressure evolution

# Forward Solver for laboratory friction experiments.
Approximates the time evolution of the 4D state vector:

- **x** = log(vᵢ / v₀)
- **y** = (τfᵢ - μ₀ * σₙ₀) / (a * σₙ₀)
- **z** = (φᵢ - ϕ₀) / (λ * β * σₙ₀)
- **u** = -pᵢ / (λ * σₙ₀)

## Supported Experiments

| Exp   | σₙ₀ (MPa) |
|-------|-----------|
| b693  | 17.003    |                         
| b694  | 15.400    |                          
| b695  | 17.909    |                          
| b696  | 14.408    |                         
| b697  | 16.389    |                          
| b698  | 17.379    |                       
| b721  | 21.985    |                       
| b722  | 14.000    |                      
| b724  | 13.600    |                     
| b726  | 15.001    |                     
| b728  | 20.002    |                     

## Quick Start
python pinn_forward/main.py --exp b693

# Friction PINN – Inverse Solver
Physics-Informed Neural Network for **inverse estimation** of effective normal stress (σₙ₀) in rate-and-state friction experiments with dilatancy and pore-pressure evolution.

Uses laboratory friction data (shear stress or apparent friction coefficient time series) + physics residuals + initial conditions to simultaneously learn:

- Neural network approximating state evolution  
- Effective normal stress σₙ₀ (as a trainable parameter in MPa)

## Key Differences from Forward Solver

- σₙ₀ is now a learnable `nn.Parameter` (initialized at 30 MPa)  
- Data loss term matches predicted friction → measured friction from file  

## Quick Start
python pinn_backward/main.py --exp b693
