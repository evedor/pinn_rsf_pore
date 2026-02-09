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
cd ./pinn_forward
python main.py --exp b693

# Friction PINN – Inverse Solver
Physics-Informed Neural Network for **inverse estimation** of effective normal stress (σₙ₀) in rate-and-state friction experiments with dilatancy and pore-pressure evolution.

Uses laboratory friction data (shear stress or apparent friction coefficient time series) + physics residuals + initial conditions to simultaneously learn:

- Neural network approximating state evolution  
- Effective normal stress σₙ₀ (as a trainable parameter in MPa)

## Key Differences from Forward Solver

- σₙ₀ is now a learnable `nn.Parameter` (initialized at 30 MPa)  
- Data loss term matches predicted friction → measured friction from file  

## Quick Start
cd ./pinn_backward
python pinn_backward/main.py --exp b693

## Data source
The initial conditions for each experiment in the forward modeling, as well as the laboratory data files required for the inversion, can be obtained in the following two ways:

1. The laboratory measurement data results can be found on the Open Science Framework (OSF) with DOI 10.17605/OSF.IO/9DQH7
2. The code to reproduce the data using the Runge-Kutta method is available at: https://github.com/Geolandi/labquakesde

We have provided a spring_slider_model_RK.jl file is used to quickly generate the corresponding files of the paper, it is adapted from https://github.com/Geolandi/labquakesde.
It should be noted that the initial conditions of the Forward Solver and the observation files of the Backward Solver are randomly selected from the results of the spring_slider_model_RK.jl file. As long as they are stable after the preheating stage, they can be randomly selected. Random selection does not affect the experimental results of the paper.
