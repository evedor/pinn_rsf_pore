# pinn_rsf_pore
Physics-Informed Neural Networks for coupled rate-and-state friction and pore-pressure evolution

# Forward simulation of laboratory friction experiments.
Approximates the time evolution of the 4D state vector:

- **x** = log(vᵢ / v₀)
- **y** = (τfᵢ - μ₀ * σₙ₀) / (a * σₙ₀)
- **z** = (φᵢ - ϕ₀) / (λ * β * σₙ₀)
- **u** = -pᵢ / (λ * σₙ₀)

## Supported Experiments

| Exp   | σₙ₀ (MPa) | Notes                     |
|-------|-----------|---------------------------|
| b693  | 17.003    |                           |
| b694  | 15.400    |                           |
| b695  | 17.909    |                           |
| b696  | 14.408    |                           |
| b697  | 16.389    |                           |
| b698  | 17.379    |                           |
| b721  | 21.985    |                           |
| b722  | 14.000    |                           |
| b724  | 13.600    |                           |
| b726  | 15.001    |                           |
| b728  | 20.002    |                           |

## Quick Start
python main.py --exp b728
