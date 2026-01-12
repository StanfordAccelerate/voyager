# Here is the code to do fitting for NLAs and generate coefficients for each range
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import gelu
from numpy.polynomial.polynomial import Polynomial

# Define fitting ranges: look up the plot and determine the ranges for fitting manually
ranges = [
    (-2.5, -0.5),
    (-0.5, 0.0),
    (0.0, 1.0),
    (1.0, 2.0)
]

# Generate dense x-axis for plotting
x_plot = torch.linspace(-5, 5, 1000)
y_true = gelu(x_plot)

# Storage for coefficients and approximated y values
coeffs = []
y_approx = torch.zeros_like(x_plot)

# Fit and apply polynomials for each range
for low, high in ranges:
    x = torch.linspace(low, high, 200)
    y = gelu(x)
    
    p = Polynomial.fit(x.numpy(), y.numpy(), deg=3, domain=[low, high])
    coefs = p.convert().coef
    while len(coefs) < 4:
        coefs = np.append(coefs, 0.0)
    coeffs.append(coefs[:4])
    
    mask = (x_plot >= low) & (x_plot < high)
    y_approx[mask] = (
        coefs[0] +
        coefs[1] * x_plot[mask] +
        coefs[2] * x_plot[mask]**2 +
        coefs[3] * x_plot[mask]**3
    )

# Apply boundary conditions
mask_lo = x_plot < -2.5
mask_hi = x_plot >= 2.0
y_approx[mask_lo] = 0.0
y_approx[mask_hi] = x_plot[mask_hi]

# Plot comparison and save
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_true, label="GELU (PyTorch)", linewidth=2)
plt.plot(x_plot, y_approx, '--', label="Fitted Approximation", linewidth=2)
plt.legend()
plt.grid(True)
plt.title("GELU Approximation vs Original (PyTorch)")
plt.xlabel("x")
plt.ylabel("GELU(x)")
plt.tight_layout()
plt.savefig("gelu_approx_vs_original.png", dpi=300)

# Print coefficients
for i, (low, high) in enumerate(ranges):
    c0, c1, c2, c3 = coeffs[i]
    print(f"For x in [{low}, {high}]: GELU(x) ≈ {c0:.8f} + {c1:.8f} * x + {c2:.8f} * x**2 + {c3:.8f} * x**3")
