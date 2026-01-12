import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import minimize
from scipy.interpolate import BSpline, PPoly
from sympy import symbols, expand, Poly

# Original GELU function
def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

# Build BSpline
def make_bspline(knots, coeffs, degree):
    t = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
    return BSpline(t, coeffs, degree)

# MSE Loss
def loss(params, x_sample, y_sample, num_free_knots, degree):
    free_knot_pos = np.sort(params[:num_free_knots])
    full_knot_pos = np.concatenate(([-2.5], free_knot_pos, [2.0]))
    coeffs = params[num_free_knots:]
    spline = make_bspline(full_knot_pos, coeffs, degree)
    return np.mean((spline(x_sample) - y_sample) ** 2)

# Configuration
degree = 2
num_total_knots = 6
num_free_knots = num_total_knots - 2  # internal knots
num_coeffs = num_total_knots + degree - 1

x_fit = np.linspace(-2.5, 2.0, 500)
y_fit = gelu(x_fit)

# Initial values
init_knots_free = np.linspace(-1.5, 1.0, num_free_knots)  # initial guess for internal knots
init_coeffs = np.linspace(0, 1, num_coeffs)
init_params = np.concatenate((init_knots_free, init_coeffs))

# Bounds
bounds_knots = [(-2.5, 2.0)] * num_free_knots
bounds_coeffs = [(-2, 2)] * num_coeffs
bounds = bounds_knots + bounds_coeffs

# Optimize
res = minimize(loss, init_params, args=(x_fit, y_fit, num_free_knots, degree),
               bounds=bounds, method='L-BFGS-B', options={'maxiter': 1000})

# Extract optimized knots and coefficients
opt_free_knots = np.sort(res.x[:num_free_knots])
opt_full_knots = np.concatenate(([-2.5], opt_free_knots, [2.0]))
opt_coeffs = res.x[num_free_knots:]
spline = make_bspline(opt_full_knots, opt_coeffs, degree)
ppoly = PPoly.from_spline(spline)

# Convert to unshifted polynomials
x = symbols('x')
segments = []
for i in range(ppoly.c.shape[1]):
    a = ppoly.x[i]
    b = ppoly.x[i + 1]
    if np.isclose(a, b):
        continue
    shifted_coeffs = ppoly.c[:, i][::-1]  # [c0, c1, c2]
    c0, c1, c2 = shifted_coeffs
    expr = c0 + c1*(x - a) + c2*(x - a)**2
    poly = Poly(expand(expr), x)
    coeffs_std = [round(float(c), 8) for c in poly.all_coeffs()[::-1]]
    segments.append((round(a, 6), round(b, 6), coeffs_std))

# Print result
print("\n=== Free-Knot Quadratic Coefficients ===")
print("Format: [c0, c1, c2]  for  c0 + c1*x + c2*x^2")
for a, b, coeffs in segments:
    print(f"[{a}, {b}): {coeffs}")

# Plot
x_vals = np.linspace(-4, 4, 1000)
y_approx = np.zeros_like(x_vals)

for i, x_val in enumerate(x_vals):
    for (a, b, coeffs) in segments:
        if a <= x_val < b:
            y_approx[i] = sum(c * x_val**p for p, c in enumerate(coeffs))
            break
    else:
        y_approx[i] = 0 if x_val < segments[0][0] else x_val

plt.figure(figsize=(8, 5))
plt.plot(x_vals, gelu(x_vals), label="Original GELU", linewidth=2)
plt.plot(x_vals, y_approx, '--', label="Free-Knot Approximation", linewidth=2)
plt.title("GELU vs Free-Knot Quadratic Spline")
plt.xlabel("x")
plt.ylabel("GELU(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("gelu_piecewise_quadratic_plot.png", dpi=300)
plt.close()
