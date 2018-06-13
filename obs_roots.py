import numpy as np
import sympy as sp
wn = 374
zet = 0.925
a = wn * zet
w = np.sqrt(1 - zet**2) * wn
T = 1e-3
phi = np.arctan(-a / w)
z = sp.symbols("z")

Gz = (z - 1) / z / (a**2 + w**2) * (
    z / (z - 1) - 
    (z**2 - z * np.exp(-a * T) / np.cos(phi) * np.cos(w * T - phi)) /
    (z**2 - 2 * z * np.exp(-a * T) * np.cos(w * T) + np.exp(-2 * a * T)))
Gz = sp.simplify(Gz)
coeffs = np.array(sp.Poly(sp.denom(Gz), z).coeffs())
print(np.roots(sp.Poly(sp.denom(Gz), z).coeffs()))
coeffs /= coeffs[0]
np.savetxt("data/observer_coeffs.txt", coeffs)
print("Observer equation is")
print(coeffs)
print("Observer roots are")
print(np.roots(coeffs))
