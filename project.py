import sympy as sp
import numpy as np
import matplotlib.pyplot as plt



###############################################################################
Tn = 1e-3
T = sp.symbols("T")
s, z = sp.symbols("s, z")
###############################################################################

###############################################################################
# read the data in
# ddy + a1 dy + a2 y = b2 u
coeff = np.loadtxt("data/sys_coeff.txt")
a1 = coeff[0]
a2 = coeff[1]
b2 = coeff[2]

# construct continous time state space
A = sp.Matrix([
  [0, 1],
  [-a2, -a1]])
B = sp.Matrix([
  [0],
  [b2]])
C = sp.Matrix([
  [1, 0]])

# compute state transistion matrix via diagonalization
M, D = A.diagonalize()
t = sp.symbols("t")
P = M * sp.exp(D * t) * M.inv()

# compute the digital state space
Ad = P.subs({t: Tn})
Bd = sp.integrate(P * B, (t, 0, T), conds="none").subs({T: Tn})
Cd = C
###############################################################################

###############################################################################
def normalize_poly(poly):
  poly = sp.simplify(poly)
  num = sp.Poly(sp.expand(sp.numer(poly)))
  den = sp.Poly(sp.expand(sp.denom(poly)))
  num /= den.coeffs()[0]
  den /= den.coeffs()[0]
  return num / den
###############################################################################

###############################################################################
# find G(z) by state space
Gz1 = normalize_poly((Cd * (z * sp.eye(2) - Ad).inv() * Bd)[0])


# find G(z) by table
# expanstion to
#          1               a          b
# ------------------- = -------- + --------
# (s - p1) * (s - p2)   (s - p1)   (s - p2)
p = np.roots([1, a1, a2])
a = 1.0 / (p[0] - p[1])
b = -a

Gz2 = b2 * (a / -p[0] * (1 - sp.exp(p[0] * T)) / (z - sp.exp(p[0] * T)) +
    b / -p[1] * (1 - sp.exp(p[1] * T)) / (z - sp.exp(p[1] * T)))
Gz2 = normalize_poly(Gz2.subs({T: Tn}))
###############################################################################

###############################################################################
lam, k1, k2 = sp.symbols("lam, k1, k2")

K = sp.Matrix([
  [k1, k2]])
keq = sp.Poly((lam * sp.eye(2) - (Ad - Bd * K)).det(), lam)
pole1 = 0.9
pole2 = 0.93
keq1 = keq.coeffs()[1] / keq.coeffs()[0] - (-pole1 + -pole2)
keq2 = keq.coeffs()[2] / keq.coeffs()[0] - (pole1 * pole2)
ksol = sp.solve([keq1, keq2], [k1, k2])
Kd = sp.Matrix([
  [ksol[k1], ksol[k2]]])

def dbode(A, B, C, T):
  z, w = sp.symbols("z, w")
  G = normalize_poly((C * (z * sp.eye(A.shape[0]) - A).inv() * B)[0])
  sp.pprint(G)
  max_w = (2.0 * np.pi / T) / 2.0
  w = np.logspace(1, np.log10(max_w), 100000)
  G_fun = sp.lambdify(z, G, "numpy")
  Gnum = G_fun(np.exp(1j * w * T))
  Gabs = np.abs(Gnum)
  Gang = np.angle(Gnum)

  _, ax = plt.subplots(nrows=2, ncols=1)
  ax[0].semilogx(w, np.log10(Gabs) * 20.0)
  ax[1].semilogx(w, Gang / np.pi * 180.0)
  plt.show()

dbode(Ad - Bd * Kd, Bd, Cd, Tn)
###############################################################################
