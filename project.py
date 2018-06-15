import sympy as sp
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time
import sys


###############################################################################
def dbode(G, z, T):
  max_w = (2.0 * np.pi / T) / 2.0
  w = np.logspace(-2, np.log10(max_w), 100000)
  G_fun = sp.lambdify(z, G, "numpy")

  Gnum = G_fun(np.exp(1j * w * T))
  Gabs = np.abs(Gnum)
  Gang = np.angle(Gnum)

  fig = plt.gcf()
  ax1 = fig.add_subplot(211)
  ax2 = fig.add_subplot(212)
  ax1.semilogx(w, np.log10(Gabs) * 20.0)
  ax2.semilogx(w, np.remainder(Gang / np.pi * 180.0, 360.0))

def dmargin(G, z, T):
  max_w = (2.0 * np.pi / T) / 2.0
  w = np.logspace(-2, np.log10(max_w), 100000)
  G_fun = sp.lambdify(z, G, "numpy")

  #gain_cross = fsolve(lambda x: np.abs(np.abs(G_fun(np.exp(1j * x * T)) -
  #  1.0)), max_w / 100.0)
  #phase_cross = fsolve(lambda x: np.abs(np.remainder(np.angle(G_fun(np.exp(1j *
  #  x * T))), 2.0 * np.pi) - np.pi), max_w / 100.0)
  gain_cross = w[np.argmin(np.abs(np.abs(G_fun(np.exp(1j * w * T))) - 1.0))]
  phase_cross = w[np.argmin(np.abs(np.remainder(np.angle(G_fun(np.exp(1j
    * w * T))), 2.0 * np.pi) - np.pi))]
  print("gain_cross = %e" % gain_cross)
  print("phase_cross = %e" % phase_cross)

  # if np.abs(np.abs(G_fun(np.exp(1j * gain_cross * T)) - 1.0)) > 1e-5:
  #   PM = float("infinity")
  # else:
  PM = np.abs(np.remainder(np.angle(G_fun(np.exp(1j * gain_cross * T))), 2.0 *
    np.pi) - np.pi) / np.pi * 180.0

  # if np.abs(np.remainder(np.angle(G_fun(np.exp(1j * phase_cross * T))), 2.0 * np.pi) - np.pi) > 1e-5:
  #   GM = float("infinity")
  # else:
  GM = np.abs(np.log10(np.abs(G_fun(np.exp(1j * phase_cross * T)))) * 20)

  print(GM)
  print(PM)
  return GM, PM

def normalize_poly(poly):
  poly = sp.simplify(poly)
  num = sp.Poly(sp.expand(sp.numer(poly)))
  den = sp.Poly(sp.expand(sp.denom(poly)))
  num /= den.coeffs()[0]
  den /= den.coeffs()[0]
  return num / den

def dsim(Ad, Bd, Cd, Nd, x0, u_fun, n):
  Ad = np.array(Ad).astype(np.float64)
  Bd = np.array(Bd).astype(np.float64)
  Cd = np.array(Cd).astype(np.float64)
  Nd = np.array(Nd).astype(np.float64)

  path = np.zeros(n)
  x = x0
  for k in range(n):
    path[k] = Cd.dot(x)[0]
    x = Ad.dot(x) + Bd.dot(Nd).dot(u_fun(k))
  return path
###############################################################################


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
A = np.array([
  [0, 1],
  [-a2, -a1]])
B = np.array([
  [0],
  [b2]])
C = np.array([
  [1, 0]])

# compute state transistion matrix via diagonalization
def diagonalize(A):
  val, vec = np.linalg.eig(A)
  D = np.diag(val)
  M = vec
  return M, D

M, D = diagonalize(A)
Minv = np.linalg.inv(M)
Ad = M.dot(np.diag(np.exp(np.diag(D) * Tn))).dot(Minv)
Bd = M.dot(np.diag((np.exp(np.diag(D) * Tn) - 1.0) / np.diag(D))).dot(Minv).dot(B)
Cd = C

Ad = sp.Matrix(Ad)
Bd = sp.Matrix(Bd)
Cd = sp.Matrix(Cd)

sp.pprint(Ad)
sp.pprint(Bd)
###############################################################################

###############################################################################
# find G(z) by state space
Gz1 = normalize_poly((Cd * (z * sp.eye(2) - Ad).inv() * Bd)[0])
sp.pprint(Gz1)


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
sp.pprint(Gz2)
###############################################################################

###############################################################################
# place the feedback poles
k1, k2 = sp.symbols("k1, k2")

K = sp.Matrix([
  [k1, k2]])
keq = sp.Poly((z * sp.eye(Ad.shape[0]) - (Ad - Bd * K)).det(), z)
pole1 = 0.9
pole2 = 0.93
keq1 = keq.coeffs()[1] / keq.coeffs()[0] - (-pole1 + -pole2)
keq2 = keq.coeffs()[2] / keq.coeffs()[0] - (pole1 * pole2)
ksol = sp.solve([keq1, keq2], [k1, k2])
Kd = sp.Matrix([
  [ksol[k1], ksol[k2]]])
print("Kd = ")
sp.pprint(Kd)

plt.figure(1)
dbode((Kd * (z * sp.eye(Ad.shape[0]) - Ad).inv() * Bd)[0], z, Tn)
dmargin((Kd * (z * sp.eye(Ad.shape[0]) - Ad).inv() * Bd)[0], z, Tn)
#plt.show()
###############################################################################

###############################################################################
# compute the feedforward gain
Nd = 1.0 / (Cd * (sp.eye(Ad.shape[0]) - (Ad - Bd * Kd)).inv() * Bd)[0]
print("Nd = %.5e" % Nd)
###############################################################################

###############################################################################
# places poles of the observer
zet = 0.925
wn = 374
l1, l2 = sp.symbols("l1, l2")
L = sp.Matrix([
  [l1],
  [l2]])
leq = sp.Poly(sp.simplify((z * sp.eye(Ad.shape[0]) - (Ad - L * Cd)).det()), z)
l_coeffs = np.loadtxt("data/observer_coeffs.txt")
print(l_coeffs)
leq1 = leq.coeffs()[1] / leq.coeffs()[0] - (l_coeffs[1])
leq2 = leq.coeffs()[2] / leq.coeffs()[0] - (l_coeffs[2])
lsol = sp.solve([leq1, leq2], [l1, l2])
sp.pprint(lsol)
Ld = sp.Matrix([
  [lsol[l1]],
  [lsol[l2]]])
print("Ld = ")
sp.pprint(Ld)
###############################################################################

###############################################################################
# get the GM and PM for the state observer system
Lz = (Kd * (z * sp.eye(Ad.shape[0]) - (Ad - Ld * Cd - Bd * Kd)).inv() * Ld * Cd
    * (z * sp.eye(Ad.shape[0]) - Ad).inv() * Bd)[0]
Lz = sp.simplify(Lz)
plt.figure(1)
dbode(Lz, z, Tn)
dmargin(Lz, z, Tn)
###############################################################################

###############################################################################
# simulate the system response
x0 = np.array([
  [0],
  [0]])
def u_fun(k):
  return 0.5
n = 300
path = dsim(Ad - Bd * Kd, Bd, Cd, Nd, x0, u_fun, n)
plt.figure(5)
plt.plot(range(n), path)

path = dsim(Ad - Ld * Cd, Ld, Cd, 1, x0, u_fun, n)
plt.figure(6)
plt.plot(range(n), path)

plt.show()
###############################################################################

