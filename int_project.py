import time
import sys

t_import1 = time.time()
import sympy as sp
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
t_import2 = time.time()
print("Importing takes %f s" % (t_import2 - t_import1))


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

  gain_cross = w[np.argmin(np.abs(np.abs(G_fun(np.exp(1j * w * T))) - 1.0))]
  phase_cross = w[np.argmin(np.abs(np.remainder(np.angle(G_fun(np.exp(1j
    * w * T))), 2.0 * np.pi) - np.pi))]
  #print("gain_cross = %e" % gain_cross)
  #print("phase_cross = %e" % phase_cross)

  PM = np.abs(np.remainder(np.angle(G_fun(np.exp(1j * gain_cross * T))), 2.0 *
    np.pi) - np.pi) / np.pi * 180.0
  GM = np.abs(np.log10(np.abs(G_fun(np.exp(1j * phase_cross * T)))) * 20)

  return GM, PM

def normalize_poly(poly):
  poly = sp.simplify(poly)
  num = sp.Poly(sp.expand(sp.numer(poly)))
  den = sp.Poly(sp.expand(sp.denom(poly)))
  num /= den.coeffs()[0]
  den /= den.coeffs()[0]
  return num / den

def dsim(A, B, C, N, x0, r_fun, n):
  A = np.array(A).astype(np.float64)
  B = np.array(B).astype(np.float64)
  C = np.array(C).astype(np.float64)
  N = np.array(N).astype(np.float64)

  y = np.zeros(n * 2)
  x = x0
  for k in range(n):
    y[2 * k] = C.dot(x)[0]
    y[2 * k + 1] = y[2 * k]
    x = A.dot(x) + B.dot(N).dot(r_fun(k))
  k = np.repeat(np.arange(n), 2)
  k[1::2] += 1
  return k, y
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

t_c2d1 = time.time()
M, D = diagonalize(A)
Minv = np.linalg.inv(M)
Ad = M.dot(np.diag(np.exp(np.diag(D) * Tn))).dot(Minv)
Bd = M.dot(np.diag((np.exp(np.diag(D) * Tn) - 1.0) / np.diag(D))).dot(Minv).dot(B)
Cd = C

Ki = 1
Aa = sp.Matrix(Ad).col_join(Ki * sp.Matrix(Cd)).row_join(sp.Matrix([
  [0],
  [0],
  [1]]))
Ba = sp.Matrix(Bd).col_join(sp.Matrix([[0]]))
Ca = sp.Matrix(Cd).row_join(sp.Matrix([[0]]))
t_c2d2 = time.time()
print("Converting c2d takes %f" % (t_c2d2 - t_c2d1))
print("A, B, C = ")
sp.pprint(Aa)
sp.pprint(Ba)
sp.pprint(Ca)
###############################################################################

###############################################################################
# find G(z) by state space
Gz1 = normalize_poly((Cd * (z * sp.eye(Ad.shape[0]) - Ad).inv() * Bd)[0])
###############################################################################

###############################################################################
# place the feedback poles
k1, k2, k3 = sp.symbols("k1, k2, k3")

t_kplace1 = time.time()
K = sp.Matrix([
  [k1, k2, k3]])
keq = sp.Poly((z * sp.eye(Aa.shape[0]) - (Aa - Ba * K)).det(), z)
p1 = 0.9
p2 = 0.93
p3 = 0.97
keq1 = keq.coeffs()[1] / keq.coeffs()[0] - (-p3 - p2 - p1)
keq2 = keq.coeffs()[2] / keq.coeffs()[0] - (p2 * p3 + p1 * p3 + p1 * p2)
keq3 = keq.coeffs()[3] / keq.coeffs()[0] - (-p1 * p2 * p3)
ksol = sp.solve([keq1, keq2, keq3], [k1, k2, k3])
Ka = sp.Matrix([
  [ksol[k1], ksol[k2], ksol[k3]]])
sp.pprint(Ka)
t_kplace2 = time.time()
print("Placing K poles takes %f" % (t_kplace2 - t_kplace1))
###############################################################################

###############################################################################
plt.figure(1)
tbode_1 = time.time()
dbode((Ka * (z * sp.eye(Aa.shape[0]) - Aa).inv() * Ba)[0], z, Tn)
GM, PM = dmargin((Ka * (z * sp.eye(Aa.shape[0]) - Aa).inv() * Ba)[0], z, Tn)
print("        GM = %f, PM = %f" % (GM, PM))
tbode_2 = time.time()
print("Bode takes %f s" % (tbode_2 - tbode_1))
###############################################################################

###############################################################################
# compute the feedforward gain
Na = sp.Matrix([
  [0],
  [0],
  [-1]])
###############################################################################

###############################################################################
# places poles of the observer
l1, l2 = sp.symbols("l1, l2")

t_lplace1 = time.time()
L = sp.Matrix([
  [l1],
  [l2]])
leq = sp.Poly(sp.simplify((z * sp.eye(Ad.shape[0]) - (Ad - L * Cd)).det()), z)
l_coeffs = np.loadtxt("data/observer_coeffs.txt")
p1 = np.roots(l_coeffs)[0]
p2 = np.roots(l_coeffs)[1]
leq1 = leq.coeffs()[1] / leq.coeffs()[0] - (-p2 - p1)
leq2 = leq.coeffs()[2] / leq.coeffs()[0] - (p2 * p1)
lsol = sp.solve([leq1, leq2], [l1, l2])
Ld = sp.Matrix([
  [lsol[l1]],
  [lsol[l2]]])
La = Ld.col_join(sp.Matrix([[0]]))
print("La = ")
sp.pprint(La)
t_lplace2 = time.time()
print("Placing L poles takes %f" % (t_lplace2 - t_lplace1))
###############################################################################

###############################################################################
# get the GM and PM for the state observer system
Lz = (Ka * (z * sp.eye(Aa.shape[0]) - (Aa - La * Ca - Ba * Ka)).inv() * La * 
    Cd * (z * sp.eye(Ad.shape[0]) - Ad).inv() * Bd)[0]
Lz = sp.simplify(Lz)

tbode_1 = time.time()
plt.figure(1)
dbode(Lz, z, Tn)
GM, PM = dmargin(Lz, z, Tn)
print("        GM = %f, PM = %f" % (GM, PM))
tbode_2 = time.time()
print("Bode takes %f s" % (tbode_2 - tbode_1))
###############################################################################

###############################################################################
# simulate the system response
t_sim1 = time.time()
x0 = np.array([
  [0],
  [0],
  [0]])
def u_fun(k):
  return 0.5
n = 600
k, y = dsim(Aa - Ba * Ka * 1.1, Na, Ca, 1, x0, u_fun, n)
plt.figure(5)
plt.plot(k, y)

k, y = dsim(Aa - La * Ca, La, Ca, 1, x0, u_fun, n)
plt.figure(6)
plt.plot(k, y)
t_sim2 = time.time()
print("Simulation takes %f" % (t_sim2 - t_sim1))

plt.show()
###############################################################################

