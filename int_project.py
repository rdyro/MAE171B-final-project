import time
import sys

t_import1 = time.time()
import sympy as sp
import numpy as np
from scipy.optimize import fsolve
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
t_import2 = time.time()
print("Importing takes %f s" % (t_import2 - t_import1))

plt.rcParams["figure.figsize"] = (15, 9)
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["legend.fontsize"] = 20

###############################################################################
def dbode(G, z, T):
  max_w = (2.0 * np.pi / T) / 2.0
  w = np.logspace(-2, np.log10(max_w), 100000)
  G_fun = sp.lambdify(z, G, "numpy")

  Gnum = G_fun(np.exp(1j * w * T))
  Gabs = np.abs(Gnum)
  Gang = np.angle(Gnum)

  f, axarr = plt.subplots(2, sharex=True)
  axarr[0].semilogx(w, np.log10(Gabs) * 20.0)
  axarr[0].set_ylabel("Gain (dB)")
  axarr[1].semilogx(w, np.remainder(Gang / np.pi * 180.0, 360.0))
  axarr[1].set_ylabel("Angle (deg)")
  axarr[1].set_xlabel("Frequency, $\\omega$ (rad/s)")

def dmargin(G, z, T):
  max_w = (2.0 * np.pi / T) / 2.0
  w = np.logspace(-2, np.log10(max_w), 100000)
  G_fun = sp.lambdify(z, G, "numpy")

  gain_cross = w[np.argmin(np.abs(np.abs(G_fun(np.exp(1j * w * T))) - 1.0))]
  phase_cross = w[np.argmin(np.abs(np.remainder(np.angle(G_fun(np.exp(1j
    * w * T))), 2.0 * np.pi) - np.pi))]
  print("gain_cross = %e" % gain_cross)
  print("phase_cross = %e" % phase_cross)

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

def dsim(A, B, K, C, N, x0, r_fun, n):
  A = np.array(A).astype(np.float64)
  B = np.array(B).astype(np.float64)
  K = np.array(K).astype(np.float64)
  C = np.array(C).astype(np.float64)
  N = np.array(N).astype(np.float64)

  y = np.zeros(n * 2)
  u = np.zeros(n * 2)
  x = x0
  for k in range(n):
    y[2 * k] = C.dot(x)[0]
    y[2 * k + 1] = y[2 * k]
    u[2 * k] = K.dot(x)
    u[2 * k + 1] = K.dot(x)
    x = (A - B.dot(K)).dot(x) + N.dot(r_fun(k))
  k = np.repeat(np.arange(n), 2)
  k[1::2] += 1
  return k, y, u
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
tbode_1 = time.time()

Lz = (Ka * (z * sp.eye(Aa.shape[0]) - Aa).inv() * Ba)[0]
Lz = normalize_poly(Lz)
Sz = normalize_poly(1 / (1 + Lz))
max_w = (2.0 * np.pi / Tn) / 2.0
w = np.logspace(-2, np.log10(max_w), 100000)
Sz_fun = sp.lambdify(z, Sz, "numpy")
Sznum = Sz_fun(np.exp(1j * w * Tn))
Szabs = np.abs(Sznum)
print("Max Sz = %.5e" % np.max(Szabs))
VGM = np.max(Szabs) / (np.max(Szabs) - 1.0)
print("VGM = %.5e" % VGM)

dbode(Lz, z, Tn)
plt.savefig("graph/pid_feedback_Lz.png")
dbode(Sz, z, Tn)
plt.savefig("graph/pid_feedback_Sz.png")

GM, PM = dmargin(Lz, z, Tn)
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
Lz = normalize_poly(Lz)
Sz = normalize_poly(1 / (1 + Lz))
max_w = (2.0 * np.pi / Tn) / 2.0
w = np.logspace(-2, np.log10(max_w), 100000)
Sz_fun = sp.lambdify(z, Sz, "numpy")
Sznum = Sz_fun(np.exp(1j * w * Tn))
Szabs = np.abs(Sznum)
print("Max Sz = %.5e" % np.max(Szabs))
VGM = np.max(Szabs) / (np.max(Szabs) - 1.0)
print("VGM = %.5e" % VGM)

tbode_1 = time.time()
dbode(Lz, z, Tn)
plt.savefig("graph/pid_observer_Lz.png")
dbode(Sz, z, Tn)
plt.savefig("graph/pid_observer_Sz.png")
GM, PM = dmargin(Lz, z, Tn)
print("        GM = %f, PM = %f" % (GM, PM))
tbode_2 = time.time()
print("Bode takes %f s" % (tbode_2 - tbode_1))
###############################################################################

###############################################################################
Aso = (Aa - Ba * Ka).row_join(-Ba * Ka * sp.Matrix([
  [1, 0],
  [0, 1],
  [0, 0]])).col_join(sp.zeros(2, 3).row_join(Ad - Ld * Cd))
Bso = Na.col_join(sp.zeros(2, 1))
Cso = Ca.row_join(sp.zeros(1, 2))
###############################################################################

###############################################################################
# simulate the system response
x0 = np.array([
  [0],
  [0],
  [0]])
n = 10000 - 10

data = np.loadtxt("data/PID_control.txt")
k_exp = (data[:, 0] / 1e-3).astype(np.int)
r_exp = data[:, 1]
y_exp = data[:, 2]
u_exp = data[:, 3]

def r_fun(k):
  return r_exp[k]

N = 100
b = 1.0 / N * np.ones(N)
a = 1

yf_exp = filtfilt(b, a, y_exp)

k, y, u = dsim(Aa, Ba, Ka, Ca, Na, x0, r_fun, n)
plt.figure(10)
plt.plot(k_exp, r_exp, label="Reference")
plt.plot(k_exp, y_exp, label="Output Raw")
plt.plot(k_exp, yf_exp, label="Ouput Filtered")
plt.plot(k, y, label="Simulated Output")
plt.xlabel("Time, k ($\\times 10^{-3}$ s)")
plt.ylabel("Angular Position (rad)")
plt.legend()
plt.savefig("graph/pid_y.png")

N = 30
b = 1.0 / N * np.ones(N)
a = 1

uf_exp = filtfilt(b, a, u_exp)

plt.figure(11)
plt.plot(k_exp, u_exp, label="Input raw")
plt.plot(k_exp, uf_exp, label="Input filtered")
plt.plot(k, u, label="Simulated Input")
plt.xlabel("Time, k ($\\times 10^{-3}$ s)")
plt.ylabel("Duty Cycle (1)")
plt.legend()
plt.savefig("graph/pid_u.png")

plt.figure(12)
plt.plot(k, u)

plt.figure(13)
plt.plot(k, y)

plt.show()
###############################################################################

