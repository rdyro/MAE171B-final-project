import sympy as sp
import numpy as np
from scipy.optimize import fsolve
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import time
import sys

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

###############################################################################
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
    u[2 * k] = K.dot(x) + N.dot(r_fun(k))
    u[2 * k + 1] = K.dot(x) + N.dot(r_fun(k))
    x = (A - B.dot(K)).dot(x) + B.dot(N).dot(r_fun(k))
  k = np.repeat(np.arange(n), 2)
  k[1::2] += 1
  return k, y, u
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

print("A = ")
print(A)
print("B = ")
print(B)
print("C = ")
print(C)

# compute state transistion matrix via diagonalization
def diagonalize(A):
  val, vec = np.linalg.eig(A)
  D = np.diag(val)
  M = vec
  return M, D

M, D = diagonalize(A)
Minv = np.linalg.inv(M)
print("M = ")
print(M)
print("D = ")
print(D)
print("M^-1 = ")
print(Minv)

Ad = M.dot(np.diag(np.exp(np.diag(D) * Tn))).dot(Minv)
Bd = M.dot(np.diag((np.exp(np.diag(D) * Tn) - 1.0) / np.diag(D))).dot(Minv).dot(B)
Cd = C

Ad = sp.Matrix(Ad)
Bd = sp.Matrix(Bd)
Cd = sp.Matrix(Cd)

print("Ad = ")
sp.pprint(Ad)
print("Bd = ")
sp.pprint(Bd)
print("Cd = ")
sp.pprint(Cd)
###############################################################################

###############################################################################
# find G(z) by state space
Gz1 = normalize_poly((Cd * (z * sp.eye(2) - Ad).inv() * Bd)[0])
print("Gz = ")
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
#sp.pprint(Gz2)
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

Lz = normalize_poly((Kd * (z * sp.eye(Ad.shape[0]) - Ad).inv() * Bd)[0])

Sz = normalize_poly(1 / (1 + Lz))
max_w = (2.0 * np.pi / Tn) / 2.0
w = np.logspace(-2, np.log10(max_w), 100000)
Sz_fun = sp.lambdify(z, Sz, "numpy")
Sznum = Sz_fun(np.exp(1j * w * Tn))
Szabs = np.abs(Sznum)
print("Max Sz = %.5e" % np.max(Szabs))
VGM = np.max(Szabs) / (np.max(Szabs) - 1.0)
print("VGM = %.5e" % VGM)

print("Lz = ")
sp.pprint(Lz)

dbode(Lz, z, Tn)
dmargin(Lz, z, Tn)
plt.savefig("graph/pd_feedback_Lz.png")

dbode(Sz, z, Tn)
dmargin(Sz, z, Tn)
plt.savefig("graph/pd_feedback_Sz.png")
###############################################################################

###############################################################################
# compute the feedforward gain
Nd = 1.0 / (Cd * (sp.eye(Ad.shape[0]) - (Ad - Bd * Kd)).inv() * Bd)[0]
Gcl = normalize_poly((Cd * (z * sp.eye(Ad.shape[0]) - (Ad - Bd * Kd)).inv() * Bd)[0])
print("Gcl = ")
sp.pprint(Gcl)
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
Lz = normalize_poly(Lz)
Sz = normalize_poly(1 / (1 + Lz))

Sz = normalize_poly(1 / (1 + Lz))
max_w = (2.0 * np.pi / Tn) / 2.0
w = np.logspace(-2, np.log10(max_w), 100000)
Sz_fun = sp.lambdify(z, Sz, "numpy")
Sznum = Sz_fun(np.exp(1j * w * Tn))
Szabs = np.abs(Sznum)
print("Max Sz = %.5e" % np.max(Szabs))
VGM = np.max(Szabs) / (np.max(Szabs) - 1.0)
print("VGM = %.5e" % VGM)

print("Lz = ")
sp.pprint(Lz)

dbode(Lz, z, Tn)
dmargin(Lz, z, Tn)
plt.savefig("graph/pd_observer_Lz.png")

dbode(Sz, z, Tn)
plt.savefig("graph/pd_observer_Sz.png")
###############################################################################

###############################################################################
# simulate the system response
x0 = np.array([
  [0],
  [0]])
n = 10000 - 10

data = np.loadtxt("data/PD_control_867.txt")
k_exp = (data[:, 0] / 1e-3).astype(np.int)
r_exp = data[:, 1] / 8.67
y_exp = data[:, 2]
u_exp = data[:, 3]

def r_fun(k):
  return r_exp[k]


N = 100
b = 1.0 / N * np.ones(N)
a = 1

yf_exp = filtfilt(b, a, y_exp)

k, y, u = dsim(Ad, Bd, Kd, Cd, Nd, x0, r_fun, n)
plt.figure(10)
plt.plot(k_exp, r_exp, label="Reference")
plt.plot(k_exp, y_exp, label="Output Raw")
plt.plot(k_exp, yf_exp, label="Ouput Filtered")
plt.plot(k, y, label="Simulated Output")
plt.xlabel("Time, k ($\\times 10^{-3}$ s)")
plt.ylabel("Angular Position (rad)")
plt.legend()
plt.savefig("graph/pd_y.png")

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
plt.savefig("graph/pd_u.png")

def r2_fun(k):
  return 0.025
k, y, u = dsim(Ad, Bd, Kd, Cd, Nd, x0, r2_fun, n)

plt.figure(12)
plt.plot(k, u)

plt.figure(13)
plt.plot(k, y)

plt.show()
###############################################################################

