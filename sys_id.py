import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt

########################### SYSTEM IDENTIFICATION #############################
# load the data from the csv file
data = np.loadtxt("data/step_response.txt")
t = data[:, 0]
u = data[:, 1]
th = data[:, 2]

# get the speed from data using central difference
dth = np.diff(th)
w = (th[2:] - th[:-2]) / (t[2:] - t[:-2])
tw = t[1:-1]

# find where the step input begins
t0_idx = np.searchsorted(u, 0.5)
t0 = t[t0_idx]

# shift the data and time to the beginning of the step
w = w[tw > t0]
tw = tw[tw > t0]
tw -= tw[0]
w = w[tw < 1.5] / 180.0 * np.pi # convert to radians
tw = tw[tw < 1.5]

# linearize data and fit a curve to it
y = w
x = tw
y_max = np.mean(y[x > 1.2])
y = y_max - y;
x = x[y > 0.0]
y = y[y > 0.0]
ylog = np.log(y)
x_max = 0.3
p = np.polyfit(x[x < x_max], ylog[x < x_max], 1) # use only

# compute k and tau from the linear fit
k = y_max
tau = 1.0 / np.fabs(p[0])
print("k =   %.5e" % k)
print("tau = %.5e" % tau)
np.savetxt("data/k_tau.txt", np.array([k, tau]).reshape((1, 2)))
###############################################################################


######################### PLOTTING ############################################
# filter data for displaying and comparing
N = 7
a = 1
b = 1.0 / N * np.ones(N)
wf = filtfilt(b, a, w)

# plot and display to compare if the fit makes sense
plt.figure(0)
plt.plot(tw, wf)
plt.plot([x[0], x[-1]], [y_max, y_max])
plt.plot(x, k * (1.0 - np.exp(-x / tau)))
###############################################################################


##################### COMPUTING INVERTED PENDULUM MODEL #######################
# compute the system coefficients
Vs = 10.7
J_inertia = 2.33e-5
K = Vs / k
R = K**2 * tau / J_inertia

l_c = 2.54e-2
m = 4.4e-2
g = 9.81
J_ip = 8.54e-5

# compute the linearized model of the inverted pendulum
a1 = K**2 / R / J_ip
a0 = -m * g * l_c / J_ip
b0 = K * Vs / R / J_ip

# save the system coefficients -> [d2y + a1 * dy + a0 * y = b0 * u]
print("")
print("a1 = %.5e" % a1)
print("a0 = %.5e" % a0)
print("b0 = %.5e" % b0)
np.savetxt("data/sys_coeff.txt", np.array([a1, a0, b0]).reshape((1, 3)))

# find the poles of the system
p = np.roots([1, a1, a0])
print("")
print("Poles of the system")
print("ddy + a1 dy + a0 dy = b0 u")
print("p1 = %.5e, p2 = %.5e" % (p[0], p[1]))
np.savetxt("data/sys_poles.txt", np.array([p[0], p[1]]).reshape((1, 2)))
###############################################################################

plt.show()
