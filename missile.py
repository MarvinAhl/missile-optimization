"""
Author: Marvin Ahlborn

This is a python optimization tool for computing the time optimal trajectory
of a BrahMos-Like Missile using GEKKO Optimization Suite and its APOPT Solver.
"""

# TODO:
# - Add more complex rotational physics with additional translational forces
# - Add drag
# - Maybe add fancy animation (not likely to happen)

import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO

# Constants
g = 9.81  # Gravity
inpm = 1/5  # Inertia per mass: Inertia = inpm * Mass (very simplified physics here)

clear_time = 0.5  # Minimum time for first phase to clear launch site
burn_time = 2.0  # Time of main engine burn
main_force = 500.0  # Force of main engine
control_force = 50.0  # Angular momentum of controlling engines

# Initial values
x0 = 0.0
y0 = 0.0
xv0 = 0.0
yv0 = 14.0  # Initial velocity to reach about 10 meters of height
phi0 = 0.0  # Angle is pointing straight up, positive angles are counted clockwise
phiv0 = 0.0
mass0 = 10.0

# Final mass
massf = 3.0

# Mass reduction per time
dmdt = (mass0 - massf) / burn_time / main_force

# Target values
xt = 500.0
yt = 0.0

# Collocation values
# Number of phases:
# - Straight Up Coast
# - Angular Acceleration Burn
# - Constant Angular Velocity Coast
# - Angular Deceleration Burn
# - Pre Main Engine Coast
# - Main Engine Burn
# - Coast to target
n = 7
p = 25  # Collocation points per phase

m = GEKKO(remote=True)

m.options.NODES = 4  # Two intermediate nodes between collocation points
m.options.SOLVER = 1  # APOPT
m.options.IMODE = 6  # MPC Direct Collocation
m.options.MAX_ITER = 500  # To prevent endless loops
m.options.MV_TYPE = 0  # MVs are constant between endpoints
m.options.DIAGLEVEL = 1  # Show some diagnostics

m.time = np.linspace(0, 1, p)

# Phase timespans
tf = [m.FV(value=1, lb=1e-3) for i in range(n)]
for i in range(n):
    tf[i].STATUS = 1  # make final times controllable

# There is a minimum length for phase 1
tf[0].LOWER = clear_time

# Psyche! Length of main engine burn is fixed by fuel of sollid rocket motor
tf[5].STATUS = 0
tf[5].value = burn_time

# control variables
x = [m.Var(value=x0, fixed_initial=False) for i in range(n)]
y = [m.Var(value=y0, fixed_initial=False) for i in range(n)]
xv = [m.Var(value=xv0, fixed_initial=False) for i in range(n)]
yv = [m.Var(value=yv0, fixed_initial=False) for i in range(n)]
phi = [m.Var(value=phi0, lb=-np.pi/2, ub=np.pi*3/2, fixed_initial=False) for i in range(n)]
phiv = [m.Var(value=phiv0, fixed_initial=False) for i in range(n)]
mass = [m.Var(value=mass0, fixed_initial=False) for i in range(n)]
main = [0, 0, 0, 0, 0, main_force, 0]
control = [0, control_force, 0, -control_force, 0, 0, 0]

# Fix startpoints
m.fix_initial(x[0], val=x0)
m.fix_initial(y[0], val=y0)
m.fix_initial(xv[0], val=xv0)
m.fix_initial(yv[0], val=yv0)
m.fix_initial(phi[0], val=phi0)
m.fix_initial(phiv[0], val=phiv0)
m.fix_initial(mass[0], val=mass0)

# Differential equations describing the system
for i in range(n):
    m.Equation(x[i].dt() == xv[i] * tf[i])
    m.Equation(xv[i].dt() == (main[i]*m.sin(phi[i])/mass[i]) * tf[i])
    m.Equation(y[i].dt() == yv[i] * tf[i])
    m.Equation(yv[i].dt() == (main[i]*m.cos(phi[i])/mass[i] - g) * tf[i])
    m.Equation(phi[i].dt() == phiv[i] * tf[i])
    m.Equation(phiv[i].dt() == (control[i]/(mass[i]*inpm)) * tf[i])
    m.Equation(mass[i].dt() == (-dmdt * main[i]) * tf[i])

# Connect phases at endpoints
for i in range(n-1):
    m.Connection(x[i+1], x[i], 1, 'end', 1, 'end')
    m.Connection(x[i+1],'calculated', pos1=1, node1=1)
    m.Connection(xv[i+1], xv[i], 1, 'end', 1, 'end')
    m.Connection(xv[i+1],'calculated', pos1=1, node1=1)
    m.Connection(y[i+1], y[i], 1, 'end', 1, 'end')
    m.Connection(y[i+1],'calculated', pos1=1, node1=1)
    m.Connection(yv[i+1], yv[i], 1, 'end', 1, 'end')
    m.Connection(yv[i+1],'calculated', pos1=1, node1=1)
    m.Connection(phi[i+1], phi[i], 1, 'end', 1, 'end')
    m.Connection(phi[i+1],'calculated', pos1=1, node1=1)
    m.Connection(phiv[i+1], phiv[i], 1, 'end', 1, 'end')
    m.Connection(phiv[i+1],'calculated', pos1=1, node1=1)
    m.Connection(mass[i+1], mass[i], 1, 'end', 1, 'end')
    m.Connection(mass[i+1],'calculated', pos1=1, node1=1)

# Final x and y values
xf = m.FV(value=xt)
xf.STATUS = 1
yf = m.FV(value=yt)
yf.STATUS = 1

m.Connection(xf, x[-1], pos2='end')
m.Connection(yf, y[-1], pos2='end')

# Angular velocity after angular desceleration burn
phivd = m.FV()
phivd.STATUS = 1

m.Connection(phivd, phiv[3], pos2='end')

# Objective consisting of hitting the target in shortest possible time
# but also reduceing rotation
m.Minimize(1e2*(xf-xt)**2 + 1e2*(yf-yt)**2 + 1e2*phivd**2 + m.sum(tf))

try:
    m.solve(disp=True)  # Wish me luck
except:
    m.open_folder()

# Reshape results
ts = np.zeros(n+1)
ts[1:] = np.cumsum([tf[i].value[0] for i in range(n)])
t = np.reshape([m.time * (end-start) + start for start, end in zip(ts[:-1], ts[1:])], n*p)

print(ts)

xs = np.array([x[i].value for i in range(n)]).reshape(n*p)
ys = np.array([y[i].value for i in range(n)]).reshape(n*p)
xvs = np.array([xv[i].value for i in range(n)]).reshape(n*p)
yvs = np.array([yv[i].value for i in range(n)]).reshape(n*p)
phis = np.array([phi[i].value for i in range(n)]).reshape(n*p)
phivs = np.array([phiv[i].value for i in range(n)]).reshape(n*p)
masss = np.array([mass[i].value for i in range(n)]).reshape(n*p)

print(f"Initial mass: {mass[0].value[0]}\tFinal mass: {mass[-1].value[-1]}")

x_dots = np.array([x[i].value[0] for i in range(n)])
y_dots = np.array([y[i].value[0] for i in range(n)])
xv_dots = np.array([xv[i].value[0] for i in range(n)])
yv_dots = np.array([yv[i].value[0] for i in range(n)])
phi_dots = np.array([phi[i].value[0] for i in range(n)])
phiv_dots = np.array([phiv[i].value[0] for i in range(n)])
mass_dots = np.array([mass[i].value[0] for i in range(n)])
t_dots = ts[:-1]

# Plot results
plt.plot(xs, ys, 'k-')
plt.plot(x_dots, y_dots, 'r.')
plt.axis('equal')
plt.grid(True)
plt.savefig('missile_trajectory.png', dpi=400)

plt.clf()

plt.figure(1)
plt.subplot(3, 2, 1)
plt.plot(t, xs, 'k-')
plt.plot(t_dots, x_dots, 'r.')
plt.grid(True)
plt.ylabel('x-Position')

plt.subplot(3, 2, 2)
plt.plot(t, ys, 'k-')
plt.plot(t_dots, y_dots, 'r.')
plt.grid(True)
plt.ylabel('y-Position')

plt.subplot(3, 2, 3)
plt.plot(t, xvs, 'k-')
plt.plot(t_dots, xv_dots, 'r.')
plt.grid(True)
plt.ylabel('x-Velocity')

plt.subplot(3, 2, 4)
plt.plot(t, yvs, 'k-')
plt.plot(t_dots, yv_dots, 'r.')
plt.grid(True)
plt.ylabel('y-Velocity')

plt.subplot(3, 2, 5)
plt.plot(t, phis, 'k-')
plt.plot(t_dots, phi_dots, 'r.')
plt.grid(True)
plt.ylabel('Angle')

plt.subplot(3, 2, 6)
plt.plot(t, phivs, 'k-')
plt.plot(t_dots, phiv_dots, 'r.')
plt.grid(True)
plt.ylabel('Angular Velocity')

plt.xlabel('Time')
plt.savefig('missile_data.png', dpi=400)