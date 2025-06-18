import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
r = 0.5
N = 1.0
N_H = 0.3
H = 0.4
I_0 = 1.0
k1 = 0.1
k2 = 0.1
kb = 0.1
C = 1.0
g = 1.0

a = 0.2
h = 0.1      # handling time of Z on F
a_F = 0.2    # attack rate of T on F
h_F = 0.1    # handling time of T on F
a_Z = 0.15   # attack rate of T on Z
h_Z = 0.1    # handling time of T on Z

e = 1.      # conversion efficiency of Z
e_F = 0.2    # conversion efficiency of T from F
e_Z = 0.2    # conversion efficiency of T from Z

m_Z = 0.2    # mortality of Z
m_T = 0.1    # mortality of T


# --- System of ODEs ---
def system(t, y):
    F, Z, T = y
    I_val = I_0 * np.exp(-(k1 * F) + k2 * C + kb)
    N_bar = (N / (N + N_H))

    dFdt = (
        r * F * N_bar * (I_val / (H + I_val))
        - (a * F * Z) / (1 + a * h * F)
        - (a_F * F * T) / (1 + a_F * h_F * F + a_Z * h_Z * Z)
    )

    dZdt = (
        (e * a * F * Z) / (1 + a * h * F)
        - (a_Z * Z * T) / (1 + a_F * h_F * F + a_Z * h_Z * Z)
        - m_Z * Z
    )

    dTdt = (
        (e_F * a_F * F * T) / (1 + a_F * h_F * F + a_Z * h_Z * Z)
        + (e_Z * a_Z * Z * T) / (1 + a_F * h_F * F + a_Z * h_Z * Z)
        - m_T * T
    )

    return np.array([dFdt, dZdt, dTdt])

# --- Runge-Kutta 4th order ---
def runge_kutta(f, y0, t0, tf, h):
    t_values = [t0]
    y_values = [np.array(y0)]
    t = t0
    y = np.array(y0)

    while t < tf:
        if t + h > tf:
            h = tf - t
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)

        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h

        t_values.append(t)
        y_values.append(y.copy())

    return np.array(t_values), np.array(y_values)


N_bar = (N / (N + N_H))
F_star = m_Z / (a * (e - m_Z * h))
          
# --- Simulation settings ---
t0 = 0
tf = 1000
h = 0.1
t_vals = np.arange(t0, tf + h, h)

F_star = m_Z / (a * (e - m_Z * h))


# --- Baseline plot ---
y0_baseline = [1.0, 0.5, 0.2]
t_vals, y_vals = runge_kutta(system, y0_baseline, t0, tf, h)

plt.figure(figsize=(10, 6))
plt.plot(t_vals, y_vals[:, 0], label='F(t)', linewidth=2)
plt.plot(t_vals, y_vals[:, 1], label='Z(t)', linewidth=2)
plt.plot(t_vals, y_vals[:, 2], label='T(t)', linewidth=2)

plt.title('Baseline Simulation', fontsize = 14)
plt.xlabel('Time', fontsize = 14)
plt.ylabel('Population', fontsize = 14)
plt.legend(fontsize = 14)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Varying initial conditions ---
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Vary F(0), fix Z(0), T(0)
F0_range = [0.2, 0.5, 1.0, 2.0]
Z0_fixed = 0.5
T0_fixed = 0.2

for F0 in F0_range:
    _, yv = runge_kutta(system, [F0, Z0_fixed, T0_fixed], t0, tf, h)
    axs[0].plot(t_vals, yv[:, 0], label=f'F(0) = {F0:.1f}')
axs[0].set_title('Varying F(0)')
axs[0].set_ylabel('F(t)')
axs[0].legend()
axs[0].grid(True)

# Vary Z(0), fix F(0), T(0)
Z0_range = [0.1, 0.5, 1.0, 2.0]
F0_fixed = 1.0
T0_fixed = 0.2
for Z0 in Z0_range:
    _, yv = runge_kutta(system, [F0_fixed, Z0, T0_fixed], t0, tf, h)
    axs[1].plot(t_vals, yv[:, 1], label=f'Z(0) = {Z0:.1f}')
axs[1].set_title('Varying Z(0)')
axs[1].set_ylabel('Z(t)')
axs[1].legend()
axs[1].grid(True)

# Vary T(0), fix F(0), Z(0)
T0_range = [0.2, 0.5, 1.0, 2.0]
F0_fixed = 1.0
Z0_fixed = 0.5
for T0 in T0_range:
    _, yv = runge_kutta(system, [F0_fixed, Z0_fixed, T0], t0, tf, h)
    axs[2].plot(t_vals, yv[:, 2], label=f'T(0) = {T0:.1f}')
axs[2].set_title('Varying T(0)')
axs[2].set_ylabel('T(t)')
axs[2].set_xlabel('Time')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
