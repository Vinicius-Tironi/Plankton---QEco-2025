import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

mu_Z = 0.2 # retificar após a introdução da tilápia
lambd = 0.2
beta = 0.3
phi = 0.5
C = 0.7
h = 0.31
v = 0.9
S = 0.05

I_n = 250
I_e = 0.01
k_1 = 0.4
k_b = 2

#Tilapia
alpha_Z = 0.09
beta_Z = 0.2
alpha_F = 0.005
beta_F = 0.1
h_F = 0.01
h_Z = 0.01
mu_T = 0.09

def system(y, t, mu_Z, lambd, beta, phi, C, h, v, S, alpha_Z, beta_Z, alpha_F, beta_F, h_F, h_Z, mu_T, I_n, I_e, k_1, k_b):
    epsilon = 1e-10
    Z, F, T = y
    dZdt = (beta * lambd * F * Z) / (1 + (lambd * h * F) + epsilon) - (alpha_Z * Z * T) / (1 + (alpha_F * h_F * F) + (alpha_Z * h_Z * Z) + epsilon) - (mu_Z * Z)
    dFdt = ( ((I_n*np.exp(-(k_1*F+k_b))-(I_e))/(I_e)) * ((phi * F)  + C * ((F*v*S)/(1+v*S) + epsilon)))  - (lambd * F * Z) / (1 + (lambd * h * F) + epsilon) - (alpha_F * F * T) / (1 + (alpha_F * h_F * F) + (alpha_Z * h_Z * Z) + epsilon)
    dTdt = (beta_F * alpha_F * F * T) / (1 + (alpha_F * h_F * F) + (alpha_Z * h_Z * Z) + epsilon) + (beta_Z * alpha_Z * Z * T) / (1 + (alpha_F * h_F * F) + (alpha_Z * h_Z * Z) + epsilon) - (mu_T * T)
    return [dZdt, dFdt, dTdt]

Z0 = 10.0
F0 = 10.0
T0 = 2.0
y0 = [Z0, F0, T0]

t = np.linspace(0, 2000, 2000)

sol = odeint(system, y0, t, args=(mu_Z, lambd, beta, phi, C, h, v, S, alpha_Z, beta_Z, alpha_F, beta_F, h_F, h_Z, mu_T, I_n, I_e, k_1, k_b))
Z, F, T = sol[:, 0], sol[:, 1], sol[:, 2]

plt.figure(figsize=(10, 10))
plt.plot(t, Z, label="Z(t) - Zooplankton", color='red')
plt.plot(t, F, label="F(t) - Phytoplankton", color='green')
plt.plot(t, T, label="T(t) - Tilápia", color='goldenrod')
plt.xlabel('Tempo (t)', fontsize=14)
plt.ylabel('Populações', fontsize=14)
plt.title('Zooplankton, Phytoplankton and Tilapia', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()