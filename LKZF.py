# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

mu_Z = 0.5
alpha = 0.4
beta = 0.5
phi = 1.0
C = 0.2
K = 20
h = 0.02
v = 0.3
S = 0.4

alpha_Z = 0.2
beta_Z = 0.2
alpha_F = 0.005
beta_F = 0.1
h_F = 0.01
h_Z = 0.01
mu_T = 0.07

F_star = (mu_Z)/((alpha)*(beta-mu_Z*h))
Z_star = ((1)/(alpha)+(mu_Z*h)/(alpha*(beta-mu_Z*h)))*((C*v*S)/(1+v*S)+phi-((phi)/(K))*(mu_Z)/(alpha*(beta-mu_Z*h)))

prop = F_star/Z_star

Z0 = 5
F0 = 10
y0 = [Z0, F0]

t = np.linspace(0, 120, 1200)
dt = t[1] - t[0]

def system(y, t, mu_Z, alpha, beta, phi, C, K, h, v, S):
    Z, F = y
    dZdt = ((beta * alpha * F * Z)/( 1 + alpha * h * F))- mu_Z * Z
    dFdt = phi * F * (1 - F/ K) + C * ((F*v*S)/(1+v*S)) - ((alpha * F * Z)/(1 + alpha * h * F))
    return [dZdt, dFdt]


sol = odeint(system, y0, t, args=(mu_Z, alpha, beta, phi, C, K, h, v, S))
Z, F = sol[:, 0], sol[:, 1]
 
dZdt = ((beta * alpha * F * Z)/( 1 + alpha * h * F))- mu_Z * Z
dFdt = phi * F * (1 - F/ K) + C * ((F*v*S)/(1+v*S)) - ((alpha * F * Z)/(1 + alpha * h * F))

#plt.plot(sol[:,0], sol[:, 1], label="Phase Space", color='blue')
#plt.xlabel('Zooplankton(t)', fontsize = '16')
#plt.ylabel('Phitoplankton(t)' , fontsize = '16')
#plt.title("Phase Space")
#plt.grid(True)
#plt.show()

plt.figure(figsize=(10, 10))
plt.plot(t, Z, label="Z(t)", color='red')
plt.plot(t, F, label="F(t)", color='green')
plt.axhline(y=F_star, color='black', linestyle='--')
plt.axhline(y=Z_star, color='black', linestyle='--')
plt.xlabel('Time (t)', fontsize = '20')
plt.ylabel('Populations', fontsize = '20')
plt.title('Zooplankton(t) e Phytoplankton(t)', fontsize = '20')
plt.legend(fontsize = '20')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(t, dZdt, label="dZ/dt", color='red')
plt.plot(t, dFdt, label="dF/dt", color='orange')
plt.xlabel('Time (t)')
plt.ylabel('Derivadas')
plt.title('dZ/dt e dF/dt')
plt.legend()
plt.grid(True)
plt.show()
