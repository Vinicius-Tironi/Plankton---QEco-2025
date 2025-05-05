# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

mu_Z = 0.3
alpha = 0.2
beta = 0.4 
gamma = 0.01 
phi = 0.9
C = 0.5 
K = 20
h = 0.5
v = 0.9 
S = 0.05

I_n = 250 
#I_0 = 0.01 
I_e = 0.01
k_1 = 0.4
k_b = 2


F_star = 1/(alpha*((beta/mu_Z)-h))
F_s = 1/(-alpha*h + (beta*alpha)/(mu_Z))
print("F* =" , F_star)

Z_star = ((1+alpha*h*F_star)/(alpha))*(phi + (C*v*S)/(1+v*S) - (phi*F_star)/(K))
Z_s = (1/alpha + (h*mu_Z)/(alpha*beta-mu_Z*alpha*h))*(phi + (C*v*S)/(1+v*S) - (phi*mu_Z)/(K*(alpha*beta-mu_Z*alpha*h)))
print("Z* = " , Z_star)

e = Z_star - Z_s
print("e = ", e)

Z0 = 1
F0 = 5
y0 = [Z0, F0]

F_t = 20

t = np.linspace(0, 200, 2000)

def system(y, t, mu_Z, alpha, beta, phi, C, K, h, v, S):
    Z, F = y
    dZdt = ((beta * alpha * F * Z)/( 1 + alpha * h * F))- mu_Z * Z
    dFdt = ( (I_n*np.exp(-(k_1*F+k_b)))-(I_e)/(I_n*np.exp(-(k_1*F+k_b))) * ((phi * F)  + C * ((F*v*S)/(1+v*S)))) - ((alpha * F * Z)/(1 + alpha * h * F)) - (gamma * F**2)

    return [dZdt, dFdt]

sol = odeint(system, y0, t, args=(mu_Z, alpha, beta, phi, C, K, h, v, S))
Z, F = sol[:, 0], sol[:, 1]

 
dZdt = ((beta * alpha * F * Z)/( 1 + alpha * h * F))- mu_Z * Z
dFdt = phi * F * (1 - F/ K) + C * ((F*v*S)/(1+v*S)) - ((alpha * F * Z)/(1 + alpha * h * F))

prop = 6.0/3.58
prop

plt.figure(figsize=(10, 10))
plt.plot(t, Z, label="Z(t)", color='red')
plt.plot(t, F, label="F(t)", color='green')
#plt.axhline(y=F_star, color='black', linestyle='--', label='F*')
#plt.axhline(y=Z_star, color='black', linestyle='--', label='Z*')
plt.xlabel('Time (t)', fontsize = '20')
plt.ylabel('Populations', fontsize = '20')
plt.title('Zooplankton(t) and Phytoplankton(t)', fontsize = '20')
plt.legend(fontsize = '20')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(t, dZdt, label="dZ/dt", color='red')
plt.plot(t, dFdt, label="dF/dt", color='orange')
plt.xlabel('Tempo (t)')
plt.ylabel('Derivadas')
plt.title('dZ/dt e dF/dt')
plt.legend()
plt.grid(True)
plt.show()
