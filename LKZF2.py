# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

mu_Z = 0.5
lambd = 0.4
beta = 0.5
phi = 1.0
C = 0.8
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
mu_T = 0.09

Z0 = 5
F0 = 10
T0 = 3
y0 = [Z0, F0, T0]


prop = 4.58/2.19

t = np.linspace(0, 300, 3000)

def system(y, t, mu_Z, lambd, beta, phi, C, K, h, v, S, alpha_Z, beta_Z, alpha_F, beta_F, h_F, h_Z, mu_T):
    Z, F, T = y
    dZdt = (beta*lambd*F*Z)/(1+(lambd*h*F))  -  (alpha_Z*Z*T)/(1+(h_F*alpha_F*F)+(h_Z*alpha_Z*Z))  -  mu_Z*Z
    dFdt = phi*F * (1- (F)/(K))  +  C*(F*v*S)/(1+(v*S))  -  (lambd*F*Z)/(1+(h*lambd*F))  -  (alpha_F*T*F)/(1+(alpha_F*h_F*F)+(alpha_Z*h_Z*Z))
    dTdt = beta_F * ((alpha_F*F*T) / (1+(alpha_F*h_F*F)+(alpha_Z*h_Z*Z)))  +  beta_Z*((alpha_Z*Z*T)/(1+(alpha_F*h_F*F)+(alpha_Z*h_Z*Z)))  -  mu_T*T
    
    return [dZdt, dFdt, dTdt]

sol = odeint(system, y0, t, args=(mu_Z, lambd, beta, phi, C, K, h, v, S, alpha_Z, beta_Z, alpha_F, beta_F, h_F, h_Z, mu_T))
Z, F, T = sol[:, 0], sol[:, 1], sol[:, 2]

 
dZdt = (beta*lambd*F*Z)/(1+(lambd*h*F))-(alpha_Z*Z*T)/(1+(h_F*alpha_F*F)+(h_Z*alpha_Z))-mu_Z*Z
dFdt = phi*F * (1- (F)/(K)) + C*(F*v*S)/(1+(v*S))-(lambd*F*Z)/(1+(h*lambd*F))-(alpha_Z*T*F)/(1+(alpha_F*h_F*F)+(alpha_Z*h_Z*Z))
dTdt = beta_F * ((alpha_F*F*T) / (1+(alpha_F*h_F*F)+(alpha_Z*h_Z*Z)))  + beta_Z*((alpha_Z*Z*T)/(1+(alpha_F*h_F*F)+(alpha_Z*h_Z*Z))) - mu_T*T
    
#plt.plot(sol[:,0], sol[:, 1], label="Phase Space", color='blue')
#plt.xlabel('Zooplankton(t)', fontsize = '16')
#plt.ylabel('Phitoplankton(t)' , fontsize = '16')
#plt.title("Phase Space")
#plt.grid(True)
#plt.show()

plt.figure(figsize=(10, 10))
plt.plot(t, Z, label="Z(t)", color='red')
plt.plot(t, F, label="F(t)", color='green')
plt.plot(t, T, label="T(t)", color='goldenrod')
plt.xlabel('Time (t)', fontsize = '20')
plt.ylabel('Populations', fontsize = '20')
plt.title('Zooplankton(t), Phytoplankton(t) and Tilápia(t)', fontsize = '20')
plt.legend(fontsize = '20')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(t, dZdt, label="dZ/dt", color='red')
plt.plot(t, dFdt, label="dF/dt", color='orange')
plt.plot(t, dTdt, label="dF/dt", color='darkmagenta')
plt.xlabel('Tempo (t)')
plt.ylabel('Derivadas')
plt.title('dZ/dt e dF/dt')
plt.legend()
plt.grid(True)
plt.show()