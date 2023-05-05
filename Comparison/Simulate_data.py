##### Creates synthetic trajectories from the van der pol oscillator with both fast and slow sampling


import numpy as np
import matplotlib.pyplot as plt
import pickle

def vdp(y, t,c):
    theta, omega = y
    dydt = [y[1], c*(1-y[0]**2)*y[1]-y[0]]
    return dydt

y0 =  np.array([ 0.84144155, -1.08920043])
ts = np.linspace(0,int(1e4),int(1e5)) 
c = 2
from scipy.integrate import odeint
ys = odeint(vdp, y0, ts,args = (c,),rtol = 1e-5, atol = 1e-5)

# plt.scatter(sol[:,0],sol[:,1],s = .1)
plt.scatter(ys[:,0],ys[:,1])

rate = 40
ys2 = ys[::rate].copy()
ts2 = ts[::rate].copy()

ys = ys[:len(ys2)]
ts = ts[:len(ts2)]
plt.scatter(ys2[:10,0],ys2[:10,1])
plt.figure(figsize = (6,5.5))



    
ts,ys = np.array(ts),np.array(ys)
plt.scatter(ys[:,0],ys[:,1],color = 'k',s = 1)
# #plt.title('Euler-Maruyama Time Trajectory with Diffusion = {}'.format(G['diff']))
with open('samples_fast.p', "wb") as f:
    pickle.dump([ts,ys], f)
    
with open('samples_slow.p', "wb") as f:
    pickle.dump([ts2,ys2], f)