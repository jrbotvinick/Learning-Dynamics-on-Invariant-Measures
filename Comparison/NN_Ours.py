

def traj_ours(TIME,name,seed):#input the desired wall clock training time and the name of the trajectory file to learn
    
    
    ###########################################################
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    from time import time
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    import copy
    from scipy.sparse import lil_matrix, csr_matrix, identity,spdiags, linalg
    import scipy.sparse as sparse
    from scipy.sparse.linalg import LinearOperator,eigs,spsolve,norm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.interpolate import RegularGridInterpolator
    from scipy.interpolate import RBFInterpolator
    import matplotlib.animation as animation
    from time import time
    from numpy.random import default_rng
    import math 
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    from torch.distributions import normal
    import torch.optim as optim
    import matplotlib.pylab as plt
    from IPython import get_ipython
    from scipy.ndimage import gaussian_filter
    import pickle
    import sys
    import torch.optim as optim
    import os
    import json
    from datetime import datetime
    from scipy.optimize import minimize, LinearConstraint
    ##########################################################


    #Load in binned trajectory data
    with open(name, "rb") as f:
        data = pickle.load(f)
        
    ys = data[1]
    G = {
    "dt": .01, #timestep
    "dx": .1, #spatial discretization
    'mu': .5,
    'gamma': -1,
    'lambda': -.1,
    'bounds': [[-4,4],[-5,5]], #bounds used in discretization
    "alpha": 1e-8, #teleportation parameter
    "diff": 1e-3, #diffusion parameter
    'Cost': 'KL',#cost function, can be L2,W2,KL
    'param': 'NN',
    'grad_scale': 1e-6,
    'opt_method': 'L-BFGS-B', #type of optimization used by scipy.optimize
    'nodes': 100,
    'act': 'tanh',
    'lr': 1e-3,
    'invert_V1': True, #Decide whether to invert V1
    'invert_V2': True, #Decide whether to invert V2
    'inverse_crime': False,
    'rescale': 1, #factor by which to rescale parameters after inversion
    'initialguess': [1e-2,'constant'], #initial guess for velocities, string can be 'constant', 'noise', 'gauss', or 'shifted'
    'TSMax': 1e2, #number of steps for plotting dynamics
    'interp': 'linear', #how to interpolate reconstructed parameter for plotting dynamics
    'filtering': 2, #standard deviation of gaussian kernel for filtering
    'maxfuniters': 100, #max number of function evals in each iteration
    'numiter': 100000000, #total number of iterations
    'plotevery': 1000000, #how often to plot updates (no plots saved if > numiters)
    'plot': False,
    'regularize': 0,
    'Witer': 150,
    'Wsize': 64,
    'tol': .0001,
    'NNweight' : 1,
    'end': False,
    }
    ########################################################################
    
    
    bounds = G['bounds']
    
    Xi=[i for i in np.arange(bounds[0][0], bounds[0][1]+G["dx"],G["dx"])]
    Yi=[i for i in np.arange(bounds[1][0], bounds[1][1]+G["dx"],G["dx"])]
    
    Xf=[i-G["dx"]/2 for i in np.arange(bounds[0][0], bounds[0][1]+ G["dx"] + G["dx"] ,G["dx"])]
    Yf=[i-G["dx"]/2 for i in np.arange(bounds[1][0], bounds[1][1]+ G["dx"] + G["dx"] ,G["dx"])]
    
    G['nx'] = len(Xi)
    G['ny']  = len(Yi)
    
    
    Xi_int=[i for i in np.arange(bounds[0][0], bounds[0][1]+2*G["dx"],G["dx"])]
    Yi_int=[i for i in np.arange(bounds[1][0], bounds[1][1]+2*G["dx"],G["dx"])]
    
    Xi = np.array(Xi)
    Yi = np.array(Yi)
    
    xv, yv  = np.meshgrid(Xi, Yi, sparse=False, indexing='ij')
    
    X = np.zeros((G['nx']*G['ny'],2))
    X[:,0] = xv.reshape(G['nx']*G['ny'], order='F')
    X[:,1] = yv.reshape(G['nx']*G['ny'], order='F')
    
    
    hist_bounds = [[Xf[0],Xf[-1]],[Yf[0],Yf[-1]]]
    Peq_true, edges = np.histogramdd(ys , range = hist_bounds, bins = [G['nx'],G['ny']],normed=True)
    Peq_true = Peq_true/sum(Peq_true.flatten())
    from scipy.ndimage import gaussian_filter
    Peq_true = gaussian_filter(Peq_true,sigma =  G['filtering'])
    plt.imshow(Peq_true.T,origin = 'lower',aspect = 'auto')
    
    plt.show()
    Peq_true = Peq_true.flatten(order = 'F')
    #cost matrix Wasserstein metric (W2)
    
    nx = G['nx']
    ny = G['ny']
    
    
    DxR,DxL,DxC = np.ones((nx,ny)),np.ones((nx,ny)),-2*np.ones((nx,ny))
    DyR,DyL,DyC = np.ones((nx,ny)),np.ones((nx,ny)),-2*np.ones((nx,ny))
    DxR[-1,:],DxL[0,:],DyR[:,-1], DyL[:,0] = 0,0,0,0
    DxC[0,:],DxC[-1,:], DyC[:,0],DyC[:,-1] = -1,-1,-1,-1
    DxR,DxL,DxC,DyR,DyL,DyC = G['diff']*DxR.flatten(order = 'F'),G['diff']*DxL.flatten(order = 'F'),G['diff']*DxC.flatten(order = 'F'),G['diff']*DyR.flatten(order = 'F'),G['diff']*DyL.flatten(order = 'F'),G['diff']*DyC.flatten(order = 'F')

    ###########Solve Forward Problem
    def diffmin(x):
        return 1.*(x<0)
    def diffmax(x):
        return 1.*(x>0)
    
    def RHS_Matrix2 (G,bounds):
        Uf = G['Vx']
        Vf = G['Vy']
        up = np.maximum(Uf,0)
        un = np.minimum(Uf,0)
        vp = np.maximum(Vf,0)
        vn = np.minimum(Vf,0)
        #Build sparse time advance operator K_mat
        #Each row (first index) is the equation for one [i,j,k] cell in terms of +-1 neighbors
        N=nx*ny
        
        # This is the equivalent of "ndgrid" in Matlab by specifing indexing='ij'  
        XXf,YYf= np.meshgrid(Xf,Yf,indexing='ij')
       
        # ufm=Uf[i-1,j-1,k-1]
        ufm = Uf[:-1,:-1]
        ufm = ufm.flatten(order='F')
      
    
        #  ufp=Uf[i,j-1,k-1]
        ufp = Uf[1:,:-1]
        ufp = ufp.flatten(order='F')
      
                   
        # vfm=Vf[i-1,j-1,k-1]
        vfm = Vf[:-1,:-1]
        vfm = vfm.flatten(order='F')
       
    
        # vfp=Vf[i-1,j,k-1]
        vfp = Vf[:-1,1:]
        vfp = vfp.flatten(order='F')
        
        T1 = spdiags(np.array([np.maximum(0,ufp)+DxR/G['dx'], np.minimum(0,ufm)- np.maximum(0,ufp)+DxC/G['dx'], -np.minimum(0,ufm)+DxL/G['dx']]), np.array([-1,0,1]), N, N) 
        T2 = spdiags(np.array([np.maximum(0,vfp)+DyR/G['dx'], np.minimum(0,vfm)- np.maximum(0,vfp)+DyC/G['dx'], -np.minimum(0,vfm)+DyL/G['dx']]), np.array([-nx,0,nx]), N, N)
       
        K_mat = (T1 + T2) /G['dx']*G['dt']
        return  K_mat
    
    
    def FWD(G,bounds):  
        K_mat = RHS_Matrix2(G,bounds)
        K_mat = K_mat.tocsr()
    
        # scale to positive
        mnV = abs(K_mat.min(axis=0).min())       
        N = K_mat.shape[0]
        speyeN = identity(N).tocsr()
        M = speyeN +(1/(2*mnV))*K_mat
        e = np.ones(N,)
    
        A = (1-G["alpha"])*M - speyeN
        b = -(G["alpha"]/N)*e
    
        x = spsolve(A, b )
        Peq = x.reshape((G['nx'],G['ny']), order='F')
       # print(np.sum(x))
    
        ##### Reshape the solution
        Peq0 = Peq * 0
        Peq0[1:-1, 1:-1] = Peq[1:-1, 1:-1]
        Peq0 = Peq0/Peq0.sum()
        Peq0 = Peq0.flatten(order = 'F')
        
    
        # return the transpose of matrix (1-G["alpha"])*M - speyeN for adjoint eqn.
        return (1-G["alpha"])*M.transpose() - speyeN, Peq0, x, K_mat
    
    
    def safe_divide(n, d):    
        return n / d if d else 0
    
    
    
    def KL(n,d):
        if n!=0 and d!= 0:
            return n*np.log(n/d)
        else:
            return 0
        
    def KLd(n,d):
        if n!=0 and d!=0:
            return -n/d
        else:
            return 0
        
    def JS(n,d):
        if n!=0 and d!= 0:
            m = (n+d)/2
            return (KL(n,m)+KL(d,m))/2
        else:
            return 0
    
    def JSd(n,d):
        if n!=0 and d!= 0:
            return .5*np.log((2*d)/(n+d))
        else:
            return 0
    
    
   
    
   
    
    
    #####Calculate Gradient
    def calc_cost_gradient(G,bounds,Peq_true):
        #### 1. Solve the forward problem
        A,Peq,_,K_mat = FWD(G,bounds)
        #### 2. compute the loss function
    
        if G['Cost'] == 'KL':
            cost = np.sum([KL(Peq_true[i],Peq[i]) for i in range(len(Peq))])
            u = np.asarray([KLd(Peq_true[i],Peq[i]) for i in range(len(Peq))])
        if G['Cost'] == 'JS':
            cost = np.sum([JS(Peq_true[i],Peq[i]) for i in range(len(Peq))])
            u = np.asarray([JSd(Peq_true[i],Peq[i]) for i in range(len(Peq))]) 
        if G['Cost'] == 'L2':
            cost = np.linalg.norm(Peq - Peq_true)**2 *0.5 
            u = Peq-Peq_true 
        #### 3. Solve the adjoint equation
        sol = spsolve(A,-u + u.dot(Peq))
        #### 4. compute the gradient
        grad1, grad2 = np.zeros((nx+1,ny+1)),np.zeros((nx+1,ny+1))
        mnV = abs(K_mat.min(axis=0).min())
        for i in range(1,nx-1):
          for j in range(1,ny-1):
            idx = i + nx*j
            dv1, dv2 = 1,1
            v1, v2 = G['Vx'][i,j], G['Vy'][i,j]
            grad1[i,j] = ((G['dt']*(1-G["alpha"])*(1/(2*mnV))*(sol[idx]-sol[idx-1])*(diffmax(v1)*Peq[idx-1]+diffmin(v1)*Peq[idx])*dv1/G['dx'])) 
            grad2[i,j] = ((G['dt']*(1-G["alpha"])*(1/(2*mnV))*(sol[idx]-sol[idx-nx])*(diffmax(v2)*Peq[idx-nx]+diffmin(v2)*Peq[idx])*dv2/G['dx'])) 
        
        
        g1 = grad1[1:-1,1:-1]
        g2 = grad2[1:-1,1:-1]
        
        return cost,g1.flatten(),g2.flatten()
    
    Iterations = 1
    iters = []
    costs = []
    
    
    def interp_col(y,C): #given a vector y in R3, return interpolated parameter values
        interp = RegularGridInterpolator((Xi, Yi), C, method= 'linear', bounds_error=False,fill_value=None)
        return interp(y)
    
    def dynamics(G):   
        def RHS_NN(x,G):    
            v = net(torch.tensor(x,dtype = torch.float).reshape(2,1)).detach().numpy()[0]
            return np.array([v[0],v[1]])
        
    
        
        print('Plotting Dynamics')
        Vx, Vy = G['Vx'].copy(), G['Vy'].copy()
      
       
    
        TimeStepMax = G['TSMax']
        IC = G['IC']
        ys = np.zeros((int(TimeStepMax),2))
        ys[0,:] = ys[0,:] = IC
        ysn = ys.copy()
        tic = time()
        noiseV = np.random.normal(0,1,int(2*TimeStepMax)).reshape((int(TimeStepMax),2))  
        for timestep in range(1,int(TimeStepMax)): 
            if timestep% int(TimeStepMax/10)==0:
              toc = time()
              print(f'time step: {timestep:4d} Elapsed time: {toc-tic:.2f}s')
            ## https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
            # ys[timestep,:] = ys[timestep-1,:] + G["dt"]* RHS_NN(ys[timestep-1,:],G) +np.sqrt(2*G['diff']*G['dt'])*noiseV[timestep-1,:]
            ysn[timestep,:] = ysn[timestep-1,:] + G["dt"]* RHS_NN(ysn[timestep-1,:],G)  
        
        G['Vx'], G['Vy'] = Vx, Vy
        
    
        return ysn
    
    
    def velocities(G):
        print('Plotting Inferred Parameter')
        Vx, Vy = G['Vx'].copy(), G['Vy'].copy()
      
        Np = 300
        Xp = np.linspace(bounds[0][0]+G['dx'],bounds[0][1]-G['dx'],Np)
        Yp = np.linspace(bounds[1][0]+G['dx'],bounds[1][1]-G['dx'],Np)
        # Np = nx
        # Xp = Xi
        # Yp = Yi
        
        
        P = np.zeros((Np,Np,2)) 
        P[:,:,0],P[:,:,1] = np.meshgrid(Xp,Yp,indexing = 'ij')
        P = np.zeros((Np,Np,2)) 
        P[:,:,0],P[:,:,1] = np.meshgrid(Xp,Yp,indexing = 'ij')
      #y_big is n by 3 where n= (nx-1)*(ny-1)*(nz-1) is # of points we evaluate the RHS
        P = P.reshape((Np**2,2)).T
        Vi = net(torch.tensor(P,dtype = torch.float)).detach().numpy()
        V1 = Vi[:,0].reshape((Np,Np))
        V2 = Vi[:,1].reshape((Np,Np))
        G['Vx'], G['Vy'] = Vx, Vy
        plt.streamplot(Xp,Yp,V1.T,V2.T)
        plt.show()
        return V1,V2
    
    
    
    m_normal = normal.Normal(0, 1)
    
    
    relu = nn.ReLU()
    sig = nn.Sigmoid()
    
    def activation(z):
        if G['act'] == 'relu':
            return relu(z)
        if G['act'] == 'sig':
            return sig(z)
        if G['act'] == 'tanh':
            return torch.tanh(z)
    
    ####### Net parameters
    learning_rate = G['lr']
    D_in = 2
    H1 = G['nodes']
    D_out = 1
    a = G['NNweight']
    ######################
    
    
    #Network for V
    class network(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 100),
                nn.Tanh(),
                nn.Linear(100, 2),
            )
        def forward(self, y):
            return self.net(y.T)
    
     
    
        
    class PDEsolver(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            """
            Forward equation
            """
            V = input.detach().numpy() # input has output from NN evaluated at the mesh
    
            G['Vx'][1:-1,1:-1] = V[:,0].reshape((nx-1,ny-1))
            G['Vy'][1:-1,1:-1] = V[:,1].reshape((nx-1,ny-1))
            g = np.zeros((2,(nx-1)*(ny-1)))
            c,g[0,:],g[1,:] = calc_cost_gradient(G,bounds,Peq_true)
    
        
    
            ctx.save_for_backward(torch.tensor(g.T, dtype = torch.float)) # save something for backward in NN
    
            return torch.tensor(c, dtype = torch.float)
        @staticmethod
        def backward(ctx, grad_output):
            g, = ctx.saved_tensors  # pull it out from what is saved earlier ctx is like "self"
            return g, None
    
    
    
    XX = torch.tensor(np.meshgrid(Xf[1:-1],Yi_int[1:-1],indexing='ij'), dtype = torch.float)
    XX = XX.reshape((2,len(Xf[1:-1])*len(Yi_int[1:-1])))
    
    YY = torch.tensor(np.meshgrid(Xi_int[1:-1],Yf[1:-1],indexing='ij'), dtype = torch.float)
    YY = YY.reshape((2,len(Xi_int[1:-1])*len(Yf[1:-1])))
    
    torch.manual_seed(seed)
    net = network()
    Vi = net(XX).detach().numpy()
    Vj = net(YY).detach().numpy()
    G['Vx'], G['Vy'] = np.zeros((nx+1,ny+1)),np.zeros((nx+1,ny+1))
    
    def bin_traj(ys):
        Peq, _ = np.histogramdd(ys , range = [[-4,4],[-5,5]], bins = [50,50],normed=True)
        Peq = Peq/sum(Peq.flatten())
        return Peq
    
    _,Peq_initial,_,_ = FWD(G,bounds)
    
    solver = PDEsolver.apply
    
    steps = G['numiter']
    
    loss_history = []
    grad_history = []
    reg_history = []
    total_history = []
    
    
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    start = time()
    for k in range(steps):
        net.train()
            
        def closure1():
            opt.zero_grad() 
            y_pred = torch.zeros((nx-1)*(ny-1),2)
            y_pred[:,0] = net(XX)[:,0]
            y_pred[:,1] = net(YY)[:,1]
            loss = solver(y_pred)
            loss.backward()
            return loss
    
        loss = opt.step(closure1)
       
        
    
        #Update velocities by NN
            
        Vi = net(XX).detach().numpy()
        Vj = net(YY).detach().numpy()
        
        G['Vx'][1:-1,1:-1] = Vi[:,0].reshape((nx-1,ny-1))
        G['Vy'][1:-1,1:-1] = Vj[:,1].reshape((nx-1,ny-1))
        # G['Vx'], G['Vy'] = np.zeros((nx+1,ny+1)),np.zeros((nx+1,ny+1))
    
    
        
    
    
        # print('Cost: ', loss.detach().numpy())
        costs.append(loss.detach().numpy())
        iters.append(k)
    
                
        if k % 1000 == 0:
            print('Iteration', k,'|', 'Cost:',loss.detach().numpy(), '|', 'Tol:', costs[-1]/costs[0])
            if np.abs(costs[-1]/costs[0])< G['tol']:
                G['end'] = True
    
    
        if k%G['plotevery'] == 0 and k!= 0:
            _,Peq,_,_ = FWD(G,bounds)
            plt.imshow(Peq.reshape(nx,ny,order = 'F').T,origin = 'lower',aspect = 'auto')
            plt.show()
            velocities(G)
            plt.show()
        end = time()
        if end-start > TIME:
            break
        k+=1
            

    # end = time()
    G['dt'] = .01
    G['TSMax'] = 1e6
    G['IC'] = np.array([ 0.84144155, -1.08920043])
    ysn = dynamics(G)
    plt.scatter(ysn[:,0],ysn[:,1],c = 'r')
    v1,v2 = velocities(G)
    return ysn, end-start
     
      
    
