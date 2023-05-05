#Code adapted from https://github.com/dynamicslab/pysindy

def traj_SINDy(reg,name):
    
    import numpy as np
    import pysindy as ps
    import pickle 
    import time
    
    import matplotlib.pyplot as plt
        
    with open(name, "rb") as f:
        data = pickle.load(f)
    ts,ys = data[0],data[1]
    # ys = ys[:-1000]
    # ts = ts[:-1000]
    start = time.time()
    
    # plt.scatter(ys[:4,0],ys[:4,1])
    poly_order = 3
    threshold = reg
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
    )
    
    model.fit(ys, t=ts)
    end = time.time()
    tt = np.linspace(0,10000,int(1e6))
    x0_train = np.array([ 0.84144155, -1.08920043])
    x_sim = model.simulate(x0_train, tt,integrator = 'odeint',integrator_kws={ "rtol": 1e-5, "atol": 1e-5})
    plt.scatter(ys[:,0],ys[:,1])
    
    plt.scatter(x_sim[:,0],x_sim[:,1])
    # plt.scatter(ys[:,0],ys[:,1])
    
    
    model.print()
    return x_sim,end-start
  
