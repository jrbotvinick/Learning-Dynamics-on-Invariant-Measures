import numpy as np
import matplotlib.pyplot as plt
from NN_Ours import traj_ours
from SINDy_func import traj_SINDy
from NODE_func import traj_NODE
import pickle 
import ot
from scipy.ndimage import gaussian_filter

#Compares our method with the Neural ODE framework. The file can be modified to run simmilar experiments with SINDy.

NAMES = ['samples_slow.p','samples_fast.p']
cc = 0

np.random.seed(1329) #1329 from before
SEEDS = []
for i in range(10):
    s = np.random.randint(int(10**6))
    SEEDS.append(s)
    
for NAME in NAMES:  
    with open(NAME, "rb") as f:
        ys_true = pickle.load(f)
    ys_true = ys_true[1]
    
    
    def bin_traj(ys):
        Peq,edges = np.histogramdd(ys , range = [[-4,4],[-5,5]], bins = [50,50],normed=True)
        # costM /= costM.max()
        Peq = Peq/sum(Peq.flatten())
        return Peq,edges
    
    
    def cost(P1,P2,edges):
        dx = edges[0][1]-edges[0][0]
        dy = edges[1][1] -edges[1][0]
        Xi = [edges[0][0]+dx/2 + dx*i for i in range(len(edges[0])-1)]
        Yi = [edges[1][0]+dy/2 + dx*i for i in range(len(edges[1])-1)]
        xv, yv = np.meshgrid(Xi, Yi, sparse=False, indexing='ij')
        X = np.zeros((len(Xi)*len(Yi),2))
        X[:,0] = xv.reshape(len(Xi)*len(Yi), order='F')
        X[:,1] = yv.reshape(len(Xi)*len(Yi), order='F')
        costM = ot.dist(X, X)
        _,log = ot.lp.emd(P1.flatten(order = 'F'), P2.flatten(order = 'F'), costM, numItermax=1000000, log=True)
        return log['cost']

    P_true,edges = bin_traj(ys_true)
    plt.imshow(P_true)
    plt.show()
  
    
    
    TIMES = [500]
    TRAJS1 = []
    TRAJS2 = []
    for TIME in TIMES:
        O_costs = []
        N_costs = []
        O_times = []
        N_times = []
        for i in range(0,10):
            seed = SEEDS[i]
            ys2,ts2 = traj_ours(TIME,NAME,seed)
            ys,ts = traj_NODE(TIME,NAME,seed)
            TRAJS1.append(ys[:int(1e5)])
            TRAJS2.append(ys2[:int(1e5)])
            P1,_ = bin_traj(ys)
            P2,_ = bin_traj(ys2)
            plt.imshow(P1)
            plt.show()
            plt.imshow(P2)
            plt.show()
            N_costs.append(cost(P1,P_true,edges))
            O_costs.append(cost(P2,P_true,edges))
            N_times.append(ts)
            O_times.append(ts2)
            print(N_costs)
            print(O_costs)
            
        with open('EXP_INFO {}'.format(cc), "wb") as f: 
               pickle.dump([O_costs,N_costs,O_times,N_times], f)
        with open('EXP_TRAJS {}'.format(cc), "wb") as f:
               pickle.dump([TRAJS1,TRAJS2], f)   
    cc+=1
