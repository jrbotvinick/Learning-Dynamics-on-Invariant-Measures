#Code adapted from https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py

def traj_NODE(TIME,name,seed):
    
    import os
    import argparse
    import time
    import numpy as np
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pickle
    
    # iterations = iters
    # print(iterations)
    
    with open(name, "rb") as f:
        data = pickle.load(f)
    ts,ys = data[0],data[1]
    ts = ts
    ys = ys
    t_test = np.linspace(0,100,int(1e4))
    parser = argparse.ArgumentParser('ODE demo')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
    parser.add_argument('--data_size', type=int, default=len(ts))
    parser.add_argument('--batch_time', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--niters', type=int, default=1000000)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--adjoint', action='store_true')
    args = parser.parse_args()
    
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint
    
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    true_y0 = torch.tensor([[ 0.84144155, -1.08920043]]).to(device)
    t = torch.tensor(ts,dtype = torch.float32)
    t_test = torch.tensor(t_test,dtype = torch.float32)
    
    true_y = torch.tensor(ys,dtype = torch.float32)
    rr = 1e-5
    aa = 1e-5
  
    def get_batch():
        s = torch.tensor([i for  i in range(args.data_size-1)])
        batch_y0 = true_y[s]  # (M, D)
        batch_t = t[:args.batch_time]  # (T)
        batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
        return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

    
    def makedirs(dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    
    
    
    import matplotlib.pyplot as plt
    
    
    def visualize(true_y, pred_y, odefunc):
        print(np.shape(pred_y))
        pp = pred_y.cpu().numpy()
        zz = true_y.cpu().numpy()
        plt.scatter(pp[:,0,0],pred_y[:,0,1])
        plt.scatter(zz[:,0],zz[:,1])
    
        plt.show()
    
    
    class ODEFunc(nn.Module):
    
        def __init__(self):
            super(ODEFunc, self).__init__()
    
            self.net = nn.Sequential(
                nn.Linear(2, 100),
                nn.Tanh(),
                nn.Linear(100, 2),
               
            )
        def forward(self, t, y):
            return self.net(y)
    
    
    torch.manual_seed(seed)
    func = ODEFunc().to(device)
    optimizer = optim.Adam(func.parameters(), lr=1e-3)
    start = time.time()
    visualize_every = 100000
    itr = 1
    while True:
        optimizer.zero_grad()
        
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t,rtol = rr, atol = aa).to(device)
        loss = torch.mean((pred_y - batch_y)**2)
        if itr == 1:
            loss0 = loss
        loss.backward()
        if itr%1000 == 0:
            print('Iter {:04d} | Total Loss Tol {:.6f}'.format(itr, loss.item()/loss0))

        optimizer.step()

        if itr % visualize_every == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t_test,rtol = rr, atol = aa)
                print('Iter {:04d} | Total Loss Tol {:.6f}'.format(itr, loss.item()/loss0))
                visualize(true_y, pred_y, func)
        itr += 1
        end = time.time()
        if end - start > TIME:
            break

            
            
            
    with torch.no_grad():
        tts = torch.tensor(np.linspace(0,10000,int(1e6)),dtype = torch.float32)
        pred_y = odeint(func, true_y0, tts,rtol = rr, atol = aa)
    # with open('samples_slow_NODE_526_itr.p', "wb") as f:
    #     pickle.dump([pred_y,end-start], f)

    return pred_y.detach().numpy().reshape(len(pred_y),2),end-start
