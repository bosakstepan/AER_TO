import local_algorithm as aer
import time
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import mat73
from typing import Union
from functions import get_pp_from_port, get_p_from_pxpy
path = "./Data/ka_08/16x10/GASr.mat"
device = "cpu"
mode = "filter2"
delta = 0
max_beta = 32
max_gamma = 0 # switch the regularization off
switch_delta = False # sets the self-resonance regularization
# load the gasr and evaluate bounds
data = mat73.loadmat(path, use_attrdict=True) 
# get the parametrization bound
TEC = data.Mesh.triangleEdgeCenters
BF = data.BF.data[:, 2].astype(int) - 1
BFCenters = TEC[BF, :]
maxx = np.max(BFCenters[:, 0])
maxy = np.max(BFCenters[:, 1])
L = maxx + maxy

port = int(data["port_c"]) - 1 # Convert to zero-based index
# get the port position
pp = get_pp_from_port(data, port)
p = get_p_from_pxpy(maxy, pp) # get the port position in px, py coordinates
#pbounds = {'lr': (0.4, 0.9), 'wd': (0.001, 0.01), 'max_i': (70, 90), 'p': (0, L)} # use to optimize port position using parametrization L
pbounds = {'lr': (0.3, 0.7), 'wd': (0.001, 0.01), 'max_i': (60, 75), 'p' : (p, p)} # port is fixed, so we optimize only lr, wd and max_i
scores = []
N_START = 3
N_ITER = 13 - N_START
index = 0


def optimized_function(lr : float, wd : float, max_i : int, p: Union[int, float]):
    aer_q = aer.AER_Q(data, p, device=device, mode=mode)
    port = aer_q.port
    print(f"Port: {port}")
    aer_q = aer_q.to(device)
    # measure times
    start = time.time() # sometime time.time() is not precise enough so use time.perf_counter()
    aer_qb, losses, betas = aer.optimize(aer_q, lr, max_beta, int(max_i), wd, max_gamma=max_gamma, switch_delta=switch_delta)
    end = time.time()
    elapsed = end - start
    _, _, _, _, w  = aer_qb(max_beta)
    global index
    name = f"qfn_{index}"
    index += 1
    torch.save(aer_qb.state_dict(), f"{name}.pth")
    # save the data into a csv
    optimization_log = pd.DataFrame({"losses": losses, "betas": betas})
    # save training time 
    optimization_log["computing_time"] = elapsed
    optimization_log["lr"] = lr
    optimization_log["wd"] = wd
    optimization_log["max_i"] = max_i
    optimization_log["w"] = w.item()
    optimization_log["port"] = port
    optimization_log.to_csv(f"{name}.csv")
    scores.append(losses[-1])
    return -1*losses[-1] # we want to minimize the loss, so we return -loss

optimizer = BayesianOptimization(
    f=optimized_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=N_START,
    n_iter=N_ITER,
)

print(optimizer.max)
# save optimizer scores
pd.DataFrame(scores).to_csv("scores.csv")



