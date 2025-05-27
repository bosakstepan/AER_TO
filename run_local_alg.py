import local_algorithm as aer
import torch
import pandas as pd
import numpy as np
import argparse
import time
import mat73
from typing import Union
np.random.seed(0)
torch.manual_seed(0)

def main(data : dict, file_name : str, p : Union[int, float], lr : float, max_beta : int, max_i : int, wd : float, mode : str = "filter1", device : str = "cpu"):
    """Local algorithm for AER optimization."""
    # load the data
    aer_q = aer.AER_Q(data, p, mode=mode, device=device)
    aer_q = aer_q.to(device)
    port = aer_q.port # display the port index
    print(f"Port: {port}")
    # measure times
    start = time.perf_counter() # sometime time.time() is not precise enough so use time.perf_counter()
    qer_qb, losses, betas = aer.optimize(aer_q, lr, max_beta, max_i, wd)
    end = time.perf_counter()
    elapsed = end - start
    Q, Qe, _, _, w  = qer_qb(max_beta)
    print(f"Q/Qlb: {Q/qer_qb.qlb}")
    print(f"Qe/Qlb: {Qe.item()/qer_qb.qlb}")
    print(f"w: {w.item()}")
    print(f"loss: {losses[-1]}")
    # save model and the optimization log
    torch.save(aer_q.state_dict(), f"{file_name}.pth")
    # save the data into a csv
    optimization_log = pd.DataFrame({"losses": losses, "betas": betas})
    optimization_log["port"] = port
    # save training time 
    optimization_log["computing_time"] = elapsed
    optimization_log.to_csv(f"{file_name}.csv")
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="aer_q")
    parser.add_argument("--data_path", type=str, default="./Data/ka_08/16x10/GASr.mat")
    parser.add_argument("--lr", type=float, default = 0.05) # Hyperparameter for the learning rate
    parser.add_argument("--wd", type=float, default = 1e-3) # Hyperparameter for the weight decay
    parser.add_argument("--max_i", type=int, default=int(2100)) # Hyperparameter for the maximum number of iterations
    parser.add_argument("--max_beta", type=int, default=64) # Hyperparameter for the maximum beta
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="filter2") # Select filter mode: filter1, filter2, or no filter
    args = parser.parse_args()
    #epochs = args.epochs
    file_name = args.file_name
    data_path = args.data_path
    lr = args.lr
    wd = args.wd
    max_i = args.max_i
    max_beta = args.max_beta
    device = args.device
    mode = args.mode
    # load the data
    data = mat73.loadmat(data_path, use_attrdict=True)
    # Chose port here - center of the longer edge is save in the data as port_c -- index of the edge
    port = int(data["port_c"]) - 1 # Convert to zero-based index
    main(data, file_name, p=port, lr=lr, max_beta=max_beta, max_i=max_i, wd=wd, mode=mode, device=device)
    pass