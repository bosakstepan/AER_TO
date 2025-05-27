# %% Q-Factor results processing - loading modules and data
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import local_algorithm as aer
import torch
import pandas as pd
import local_algorithm as aer
import mat73
from plot_funcitons import plot_topology, get_rgb_values
#!%load_ext autoreload
#!autoreload 2
#!%matplotlib inline
path_data = "./Data/ka_08/16x10/GASr.mat"
data = mat73.loadmat(path_data, use_attrdict=True) 
max_beta = 64
delta = 1
np.random.seed(0)
torch.manual_seed(0)
#%% Load the model
model_path = "./docsrc/ka_08/filter2/aer_q.pth"
optimization_log_path = "./docsrc/ka_08/filter2/aer_q.csv"
optimization_log = pd.read_csv(optimization_log_path)
optimization_log.head()
optimization_log.describe()
port = int(optimization_log["port"].values[0])

# load the model
aer_q = aer.AER_Q(data, p = port, mode="filter2", device='cpu')
aer_q.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
# test the model
with torch.no_grad():
    Q, Qe, I, gc, w  = aer_q(max_beta)
    # find index of minimum Qhat
    print(f"Q/Qlb: {Q/aer_q.qlb}")
    print(f"Qe/Qlb: {Qe/aer_q.qlb}")
    print(f"w: {w}") 
    
#%% Process CSV files
# Plot losses in log scale, forward and gradient times, learning rates and betas in three subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(optimization_log["losses"])
ax[0].set_yscale("log")
ax[0].set_title("Losses")
ax[1].plot(optimization_log["betas"])
ax[1].set_title("Betas")
plt.show()
print(f"Computing time: {optimization_log['computing_time'][0]}")
print(f"Steps: {len(optimization_log)}")
    
# %% Threshold the vector g and calculate Q-factor

gt = torch.round(gc)
globIndxBf = torch.where(gt == 1)[0]
Zt = aer_q.z[torch.meshgrid(globIndxBf, globIndxBf, indexing='ij')]
VT = aer_q.v[globIndxBf]
iT = torch.linalg.solve(Zt, VT)
xeT = aer_q.xe[torch.meshgrid(globIndxBf, globIndxBf, indexing='ij')]
xmT = aer_q.xm[torch.meshgrid(globIndxBf, globIndxBf, indexing='ij')]
r0T = aer_q.r0[torch.meshgrid(globIndxBf, globIndxBf, indexing='ij')]
itH = iT.transpose(0, 1).conj()
We = torch.real(itH @ xeT @ iT)
Wm = torch.real(itH @ xmT @ iT)
Prad = torch.real(itH @ r0T @ iT)
Qt = torch.max(We, Wm)/Prad
Qet = torch.abs(We - Wm) / Prad
print(f"Q/Qlb thresholded: {(Qt/aer_q.qlb).item()}")
print(f"Qe/Qlb thresholded: {(Qet/aer_q.qlb).item()}")

rgb_gray = get_rgb_values(gc, port)
rgb_bw = get_rgb_values(gt, port)
mesh = data.Mesh
BF = data.BF


# Plot the topology with the optimized design values
plot_topology(data.Mesh, rgb_gray, data.BF, port=port)
plot_topology(data.Mesh, rgb_bw, data.BF, port=port)
