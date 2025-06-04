import torch
import torch.nn as nn
from typing import Union
import functions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

class AER_Q(nn.Module):
    def __init__(self, data : dict, p: Union[int, float], device = 'cpu', mode : str ="filter1"):
        super(AER_Q, self).__init__()     
        self.device = device # device to run the model on
        # Find port index - if int we have port index, if float we have port position
        self.maxx, self.maxy = self.get_L(data)
        if isinstance(p, int):
            self.port = p
        elif isinstance(p, float):
            pp = self.get_pxpy(p)
            self.port = self.get_port(data, pp)
        else:
            raise ValueError("pp must be an int or a numpy array")        
        # MoM matricies and constants
        self.qlb = float(data.QlbTM) # Lower bound for Q
        self.z = torch.tensor(data.Z, dtype=torch.complex64, requires_grad=False).to(device)
        self.omw = torch.tensor(data.omW, dtype=torch.complex64, requires_grad=False).to(device)
        self.x0 = torch.imag(self.z).to(torch.complex64).to(device)
        self.r0 = torch.real(self.z).to(torch.complex64).to(device)
        self.xe = torch.tensor(data.Xe, dtype=torch.complex64, requires_grad=False).to(device)
        self.xm = torch.tensor(data.Xm, dtype=torch.complex64, requires_grad=False).to(device)
        self.lmat_diag = torch.diag(torch.tensor(data.Lmat, dtype=torch.complex64, requires_grad=False)).to(device)
        self.v = torch.zeros(self.z.shape[0], 1, dtype=torch.complex64, requires_grad=False).to(device)
        self.v[self.port] = 1
        self.N = int(self.z.shape[0]) # Number of DOFs
        self.h = torch.tensor(data.H, dtype=torch.float32, requires_grad=False).to(device) # Density filters
        self.mask = torch.ones(self.N, dtype=torch.bool, device=self.device) # mask for the port
        self.mask[self.port] = False
        self.Mesh = data.Mesh # Mesh data
        self.BF = data.BF # Boundary face data
        # optimization parameters
        self.weights = nn.Parameter(0.5*torch.ones(self.N-1, 1, dtype=torch.float32, requires_grad=True))
        self.ni = torch.tensor(0.5, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # Optimization mode
        self.mode = mode # mode of the forward function -- two filters are available
        
    def get_pxpy(self, p : float):
        """ Get the position of the port in px, py coordinates -- parametrization of the position."""
        py = np.clip(self.maxy - p, 0, self.maxy)
        px = np.clip(p - self.maxy, 0, self.maxx)
        return np.array([px, py, 0])

    def get_port(self, data : dict, pp : np.ndarray):
        """ Get the port index based on the position."""
        TEC = data.Mesh.triangleEdgeCenters
        BF = data.BF.data[:, 2].astype(int) - 1
        BFCenters = TEC[BF, :]
        mask = (((BFCenters[:, 0] == 0) & (BFCenters[:, 1] >= 0)) |
                ((BFCenters[:, 1] == 0) & (BFCenters[:, 0] >= 0)))
        BFCentersf = BFCenters[mask, :]
        filtered_indices = np.where(mask)[0]  # indices of BFCentersf in BFCenters
        # Find closest point
        distances = np.linalg.norm(BFCentersf - pp, axis=1)
        min_idx_in_filtered = int(np.argmin(distances))
        port = filtered_indices[min_idx_in_filtered]
        return port

    def get_L(self, data : dict):
        """ Get the size of the mesh for parametrization."""
        TEC = data.Mesh.triangleEdgeCenters
        BF = data.BF.data[:, 2].astype(int) - 1
        BFCenters = TEC[BF, :]
        maxx = np.max(BFCenters[:, 0])
        maxy = np.max(BFCenters[:, 1])
        return maxx, maxy

    def q(self, I):
        """ Calculate the quality factor Q and tuning Qe factor."""
        iH = I.transpose(0, 1).conj()
        We = torch.real(iH @ self.xe @ I)
        Wm = torch.real(iH @ self.xm @ I)
        Prad = torch.real(iH @ self.r0 @ I)
        Q = torch.max(We, Wm)/Prad
        Qe = torch.abs(We - Wm) / Prad
        return Q.squeeze(), Qe.squeeze()
        
    def w(self, rho):
        """ Regularization function."""
        bl = torch.sum(4*rho*(1-rho)/rho.shape[0])
        return bl
    
    def ff(self, rho, rMet :float = 0.1, rVac : float = 10e5):
        """ Resistivity projection function."""
        Rs = rVac * (rMet / rVac) ** rho
        Rs = Rs.squeeze() * self.lmat_diag
        Zmod = torch.diag(Rs) + self.z
        iF = torch.linalg.solve(Zmod, self.v)
        return iF
    
    def forward(self, beta : float, p : int = 1):
        """ Forward function for the AER_Q model."""
        gct = torch.sigmoid(self.weights)
        gc = torch.ones(self.N, 1, dtype=torch.float32, device=self.device) # Copy of g
        gc[self.mask] = gct
        gc = gc/(1 + p*(1 - gc)) # Penalization function
        # switch between the two modes -- filter1 and filter2 and no filter
        match self.mode:
            case "filter1":
                gc = self.h[0, :, :] @ gc
            case "filter2":
                gc = self.h[1, :, :] @ gc
            case _:
                gc = gc
        gc = (torch.tanh(beta*self.ni) +  torch.tanh(beta*(gc-self.ni))) / (torch.tanh(beta*self.ni) + torch.tanh(beta*(1-self.ni))) # thresholding function
        gc[self.port] = 1 # Port is always 1
        iF = self.ff(gc) # resistivity projection
        Q, Qe = self.q(iF) # Calculate Q and Qe
        w = self.w(gc) # Regularization term
        return Q, Qe, iF, gc, w

def optimize(aer_q : AER_Q, lr : float, max_beta : int, max_i : int, wd : float, max_gamma : float = 1, di : float = 0.3, plot = True):
    optimizer = torch.optim.AdamW(aer_q.parameters(), lr=lr, weight_decay=wd)
    losses = []
    betas = []
    beta = 1
    gamma = 0
    delta = 0
    i = 0
    if plot:
        fig = plt.figure(figsize=(12, 4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])  # Equal widths

        ax1 = fig.add_subplot(gs[0], projection='3d')  # 3D plot
        ax2 = fig.add_subplot(gs[1])                   # 2D plot (loss)
        ax3 = fig.add_subplot(gs[2])                   # 2D plot (beta)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_yscale("log")
        loss_line, = ax2.plot([], [], 'b-')  # initialize empty line
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Beta")
        beta_line, = ax3.plot([], [], 'g-')  # initialize empty line
        plt.tight_layout()
        plt.ion()
    while True:
        optimizer.zero_grad()
        Q, Qe, _, gc, w = aer_q(beta)
        if plot:
            x_vals = list(range(len(losses)))
            loss_line.set_data(x_vals, losses)
            beta_line.set_data(x_vals, betas)

            ax2.relim()
            ax2.autoscale_view()
            ax3.relim()
            ax3.autoscale_view()
            ax1.cla()
            x_t = functions.get_rgb_values(gc, aer_q.port)
            functions.plot_topology(aer_q.Mesh, ax1, x_t, aer_q.BF, port=aer_q.port)
            plt.pause(0.05)
        loss = ((Q + delta * Qe) / aer_q.qlb) + gamma*w
        # save progress
        betas.append(beta)
        loss_value = loss.item()
        losses.append(loss_value)
        i += 1
        if i > max_i-1:
            if beta < max_beta:
                beta *= 2
                i = 0
                delta += di
            elif gamma == 0:
                if max_gamma == 0:
                    break
                gamma = max_gamma
                i = 0
            else:
                break
        if len(losses) % 20 == 0:
            with torch.no_grad():
                print(f'Epoch {len(losses)}, Loss: {loss.item()}, Qe/Qlb: {Qe.item()/aer_q.qlb}, Q/Qlb: {Q.item()/aer_q.qlb}, beta: {beta}, delta: {delta}, gamma: {gamma}, w: {w.item()}')
        loss.backward() # AD
        optimizer.step() # AdamW step
    if plot:
        plt.ioff()
        plt.show(block=False)
    # Final results
    return aer_q, losses, betas
    