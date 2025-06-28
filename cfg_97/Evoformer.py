import torch
from torch import nn
from torch.nn import functional as F
import basic,EvoPair,EvoMSA # Assuming these are available
import math,sys
from torch.utils.checkpoint import checkpoint


class EvoBlock(nn.Module):
    def __init__(self,m_dim,z_dim,docheck=False):
        super(EvoBlock,self).__init__()
        self.msa_row=EvoMSA.MSARow(m_dim,z_dim)
        self.msa_col=EvoMSA.MSACol(m_dim)
        self.msa_trans=EvoMSA.MSATrans(m_dim)
        self.msa_opm=EvoMSA.MSAOPM(m_dim,z_dim)
        self.pair_triout=EvoPair.TriOut(z_dim)
        self.pair_triin =EvoPair.TriIn(z_dim)
        self.pair_tristart=EvoPair.TriAttStart(z_dim)
        self.pair_triend  =EvoPair.TriAttEnd(z_dim)
        self.pair_trans = EvoPair.PairTrans(z_dim)
        self.docheck=docheck

    def layerfunc_msa_row(self,m,z):
        return self.msa_row(m,z) + m
    def layerfunc_msa_col(self,m):
        return self.msa_col(m) + m
    def layerfunc_msa_trans(self,m):
        return self.msa_trans(m) + m
    def layerfunc_msa_opm(self,m,z):
        return self.msa_opm(m) + z

    def layerfunc_pair_triout(self,z):
        return self.pair_triout(z) + z
    def layerfunc_pair_triin(self,z):
        return self.pair_triin(z) + z
    def layerfunc_pair_tristart(self,z):
        return self.pair_tristart(z) + z
    def layerfunc_pair_triend(self,z):
        return self.pair_triend(z) + z
    def layerfunc_pair_trans(self,z):
        return self.pair_trans(z) + z

    def forward(self, m, z):
        if True: # Current code path always takes this branch
            m_in = m
            z_in = z
            m = m + self.msa_row(m,z)
            m = m + self.msa_col(m)
            m = m + self.msa_trans(m)
            z = z + self.msa_opm(m_in) # OPM typically uses the MSA representation input to the block
            z = z + self.pair_triout(z)
            z = z + self.pair_triin(z)
            z = z + self.pair_tristart(z)
            z = z + self.pair_triend(z)
            z = z + self.pair_trans(z)
            return m,z
        else: # This branch is currently unused
            m=checkpoint(self.layerfunc_msa_row,m,z, use_reentrant=False)
            m=checkpoint(self.layerfunc_msa_col,m, use_reentrant=False)
            m=checkpoint(self.layerfunc_msa_trans,m, use_reentrant=False)
            z=checkpoint(self.layerfunc_msa_opm,m,z, use_reentrant=False) # Pass m_in here if OPM needs block input m
            z=checkpoint(self.layerfunc_pair_triout,z, use_reentrant=False)
            z=checkpoint(self.layerfunc_pair_triin,z, use_reentrant=False)
            z=checkpoint(self.layerfunc_pair_tristart,z, use_reentrant=False)
            z=checkpoint(self.layerfunc_pair_triend,z, use_reentrant=False)
            z=checkpoint(self.layerfunc_pair_trans,z, use_reentrant=False)
            return m,z


class Evoformer(nn.Module):
    def __init__(self,m_dim,z_dim,docheck=False):
        super(Evoformer,self).__init__()
        self.layers=[16]
        self.docheck=docheck
        # if docheck:
            # print('Evoformer will do checkpoint (referring to its own loop if that logic were active)')
        # EvoBlock instances are created with docheck=True, triggering their print statement
        self.evos=nn.ModuleList([EvoBlock(m_dim,z_dim,True) for i in range(self.layers[0])])

    def layerfunc(self,layermodule,m,z): # Not used by current active forward
        m_,z_=layermodule(m,z)
        return m_,z_

    def forward_n(self, m, z, starti, endi):
        for i in range(starti,endi):
            # print(f"Evoformer.forward_n calling EvoBlock {i}")
            m,z=self.evos[i](m, z)
        return m,z

    def forward(self, m, z):
        print(f"Evoformer START - m: {m.shape}, z: {z.shape}")
        # Explicitly set use_reentrant=False for checkpoint calls.
        m,z = checkpoint(self.forward_n, m, z, 0, 3, use_reentrant=False)
        m,z = checkpoint(self.forward_n, m, z, 3, 6, use_reentrant=False)
        m,z = checkpoint(self.forward_n, m, z, 6, 10, use_reentrant=False)
        m,z = checkpoint(self.forward_n, m, z, 10, 13, use_reentrant=False)
        m,z = checkpoint(self.forward_n, m, z, 13, 16, use_reentrant=False)
        return m,z

if __name__ == "__main__":
    N=10
    L=30
    m_dim=16
    z_dim=8
    m_test=torch.rand(N,L,m_dim)
    z_test=torch.rand(L,L,z_dim)
    model = Evoformer(m_dim,z_dim)
    m_out,z_out=model(m_test,z_test)
