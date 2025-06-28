import torch
from torch import nn
from torch.nn import functional as F
import basic,IPA # cfg_97 specific imports
import math,os
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation

# DEBUG_IDENTIFIER = "STRUCTURE_DEBUG_PLACEHOLDER" # Removed

expdir_cfg_97 = os.path.dirname(os.path.abspath(__file__))
newconfig_path_cfg_97 = os.path.join(expdir_cfg_97, 'newconfig')

attdrop = False
denoisee2e = False
ss_type = ''

if os.path.exists(newconfig_path_cfg_97):
    try:
        lines = open(newconfig_path_cfg_97).readlines()
        if len(lines) >= 3:
            attdrop = lines[0].strip().split()[-1] == '1'
            denoisee2e = lines[1].strip().split()[-1] == '1'
            ss_type =  lines[2].strip().split()[-1]
        # else:
            # print(f"Warning: {newconfig_path_cfg_97} has fewer than 3 lines. Using default Evoformer configs.")
    except Exception as e:
        # print(f"Warning: Could not read {newconfig_path_cfg_97}. Using default Evoformer configs. Error: {e}")
        pass # Keep it silent for now
# else:
    # print(f"Warning: {newconfig_path_cfg_97} not found. Using default Evoformer configs for attdrop, denoisee2e, ss_type.")


class TransitionModule(nn.Module):
    def __init__(self,c):
        super(TransitionModule,self).__init__()
        self.c=c
        self.norm1=nn.LayerNorm(c)
        self.linear1=basic.Linear(c,c)
        self.linear2=basic.Linear(c,c)
        self.linear3=basic.Linear(c,c)
        self.norm2=nn.LayerNorm(c)
    def forward(self,s_):
        s_norm1 = self.norm1(s_)
        s_lin1_relu = F.relu(self.linear1(s_norm1))
        s_lin2_relu = F.relu(self.linear2(s_lin1_relu))
        s_res_lin3 = self.linear3(s_lin2_relu)
        s_out = s_ + s_res_lin3
        s_final = self.norm2(s_out)
        return s_final

class BackboneUpdate(nn.Module):
    def __init__(self,indim):
        super(BackboneUpdate,self).__init__()
        self.indim=indim
        self.linear=basic.Linear(indim,6)
        torch.nn.init.zeros_(self.linear.linear.weight)
        torch.nn.init.zeros_(self.linear.linear.bias)

    def forward(self,s,L):
        pred_affine = self.linear(s)
        rot_quaternion_part = pred_affine[...,:3]
        rot = basic.quat2rot(rot_quaternion_part,L)
        trans = pred_affine[...,3:]
        return rot,trans

class TorsionNet(nn.Module):
    def __init__(self,s_dim,c):
        super(TorsionNet,self).__init__()
        self.s_dim=s_dim
        self.c=c
        self.linear1=basic.Linear(s_dim,c)
        self.linear2=basic.Linear(c,c)
        self.linear3=basic.Linear(c,c)
        self.linear4=basic.Linear(c,c)
        self.linear5=basic.Linear(c,c)
        self.linear6=basic.Linear(c,c)
        self.linear7_1=basic.Linear(c,1)
        self.linear7_2=basic.Linear(c,2)
        self.linear7_3=basic.Linear(c,2)

    def forward(self,s_init,s):
        a = F.relu(self.linear1(s_init) + self.linear2(s))
        a_res = F.relu(self.linear3(a))
        a_res = self.linear4(a_res)
        a = a + a_res
        a_res2 = F.relu(self.linear5(a))
        a_res2 = self.linear6(a_res2)
        a = a + a_res2
        a_relu = F.relu(a)
        bondlength = self.linear7_1(a_relu)
        angle_unnorm = self.linear7_2(a_relu)
        torsion_unnorm = self.linear7_3(a_relu)
        angle_norm = torch.norm(angle_unnorm,dim=-1,keepdim=True)
        angle = angle_unnorm / (angle_norm+1e-8)
        torsion_norm = torch.norm(torsion_unnorm,dim=-1,keepdim=True)
        torsion = torsion_unnorm / (torsion_norm+1e-8)
        return bondlength, angle, angle_norm, torsion, torsion_norm


class lddtpredictor(nn.Module):
    def __init__(self,s_dim,z_dim):
        super(lddtpredictor,self).__init__()
        self.lineara=basic.Linear(s_dim,z_dim)
        self.linearb=basic.Linear(s_dim,z_dim)
        self.linear2=basic.Linear(z_dim,z_dim)
        self.linear3=basic.Linear(z_dim,z_dim)
        self.linear4=basic.Linear(z_dim,z_dim)
        self.linear5=basic.Linear(3,z_dim)
        self.linear6=basic.Linear(z_dim,z_dim)
        self.linear7=basic.Linear(z_dim,z_dim)
        self.linear8=basic.Linear(z_dim,z_dim)
        self.lastlinear = basic.Linear(z_dim,5)

    def forward(self,rot,trans,s,z):
        feat_s_a = self.lineara(s)
        feat_s_b = self.linearb(s)
        zs_from_s = feat_s_a[None,:,:] + feat_s_b[:,None,:]
        zs_combined_z = zs_from_s + self.linear2(z)
        zs_mlp1 = F.relu(self.linear3(F.relu(zs_combined_z)))
        zs = zs_from_s + self.linear4(zs_mlp1)
        rel_pos_global = trans[:,None,:] - trans[None,:,:]
        spatial_feat = F.relu(self.linear5(F.relu(rel_pos_global)))
        zs = zs + self.linear6(spatial_feat)
        zs_mlp2 = F.relu(self.linear7(F.relu(zs)))
        zs = zs + self.linear8(zs_mlp2)
        logits = self.lastlinear(zs)
        return F.log_softmax(logits,dim=-1)

    def compute_plddt(self,predlogsoft):
        explddt = torch.exp(predlogsoft)
        lddt = (explddt[...,:1].sum(dim=-1) + \
                explddt[...,:2].sum(dim=-1) + \
                explddt[...,:3].sum(dim=-1) + \
                explddt[...,:4].sum(dim=-1))*0.25
        L = lddt.shape[0]
        mask_diag = ~torch.eye(L, dtype=torch.bool, device=lddt.device)
        plddt_residue = torch.sum(lddt * mask_diag, dim=1) / (L - 1 + 1e-8)
        return plddt_residue


class StructureModule(nn.Module):
    def __init__(self,s_dim,z_dim,N_layer,c, global_rnalm_instance=None, **kwargs):
        super(StructureModule,self).__init__()
        self.s_dim=s_dim
        self.z_dim=z_dim
        self.N_layer=N_layer
        self.c=c
        self.layernorm_s=nn.LayerNorm(s_dim)
        self.layernorm_z=nn.LayerNorm(z_dim)
        # Assuming IPA.InvariantPointAttention.forward also needs current_debug_identifier removed if it had it
        self.ipa=IPA.InvariantPointAttention(dim_in=self.c, dim_z=self.z_dim, N_head=64, c=16)
        self.transition = TransitionModule(self.c)
        self.bbupdate = BackboneUpdate(self.c)
        self.torsionnet=TorsionNet(s_dim,self.c)
        self.lddt_preder= lddtpredictor(s_dim,z_dim)
        self._init_T_static()
        self.baseframe = self._base_frame_static().double()

    @staticmethod
    def _base_frame_static():
        x1=torch.FloatTensor([-4.2145e-01,  3.7763e+00,0])[None,:]
        x2=torch.FloatTensor([0,0,0])[None,:]
        x3=torch.FloatTensor([3.3910e+00,0,0])[None,:]
        x=torch.cat([x1,x2,x3])
        return x

    def _init_T_static(self):
        StructureModule.trans_init_static = torch.zeros(3)[None,:]
        StructureModule.rot_init_static = torch.eye(3)[None,:,:]

    def pred(self,s_init,z,base_x_template):
        if StructureModule.trans_init_static.device != s_init.device:
            StructureModule.trans_init_static = StructureModule.trans_init_static.to(s_init.device)
        if StructureModule.rot_init_static.device != s_init.device:
            StructureModule.rot_init_static = StructureModule.rot_init_static.to(s_init.device)
        if self.baseframe.device != s_init.device:
            self.baseframe = self.baseframe.to(s_init.device)

        L=s_init.shape[0]
        rot = StructureModule.rot_init_static.repeat(L,1,1)
        trans = StructureModule.trans_init_static.repeat(L,1)
        s = self.layernorm_s(s_init)
        z_normed = self.layernorm_z(z)

        for layer_idx in range(self.N_layer):
            # Assuming self.ipa.forward no longer takes current_debug_identifier
            s_ipa_out = self.ipa(s,z_normed,rot,trans)
            s = s + s_ipa_out
            s = self.transition(s)
            rot_tmp,trans_tmp = self.bbupdate(s,L)
            rot,trans = basic.update_transform(rot_tmp,trans_tmp,rot,trans)

        s_ipa_out_final = self.ipa(s,z_normed,rot,trans)
        s = s + s_ipa_out_final
        s = self.transition(s)
        rot_tmp_final,trans_tmp_final = self.bbupdate(s,L)
        rot,trans = basic.update_transform(rot_tmp_final,trans_tmp_final,rot,trans)
        predx_frame_coords = basic.batch_atom_transform(base_x_template,rot,trans)
        predlogsoft_lddt = self.lddt_preder(rot,trans,s,z_normed)
        plddt_scores = self.lddt_preder.compute_plddt(predlogsoft_lddt)
        return predx_frame_coords, rot, trans, plddt_scores