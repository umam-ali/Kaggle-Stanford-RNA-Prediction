#! /nfs/amino-home/liyangum/miniconda3/bin/python
import numpy
import torch
torch.manual_seed(6) # my lucky number
#torch.set_num_threads(1)
import torch.autograd as autograd
import numpy as np 
np.random.seed(9) # my unlucky number
import random
random.seed(9)
import Cubic,Potential
import operations
import os,json,sys

import a2b,rigid
import torch.optim as opt
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from scipy.optimize import fmin_l_bfgs_b,fmin_cg,fmin_bfgs
from scipy.optimize import minimize
import lbfgs_rosetta
import pickle
Scale_factor=1.0
USEGEO=False





def readconfig(configfile=''):
    config=[]
    expdir=os.path.dirname(os.path.abspath(__file__))
    if configfile=='':
        configfile=os.path.join(expdir,'lib','ddf.json')
    config=json.load(open(configfile,'r'))
    return config 

    
class Structure:
    def __init__(self,fastafile,geofiles,saveprefix,initial_ret,foldconfig):
        self.config=readconfig(foldconfig)
        self.seqfile=fastafile
        self.init_ret = initial_ret
        self.foldconfig = foldconfig
        # try:
        #     self.geos = np.load(geofile,allow_pickle=True).item()
        # excpet:
        #    pass
        #self.geo = np.load(geofile,allow_pickle=True).item()
        self.geofiles = geofiles
        self.rets = [pickle.load(open(refile,'rb')) for refile  in geofiles]
        self.txs=[]
        for ret in self.rets:
            self.txs.append( torch.from_numpy(  ret['coor']   ).double()     )
        self.handle_geo()
        self.pair =[]
        for ret in self.rets:
            self.pair.append( torch.from_numpy(  ret['plddt']   ).double()     )
        self.saveprefix=saveprefix
        self.seq=open(fastafile).readlines()[1].strip()
        self.L=len(self.seq)
        basenpy = np.load( os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','base.npy'  )  )
        self.basex = operations.Get_base(self.seq,basenpy)
        othernpy = np.load( os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','other2.npy'  )  )
        self.otherx = operations.Get_base(self.seq,othernpy)
        sidenpy = np.load( os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','side.npy'  )  )
        self.sidex = operations.Get_base(self.seq,sidenpy)        
        #self.txs=[]
        # for coorfile in coornpys:
        #     self.txs.append( torch.from_numpy(np.load(coorfile)).double()     )
        
        self.init_mask()
        self.init_paras()
        self._init_fape()

        self.local_weight = torch.ones(self.L,self.L) # fape of neighboring pairs control the trosion
        for i in range(self.L):
            for j in range(i+1,min(self.L,i+2)):
                self.local_weight[i,j] = self.local_weight[j,i] = 4
            for j in range(i+2,min(self.L,i+3)):
                self.local_weight[i,j] = self.local_weight[j,i] = 3
            for j in range(i+3,min(self.L,i+4)):
                self.local_weight[i,j] = self.local_weight[j,i] = 2

    def _init_fape(self):
        self.tx2ds=[]
        for tx in self.txs:
            true_rot,true_trans   = operations.Kabsch_rigid(self.basex,tx[:,0],tx[:,1],tx[:,2])
            true_x2 = tx[:,None,:,:] - true_trans[None,:,None,:]
            true_x2 = torch.einsum('ijnd,jde->ijne',true_x2,true_rot.transpose(-1,-2))
            self.tx2ds.append(true_x2)
    def handle_geo(self):
        oldkeys=['dist_p','dist_c','dist_n']
        newkeys=['pp','cc','nn']
        self.geos=[]
        for ret in self.rets:
            geo = {}
            for nk,ok in zip(newkeys,oldkeys):
                geo[nk] = ret[ok].astype(np.float64)  + 0
            self.geos.append(geo)



    def init_mask(self):
        halfmask=np.zeros([self.L,self.L])
        fullmask=np.zeros([self.L,self.L])
        for i in range(self.L):
            for j in range(i+1,self.L):
                halfmask[i,j]=1
                fullmask[i,j]=1
                fullmask[j,i]=1
        self.halfmask=torch.DoubleTensor(halfmask) > 0.5
        self.fullmask=torch.DoubleTensor(fullmask) > 0.5
        self.clash_mask = torch.zeros([self.L,self.L,22,22])
        for i in range(self.L):
            for j in range(i+1,self.L):
                self.clash_mask[i,j]=1
        # for i in range(self.L-1):
        #     self.clash_mask[i,i+1,5,0]=0
        #     self.clash_mask[i,i+1,0,5]=0
        for i in range(self.L):
             self.clash_mask[i,i,:6,7:]=1

        for i in range(self.L-1):
            self.clash_mask[i,i+1,:,0]=0
            self.clash_mask[i,i+1,0,:]=0
            self.clash_mask[i,i+1,:,5]=0
            self.clash_mask[i,i+1,5,:]=0

        self.side_mask = rigid.side_mask(self.seq)
        self.side_mask = self.side_mask[:,None,:,None] * self.side_mask[None,:,None,:]
        self.clash_mask = (self.clash_mask > 0.5) * (self.side_mask > 0.5)

        self.geo_confimask_cc = []
        self.geo_confimask_pp = []
        self.geo_confimask_nn = []
        for geo in self.geos:
            confimask_cc = torch.DoubleTensor(geo['cc'][:,:,-1]) < 0.5
            confimask_pp = torch.DoubleTensor(geo['pp'][:,:,-1]) < 0.5
            confimask_nn = torch.DoubleTensor(geo['nn'][:,:,-1]) < 0.5
            self.geo_confimask_cc.append(confimask_cc)
            self.geo_confimask_pp.append(confimask_pp)
            self.geo_confimask_nn.append(confimask_nn)

        # self.confimask_pccp =  torch.DoubleTensor(self.geo['pccp'][:,:,-1]) < 0.5
        # self.dynamic_pccp   =  self.confimask_pccp*self.halfmask
        # self.pccpi,self.pccpj = torch.where(self.dynamic_pccp > 0.5)
        # self.dynamic_pccp_np=self.dynamic_pccp.numpy()

        # self.confimask_cnnc =  torch.DoubleTensor(self.geo['cnnc'][:,:,-1]) < 0.5
        # self.dynamic_cnnc   =  self.confimask_cnnc*self.halfmask
        # self.cnnci,self.cnncj = torch.where(self.dynamic_cnnc > 0.5)
        # self.dynamic_cnnc_np=self.dynamic_cnnc.numpy()

        # self.confimask_pnnp=  torch.DoubleTensor(self.geo['pnnp'][:,:,-1]) < 0.5
        # self.dynamic_pnnp   =  self.confimask_pnnp*self.halfmask
        # self.pnnpi,self.pnnpj = torch.where(self.dynamic_pnnp > 0.5)
        # self.dynamic_pnnp_np=self.dynamic_pnnp.numpy()


    def init_paras(self):
        self.geo_cc = []
        self.geo_pp = []
        self.geo_nn = []
        for geo in self.geos:
            cc_cs,cc_decs=Cubic.dis_cubic(geo['cc'],2,40,36)
            pp_cs,pp_decs=Cubic.dis_cubic(geo['pp'],2,40,36)
            nn_cs,nn_decs=Cubic.dis_cubic(geo['nn'],2,40,36)
            self.geo_cc.append([cc_cs,cc_decs])
            self.geo_pp.append([pp_cs,pp_decs])
            self.geo_nn.append([nn_cs,nn_decs])

        # self.pccp_cs,self.pccp_decs=Cubic.torsion_cubic(self.geo['pccp'],-math.pi,math.pi,36)
        # self.pccp_cs,self.pccp_decs=self.pccp_cs[self.dynamic_pccp_np],self.pccp_decs[self.dynamic_pccp_np]
        # self.pccp_coe=torch.DoubleTensor(np.array([acs.c for acs in self.pccp_cs]))
        # self.pccp_x = torch.DoubleTensor(np.array([acs.x for acs in self.pccp_cs]))

        # self.cnnc_cs,self.cnnc_decs=Cubic.torsion_cubic(self.geo['cnnc'],-math.pi,math.pi,36)
        # self.cnnc_cs,self.cnnc_decs=self.cnnc_cs[self.dynamic_cnnc_np],self.cnnc_decs[self.dynamic_cnnc_np]
        # self.cnnc_coe=torch.DoubleTensor(np.array([acs.c for acs in self.cnnc_cs]))
        # self.cnnc_x = torch.DoubleTensor(np.array([acs.x for acs in self.cnnc_cs]))


        # self.pnnp_cs,self.pnnp_decs=Cubic.torsion_cubic(self.geo['pnnp'],-math.pi,math.pi,36)
        # self.pnnp_cs,self.pnnp_decs=self.pnnp_cs[self.dynamic_pnnp_np],self.pnnp_decs[self.dynamic_pnnp_np]
        # self.pnnp_coe=torch.DoubleTensor(np.array([acs.c for acs in self.pnnp_cs]))
        # self.pnnp_x = torch.DoubleTensor(np.array([acs.x for acs in self.pnnp_cs]))
     

    def init_quat(self,ii):
        x = torch.rand([self.L,21])
        x[:,18:] = self.txs[ii].mean(dim=1)
        init_coor = self.txs[ii] 
        biasq=torch.mean(init_coor,dim=1,keepdim=True)
        q=init_coor-biasq
        m = torch.einsum('bnz,bny->bzy',self.basex,q).reshape([self.L,-1])

        x[:,:9] = x[:,9:18] = m
        x.requires_grad=True
        return x
        
    def init_quat_safe(self,ii):
        x = torch.rand([self.L,21])
        x[:,18:] = self.txs[ii].mean(dim=1)
        init_coor = self.txs[ii] 
        biasq=torch.mean(init_coor,dim=1,keepdim=True)
        q=init_coor-biasq + torch.rand([self.L,3,3])
        m = (torch.einsum('bnz,bny->bzy',self.basex,q) + torch.eye(3)[None,:,:])  .reshape([self.L,-1])

        x[:,:9] = x[:,9:18] = m
        x.requires_grad=True
        return x

    def compute_bb_clash(self,coor,other_coor):
        com_coor = torch.cat([coor,other_coor],dim=1)
        com_dis  = (com_coor[:,None,:,None,:] - com_coor[None,:,None,:,:]).norm(dim=-1)
        dynamicmask2_vdw= (com_dis <= 3.15) * (self.clash_mask)
        #vdw_dynamic=torch.nn.functional.softplus(3.15-com_dis[dynamicmask2_vdw])
        vdw_dynamic = Potential.LJpotential(com_dis[dynamicmask2_vdw],3.15)
        return vdw_dynamic.sum()*self.config['weight_vdw']

    def compute_full_clash(self,coor,other_coor,side_coor):
        com_coor = torch.cat([coor[:,:2],other_coor,side_coor],dim=1)
        com_dis  = (com_coor[:,None,:,None,:] - com_coor[None,:,None,:,:]).norm(dim=-1)
        dynamicmask2_vdw= (com_dis <= 2.5) * (self.clash_mask)
        #vdw_dynamic=torch.nn.functional.softplus(3.15-com_dis[dynamicmask2_vdw])
        vdw_dynamic = Potential.LJpotential(com_dis[dynamicmask2_vdw],2.5)
        return vdw_dynamic.sum()*self.config['weight_vdw']




    def compute_cc_energy(self, coor):
        min_dis, max_dis, bin_num = 2, 40, 36  # These are not used by the new Potential.cubic_distance
        c_atoms = coor[:, 1]
        # upper_th is used for dynamicmask_cc, so it's still relevant for selecting which distances to score
        upper_th = max_dis - ((max_dis - min_dis) / bin_num) * 0.5 
        # lower_th = 3.10 # Not directly used in this snippet, but was in original context

        cc_map = operations.pair_distance(c_atoms, c_atoms) # Assuming this returns a Tensor

        total_ecb_for_all_geos = torch.tensor(0.0, device=coor.device, dtype=coor.dtype)

        for geo_splines_data, confimask_cc_tensor in zip(self.geo_cc, self.geo_confimask_cc):
            # cc_cs_obj_array is the 2D numpy array of interp1d callable objects
            cc_cs_obj_array = geo_splines_data[0] # Assuming the callables are the first element

            # Ensure confimask_cc_tensor is on the same device as cc_map for boolean operations
            confimask_cc_tensor = confimask_cc_tensor.to(cc_map.device)
            
            dynamicmask_cc = (cc_map <= upper_th) & \
                             (confimask_cc_tensor) & \
                             (self.fullmask.to(cc_map.device)) & \
                             (cc_map >= 2.5)
            
            # This is a PyTorch tensor mask
            # dynamicmask_cc_np = dynamicmask_cc.cpu().numpy() # Use PyTorch mask directly if possible

            # Initialize E_cb for this specific geo data
            E_cb_current_geo = torch.tensor(0.0, device=coor.device, dtype=coor.dtype)

            # Get distances and corresponding splines
            distances_to_evaluate = cc_map[dynamicmask_cc]

            if distances_to_evaluate.numel() > 0:
                # cc_cs_obj_array is a 2D NumPy array of interp1d objects.
                # We need to select the interp1d objects corresponding to the True values in dynamicmask_cc.
                # It's often easier to work with NumPy for indexing arrays of Python objects.
                dynamicmask_cc_np_for_spline_selection = dynamicmask_cc.cpu().numpy()
                
                # This flattens the 2D array of splines and then selects using the flattened mask
                flat_spline_callables = list(cc_cs_obj_array[dynamicmask_cc_np_for_spline_selection])

                if flat_spline_callables:
                    # Potential.cubic_distance now expects a list of callables
                    spline_energies = Potential.cubic_distance(
                        distances_to_evaluate, 
                        flat_spline_callables, 
                        min_dis, max_dis, bin_num  # These are now ignored by the Potential.cubic_distance
                    )
                    E_cb_current_geo = spline_energies.sum() * self.config['weight_cc'] * 0.5
                # else: E_cb_current_geo remains 0
            
            # Penalty term for very short distances (applied per geo data)
            penalty_factor = torch.tensor(5.0, device=coor.device, dtype=coor.dtype) # Ensure tensor
            penalty_term = ((cc_map <= 2.5) & \
                            (self.fullmask.to(cc_map.device)) & \
                            (confimask_cc_tensor)
                           ).sum() * penalty_factor * self.config['weight_cc']
            
            E_cb_current_geo = E_cb_current_geo + penalty_term
            total_ecb_for_all_geos = total_ecb_for_all_geos + E_cb_current_geo
            
        return total_ecb_for_all_geos

    # You would define compute_pp_energy and compute_nn_energy similarly
    def compute_pp_energy(self, coor):
        min_dis, max_dis, bin_num = 2, 40, 36
        p_atoms = coor[:, 0] # P atoms
        upper_th = max_dis - ((max_dis - min_dis) / bin_num) * 0.5
        
        pp_map = operations.pair_distance(p_atoms, p_atoms)

        total_epb_for_all_geos = torch.tensor(0.0, device=coor.device, dtype=coor.dtype)

        # Assuming self.geo_pp and self.geo_confimask_pp are structured like their cc counterparts
        for geo_splines_data, confimask_pp_tensor in zip(self.geo_pp, self.geo_confimask_pp):
            pp_cs_obj_array = geo_splines_data[0]
            confimask_pp_tensor = confimask_pp_tensor.to(pp_map.device)

            dynamicmask_pp = (pp_map <= upper_th) & \
                             (confimask_pp_tensor) & \
                             (self.fullmask.to(pp_map.device)) & \
                             (pp_map >= 2.5)

            E_pb_current_geo = torch.tensor(0.0, device=coor.device, dtype=coor.dtype)
            distances_to_evaluate = pp_map[dynamicmask_pp]

            if distances_to_evaluate.numel() > 0:
                dynamicmask_pp_np_for_spline_selection = dynamicmask_pp.cpu().numpy()
                flat_spline_callables = list(pp_cs_obj_array[dynamicmask_pp_np_for_spline_selection])

                if flat_spline_callables:
                    spline_energies = Potential.cubic_distance(
                        distances_to_evaluate,
                        flat_spline_callables,
                        min_dis, max_dis, bin_num
                    )
                    E_pb_current_geo = spline_energies.sum() * self.config['weight_pp'] * 0.5
            
            penalty_factor = torch.tensor(5.0, device=coor.device, dtype=coor.dtype)
            penalty_term = ((pp_map <= 2.5) & \
                            (self.fullmask.to(pp_map.device)) & \
                            (confimask_pp_tensor)
                           ).sum() * penalty_factor * self.config['weight_pp']
            
            E_pb_current_geo = E_pb_current_geo + penalty_term
            total_epb_for_all_geos = total_epb_for_all_geos + E_pb_current_geo
            
        return total_epb_for_all_geos

    def compute_nn_energy(self, coor):
        min_dis, max_dis, bin_num = 2, 40, 36
        n_atoms = coor[:, -1] # N atoms (assuming last atom in the backbone representation)
        upper_th = max_dis - ((max_dis - min_dis) / bin_num) * 0.5
        
        nn_map = operations.pair_distance(n_atoms, n_atoms)

        total_enb_for_all_geos = torch.tensor(0.0, device=coor.device, dtype=coor.dtype)

        # Assuming self.geo_nn and self.geo_confimask_nn are structured like their cc counterparts
        for geo_splines_data, confimask_nn_tensor in zip(self.geo_nn, self.geo_confimask_nn):
            nn_cs_obj_array = geo_splines_data[0]
            confimask_nn_tensor = confimask_nn_tensor.to(nn_map.device)

            dynamicmask_nn = (nn_map <= upper_th) & \
                             (confimask_nn_tensor) & \
                             (self.fullmask.to(nn_map.device)) & \
                             (nn_map >= 2.5)

            E_nb_current_geo = torch.tensor(0.0, device=coor.device, dtype=coor.dtype)
            distances_to_evaluate = nn_map[dynamicmask_nn]

            if distances_to_evaluate.numel() > 0:
                dynamicmask_nn_np_for_spline_selection = dynamicmask_nn.cpu().numpy()
                flat_spline_callables = list(nn_cs_obj_array[dynamicmask_nn_np_for_spline_selection])

                if flat_spline_callables:
                    spline_energies = Potential.cubic_distance(
                        distances_to_evaluate,
                        flat_spline_callables,
                        min_dis, max_dis, bin_num
                    )
                    E_nb_current_geo = spline_energies.sum() * self.config['weight_nn'] * 0.5
            
            penalty_factor = torch.tensor(5.0, device=coor.device, dtype=coor.dtype)
            penalty_term = ((nn_map <= 2.5) & \
                            (self.fullmask.to(nn_map.device)) & \
                            (confimask_nn_tensor)
                           ).sum() * penalty_factor * self.config['weight_nn']
            
            E_nb_current_geo = E_nb_current_geo + penalty_term
            total_enb_for_all_geos = total_enb_for_all_geos + E_nb_current_geo
            
        return total_enb_for_all_geos
    def compute_pccp_energy(self,coor):
        p_atoms=coor[:,0]
        c_atoms=coor[:,1]
        pccpmap=operations.dihedral( p_atoms[self.pccpi], c_atoms[self.pccpi], c_atoms[self.pccpj] ,p_atoms[self.pccpj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.pccp_coe,self.pccp_x,36)
        return neg_log.sum()*self.config['weight_pccp']

    def compute_cnnc_energy(self,coor):
        n_atoms=coor[:,-1]
        c_atoms=coor[:,1]
        pccpmap=operations.dihedral( c_atoms[self.cnnci], n_atoms[self.cnnci], n_atoms[self.cnncj] ,c_atoms[self.cnncj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.cnnc_coe,self.cnnc_x,36)
        return neg_log.sum()*self.config['weight_cnnc']

    def compute_pnnp_energy(self,coor):
        n_atoms=coor[:,-1]
        p_atoms=coor[:,0]
        pccpmap=operations.dihedral( p_atoms[self.pnnpi], n_atoms[self.pnnpi], n_atoms[self.pnnpj] ,p_atoms[self.pnnpj]                  )
        neg_log = Potential.cubic_torsion(pccpmap,self.pnnp_coe,self.pnnp_x,36)
        return neg_log.sum()*self.config['weight_pnnp']

    def compute_pcc_energy(self,coor):
        p_atoms=coor[:,1]
        c_atoms=coor[:,2]
        pccmap=operations.angle( p_atoms[self.pcci], c_atoms[self.pcci], c_atoms[self.pccj]                   )
        neg_log = Potential.cubic_angle(pccmap,self.pcc_coe,self.pcc_x,12)
        return neg_log.sum()*self.config['weight_pcc']

    def compute_fape_energy(self,coor,ep=1e-3,epmax=20):
        energy= 0
        for tx in self.tx2ds:
            px_mean = coor[:,[1]]
            p_rot   = operations.rigidFrom3Points(coor)
            p_tran  = px_mean[:,0]
            pred_x2 = coor[:,None,:,:] - p_tran[None,:,None,:] # Lx Lrot N , 3
            pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep )
            energy = energy + torch.sum(  torch.clamp(errmap,max=epmax)        )
        return energy * self.config['weight_fape']

    def compute_bond_energy(self,coor,other_coor):
        # 3.87
        o3 = other_coor[:-1,-2]
        p  = coor[1:,0]
        dis = (o3-p).norm(dim=-1)
        energy = ((dis-1.607)**2).sum()
        return energy * self.config['weight_bond']

    def tooth_func(self,errmap, ep = 0.05):
        return -1/(errmap/10+ep) + (1/ep)
    def reweight_func(self,ww):
        reweighting = torch.pow(ww,self.config['pair_weight_power'])
        reweighting[ww < self.config['pair_weight_min']] = 0
        return reweighting
    def compute_fape_energy_fromquat(self,x,coor,ep=1e-6,epmax=100):
        energy= 0
        p_rot,px_mean = a2b.Non2rot(x[:,:9],x.shape[0]),x[:,9:]
        pred_x2 = coor[:,None,:,:] - px_mean[None,:,None,:] # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
        #coor  = a2b.quat2b(x)
        for tx,weightplddt in zip(self.tx2ds,self.pair):
            # px_mean = coor[:,[1]]
            # p_rot   = operations.rigidFrom3Points(coor)
            # p_tran  = px_mean[:,0]

            tamplate_dist_map = torch.min( tx.norm(dim=-1), dim=2   )[0]
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep ) 
            #energy = energy + torch.sum(  torch.clamp(errmap,max=epmax) * self.reweight_func(weightplddt[...,None])    )
            #energy = energy + torch.sum(  self.tooth_func(errmap) * weightplddt[...,None]    )
            energy = energy + torch.sum( ( (torch.clamp(errmap,max=self.config['FAPE_max'])**self.config['pair_error_power'])  * self.reweight_func(weightplddt[...,None]) * self.local_weight[...,None] )[tamplate_dist_map>self.config['pair_rest_min_dist']]    )

        return energy * self.config['weight_fape']
    def energy(self,rama):
        #rama=torch.cat([rama,self.betas],dim=-1)
        
        coor=a2b.quat2b(self.basex,rama[:,9:])
        other_coor = a2b.quat2b(self.otherx,rama[:,9:])
        side_coor = a2b.quat2b(self.sidex,torch.cat([rama[:,:9],coor[:,-1]],dim=-1))

        #print(coor.shape,other_coor.shape,side_coor.shape)
        
        if self.config['weight_cc']>0:
            E_cc= self.compute_cc_energy(coor) / len(self.rets)
        else:
            E_cc=0
        if self.config['weight_pp']>0:
            E_pp= self.compute_pp_energy(coor) / len(self.rets)
        else:
            E_pp=0
        if self.config['weight_nn']>0:
            E_nn= self.compute_nn_energy(coor) / len(self.rets)
        else:
            E_nn=0

        if self.config['weight_pccp']>0:
            E_pccp= self.compute_pccp_energy(coor) / len(self.rets)
        else:
            E_pccp=0

        if self.config['weight_cnnc']>0:
            E_cnnc= self.compute_cnnc_energy(coor)  / len(self.rets)
        else:
            E_cnnc=0

        if self.config['weight_pnnp']>0:
            E_pnnp= self.compute_pnnp_energy(coor) / len(self.rets)
        else:
            E_pnnp=0

        if self.config['weight_vdw']>0:
            E_vdw= self.compute_full_clash(coor,other_coor,side_coor)
        else:
            E_vdw=0

        if self.config['weight_fape']>0:
            E_fape= self.compute_fape_energy_fromquat(rama[:,9:],coor) / len(self.rets)
        else:
            E_fape=0
        if self.config['weight_bond']>0:
            E_bond= self.compute_bond_energy(coor,other_coor)
        else:
            E_bond=0
        return  E_vdw + E_fape + E_bond + E_pp + E_cc + E_nn + E_pccp + E_cnnc + E_pnnp

    def energy1(self,rama):
        #rama=torch.cat([rama,self.betas],dim=-1)
        
        coor=a2b.quat2b(self.basex,rama[:,9:])
        other_coor = a2b.quat2b(self.otherx,rama[:,9:])
        side_coor = a2b.quat2b(self.sidex,torch.cat([rama[:,:9],coor[:,-1]],dim=-1))

        #print(coor.shape,other_coor.shape,side_coor.shape)
        
        if self.config['weight_cc']>0:
            E_cc= self.compute_cc_energy(coor) / len(self.rets)
        else:
            E_cc=0
        if self.config['weight_pp']>0:
            E_pp= self.compute_pp_energy(coor) / len(self.rets)
        else:
            E_pp=0
        if self.config['weight_nn']>0:
            E_nn= self.compute_nn_energy(coor) / len(self.rets)
        else:
            E_nn=0

        if self.config['weight_fape']>0:
            E_fape= self.compute_fape_energy_fromquat(rama[:,9:],coor) / len(self.rets)
        else:
            E_fape=0

        return  E_fape  + E_pp + E_cc + E_nn 


    def obj_func_grad_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama.requires_grad=True
        if rama.grad:
            rama.grad.zero_()
        f=self.energy(rama.view(self.L,21))*Scale_factor
        grad_value=autograd.grad(f,rama)[0]
        return grad_value.data.numpy().astype(np.float64)
    def obj_func_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama=rama.view(self.L,21)
        with torch.no_grad():
            f=self.energy(rama)*Scale_factor
            #print('score',f)
            return f.item()
    def obj_func_np1(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama=rama.view(self.L,21)
        with torch.no_grad():
            f=self.energy1(rama)
            print("score",f)
            f=f*Scale_factor
            return f.item()


    def foldning(self,best_key):
        minenergy=best_key
        count=0
        for tx in self.txs:
            #self.outpdb_coor(tx,self.saveprefix+f'.{str(count)}.pdb',energystr=str(0))
            count+=1
        minirama=None
        if True:

        #for ilter in range( len(self.txs)):
            ilter = self.init_ret
            selected_ret = self.geofiles[ilter]
            try:
                rama=self.init_quat(ilter).data.numpy()
                #self.outpdb(torch.DoubleTensor(rama) ,self.saveprefix+f'_{str(ilter)}'+'.pdb',energystr='')
                # self.config=readconfig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','vdw.json'))
                # rama = fmin_l_bfgs_b(func=self.obj_func_np, x0=rama,  fprime=self.obj_func_grad_np,iprint=10)[0]
                # rama = rama.flatten()
            except:
                rama=self.init_quat_safe(ilter).data.numpy()
                #self.outpdb(torch.DoubleTensor(rama) ,self.saveprefix+f'_init'+str(ilter)+'.pdb',energystr='')
            
            for i in range(1):
                self.config=readconfig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','vdw.json'))
                rama = fmin_l_bfgs_b(func=self.obj_func_np, x0=rama,  fprime=self.obj_func_grad_np,iprint=10)[0]
                rama = rama.flatten()
                    
                self.config=readconfig(self.foldconfig)
                geoscale = self.config['geo_scale']
                self.config['weight_pp'] =geoscale * self.config['weight_pp']
                self.config['weight_cc'] =geoscale * self.config['weight_cc']
                self.config['weight_nn'] =geoscale * self.config['weight_nn']
                self.config['weight_pccp'] =geoscale * self.config['weight_pccp']
                self.config['weight_cnnc'] =geoscale * self.config['weight_cnnc']
                self.config['weight_pnnp'] =geoscale * self.config['weight_pnnp']
                for i in range(1):
                    rama = fmin_l_bfgs_b(func=self.obj_func_np, x0=rama, fprime=self.obj_func_grad_np,maxfun=150, maxiter=50)[0]
                    # Add maxfun and/or maxiter. The default iprint=10 means it prints progress every 10 iterations.
                    line_min = lbfgs_rosetta.ArmijoLineMinimization(self.obj_func_np,self.obj_func_grad_np,True,len(rama),120)
                    lbfgs_opt = lbfgs_rosetta.lbfgs(self.obj_func_np,self.obj_func_grad_np)
                    rama=lbfgs_opt.run(rama,256,lbfgs_rosetta.absolute_converge_test,line_min,250,self.obj_func_np,self.obj_func_grad_np,1e-9)
                    # Example of modifying a call in Structure.foldning()
                    rama = fmin_l_bfgs_b(func=self.obj_func_np, x0=rama, fprime=self.obj_func_grad_np,maxfun=150, maxiter=50)[0]
                    # Add maxfun and/or maxiter. The default iprint=10 means it prints progress every 10 iterations.

                newrama=rama+0.0
                newrama=torch.DoubleTensor(newrama) 
                current_energy =self.obj_func_np1(rama)
                #self.outpdb(newrama,self.saveprefix+f'_{str(ilter)}'+'.pdb',energystr=str(current_energy))
                if current_energy < minenergy:
                    print(current_energy,minenergy)
                    minenergy=current_energy
                    self.outpdb(newrama,self.saveprefix+'.pdb',energystr=str(current_energy))
                else:
                    print("energy not decreased")


    def outpdb(self,rama,savefile,start=0,end=10000,energystr=''):
        #rama=torch.cat([rama.view(self.L,2),self.betas],dim=-1)
        coor_np=a2b.quat2b(self.basex,rama.view(self.L,21)[:,9:]).data.numpy()
        other_np=a2b.quat2b(self.otherx,rama.view(self.L,21)[:,9:]).data.numpy()
        shaped_rama=rama.view(self.L,21)
        coor = torch.FloatTensor(coor_np)
        side_coor_NP = a2b.quat2b(self.sidex,torch.cat([shaped_rama[:,:9],coor[:,-1]],dim=-1)).data.numpy()
        
        Atom_name=[' P  '," C4'",' N1 ']
        Other_Atom_name = [" O5'"," C5'"," C3'"," O3'"," C1'"]
        other_last_name = ['O',"C","C","O","C"]

        side_atoms=         [' N1 ',' C2 ',' O2 ',' N2 ',' N3 ',' N4 ',' C4 ',' O4 ',' C5 ',' C6 ',' O6 ',' N6 ',' N7 ',' N8 ',' N9 ']
        side_last_name =    ['N',      "C",   "O",   "N",   "N",   'N',   'C',   'O',   'C',   'C',   'O',   'N',    'N', 'N','N']

        base_dict = rigid.base_table()
        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']
                #atoms = ['P','C4']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']
            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                #outs=('ATOM  ',count,Atom_name[j],'ALA','A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],1.0,90,last_name[j],'')
                #print(outs)
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1

            for j in range(other_np.shape[1]):
                outs=('ATOM  ',count,Other_Atom_name[j],self.seq[i],'A',i+1,other_np[i][j][0],other_np[i][j][1],other_np[i][j][2],0,0,other_last_name[j],'')
                #outs=('ATOM  ',count,Atom_name[j],'ALA','A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],1.0,90,last_name[j],'')
                #print(outs)
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1



                #count+=1
            
        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()
    def outpdb_coor(self,coor_np,savefile,start=0,end=1000,energystr=''):
        #rama=torch.cat([rama.view(self.L,2),self.betas],dim=-1)
        Atom_name=[' P  '," C4'",' N1 ']
        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']
                #atoms = ['P','C4']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']
            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                #outs=('ATOM  ',count,Atom_name[j],'ALA','A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],1.0,90,last_name[j],'')
                #print(outs)
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                count+=1
            
        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()

if __name__ == '__main__': 

    fastafile=sys.argv[1]
    saveprefix=sys.argv[2]
    retdirs  =sys.argv[3]
    ret_score = sys.argv[4]
    foldconfig = sys.argv[5]

    savepare = os.path.dirname(saveprefix)
    if not os.path.isdir(savepare):
        os.makedirs(savepare)

    num_of_models = readconfig(foldconfig)['num_of_models']
    #coornpys =sys.argv[4:]

    score_dict = readconfig(ret_score)
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1])
    lowest_n_keys = [item[0] for item in sorted_items][:num_of_models]
    bestkey = lowest_n_keys[0] + ''
    lowest_n_keys.sort()
    bestindex = lowest_n_keys.index(bestkey)
    initial_energy_for_foldning = score_dict[bestkey] # This is your 'x'
    print(initial_energy_for_foldning)



    current_ret = bestkey
    retfiles = [os.path.join(retdirs,afile) for afile in lowest_n_keys]
    stru=Structure(fastafile,retfiles,saveprefix+'_from_'+current_ret,bestindex,foldconfig)

    
    stru.foldning(initial_energy_for_foldning)
