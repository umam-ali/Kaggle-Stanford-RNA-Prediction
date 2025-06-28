from numpy import select
import torch
from torch import nn
from torch.nn import functional as F

import basic,Structure,Evoformer,EvoPair,EvoMSA
import math,sys,os
from torch.utils.checkpoint import checkpoint
import numpy as np
import EvoPair


from RNALM2 import Model
device = sys.argv[1]
expdir=os.path.dirname(os.path.abspath(__file__))
_DRFOLD2_ROOT_FOR_CACHE = r"C:\Users\Umam\DRFOLD2" # Or get from env var if set by main
_PREMSA_CACHE_PATH = os.path.join(_DRFOLD2_ROOT_FOR_CACHE, "runtime_cache_v2", "premsa_internals")
_MAX_LEN_FOR_PREMSA_INTERNAL_CACHING = 480 # Needs to be consistent with main script logic
os.makedirs(_PREMSA_CACHE_PATH, exist_ok=True)
RNAlm=None
_PRECOMPUTED_ASSET_DIR = os.path.join(r"C:\Users\Umam\DRfold2", "model_assets_precomputed")
_MAX_L_FOR_PREMSA_STATIC = 2000 # Must match generation script

# Attempt to load at module import time, fall back to compute if files not found
try:
    _CACHED_PREMSA_POS = torch.load(os.path.join(_PRECOMPUTED_ASSET_DIR, f"static_premsa_pos_L{_MAX_L_FOR_PREMSA_STATIC}.pt"), map_location='cpu')
    _CACHED_PREMSA_APOS = torch.load(os.path.join(_PRECOMPUTED_ASSET_DIR, f"static_premsa_apos_L{_MAX_L_FOR_PREMSA_STATIC}.pt"), map_location='cpu')
    print(f"[EvoMSA2XYZ Module] Successfully loaded precomputed PreMSA pos/apos tensors.")
except Exception as e:
    print(f"[EvoMSA2XYZ Module WARNING] Could not load precomputed PreMSA pos/apos tensors: {e}. Will recompute them if PreMSA is used.")
    # Define the compute functions here as a fallback if needed, or ensure PreMSA can compute them
    def _fallback_compute_pos(maxL=_MAX_L_FOR_PREMSA_STATIC):
        a = torch.arange(maxL); b = (a[None,:]-a[:,None]).clamp(-32,32)
        return F.one_hot(b+32,65).float()
    def _fallback_compute_apos(maxL=_MAX_L_FOR_PREMSA_STATIC):
        d_r = torch.arange(maxL); m_b = 14
        return (((d_r[:,None] & (1 << torch.arange(m_b)))) > 0).float()
    _CACHED_PREMSA_POS = _fallback_compute_pos()
    _CACHED_PREMSA_APOS = _fallback_compute_apos()


lmaadic = {
            'A':0,'G':1,'C':2,'U':3,'a':0,'g':1,'c':2,'u':3,'T':3,'t':3,'-':4
            } 
def one_d(idx_, d, max_len=2056*8):
    idx = idx_[None]
    K = torch.arange(d//2).to(idx.device)
    sin_e = torch.sin(idx[..., None] * math.pi / (max_len**(2*K[None]/d))).to(idx.device)
    cos_e = torch.cos(idx[..., None] * math.pi / (max_len**(2*K[None]/d))).to(idx.device)
    return torch.cat([sin_e, cos_e], axis=-1)[0]

def emb_alphas(alphas,seq_idx):
    data = [
        ("",' '.join(alphas)),
    ] 
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract embeddings (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.to(device),repr_layers=[12])
    token_embeddings = results["representations"][12][0,1:-1]
    return token_embeddings
def rnalm_alphas(alphas,seq_idx):
    global RNAlm # Explicitly state we are using the module-level RNAlm
    if RNAlm is None:
        raise RuntimeError(
            "RNAlm instance has not been set/patched in EvoMSA2XYZ module. "
            "The main script should patch EvoMSA2XYZ.RNAlm with the global instance."
        )
    seq  = ''.join(alphas)
    seqnpy=np.zeros(len(seq),dtype=int) + lmaadic['-']
    seq1=np.array(list(seq))  
    keys = list(lmaadic.keys())
    for akey in keys:
        seqnpy[seq1==akey] = lmaadic[akey]
    seqnpy = np.eye(lmaadic['-']+1)[seqnpy]
    model_device = next(RNAlm.parameters()).device

    fea  = {'aa':torch.FloatTensor(seqnpy).to(model_device),
            'idx':seq_idx.to(model_device), # Ensure seq_idx is also moved
            'mask':torch.zeros(len(seq), dtype=torch.float).to(model_device) }
    with torch.no_grad():
        lms,lmz = RNAlm.embedding(fea)
    print(f"lms{lms.sum().item()}")
    return lms,lmz

def batch_emb_alphas(alphas,seq_idx):
    L = len(alphas)
    data = [(str(i),' '.join(alphas)) for i in range(L)] 

    batch_labels, batch_strs, batch_tokenss = batch_converter(data)
    for i in range(L):
        batch_tokenss[i,i+1] = 24

    # Extract embeddings (on CPU)
    token_embeddings=[]
    batch_tokenss = batch_tokenss.to(device)
    with torch.no_grad():
        for batch_tokens in batch_tokenss:
            results = model(batch_tokens,repr_layers=[12])
            token_embeddings.append(results["representations"][12][0,1:-1])
    return torch.stack(token_embeddings)

class PreMSA(nn.Module):
    def __init__(self,seq_dim,msa_dim,m_dim,z_dim):
        super(PreMSA,self).__init__()
        self.msalinear=basic.Linear(msa_dim,m_dim)
        self.qlinear  =basic.Linear(seq_dim,z_dim)
        self.klinear  =basic.Linear(seq_dim,z_dim)
        self.slinear  =basic.Linear(seq_dim,m_dim)
        global _CACHED_PREMSA_POS, _CACHED_PREMSA_APOS
        if _CACHED_PREMSA_POS is None or _CACHED_PREMSA_APOS is None:
            raise RuntimeError("PreMSA static pos/apos tensors are not available.")
        self.pos = _CACHED_PREMSA_POS
        self.pos1d = _CACHED_PREMSA_APOS
        self.poslinear=basic.Linear(64,z_dim)
        self.poslinear2=basic.Linear(64,m_dim)

        self.fm_layer = basic.Linear(640,m_dim)
        self.lm_layer_s = basic.Linear(512,m_dim)
        self.lm_layer_z = basic.Linear(128,z_dim)
    def tocuda(self,device):
        self.to(device)
        self.pos.to(device)
    def compute_apos(self,maxL=2000):
        d = torch.arange(maxL)
        m = 14
        d =(((d[:,None] & (1 << np.arange(m)))) > 0).float()
        return d

    def compute_pos(self,maxL=2000):
        a = torch.arange(maxL)
        b = (a[None,:]-a[:,None]).clamp(-32,32)
        return F.one_hot(b+32,65)


    def _get_rnalm_embeddings_via_cache_or_live(self, alphas_list, idx_tensor, current_process_device):
        seq_str = "".join(alphas_list)
        L = len(seq_str)
        
        if L > _MAX_LEN_FOR_PREMSA_INTERNAL_CACHING: # Check length against module-level const
            return rnalm_alphas(alphas_list, idx_tensor) # Call existing module-level function

        import hashlib # Ensure imported
        seq_hash = hashlib.md5(seq_str.encode()).hexdigest()
        s_lm_file = os.path.join(_PREMSA_CACHE_PATH, f"rnalm_L{L}_{seq_hash}_slm.pt") # Use module-level const
        z_lm_file = os.path.join(_PREMSA_CACHE_PATH, f"rnalm_L{L}_{seq_hash}_zlm.pt")

        if os.path.exists(s_lm_file) and os.path.exists(z_lm_file):
            try:
                s_lm = torch.load(s_lm_file, map_location=current_process_device)
                z_lm = torch.load(z_lm_file, map_location=current_process_device)
                return s_lm, z_lm
            except Exception as e:
                print(f"    [PreMSA Cache Error] Load failed for RNAlm L{L} key {seq_hash}: {e}. Recomputing.")
        
        s_lm, z_lm = rnalm_alphas(alphas_list, idx_tensor) # Call module-level function for live compute

        # Ensure outputs are on the correct device before saving/using if rnalm_alphas doesn't guarantee it
        # (Though my rnalm_alphas example does try to put output on RNAlm's device)
        s_lm = s_lm.to(current_process_device)
        z_lm = z_lm.to(current_process_device)

        try:
            torch.save(s_lm, s_lm_file)
            torch.save(z_lm, z_lm_file)
        except Exception as e_save:
            print(f"    [PreMSA Cache Error] Save failed for RNAlm L{L} key {seq_hash}: {e_save}")
        return s_lm, z_lm

    # Caching helper for one_d (similar structure)
    def _get_one_d_via_cache_or_live(self, input_tensor, d_val, cache_name_prefix, L_for_key, current_process_device):
        # global one_d # Make sure one_d function is accessible
        if L_for_key > _MAX_LEN_FOR_PREMSA_INTERNAL_CACHING:
            return one_d(input_tensor, d_val) # Call module-level one_d

        # (Cache key generation, load, compute, save logic as before for one_d)
        # For compute step: output = one_d(input_tensor, d_val)
        # ... (implementation similar to _get_rnalm_embeddings_via_cache_or_live)
        import hashlib
        input_hash = hashlib.md5(str(input_tensor.sum().item()).encode() + str(input_tensor.shape).encode()).hexdigest()
        cache_key = f"oned_{cache_name_prefix}_L{L_for_key}_d{d_val}_{input_hash}"
        cache_file = os.path.join(_PREMSA_CACHE_PATH, f"{cache_key}.pt")

        if os.path.exists(cache_file):
            try: 
                return torch.load(cache_file, map_location=current_process_device)
            except Exception as e: print(f"    [PreMSA Cache Error] Load failed for one_d {cache_key}: {e}. Recomputing.")
        
        output = one_d(input_tensor, d_val) # Call module-level one_d
        output = output.to(current_process_device) # Ensure device
        try: torch.save(output, cache_file)
        except Exception as e_save: print(f"    [PreMSA Cache Error] Save failed for one_d {cache_key}: {e_save}")
        return output


    def forward(self, seq, msa, idx, alphas): # alphas is list of chars, idx is tensor

        device = msa.device 
        L = idx.shape[0]

        s_lm, z_lm = self._get_rnalm_embeddings_via_cache_or_live(alphas, idx, device)
        one_d_idx_output = self._get_one_d_via_cache_or_live(idx, 64, "idxpos", L, device)
        p = self.poslinear2(one_d_idx_output)

        seq_idx_expanded = idx[None]
        relative_pos_diff = seq_idx_expanded[:, :, None] - seq_idx_expanded[:, None, :]
        relative_pos_flat = relative_pos_diff.reshape([1, L*L])
        relative_pos_encoded = self._get_one_d_via_cache_or_live(relative_pos_flat, 64, "relpos", L, device)

        # --- Rest of PreMSA.forward logic using s_lm, z_lm, p, relative_pos_encoded ---
        if self.pos.device != device: self.pos = self.pos.to(device)
        if self.pos1d.device != device: self.pos1d = self.pos1d.to(device)

        s_target = self.slinear(seq)
        m_from_msa = self.msalinear(msa)
        m = m_from_msa + s_target[None,:,:] + p[None,:,:]
        m = m + self.lm_layer_s(s_lm)[None]

        sq_target = self.qlinear(seq)
        sk_target = self.klinear(seq)
        z_pair_val = sq_target[None,:,:] + sk_target[:,None,:]
        z_pair_val = z_pair_val + self.poslinear(relative_pos_encoded.reshape([1, L, L, -1])[0])
        z_pair_val = z_pair_val + self.lm_layer_z(z_lm)
        return m, z_pair_val
    
def fourier_encode_dist(x, num_encodings = 20, include_self = True):
    # from https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch.py
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x






class RecyclingEmbedder(nn.Module):
    def __init__(self,m_dim,z_dim,dis_encoding_dim):
        super(RecyclingEmbedder,self).__init__()  
        self.linear = basic.Linear(dis_encoding_dim*2+1,z_dim)
        self.dis_encoding_dim=dis_encoding_dim
        self.normz = nn.LayerNorm(z_dim)
        self.normm = nn.LayerNorm(m_dim)
        self.dist_linear = nn.Sequential(
                        nn.Linear(36+2, z_dim),
                        nn.ReLU(),
                        nn.Linear(z_dim, z_dim),
                    )
        self.hb_linear = nn.Sequential(
                        nn.Linear(6, z_dim),
                        nn.ReLU(),
                        nn.Linear(z_dim, z_dim),
                    )
    def forward(self,m,z,x,previous_dist,previous_hb,first):
        cb = x[:,-1]
        dismap=(cb[:,None,:]-cb[None,:,:]).norm(dim=-1)
        dis_z = fourier_encode_dist(dismap,self.dis_encoding_dim)
        if first:
            return 0,self.linear(dis_z)   
        else:
            z = self.normz(z) + self.linear(dis_z)   + self.dist_linear(previous_dist) +  self.hb_linear(previous_hb)
            m = self.normm(m)
            return m,z 
        


class zBlock(nn.Module):
    def __init__(self,z_dim):
        super(zBlock,self).__init__()
        self.pair_triout=EvoPair.TriOut(z_dim)
        self.pair_triin =EvoPair.TriIn(z_dim)
        self.pair_tristart=EvoPair.TriAttStart(z_dim)
        self.pair_triend  =EvoPair.TriAttEnd(z_dim)
        self.pair_trans = EvoPair.PairTrans(z_dim)
    def forward(self,z):
        z = z + self.pair_triout(z)
        z = z + self.pair_triin(z)
        z = z + self.pair_tristart(z)
        z = z + self.pair_triend(z)
        z = z + self.pair_trans(z)
        return z
class ssAttention(nn.Module):
    def __init__(self, z_dim,N_head=8,c=8) -> None:
        super(ssAttention,self).__init__()
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm1=nn.LayerNorm(z_dim)
        self.qlinear = basic.LinearNoBias(z_dim,N_head*c)
        self.klinear = basic.LinearNoBias(z_dim,N_head*c)
        self.vlinear = basic.LinearNoBias(z_dim,N_head*c)

        self.glinear = basic.Linear(z_dim,N_head*c)
        self.olinear = basic.Linear(N_head*c,z_dim)
    def forward(self,z):
        N,L,_,D = z.shape
        z = self.norm1(z)
        q = self.qlinear(z).reshape(N,L,L,self.N_head,self.c) 
        k = self.klinear(z).reshape(N,L,L,self.N_head,self.c) #s rv h c 
        v = self.vlinear(z).reshape(N,L,L,self.N_head,self.c)
        g = torch.sigmoid(self.glinear(z)).reshape(N,L,L,self.N_head,self.c)
        att = torch.einsum('iabhc,jabhc->ijabh',q,k)* (self.sq_c)
        att=F.softmax(att,dim=1)
        o = torch.einsum('ijabh,jabhc->iabhc',att,v) * g
        z = self.olinear(o.reshape(N,L,L,-1))   
        return z
class ssModule(nn.Module):
    def __init__(self,z_dim,N_head=8,c=8):
        super(ssModule,self).__init__()
        self.z_dim = z_dim
        self.emblinear = nn.Linear(1,z_dim)
        self.block1 = zBlock(z_dim)
        self.block2 = zBlock(z_dim)
        self.ssatt1  = ssAttention(z_dim)
        self.trans1  = EvoPair.PairTrans(z_dim,1)
        self.ssatt2  = ssAttention(z_dim)
        self.trans2  = EvoPair.PairTrans(z_dim,1)
        self.ssatt3  = ssAttention(z_dim)
        self.trans3  = EvoPair.PairTrans(z_dim,1)
    def batchz(self,z,bk):
        N = z.shape[0]
        slist = [bk(z[n]) for n in range(N)]
        return torch.stack(slist)
        


    def forward(self,ss):
        # ss N x l x l 
        z = self.emblinear(ss[...,None]) # N x l x l  x z
        z = self.batchz(z,self.block1)
        #z = self.batchz(z,self.block2) # N x l x l  x z
        z = z + self.ssatt1(z)
        z = z + self.trans1(z)
        z = z + self.ssatt2(z)
        z = z + self.trans2(z)
        z = z + self.ssatt3(z)
        z = z + self.trans3(z)
        return z





class MSA2xyzIteration(nn.Module):
    def __init__(self,seq_dim,msa_dim,N_ensemble,m_dim=64,s_dim=128,z_dim=64,docheck=True):
        super(MSA2xyzIteration,self).__init__()
        self.msa_dim=msa_dim
        self.m_dim=m_dim
        self.z_dim=z_dim
        self.seq_dim=seq_dim
        self.N_ensemble=N_ensemble
        self.dis_dim=36 + 2 
        self.pre_z=ssModule(z_dim)
        self.premsa=PreMSA(seq_dim,msa_dim,m_dim,z_dim)
        self.re_emb=RecyclingEmbedder(m_dim,z_dim,dis_encoding_dim=64)
        #self.ex_emb = RecyclingPoolEmbedder(m_dim = m_dim,s_dim = s_dim,z_dim = z_dim,dis_encoding_dim=32, dis_dim = self.dis_dim)
        self.evmodel=Evoformer.Evoformer(m_dim,z_dim,True)    
        self.slinear=basic.Linear(z_dim,s_dim)
    def pred(self,msa_,idx,ss_,m1_pre,z_pre,pre_x,cycle_index,alphas,previous_dis,previous_hb):
        m1_all,z_all,s_all=0,0,0
        N,L,_=msa_.shape
        for i in range(self.N_ensemble):
            msa_mask = torch.zeros(N,L).to(msa_.device)
            msa_true = msa_ + 0
            seq = msa_true[0]*1.0 # 22-dim
            msa = torch.cat([msa_true*(1-msa_mask[:,:,None]),msa_mask[:,:,None]],dim=-1)
            m,z=self.premsa(seq,msa,idx,alphas)
            if ss_ is None:
                ss = 0
            else:
                ss = torch.mean( self.pre_z(ss_),dim=0)
            #ss = self.pre_z(ss_)
            z  = z+ss
            m1_,z_=self.re_emb(m1_pre,z_pre,pre_x,previous_dis,previous_hb,cycle_index==0) #already added residually
            #ex_s,ex_z =self.ex_emb(previous_s,previous_z,previous_dis,previous_x,previous_hb)
            z = z+z_
            m=torch.cat([(m[0]+m1_)[None,...],m[1:]],dim=0)
            m,z=self.evmodel(m,z)
            s = self.slinear(m[0])
            m1_all =m1_all + m[0]
            z_all  =z_all  + z
            s_all  =s_all + s
        return m1_all/self.N_ensemble,z_all/self.N_ensemble,s_all/self.N_ensemble


class MSA2XYZ(nn.Module):
    def __init__(self,seq_dim,msa_dim,N_ensemble,N_cycle,m_dim=64,s_dim=128,z_dim=64,docheck=True):
        super(MSA2XYZ,self).__init__()
        self.msa_dim=msa_dim
        self.m_dim=m_dim
        self.z_dim=z_dim
        self.dis_dim=36 + 2
        self.N_cycle=N_cycle
        self.msaxyzone = MSA2xyzIteration(seq_dim,msa_dim,N_ensemble,m_dim=m_dim,s_dim=s_dim,z_dim=z_dim)
        self.msa_predor=basic.Linear(m_dim,msa_dim-1)
        self.pdis_predor=basic.Linear(z_dim,self.dis_dim)
        self.cdis_predor=basic.Linear(z_dim,self.dis_dim)
        self.ndis_predor=basic.Linear(z_dim,self.dis_dim)
        self.hb_predor=basic.Linear(z_dim,6)
        self.slinear=basic.Linear(m_dim,s_dim)
        
        self.structurenet=Structure.StructureModule(s_dim,z_dim,4,s_dim) #s_dim,z_dim,N_layer,c)

    def pred(self,msa_,idx,ss,base_x,alphas):
        plddts =[]
        predxs={}
        L=msa_.shape[1]
        m1_pre,z_pre=0,0
        x_pre=torch.zeros(L,3,3).to(msa_.device)
        previous_dis = torch.zeros([L,L,38]).to(msa_.device) 
        previous_dis[...,0] = 1
        previous_hb = torch.zeros([L,L,6]).to(msa_.device)
        previous_hb[...,0] = 1
        ret = {}
        for i in range(self.N_cycle):
            m1,z,s=self.msaxyzone.pred(msa_,idx,ss,m1_pre,z_pre,x_pre,i,alphas,previous_dis,previous_hb)
            x,_,_,plddt = self.structurenet.pred(s,z,base_x)
            pred_disn = F.softmax(self.ndis_predor(z),dim=-1)  
            pred_hb   = F.sigmoid(self.hb_predor(z)) 
            plddts.append(plddt)
            m1_pre=m1.detach()
            z_pre = z.detach()
            x_pre = x.detach()
            predxs[i]=x_pre.cpu().detach()
            previous_dis=pred_disn.detach()
            previous_hb=pred_hb.detach()
            if i == self.N_cycle - 1:
                ret['coor'] = x_pre.detach().cpu().numpy()
                ret['dist_p'] = F.softmax(self.pdis_predor(z),dim=-1).detach().cpu().numpy().astype(np.float16)  
                ret['dist_c'] = F.softmax(self.cdis_predor(z),dim=-1).detach().cpu().numpy().astype(np.float16)
                ret['dist_n'] = F.softmax(self.ndis_predor(z),dim=-1).detach().cpu().numpy().astype(np.float16)
                ret['plddt'] = plddt.detach().cpu().numpy()
        ########################last cycle###########

        return ret