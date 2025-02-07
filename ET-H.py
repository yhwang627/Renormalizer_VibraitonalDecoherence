from renormalizer.model import Model, Op
from renormalizer.mps import Mps, Mpo, MpDm, ThermalProp, gs
from renormalizer.utils.constant import *
from renormalizer.model.basis import BasisMultiElectron, BasisSHO
from renormalizer.utils import EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod,OptimizeConfig
from renormalizer.vibronic import VibronicModelDynamics,VibronicModelDynamics_ET
from renormalizer.utils import log, Quantity, constant
# from renormalizer.sbm.lib import SpectralDensityFunction

import logging
import itertools 
import numpy as np

#import os 
logger = logging.getLogger(__name__)

#log_level = int(os.environ.get("RENO_LOG_LEVEL", logging.DEBUG))

#init_log(log_level)
logging.basicConfig(level=logging.DEBUG)


dump_dir = "./"
job_name = "test"  ####################
log.register_file_output(dump_dir+job_name+".log", mode="w")



# read parameters from the electron-vibraion coupling (EVC) file
def single_mol_model(fdusin, projector=0):
    w0 = []
    d0 = []      # project on PES0 normal coordinates
    w1 = []
    d1 = []      # project on PES1 normal coordinates 
    s021 = [] 
    s120 = []
    with open(fdusin, "r") as f:
        lines = f.readlines()
        read_start = False
        for line in lines:
            if line[:6] == "------": 
                if read_start:
                    break
                else:
                    read_start = True
                    continue
            if read_start:
                split_line = line.split()
                w0.append(float(split_line[3]))
                d0.append(float(split_line[4]))
                w1.append(float(split_line[9]))       
                d1.append(float(split_line[10]))       
        
        nmodes = len(w0)

        start_s120 = start_s021 = False
        for iline, line in enumerate(lines):
            if line.rstrip().lstrip() == "BEGIN_DUSH_DATA_1":
                start_s120 = True
                start_s021 = False
            if line.rstrip().lstrip() == "BEGIN_DUSH_DATA_2":
                start_s021 = True
                start_s120 = False
            if start_s120 or start_s021:
                if line.split()[0] == "MODE":
                    res = [] 
                    for subline in lines[iline+1:iline+int(np.ceil(nmodes/10))+1]:
                        res.extend([float(i) for i in subline.split()])
                    if start_s120:
                        s120.append(res)   
                    elif start_s021:
                        s021.append(res)   
    
    s021 = np.stack(s021, axis=0)               
    s120 = np.stack(s120, axis=0)
    #assert np.allclose(s021.dot(s021.T), np.eye(nmodes), atol=1e-6)
    #assert np.allclose(s120.dot(s120.T), np.eye(nmodes), atol=1e-6)
    #assert np.allclose(s120, s021.T)
    nmodes -= projector
    w0 = np.array(w0[projector:]) * cm2au
    w1 = np.array(w1[projector:]) * cm2au
    d0 = np.array(d0[projector:])
    d1 = np.array(d1[projector:])
    
    s021 = s021[projector:, projector:]
    s120 = s120[projector:, projector:]

    return w0, d0, s021, w1, d1, s120 



#excitation energy
excitation_energy = Quantity(10000,"cm^{-1}").as_au()
Delta =  Quantity(0.08,"eV").as_au()
#electron-electron coupling
J =  Quantity(400,"cm^{-1}").as_au()


# read vibration parameters (\omega: w & displacement:d)
#A: acceptor   D:  donor
#c: cation  a:anion   s0: ground state   s1: excited state
loc = "./par/"
w_As0, d_As0_As1, _, w_As1, d_As1_As0 ,_  = single_mol_model(loc+"As0_As1.dat", projector=0)
_, d_As0_Aa, _, w_Aa, d_Aa_As0, _  = single_mol_model(loc+"As0_Aa.dat", projector=0)
w_Ds0, d_Ds0_Dc, _, w_Dc, d_Dc_Ds0, _  = single_mol_model(loc+"Ds0_Dc.dat", projector=0)



#Unit already transfered in single_mol_model() function 
"""
#tranfer omega from cm^-1 unit to a.u. unit
w_list = [w_As0,w_As1,w_Aa,w_Ds0,w_Dc]
new_w_list =[]
for ilist in w_list:
    w_tmp =[]
    for i in range(len(ilist)):
        w_tmp.append(Quantity(ilist[i],"cm^{-1}").as_au())
    new_w_list.append(np.array(w_tmp))

w_As0 = new_w_list[0]
w_As1 = new_w_list[1]
w_Asc = new_w_list[2]
w_Ds0 = new_w_list[3]
w_Dc = new_w_list[4]
"""

# screen modes with small HR factor
s_tol = 1e-2
w_threshold = 100/constant.au2cm
s_As0_As1 = (-np.sqrt(w_As0/2) * d_As0_As1)**2
s_As0_Aa = (-np.sqrt(w_As0/2) * d_As0_Aa)**2
s_Ds0_Dc = (-np.sqrt(w_Ds0/2) * d_Ds0_Dc)**2
#logger.info(s_As0_As1,w_As0,d_As0_As1)

# A part
logger.info(f"Acceptor part before HR screening, # of modes: {len(w_As0)}")
#index = set(np.where(s_As0_As1>s_tol)[0]) | set(np.where(s_As0_Aa>s_tol)[0]) 
index = (set(np.where(s_As0_As1>s_tol)[0]) | set(np.where(s_As0_Aa>s_tol)[0])) & set(np.where(w_As0>w_threshold)[0])  
logger.info(f"Acceptor part screened modes")
logger.info(set(range(len(w_As0))) - index)
index = list(index)
w_As0 = w_As0[index]
d_As0_As1 = d_As0_As1[index]
w_As1 = w_As1[index]
d_As1_As0 = d_As1_As0[index]

d_As0_Aa = d_As0_Aa[index]
w_Aa = w_Aa[index]
d_Aa_As0 = d_Aa_As0[index]
logger.info(f"Acceptor part after HR screening, # of modes: {len(w_As0)}")
logger.info(f"Acceptor part after HR screening, modes frequency: {w_As0*constant.au2cm}")


# D part
logger.info(f"Donor part before HR screening, # of modes: {len(w_Ds0)}")
#index = set(np.where(s_Ds0_Dc>s_tol)[0])  
index = set(np.where(s_Ds0_Dc>s_tol)[0]) & set(np.where(w_Ds0>w_threshold)[0]) 
logger.info(f"Donor part screened modes")
logger.info(set(range(len(w_Ds0))) - index)
index = list(index)
w_Ds0 = w_Ds0[index]
d_Ds0_Dc = d_Ds0_Dc[index]
w_Dc = w_Dc[index]
d_Dc_Ds0 = d_Dc_Ds0[index]
logger.info(f"Donor part after HR screening, # of modes: {len(w_Ds0)}")
logger.info(f"Donor part after HR screening, modes frequency: {w_Ds0*constant.au2cm}")

logger.info("Used modes number: Acceptor modes: {}    freq: {} cm-1 reorganziation energy EX; {}   ET: {}".format(len(w_As0),w_As0*constant.au2cm,w_As0**2*d_As0_As1**2/2*constant.au2cm,w_As0**2*(d_As0_As1-d_As0_Aa)**2/2*constant.au2cm))
logger.info("Used modes number: Donor modes: {}    freq: {} cm-1  reorganziation energy EX; {}   ET: {}".format(len(w_Ds0),w_Ds0*constant.au2cm,w_Ds0**2*0**2/2*constant.au2cm,w_Ds0**2*(d_Ds0_Dc)**2/2*constant.au2cm))







sp_modes_freq= np.array([[160,180],[310,320],[1320,1330],[2330,2340]])/constant.au2cm
re_EX = np.zeros(sp_modes_freq.shape[0])
re_GA = np.zeros(sp_modes_freq.shape[0])
w_sp = np.zeros(sp_modes_freq.shape[0])
d_Ex_sp = np.zeros(sp_modes_freq.shape[0])
d_GA_sp = np.zeros(sp_modes_freq.shape[0])
index = []
for i in range(len(w_As0)):
    is_spmode =False
    for i_spmode in range(sp_modes_freq.shape[0]):
        if w_As0[i] >= sp_modes_freq[i_spmode,0]  and w_As0[i] <= sp_modes_freq[i_spmode,1]:
            is_spmode = True 
            w_sp[i_spmode] = w_As0[i]
            d_Ex_sp[i_spmode] = d_As0_As1[i]
            d_GA_sp[i_spmode] = d_As0_Aa[i]
            break


    if is_spmode is False:
        index.append(i)



#delete sp modes here
w_As0 = w_As0[index]
d_As0_As1 = d_As0_As1[index]
d_As0_Aa = d_As0_Aa[index]








#sort the mode according to the freq ascending order

sorted_indice = np.argsort(w_As0)

w_As0 = w_As0[sorted_indice]
d_As0_As1 = d_As0_As1[sorted_indice]
d_As0_Aa = d_As0_Aa[sorted_indice]



sorted_indice = np.argsort(w_Ds0)

w_Ds0 = w_Ds0[sorted_indice]
d_Ds0_Dc = d_Ds0_Dc[sorted_indice]








# #NO FREQ DISTORSION 
w_As1 = w_As0.copy()
w_Aa  = w_As0.copy()
w_Dc = w_Ds0.copy()







########Print Modes used in Calculation###################################
logger.info("Used modes number: Special modes: {}    freq: {} cm-1  reorganziation energy EX; {}   ET: {} GA {}".format(len(w_sp),w_sp*constant.au2cm,w_sp**2*d_Ex_sp**2/2*constant.au2cm,w_sp**2*(d_Ex_sp-d_GA_sp)**2/2*constant.au2cm,w_sp**2*(d_GA_sp)**2/2*constant.au2cm))
logger.info("Used modes number: Acceptor modes: {}    freq: {} cm-1 reorganziation energy EX; {}   ET: {}".format(len(w_As0),w_As0*constant.au2cm,w_As0**2*d_As0_As1**2/2*constant.au2cm,w_As0**2*(d_As0_As1-d_As0_Aa)**2/2*constant.au2cm))
logger.info("Used modes number: Donor modes: {}    freq: {} cm-1  reorganziation energy EX; {}   ET: {}".format(len(w_Ds0),w_Ds0*constant.au2cm,w_Ds0**2*0**2/2*constant.au2cm,w_Ds0**2*(d_Ds0_Dc)**2/2*constant.au2cm))





#Temperature for thermal field dynamics, not used here for zero temeprature dynamics

# T = Quantity(300,"K")
# beta = T.to_beta()



# build ham_e to collect the parameters for the electronic state
e_ndof = 3
ham_e = np.zeros((e_ndof, e_ndof))
e_dofs = ["G", "R", "P"]

# parameters
e_G = 0
e_R = 0 + excitation_energy
e_P = 0 + excitation_energy - Delta
v_R_P = J * -1

# prepare parameter of elctronic part as ham_e
for idx, e_dof in enumerate(e_dofs):
    factor = locals().get(f"e_{e_dof}")
    if factor is not None:
        ham_e[idx, idx] = factor

for idx, jdx in itertools.permutations(range(e_ndof), 2):
    factor1 = locals().get(f"v_{e_dofs[idx]}_{e_dofs[jdx]}")
    factor2 = locals().get(f"v_{e_dofs[jdx]}_{e_dofs[idx]}")
    if factor1 is not None:
        ham_e[idx, jdx] = factor1
    elif factor2 is not None:
        ham_e[idx, jdx] = factor2

logger.info("ham_e:")
logger.info(f"{ham_e}")
print("ham_e:")
print(f"{ham_e}")


#######################################
# construct the model  "ham_terms"
#######################################


ham_terms = []

# electronic terms
for idx, jdx in itertools.product(range(e_ndof), repeat=2):
    if ham_e[idx, jdx] != 0:
        op = Op("a^\dagger a", [e_dofs[idx], e_dofs[jdx]],
                factor=ham_e[idx, jdx], qn=[1, -1])
        ham_terms.append(op)

# kinetic, potential terms
# inculding all modes D,A
# All modes are treated as harmonic oscillators

#kinetic terms
for imode, w in enumerate(w_As0):
    ham_terms.append(Op("p^2", [f"v_A_{imode}"], factor=0.5, qn=[0]))
for imode, w in enumerate(w_Ds0):
    ham_terms.append(Op("p^2", [f"v_D_{imode}"], factor=0.5, qn=[0]))
    
##sp
for imode, w in enumerate(w_sp):
    ham_terms.append(Op("p^2", [f"v_sp_{imode}"], factor=0.5, qn=[0]))

#potential terms, electron-vibrational coupling terms &  reorganization energy terms 
# \sum_{i,n} (1/2 \omega_{i,n}^2 q_{n}^2 + \omega_{i,n}^2 q_{n} \Deltaq +  \omega_{i,n}^2  \Deltaq^2) a^\dagger_{i} a_{i}

#vibration frequency of the same mode on different PES is different 
# no off-diagonal coupling between vibrational modes now

# Ground State  AD
for imode, w in enumerate(w_As0):
    ham_terms.append(Op("x^2 a^\dagger a", [f"v_A_{imode}","G","G"], factor=0.5*w**2, qn=[0,1,-1]))
for imode, w in enumerate(w_Ds0):
    ham_terms.append(Op("x^2 a^\dagger a", [f"v_D_{imode}","G","G"], factor=0.5*w**2, qn=[0,1,-1]))

##sp
for imode, w in enumerate(w_sp):
    ham_terms.append(Op("x^2 a^\dagger a", [f"v_sp_{imode}","G","G"], factor=0.5*w**2, qn=[0,1,-1]))

# Reactant State  A* D
for imode, w in enumerate(w_As1):
    ham_terms.append(Op("x^2 a^\dagger a", [f"v_A_{imode}","R","R"], factor=0.5*w**2, qn=[0,1,-1]))
    ham_terms.append(Op("x a^\dagger a", [f"v_A_{imode}","R","R"], factor=w**2*d_As0_As1[imode], qn=[0,1,-1]))
for imode, w in enumerate(w_Ds0):
    ham_terms.append(Op("x^2 a^\dagger a", [f"v_D_{imode}","R","R"], factor=0.5*w**2, qn=[0,1,-1]))
ham_terms.append(Op("a^\dagger a", ["R","R"], factor=np.sum(0.5*w_As1**2*d_As0_As1**2), qn=[1,-1]))

##sp
for imode, w in enumerate(w_sp):
    ham_terms.append(Op("x^2 a^\dagger a", [f"v_sp_{imode}","R","R"], factor=0.5*w**2, qn=[0,1,-1]))
    ham_terms.append(Op("x a^\dagger a", [f"v_sp_{imode}","R","R"], factor=w**2*d_Ex_sp[imode], qn=[0,1,-1]))
ham_terms.append(Op("a^\dagger a", ["R","R"], factor=np.sum(0.5*w_sp**2*d_Ex_sp**2), qn=[1,-1]))


# Product State  A- D+
for imode, w in enumerate(w_Aa):
    ham_terms.append(Op("x^2 a^\dagger a", [f"v_A_{imode}","P","P"], factor=0.5*w**2, qn=[0,1,-1]))
    ham_terms.append(Op("x a^\dagger a", [f"v_A_{imode}","P","P"], factor=w**2*d_As0_Aa[imode], qn=[0,1,-1]))
for imode, w in enumerate(w_Dc):
    ham_terms.append(Op("x^2 a^\dagger a", [f"v_D_{imode}","P","P"], factor=0.5*w**2, qn=[0,1,-1]))
    ham_terms.append(Op("x a^\dagger a", [f"v_D_{imode}","P","P"], factor=w**2*d_Ds0_Dc[imode], qn=[0,1,-1]))
ham_terms.append(Op("a^\dagger a", ["P","P"], factor=np.sum(0.5*w_Aa**2*d_As0_Aa**2), qn=[1,-1]))
ham_terms.append(Op("a^\dagger a", ["P","P"], factor=np.sum(0.5*w_Dc**2*d_Ds0_Dc**2), qn=[1,-1]))

##sp
for imode, w in enumerate(w_sp):
    ham_terms.append(Op("x^2 a^\dagger a", [f"v_sp_{imode}","P","P"], factor=0.5*w**2, qn=[0,1,-1]))
    ham_terms.append(Op("x a^\dagger a", [f"v_sp_{imode}","P","P"], factor=w**2*d_GA_sp[imode], qn=[0,1,-1]))
ham_terms.append(Op("a^\dagger a", ["P","P"], factor=np.sum(0.5*w_sp**2*d_GA_sp**2), qn=[1,-1]))






# Basis to expand the operator "basis"
nbas = 20
basis = []
basis.append(BasisMultiElectron(e_dofs, [0,1,1]))
for imode, w in enumerate(w_sp):
    basis.append(BasisSHO(f"v_sp_{imode}", w, nbas))
for imode, w in enumerate(w_As0):
    basis.append(BasisSHO(f"v_A_{imode}", w, nbas))
for imode, w in enumerate(w_Ds0):
    basis.append(BasisSHO(f"v_D_{imode}", w, nbas))
#basis.append(BasisMultiElectron(e_dofs, [0,1,1]))

# nbas= 10
# for imode, w in enumerate(omega):
#     basis.append(BasisSHO(f"v_sys_{imode}", w, nbas))


# Build model instance, constrcuct MPO according to "ham_terms" and  "basis"
model = Model(basis, ham_terms)
mpo = Mpo(model)
logger.info(f"nsites: {mpo.site_num}")
logger.info(f"mpo bond dimension: {mpo.bond_dims}")

# init state MPS
#################### temperature



# Photoinduced excitation init condition
init_condition = {"G": 1}
mps = Mps.hartree_product_state(model, condition=init_condition)
logger.info("Mps before optimization: Energy  {},    E_occupation  {}".format(mps.expectation(mpo),mps.e_occupations))
logger.info("Mps before optimization: Vibrational_amplitude_nm  {}".format(mps.vibrational_amplitude_nm))
# mps.optimize_config = OptimizeConfig(procedure = ([[10, 0.4], [20, 0.2]]+[[30,0]]*100))




# Also we can use ground state optimization to build a equlibrium state after excitation and dissipation
# only sys-vibrations
# both sys-vibrations and bath-vibrations

# Virtual bond dimension M for MPS
compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=30)

evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)

#With amplitude dump
job = VibronicModelDynamics_ET(model,
                            compress_config=compress_config,
                            evolve_config=evolve_config,
                            mps0=mps,
                            h_mpo=mpo,
                            dump_mps=None,
                            dump_dir=dump_dir, job_name=job_name,
                            auto_expand=True)

job.info_interval = 1
time_step_fs = 0.4
job.evolve(evolve_dt=time_step_fs * fs2au, nsteps=2500)


