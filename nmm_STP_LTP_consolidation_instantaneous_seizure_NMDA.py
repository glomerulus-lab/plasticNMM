'''NMM including short-term presynaptic plasticity, calcium dependent long term presynaptic and postsynaptic plasticity, consolidation, and GABAergic modulation under epileptic activity. '''
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Arial",
    "font.size": 10,
     "lines.linewidth":0.5
})

from parameterList import parameters_epileptic_NMDA as parameters

def func_sigm(x):
    # Defined page 6 in paragraph
    # Converts potential to firing rate, sigmoid transfer function
    # S(V) in eqn 10
    sigm = 5 / (1 + np.exp(0.56 * (6 - x)))
    return sigm

def func_omega(x, gamma, beta, alpha):
    # Eqn 2
    # Generic sigmoid
    # Note: alpha is theta in paper
    omega = gamma / (1 + np.exp(-beta * (x - alpha)))
    return omega

def func_sigm_NMDApost(vpost, par):
    # Eqn 5
    # Postsynaptic voltage gate of NMDA current
    s = par.sigm_NMDApost_amplitude / \
        (1 + np.exp(par.sigm_NMDApost_slope*(par.sigm_NMDApost_threshold - vpost)))
    return s

def PSP(firing_rate, y, yprime, A, tau_inverse):
    # Eqn 10 generic form
    # firing_rate = S(V)
    # tau_inverse = 1 / tau
    return (A * tau_inverse * firing_rate - 2 * tau_inverse * yprime - (tau_inverse ** 2) * y)
    
def dW(delta_t):
    # White noise derivative
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

def NMM(initial1, initial2, par, vm1, vm2):
    # inputs: initial1, initial2 state vectors
    # initial1: len 13
    dp1 = np.zeros(13) # d/dt of pop 1 state vector
    dp2 = np.zeros(20) # d/dt of pop 2 state vector
     
    y0_1, y5_1, y1_1, y6_1, y2_1, y7_1, y3_1, y8_1, r_1, u_1, Use_1, B_1, n_1 = initial1
    y0_2, y5_2, y1_2, y6_2, y2_2, y7_2, y3_2, y8_2, yinp1p2_ampa, zinp1p2_ampa, yinp1p2_nmda, zinp1p2_nmda, Ca_2, rho_2, Wp1top2_3, yinp1p2_nmda_ext, yinp1p2_nmda_ext_prime, K, B_2, n_2  = initial2
    
    Inputp1top2 = func_sigm(vm1) # F_pre in eqn 9
    
    # POP1
    # PYRAMIDAL - P population
    # Eqn 10A : y0_1 = y_p in paper
    dp1[0] = y5_1 # d(y0_1) / dt = y5_1
    dp1[1] = PSP(func_sigm(vm1), y0_1, y5_1, par.A_1, par.aa) # d(y5_1) / dt = y0_1''
    # Excitatory - P' population
    # Eqn 10B : y1_1 = y_p' in paper
    dp1[2] = y6_1 # d(y1_1) / dt = y6_1
    dp1[3] = par.A_1 * par.aa * par.pm_1 + \ # + pm_1 term
        PSP(par.c2_1 * func_sigm(par.c1_1 * y0_1), y1_1, y6_1, par.A_1, par.aa)
    # GABAslow - SOM, somatostatin
    # Eqn 10C
    dp1[4] = y7_1 # d(y2_1) / dt = y7_1
    dp1[5] = PSP(func_sigm(par.c3_1 * y0_1), y2_1, y7_1, B_1, par.bb)
    # GABAfast - PV, parvalbumin
    # Eqn 10D
    dp1[6] = y8_1 # d(y3_1) / dt = y8_1
    dp1[7] = PSP(func_sigm(par.c5_1 * y0_1 - par.c6_1 * y2_1), y3_1, y8_1, par.G_1, par.gg)
    
    # the direct impact of the STP and its consolidation is on POP2.
    # STP consolidation is driven by rho_pop2topop1 (rho_2)
    # Short-term plasticity
    # variable names matching paper
    dp1[8] = (1 - r_1) / par.tau_r - u_1 * r_1 * Inputp1top2 # Eqn 9A
    dp1[9] = (Use_1 - u_1) / par.tau_f + Use_1 * (1 - u_1) * Inputp1top2 # Eqn 9B
       
    # Long-term pre-synaptic plasticity
    # Eqn 9E : Use_1 = U_s, par.Use = U_s^d, par.Use_max = U_s^p, rho_2 = rho in paper
    dp1[10] = (par.Use - Use_1 + rho_2 * (par.Use_max - par.Use)) / par.tau_Use
    
    # SOM IPSP amplitude to get seizure phases
    # 2-variable slow/fast subsystem
    # Eqn 11A
    dp1[11] = par.delta * (n_1 - (-par.m1 * ((B_1 - par.p1) ** 2) * (1 / (1 + np.exp(-par.r1 * (par.p1 - B_1)))) + (1 / (1 + np.exp(par.r2 * (-par.p2 + B_1)))) + par.m3 * ((B_1 - par.p3) ** 2) * (1 / (1 + np.exp(-par.r3 * (-par.p3 + B_1))))))
    # Eqn 11B
    dp1[12] = par.eps_n * (-n_1 + (par.nk + par.np / (1 + np.exp(-par.nr * (par.btr_1 - B_1)))))
    
    ########################################################################################################
    # POP2
    # PYRAMIDAL
    dp2[0] = y5_2
    dp2[1] = PSP(func_sigm(vm2), y0_2, y5_2, par.A_2, par.aa)
    #Excitatory
    dp2[2] = y6_2
    dp2[3] = par.A_2 * par.aa * (par.pm_2) + PSP(par.c2_2 * func_sigm(par.c1_2 * y0_2), y1_2, y6_2, par.A_2, par.aa)
    # GABAslow - 
    dp2[4] = y7_2
    dp2[5] = PSP(func_sigm(par.c3_2 * y0_2), y2_2, y7_2, B_2, par.bb)
    # GABAfast - PV
    dp2[6] = y8_2
    dp2[7] = PSP(func_sigm(par.c5_2 * y0_2 - par.c6_2 * y2_2), y3_2, y8_2,
                 par.g_kdrive * (1 - K) + par.G_2, par.gg)

    ########################################################
    
    # AMPAERGIC INPUT
    # Eqn 12
    dp2[8] = zinp1p2_ampa
    dp2[9] = PSP(Inputp1top2, yinp1p2_ampa, zinp1p2_ampa, r_1 * u_1 * par.A_2_ampa, par.aa_ampa)

    # NMDAERGIC INPUT
    # Eqn 13
    dp2[10] = zinp1p2_nmda
    dp2[11] = PSP(Inputp1top2, yinp1p2_nmda, zinp1p2_nmda, r_1 * u_1 * par.A_nmda, par.aa_nmda)
    
    # Ca equation
    # Eqn 3 with I_NMDA from 4
    # par.Cp1top2_3 * yinp1p2_nmda = G(F_pre) in paper
    dp2[12] = (par.Cp1top2_3 * yinp1p2_nmda) * par.sigm_NMDApost_Ca_factor * func_sigm_NMDApost(vm2, par) - Ca_2 / par.tauCa # d(Ca_2) / dt

    # to model consolidation
    # Eqn 9D
    dp2[13] = (-rho_2 * (1 - rho_2) * (0.5 - rho_2) + \
               (1 - rho_2) * \
               func_omega(Ca_2, par.omega_gamma2, par.omega_beta2, par.omega_alpha2) \
               - \
               rho_2 * func_omega(Ca_2, par.omega_gamma1, par.omega_beta1, par.omega_alpha1)) \
               / par.tauP
    
    # post-synaptic long-term plasticity
    # Eqn 9F
    # Wp1top2_3 = C~_AMPA in paper
    dp2[14] = (par.Cp1top2_3 - Wp1top2_3 + rho_2 * (par.Cp1top2_3_max - par.Cp1top2_3))/par.tau_Wp1top2_3
    
    ########################################################
    
    # extrasynaptic NMDAERGIC INPUT
    # Eqn 14
    # yinp1p2_nmda_ext = y_{NMDA,ext} in paper
    # Note: if statement turns off NMDA spillover when u_1 is small, not in paper
    dp2[15] = yinp1p2_nmda_ext_prime
    dp2[16] = PSP(Inputp1top2, yinp1p2_nmda_ext, yinp1p2_nmda_ext_prime,
                  r_1*(u_1 if u_1 > 0.7 else 0) * par.A_nmda_ext, par.aa_nmda_ext)
    
    # auxiliary variable for KCC2 and GABA internalization
    # shuts down PV in pop 2
    # Eqn 15, dK/dt
    dp2[17] = (-K * (0.5 - K) * (1 - K) - par.k * par.Cp1top2_3 * yinp1p2_nmda_ext * func_sigm_NMDApost(vm2, par)) / par.tau_K

    # Eqn 11A 
    dp2[18] = par.delta*(n_2 -1*(-par.m1*((B_2-par.p1)**2)*(1/(1+np.exp(-par.r1*(par.p1-B_2)))) + (1/(1+np.exp(par.r2*(-par.p2+B_2))))+ par.m3*((B_2-par.p3)**2)*(1/(1+np.exp(-par.r3*(-par.p3+B_2))))))
    # Eqn 11B
    dp2[19] = par.eps_n*(-n_2 + (par.nk + par.np/(1+np.exp(-par.nr*(par.btr_2 - par.n_kdrive*(1-K) - B_2)))))

    # Total membrane potential equations
    # Pop 1 : page 7
    vm1 = y1_1 - par.c4_1 * y2_1 - par.c7_1 * y3_1
    # Pop 2: page 7 & page 11 bottom
    vm2 = Wp1top2_3 * yinp1p2_ampa + \ # AMPA
        par.Cp1top2_3 * yinp1p2_nmda * func_sigm_NMDApost(vm2, par) + \ # synaptic NMDA
        par.Cp1top2_3 * yinp1p2_nmda_ext * func_sigm_NMDApost(vm2, par) + \ # extrasynaptic
        y1_2 - par.c4_2 * y2_2 - par.c7_2 * y3_2 
    
    return dp1, dp2, vm1, vm2

def initial_cond(par):
    
    initial_conditions_1 = np.zeros(13)
    initial_conditions_2 = np.zeros(20)
    
    initial_conditions_1[8] = 1
    initial_conditions_1[9] = par.u_0
    initial_conditions_1[10] = par.Use_0
    initial_conditions_1[11] = 35 # for btr_1= 34
    initial_conditions_1[12] = 0.022

    initial_conditions_2[13] = par.p_0
    initial_conditions_2[14] = par.Wp1top2_3_0
    initial_conditions_2[17] = 1
    initial_conditions_2[18] = 44.8 # for btr_2 = 44
    initial_conditions_2[19] = 0.6
    
    return initial_conditions_1, initial_conditions_2

def generate_noise_vector(par):
    return np.random.normal(loc=0.0, scale=np.sqrt(par.dt), size = (int(par.t_end/par.dt), 2))

def calcul_euler_maruyama(par, k):
    t = np.arange(0, par.t_end, par.dt)
    solp1, solp2 = np.zeros((int(t.size), 13)), np.zeros((int(t.size), 20))
    solp1[0], solp2[0] = initial_cond(par)

    vmp1, vmp2 = np.ones(int(t.size)), np.ones(int(t.size))
    bnoise1 = np.copy(k)
    bnoise1[300*10000:-1] = 0
    
    bnoise2 = np.copy(k)
    bnoise2[0:300*10000] = 0
    
    for i in range(1, t.size):
        
        dp1, dp2, vmp1[i], vmp2[i] = NMM(solp1[i - 1], solp2[i - 1], par, vmp1[i-1], vmp2[i-1])
        solp1[i] = solp1[i - 1] + par.dt * dp1
        solp2[i] = solp2[i - 1] + par.dt * dp2

        solp1[i, 3] = solp1[i, 3] + par.A_1 * par.aa * par.ps * k[i, 0]
        solp2[i, 3] = solp2[i, 3] + par.A_2 * par.aa * par.ps * k[i, 1]
        
        solp1[i, 11] = solp1[i,11] + 1*bnoise1[i, 0]
        solp2[i, 18] = solp2[i, 18] + 1*bnoise2[i, 0]

    return t, vmp1, vmp2, solp1, solp2


def plot_temporal(par, t, solp1, solp2, vmp1, vmp2):

    f, ax = plt.subplots(8, 1, figsize=(15, 13), sharex=True)

    ax[0].plot(t, vmp1, label='Pop 1')
    ax[1].plot(t, vmp2, label='Pop 2', color = 'orange')
        
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel('Membrane potential')
    ax[0].set_xlim([0, par.t_end])
    ax[1].legend(loc="upper right")
    ax[1].set_ylabel('Membrane potential')
    ax[1].set_xlim([0, par.t_end])

    ax[2].plot(t, solp2[:, 12], label = 'Pop2 - [Ca]')
    ax[2].plot(t, par.omega_alpha1*np.ones(int(t.size)), ':')
    ax[2].plot(t, par.omega_alpha2*np.ones(int(t.size)), ':')
    ax[2].legend(loc="upper right")
    
    ax[2].set_ylabel('[Ca]')

    ax[3].plot(t, solp2[:, 10]*par.Cp1top2_3*func_sigm_NMDApost(vmp2, par), label = "NMDA pop1->2")
    ax[3].plot(t, solp2[:, 15]*par.Cp1top2_3*func_sigm_NMDApost(vmp2, par), label = "NMDAext pop1->2")
    ax[3].legend(loc="upper right")
    ax[3].set_ylabel('NMDA')
    
    ax[4].plot(t, solp2[:, 13], label =r'$\rho_2(t)$')
    
    ax[4].plot(t, 0.5* np.ones(int(t.size)), ':', color='black')
    ax[4].set_ylabel(r'$\rho$', rotation = 0)

    ax[5].plot(t, solp2[:, 17], label='K(t)')
    ax[5].legend(loc="upper right")

    ax[6].plot(t, solp2[:, 18], label='B2(t)')
    ax[6].legend(loc="upper right")

    ax[7].plot(t, solp1[:, 8]*solp1[:, 9]*solp2[:, 14], label = 'r1.u1.W1-2')
    ax[7].legend(loc="upper right")
    ax[7].set_ylabel('r*u*w')


    ax[-1].set_xlabel('Time (sec)')
    
def main():
    par = parameters()

    k = generate_noise_vector(par)
    print(np.shape(k))
    
    filename = 'figure'
    
    t, vmp1, vmp2, solp1, solp2 = calcul_euler_maruyama(par, k[:, [0, 1]])
       
    solutionp = np.zeros((int(t.size),3))
    solutionp[:,0] = t
    solutionp[:,1] = vmp1 # 
    solutionp[:,2] = vmp2 # 
        
    solutionall = np.concatenate((solutionp[:,0:3], solp1, solp2), axis = 1)
    np.save(filename, solutionall)
        
    plot_temporal(par, t, solp1, solp2, vmp1, vmp2)
    plt.savefig(filename+'.png', dpi = 300)
    plt.show()

if __name__ == '__main__':
    main()
