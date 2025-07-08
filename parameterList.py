import numpy as np 

class parameters_epileptic_NMDA:
    
    def __init__(self):

        self.dt = 1/10000
        self.t_end = 1000
        self.t_seizure_duration = 10
        self.pm_1 = 90
        self.pm_2 = 70
        self.ps = 2.

        self.A_1 = 5
        self.B_1 = 33.
        self.G_1 = 20.
        
        self.A_2 = 5
        self.B_2 = 45.
        self.G_2 = 2.
        
        self.aa = 100.
        self.bb = 30.
        self.gg = 350.

        self.c1_1 = 135.
        self.c2_1 = 108.
        self.c3_1 = 35.
        self.c4_1 = 25
        self.c5_1 = 200.
        self.c6_1 = 120.
        self.c7_1 = 200.
        
        self.c1_2 = 135.
        self.c2_2 = 108.
        self.c3_2 = 35.
        self.c4_2 = 25
        self.c5_2 = 200.
        self.c6_2 = 120.
        self.c7_2 = 200
                
        self.Cp1top2_3 = 50
        self.Cp1top2_5 = 0.
        self.Cp1top2_7 = 0.
        self.Cp1top2_9 = 0.
        
        self.Cp2top1_3 = 0.
        self.Cp2top1_5 = 0.
        self.Cp2top1_7 = 0.
        self.Cp2top1_9 = 0.
        
        self.A_1_ampa = 2*self.A_1
        self.A_2_ampa = 2*self.A_2
        self.aa_ampa = 200 # ( 1/200 =0.005 sec)
        self.A_nmda = 2
        self.aa_nmda = 50 # ( 1/50 =0.020 sec)
        self.A_nmda_ext = 1
        self.aa_nmda_ext = 25 # ( 1/25 =0.040 sec)
        
        self.theta0 = 2
        self.sigm_NMDApost_slope = 1
        self.sigm_NMDApost_threshold = 5
        self.sigm_NMDApost_amplitude = 1
        
        self.sigm_NMDApost_Ca_factor = 10
        self.Cp1top2_3_max = 100
        self.Cp2top1_3_max = 0
        self.omega_beta2 = 80
        self.omega_alpha2 = 0.4
        self.omega_beta1 = 80
        self.omega_alpha1 = 0.1
        self.omega_gamma2 = 5
        self.omega_gamma1 = 1
        
        self.Use = 0.4
        self.tau_r = 0.200 # 200 ms
        self.tau_f = 0.050 # 50 ms
        self.Use_max = 0.8

        self.u_0 = self.Use
        self.Use_0 = self.Use
        self.p_0 = 0
        self.Wp1top2_3_0 = self.Cp1top2_3
    
        self.tauCa = 0.05 # (physio value: 50 ms)
        self.tauP = 50 # 70 sec
        self.tau_Use = 100 # 100 sec
        self.tau_Wp1top2_3 = 100  # 100 sec
        
        self.k = 1
        self.tau_K = 10

        self.g_kdrive = 20
        self.n_kdrive = 10
        self.nk = -0.2
        self.np = 1.4
        self.nr = 2.
        self.btr_2 = 44
        self.eps_n = 0.05
                
        self.btr_1 = 34

        self.delta = 50
        self.p1 = 25
        self.r1= 1
        self.m1 = 0.0015
        self.p2 = 30
        self.r2 = 1
        self.p3 = 33
        self.r3 = 1
        self.m3 = 0.003
