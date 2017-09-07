class mockSimulation:
        ## public
        def __init__(self):
                self.theta_mean = [0, 0]
                self.theta_std = [0, 0]
                self.r = 1#number of rounds
                self.s = 1# length of simulations
                self.N = 1# number of parallel simulations
                self.msm = None
                
        def run_multipleSim(self):
                return True
        def runNxtRound(self):
                return True
                
        
        ## private
        def PreAll(self, trj):
                """
                Pre-Sampling:
                        choose states with minimum counts or newly discovered states
                        
                output:
                        trj with shape of [[Xs][Ys]]
                """
                import numpy as np
                comb_trj = np.concatenate(trj)

                #for theta in range(len(trj)):
                #        comb_trj.append(np.concatenate(np.concatenate(trj[theta])))
                #trj_Sp = np.array(comb_trj) # pick all
                
                return trj_Sp


        def PreSamp_MC(self, trj, N = 20):
                """
                Pre-Sampling for Monte Carlo simulations:
                        choose states with minimum counts or newly discovered states
                        
                output:
                        trj with shape of 
                """
                import numpy as np
                cl_trjs = trj             
                unique, counts = np.unique(cl_trjs, return_counts=True)
                leastPop = counts.argsort()[:N]
                init_cl = [unique[i] for i in leastPop]
                return init_cl

                
        def map(self, trj_Ps):
                """

                output:
                      n_ec x n_frames
                """
                # map coordinate space to reaction coorinates space
                import numpy as np
                trj_Ps_theta = []
                msm = self.msm
                for frame in trj_Ps:
                        theta = np.loadtxt('MSMStatesAllVals_1000/cluster'+str(msm.mapping_[int(frame)])+'-1')
                        trj_Ps_theta.append(theta)

                # change the format
                trj_Ps_theta_2 = []
                ##############
                trj_Ps_theta = np.array(trj_Ps_theta) 
                for theta_index in range(len(trj_Ps_theta[0])):
                        trj_Ps_theta_2.append(trj_Ps_theta[:,theta_index])
                return trj_Ps_theta_2

        
        def run(self, inits, nstepmax = 10):
                """
                Parameters
                ----------
                initi : 
                        initial state (singe state)
                msm :
                        reference MSM
                s :
                        lenght (number of steps) of each simulation	
                
                output :
                        final trajectory
                """
                import numpy as np
                import msmbuilder as msmb
		
                #msm = self.msm
                tp = self.tp
                N = len(inits)
                trjs = np.empty([N, nstepmax])
                for n in range(N):
                        init = np.int(inits[n])
                        trj = msmb.msm_analysis.sample(tp, init, nstepmax)
                        #trj = msm.sample_discrete(state=init, n_steps=nstepmax, random_state=None)
                        trjs[n] = trj
                return trjs
                

        def isActive_singleRound(self, trjs):
                time = -1
                n_parTrjs = len(trjs)
                for trj in trjs:
                        for frame in range(len(trj)):
                                if self.isActive(trj[frame]):
                                        time = n_parTrjs * frame
                                        return time
                return time



	
