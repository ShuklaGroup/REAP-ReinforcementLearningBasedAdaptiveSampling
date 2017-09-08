class mockSimulation:
        ## public
        def __init__(self):
                self.theta_mean = [0, 0]
                self.theta_std = [0, 0]
                self.r = 1#number of rounds
                self.s = 1# length of simulations
                self.N = 1# number of parallel simulations
                self.tp = None
		self.x = None
		self.y = None
                
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
                return trj

                
        def map(self, trj_Ps):
                """

                output:
                      n_ec x n_frames
                """
                # map coordinate space to reaction coorinates space
                import numpy as np
                trj_x = []
		trj_y = []
                x = self.x
		y = self.y
                for frame in trj_Ps:
			frame = int(frame)
			trj_x.append(x[frame])
			trj_y.append(y[frame])

                return trj_x, trj_y

        
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
                

        def pltPoints(self, x, y):
		import matplotlib.pyplot as plt
		import numpy as np
		
		figName = 'fig.png'
		plt.rcParams.update({'font.size':30})
		plt.rc('xtick', labelsize=30)
		plt.rc('ytick', labelsize=30)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(x, y, color='darkorange', s=5, alpha=0.2)
		plt.xlabel(r'$A-loop RMSD$')
		plt.ylabel(r'$K-E - R-E$')
		fig.savefig('fig.png', dpi=1000, bbox_inches='tight')

                return 



	
