# ipython2 SLmain.py
class mockSimulation:
        ## public
        def __init__(self):
                self.theta_mean = [0, 0]
                self.theta_std = [0, 0]
                self.r = 1#number of rounds
                self.s = 1# length of simulations
                self.N = 1# number of parallel simulations
                self.tp = None
                self.mapping = None
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
                map = self.mapping
                for MacroFrame in trj_Ps:
                    microFrame = map[int(MacroFrame)]
                    trj_x.append(x[microFrame])
                    trj_y.append(y[microFrame])

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

            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            
            #ax.scatter(x , y, color='darkorange', s=10, alpha=0.2)
            ax.scatter(x + np.random.normal(0, 0.06/4, len(x)), y+np.random.normal(0, 0.06, len(y)), color='darkorange', s=10, alpha=0.2)
            plt.xlabel(r'$RMSD of A-loop (nm)$')
            plt.ylabel(r'$d_E310-R409 - d_K295 - E310$')
            #plt.ylabel(r'$d_E_310_-_R_409  - d_K_295_-_E_310 (nm)$')
            plt.ylim([-2, 2])
            plt.xlim([0, 1])
#            plt.show()
            fig.savefig('fig.png', dpi=1000, bbox_inches='tight')

            return 
