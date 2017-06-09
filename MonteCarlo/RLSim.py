class mockSimulation:
        ## public
        def __init__(self):
                self.theta_mean = [0, 0]
                self.theta_std = [0, 0]
                self.r = 1#number of rounds
                self.s = 1# length of simulations
                self.N = 1# number of parallel simulations
                
                
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
                comb_trj = []
                for theta in range(len(trj)):
                        comb_trj.append(np.concatenate(np.concatenate(trj[theta])))
                trj_Sp = np.array(comb_trj) # pick all
                
                return trj_Sp


        def PreSamp(self, trj, starting_n=10, myn_clusters = 40, N = 2):
                """
                Pre-Sampling:
                        choose states with minimum counts or newly discovered states
                        
                output:
                        trj with shape of [[Xs][Ys]]
                """
                import numpy as np
                comb_trj = trj
                
                from sklearn.cluster import KMeans

                comb_trj_xy = np.array([[comb_trj[0][i], comb_trj[1][i]] for i in range(len(comb_trj[0]))])
                cluster = KMeans(n_clusters=myn_clusters)
                cluster.fit(comb_trj_xy)
                cl_trjs = cluster.labels_
                
                #if method=='leastPop': # N: number of chosen min pop clusters
                
                unique, counts = np.unique(cl_trjs, return_counts=True)
                leastPop = counts.argsort()[:N]
                init_cl = [unique[i] for i in leastPop]

        
                counter = 0
                init_index = []
                init_trj_xy = []
                for i in range(len(cl_trjs)):
                        if cl_trjs[i] in init_cl:
                                counter = counter + 1
                                init_index.append(i)
                                init_trj_xy.append(comb_trj_xy[i])
                init_trj = [[init_trj_xy[i][0] for i in range(len(init_trj_xy))], [init_trj_xy[i][1] for i in range(len(init_trj_xy))]]     
                trj_Sp = init_trj

                while len(trj_Sp[0])<starting_n:
                        print('trj_Sp<starting_n')
                        print(len(trj_Sp[0]), starting_n)
                        trj_Sp = np.array([np.concatenate([trj_Sp[0], trj_Sp[0]]), np.concatenate([trj_Sp[1], trj_Sp[1]])])
                

                return trj_Sp
                
        def map(self, trj_Sp):
                # map coordinate space to reaction coorinates space
                trj_Sp_theta = trj_Sp
                return trj_Sp_theta

        def reward_state(self, S, theta_mean, theta_std, W_):
                
                r_s = 0
                for k in range(len(W_)):
                        if (S[k] - theta_mean[k]) < 0: 
                                r_s = r_s + W_[k][0]*(abs(S[k] - theta_mean[k])/theta_std[k])
                        else:
                                r_s = r_s + W_[k][1]*(abs(S[k] - theta_mean[k])/theta_std[k])
                return r_s

        def reward_state_withoutStd(self, S, theta_mean, theta_std, W_):
                
                r_s = 0
                for k in range(len(W_)):
                        if (S[k] - theta_mean[k]) < 0: 
                                r_s = r_s + W_[k][0]*(abs(S[k] - theta_mean[k]))
                        else:
                                r_s = r_s + W_[k][1]*(abs(S[k] - theta_mean[k]))
                return r_s


        def updateStat(self, trj_Sp_theta):      
                import numpy as np
                theta_mean = []
                theta_std = []
                for theta in range(len(trj_Sp_theta)):
                        theta_mean.append(np.mean(trj_Sp_theta[theta]))
                        theta_std.append(np.std(trj_Sp_theta[theta]))
                self.theta_std = theta_std
                self.theta_mean = theta_mean
        

        def reward_trj(self, trj_Sp_theta, W_):
                """
                
                """
                import numpy as np
                #theta_mean = []
                #theta_std = []
                #for theta in range(len(W_)):
                #        theta_mean.append(np.mean(trj_Sp_theta[theta]))
                #        theta_std.append(np.std(trj_Sp_theta[theta]))
                

                r = []
                # for over all dicovered states
                trj_Sp_theta = np.array(trj_Sp_theta)
                for state_index in range(len(trj_Sp_theta[0])):
                        #print('trj_Sp_theta', trj_Sp_theta)
                        state_theta = trj_Sp_theta[:, state_index]
                        r_s = self.reward_state(state_theta, self.theta_mean, self.theta_std, W_)
                        
                        r.append(r_s)
                        
                R = np.sum(np.array(r))
                return R
                
        
        
        def updateW(self, trj_Sp_theta, W_0):
                """
                update weigths 
                prior_weigths = W_0
                """
                def fun(x):
                        global trj_Sp_theta_z
                        W_0 = [[x[0], x[1]],[x[2], x[3]]]
                        r_0 = self.reward_trj(trj_Sp_theta, W_0)
                        return -1*r_0                        
                import numpy as np
                from scipy.optimize import minimize
                
                global trj_Sp_theta_z 
                trj_Sp_theta_z = trj_Sp_theta
                alpha = 0.2
                delta = alpha
                cons = ({'type': 'eq',
                          'fun' : lambda x: np.array([x[0]+x[1]+x[2]+x[3]-1])},
                         {'type': 'ineq',
                          'fun' : lambda x: np.array([x[1]])},
                         {'type': 'ineq',
                          'fun' : lambda x: np.array([x[0]])},
                         {'type': 'ineq',
                          'fun' : lambda x: np.array([x[2]])},
                         {'type': 'ineq',
                          'fun' : lambda x: np.array([x[3]])},
                          {'type': 'ineq',
                          'fun' : lambda x: np.array([x[0]-x0[0]+delta])},
                          {'type': 'ineq',
                          'fun' : lambda x: np.array([x0[0]-x[0]+delta])},
                          {'type': 'ineq',
                          'fun' : lambda x: np.array([x[1]-x0[1]+delta])},
                          {'type': 'ineq',
                          'fun' : lambda x: np.array([x0[1]-x[1]+delta])},
                          {'type': 'ineq',
                          'fun' : lambda x: np.array([x[2]-x0[2]+delta])},
                          {'type': 'ineq',
                          'fun' : lambda x: np.array([x0[2]-x[2]+delta])},
                          {'type': 'ineq',
                          'fun' : lambda x: np.array([x[3]-x0[3]+delta])},
                          {'type': 'ineq',
                          'fun' : lambda x: np.array([x0[3]-x[3]+delta])},
                          )



                x0 = [W_0[0][0], W_0[0][1], W_0[1][0], W_0[1][1]]    
                
                res = minimize(fun, x0, constraints=cons)
                x = res.x
                W = [[x[0], x[1]],[x[2], x[3]]]
                return W
                
        def findStarting(self, trj_Sp_theta, trj_Sp, W_1, starting_n=10 , method = 'RL'):
                # get new starting points (in theta domain) using new reward function based on updated weigths (W_1)
                import numpy as np
                """
                print(len(trj_Sp_theta[0]), starting_n)
                
                while len(trj_Sp_theta[0])<starting_n:
                        print(len(trj_Sp_theta[0]), starting_n)
                        print('trj_Sp_theta<starting_n')
                        trj_Sp_theta = np.array([np.concatenate([trj_Sp_theta[0], trj_Sp_theta[0]]), np.concatenate([trj_Sp_theta[1], trj_Sp_theta[1]])])
        
                while len(trj_Sp[0])<starting_n:
                        print('trj_Sp<starting_n')
                        print(len(trj_Sp[0]), starting_n)
                        trj_Sp = np.array([np.concatenate([trj_Sp[0], trj_Sp[0]]), np.concatenate([trj_Sp[1], trj_Sp[1]])])
                
                #print('trj_Sp', trj_Sp)
                """
                
                theta_mean = []
                theta_std = []
                for theta in range(len(W_1)):
                        theta_mean.append(np.mean(trj_Sp_theta[theta]))
                        theta_std.append(np.std(trj_Sp_theta[theta]))
                        
                ranks = {}
                for state_index in range(len(trj_Sp_theta[0])):
                        #print(len(trj_Sp_theta[0]))
                        state_theta = trj_Sp_theta[:,state_index]
                        
                        r = self.reward_state( state_theta, theta_mean, theta_std, W_1)
                        
                        ranks[state_index] = r

                newPoints_index0 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[0:starting_n] 
                newPoints_index = np.array(newPoints_index0)[:,0]   
                
                n_coord = len(trj_Sp)
                                     
                newPoints = []
                for coord in range(n_coord):
                          newPoints.append([trj_Sp[coord][int(i)] for i in newPoints_index])                                   
                return newPoints
        
        
        def creatPotentioal(self):
                return True
                
        def run(self, init, nstepmax = 10):
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
                msm = self.msm
                
                trj = np.empty([N,s])
                for n in range(N):
                        init = np.int(inits[n])
                        trj = msm.sample_discrete(state=init, n_steps=nstepmax, random_state=None)
                
                return trjs
                
    





