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
                #trj_Sp_theta = trj_Sp

                # change the format
                trj_Ps_theta_2 = []
                ##############
                trj_Ps_theta = np.array(trj_Ps_theta) 
                for theta_index in range(len(trj_Ps_theta[0])):
                        trj_Ps_theta_2.append(trj_Ps_theta[:,theta_index])
                return trj_Ps_theta_2

        def reward_state(self, S, theta_mean, theta_std, W_):
                """
                with direction
                """
                r_s = 0
                for k in range(len(W_)):
                        """
                        r_s = r_s + W_[k]*(abs(S[k] - theta_mean[k])/theta_std[k]) #No direction
                        """
                        if (S[k] - theta_mean[k]) < 0: 
                                r_s = r_s + W_[k][0]*(abs(S[k] - theta_mean[k])/theta_std[k])
                        else:
                                r_s = r_s + W_[k][1]*(abs(S[k] - theta_mean[k])/theta_std[k])
                        
                return r_s
 
        def reward_state_noDir(self, S, theta_mean, theta_std, W_):
                # no direction
                r_s = 0
                for k in range(len(W_)):
                        r_s = r_s + W_[k]*(abs(S[k] - theta_mean[k])/theta_std[k]) #No direction
                return r_s

        def reward_state_withoutStd(self, S, theta_mean, theta_std, W_):
                
                r_s = 0
                for k in range(len(W_)):
                        
                        r_s = r_s + W_[k]*(abs(S[k] - theta_mean[k])) # no direction
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

                r = []
                # for over all dicovered states
                trj_Sp_theta = np.array(trj_Sp_theta)
                for state_index in range(len(trj_Sp_theta[0])):
                        state_theta = trj_Sp_theta[:, state_index]
                        r_s = self.reward_state(state_theta, self.theta_mean, self.theta_std, W_)
                        
                        r.append(r_s)
                        
                R = np.sum(np.array(r))
                return R
                
        
        
        def updateW(self, trj_Sp_theta, W_0):
                """
                update weigths 
                prior_weigths = W_0
                with considering direction
                """
                def fun(x):
                        global trj_Sp_theta_z
                        global n_ec
                        import numpy as np
                        x = np.array(x)
                        W_0 = x.reshape(n_ec, 2)
                        # W_0 = x
                        r_0 = self.reward_trj(trj_Sp_theta, W_0)
                        return -1*r_0     
                
                import numpy as np
                from scipy.optimize import minimize
                
                global trj_Sp_theta_z 
                global n_ec
                
                trj_Sp_theta_z = trj_Sp_theta
                alpha = 0.2
                delta = alpha
                cons = ({'type': 'eq',
                          'fun' : lambda x: np.array([np.sum(x)-1])},
                         {'type': 'ineq',
                          'fun' : lambda x: np.array([np.min(x)])},
                         {'type': 'ineq',
                          'fun' : lambda x: np.array([np.abs(np.sum(x-x0))+delta])})

                #x0 = W_0
                x0 = np.concatenate(W_0)
                res = minimize(fun, x0, constraints=cons)

                x = res.x
                x = x/(np.sum(x))
                W = x.reshape(n_ec, 2)
                
                #W = x
                return W
 
        def updateW_noDir(self, trj_Sp_theta, W_0):
                """
                update weigths 
                prior_weigths = W_0
                no direction
                """
                def fun(x):
                        global trj_Sp_theta_z

                        W_0 = x
                        r_0 = self.reward_trj(trj_Sp_theta, W_0)
                        return -1*r_0                        
                import numpy as np
                from scipy.optimize import minimize
                
                global trj_Sp_theta_z 
                trj_Sp_theta_z = trj_Sp_theta
                alpha = 0.2
                delta = alpha
                cons = ({'type': 'eq',
                          'fun' : lambda x: np.array([np.sum(x)-1])},
                         {'type': 'ineq',
                          'fun' : lambda x: np.array([np.min(x)])},
                         {'type': 'ineq',
                          'fun' : lambda x: np.array([np.abs(np.sum(x-x0))+delta])})

                x0 = W_0
                res = minimize(fun, x0, constraints=cons)

                x = res.x

                W = x
                return W    
	
	
        def findStarting(self, trj_Ps_theta, trj_Ps, W_1, starting_n=10 , method = 'RL'):
                """
                trj_Ps_theta: 
                         size n_theta x n_frames
                trj_Ps:
                """
                # get new starting points (in theta domain) using new reward function based on updated weigths (W_1)
                import numpy as np               
                theta_mean = []
                theta_std = []
                for theta in range(len(W_1)):
                        theta_mean.append(np.mean(trj_Ps_theta[theta]))
                        theta_std.append(np.std(trj_Ps_theta[theta]))
                        
                ranks = {}
                trj_Ps_theta = np.array(trj_Ps_theta)
                for state_index in range(len(trj_Ps_theta[0])):
                        state_theta = trj_Ps_theta[:,state_index]
                        
                        r = self.reward_state( state_theta, theta_mean, theta_std, W_1)
                        
                        ranks[state_index] = r

                newPoints_index0 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[0:starting_n] 
                newPoints_index = np.array(newPoints_index0)[:,0]   
                

                n_coord = 1                     
                newPoints = [trj_Ps[int(i)] for i in newPoints_index]                              
                return newPoints
        
        
        def creatPotentioal(self):
                return True
                
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
                msm = self.msm
                N = len(inits)
                trjs = np.empty([N, nstepmax])
                for n in range(N):
                        init = np.int(inits[n])
                        trj = msm.sample_discrete(state=init, n_steps=nstepmax, random_state=None)
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

########################################################################################################################
# From calculated and saved files reads if the state is an active state or not
########################################################################################################################	

        def isActive(self, state):
                import numpy as np
                isActive = np.load('isActive/isActive'+str(int(state))+'.npy')
                return isActive


        def runSimulation(self, R=3,N=10,s=8, method='RL'):
                import numpy as np
                global n_ec
                activeTime = -1
                init = 132
                inits = [init for i in range(N)]
                n_ec = 50
                #W_0 = [1/n_ec for i in range(n_ec)] # no direction
                W_0 = [[1/(2*n_ec), 1/(2*n_ec)] for i in range(n_ec)] # consider direction
                Ws = []
                trj1 = self.run(inits, nstepmax = s)
                comb_trj1 = np.concatenate(trj1)
                trjs = comb_trj1
                trj1_Ps = self.PreSamp_MC(trj1, N = 3*N) # pre analysis , 1 x n_frames
                trj1_Ps_theta = self.map(trj1_Ps)
                newPoints = self.findStarting(trj1_Ps_theta, trj1_Ps, W_0, starting_n = N , method = 'RL')
                trjs_theta = trj1_Ps_theta
                trjs_Ps_theta = trj1_Ps_theta
                
                count = 1
                for round in range(R):
                        self.updateStat(trjs_theta) # based on all trajectories
                        W_1 = self.updateW(trjs_Ps_theta, W_0)
                        W_0 = W_1
                        Ws.append(W_0)
                        
                        trj1 = self.run(newPoints, nstepmax = s) # N (number of parallel) x n_all_frames
                        isActive = self.isActive_singleRound(trj1)
                        trj1 = np.concatenate(trj1) # 1 x n_all_frames
                        
                        if int(isActive)!=-1:
                                print('Active')
                                activeTime = (round)*N*s+isActive
                                break
                                
                        com_trjs = np.concatenate((trjs, trj1))
                        trjs = np.array(com_trjs)
                        trjs_theta = np.array(self.map(trjs))
                        trjs_Ps = self.PreSamp_MC(trjs, N = 4*N)
                        trjs_Ps_theta = np.array(self.map(trjs_Ps))
                        newPoints = self.findStarting(trjs_Ps_theta, trjs_Ps, W_1, starting_n = N , method = 'RL')
                        count = count + 1
                        
                np.save('activeTime_'+'r'+str(R)+'N'+str(N)+'s'+str(s), activeTime)
                np.save('w_'+'r'+str(R)+'N'+str(N)+'s'+str(s), Ws)
                return activeTime
                        

        def multiSim_timeCal_script(self, method='RL'):

                # T_len = [1,2,3,4,5,6,7,8,9] # lenght of trajectories
                T_n = range(10,1010,10) # number of trajectories
                T_len = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000]
                N=10
                l = len(T_len)
                n = len(T_n)
                count = 1
                for i in range(l):
                        for j in range(n):
                                T_len1 = T_len[i]
                                T_n1 = T_n[j]
                                r=T_n1/N
                                N=10
                                s=T_len1
                                myfile = open('run_'+'r'+str(r)+'N'+str(N)+'s'+str(s)+'.py','w')
                                myfile.write('import pickle \n')
                                myfile.write('import RLSim as rl \n')
                                myfile.write('import numpy as np \n')
                                myfile.write('msm =  pickle.load(open(\'MSM150.pkl\',\'rb\')) \n')
                                myfile.write('my_sim = rl.mockSimulation() \n')
                                myfile.write('my_sim.msm = msm \n')
                                myfile.write('my_sim.runSimulation(s='+str(s)+', R='+ str(int(r)) +', N='+ str(N)+') \n')
                                myfile.close()
                                myRun = open('Run_'+str(count),'w')
                                myRun.write('ipython run_'+'r'+str(r)+'N'+str(N)+'s'+str(s)+'.py','w')
                                myRun.close()
                                count = count + 1
                return

        def collect_times(self):
                import numpy as np
                #T_len = [1,2,3,4,5,6,7,8,9] # lenght of trajectories
                T_len = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000]
                T_n = range(10,1010,10) # number of trajectories
                N=10
                l = len(T_len)
                n = len(T_n)
                time = np.empty([l, n])
                for i in range(l):
                        for j in range(n):
                                T_len1 = T_len[i]
                                T_n1 = T_n[j]
                                r=T_n1/N
                                N=10
                                s=T_len1
                                t = np.load('activeTime_'+'r'+str(r)+'N'+str(N)+'s'+str(s)+'.npy')
                                print(t)
                                time[i][j] = t.item()
                np.savetxt('times.txt', time)
                return time

	

def pltTimes(times, filename='pcolormesh_timeReachingActive.png'):
	
	import matplotlib.pyplot as plt
	from matplotlib.colors import LogNorm
	font = {'family':'Times New Roman', 'size': 22}
	plt.rc('font', **font)

	T_len = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000]
	T_n = range(10,1000,10)
	
	fig, ax = plt.subplots(1)

	p = ax.pcolormesh(times, norm=LogNorm(vmin=1000, vmax=5000000))
	xticks = range(len(T_n))
	yticks = range(len(T_len))
	
	T_n1 = [1, 20,40,60,80,100] # Number of rounds
	ax.set_xticklabels(T_n1)
	T_len1 = [' ', 5,10,50,100,500,1000, 5000]
	T_len1 = [' ', 5,100,50,' ' ,500, 5000]
	
	ax.set_yticklabels(T_len1)
	
	cbar = fig.colorbar(p, label='ms')
	cbar.ax.set_yticklabels([0.05,0.5,5,50])
	ax.set(xlabel='# Rounds of Simulation', ylabel=r'Trajectory Length/$\tau$')
	ax.set_xlim([0,99])
	ax.set_ylim([0,32])
	
	fig.set_size_inches(9, 7)

	fig.savefig(filename, dpi=300)
	fig.show()

