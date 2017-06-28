class mockSimulation:
        ## public
        def __init__(self):
                self.theta_mean = []
                self.theta_std = []
                self.r = 1#number of rounds
                self.s = 1# length of simulations
                self.N = 1# number of parallel simulations
                self.msm = None           
        
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
                return trj_Sp


        def PreSamp(self, trj, starting_n=1, myn_clusters = 40): #?!!!!
                """
                Pre-Sampling:
                        choose states with minimum counts or newly discovered states
                        
                output:
                        trj with shape of [[Xs][Ys]]
                        index of the trj
                """
                import numpy as np
                comb_trj = trj
                
                # clustering
                from sklearn.cluster import KMeans
                comb_trj_xy = np.array([[comb_trj[0][i], comb_trj[1][i]] for i in range(len(comb_trj[0]))])
                cluster = KMeans(n_clusters=myn_clusters)
                cluster.fit(comb_trj_xy)
                cl_trjs = cluster.labels_
                
                # finding least count cluster
                N = 1
                unique, counts = np.unique(cl_trjs, return_counts=True)
                leastPop = counts.argsort()[:N]
                init_cl = [unique[i] for i in leastPop]
                
                # 
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

                # if number of states with least count is less than desired number of starting states
                while len(trj_Sp[0])<starting_n:
                        print('trj_Sp<starting_n')
                        print(len(trj_Sp[0]), starting_n)
                        trj_Sp = np.array([np.concatenate([trj_Sp[0], trj_Sp[0]]), np.concatenate([trj_Sp[1], trj_Sp[1]])])

                return trj_Sp, init_index
        

                
        def map(self, trj):
                """
                trj:
                      mdtraj pbject
                output:
                      n_ec x n_frames
                """
                # map coordinate space to reaction coorinates space
                import mdtraj as md
                phi = md.compute_phi(trj)[1]
                psi = md.compute_psi(trj)[1]
                trj_theta = []
                trj_theta.append(phi)
                trj_theta.append(psi)
                return trj_theta
 
        def reward_state(self, S, theta_mean, theta_std, W_):
                # no direction
                r_s = 0
                for k in range(len(W_)):
                        r_s = r_s + W_[k]*(abs(S[k] - theta_mean[k])/theta_std[k]) #No direction
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
                with direction depending on self.reward_state
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
        
        def findStarting(self, trj_Ps_theta, index_orig, W_1, starting_n=1 , method = 'RL'):
                """
                trj_Ps_theta: 
                         size n_theta x n_frames
                trj_Ps:
                """
                # get new starting points (in theta domain) using new reward function based on updated weigths (W_1)
                
                # calculate stat for the pre-sampled trj
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
                        
                        r = self.reward_state(state_theta, theta_mean, theta_std, W_1)
                        
                        ranks[state_index] = r

                newPoints_index0 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[0:starting_n] 
                newPoints_index = np.array(newPoints_index0)[:,0]   
                

                n_coord = 1                     
                newPoints = [trj_Ps[int(i)] for i in newPoints_index]
                newPoints_index_orig = [index_orig[int(i)] for i in newPoints_index]
                return newPoints, newPoints_index_orig
        
        
               
        def run(self, production_steps = 200, start='ala2_1stFrame.pdb', production='ala2_production.pdb'): #### ?!!!!!!!!!!!!!!!!
                from __future__ import print_function
                from simtk.openmm import app
                import simtk.openmm as mm
                from simtk import unit
                from sys import stdout
                 
                nonbondedCutoff = 1.0*unit.nanometers
                timestep = 2.0*unit.femtoseconds
                temperature = 300*unit.kelvin
                save_frequency = 100 #Every 100 steps, save the trajectory

                pdb = app.PDBFile(start)
                forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
                ala2_model = app.Modeller(pdb.topology, pdb.positions)

                #Hydrogens will be constrained after equilibrating
                system = forcefield.createSystem(ala2_model.topology, nonbondedMethod=app.PME, 
                    nonbondedCutoff=nonbondedCutoff, constraints=app.HBonds, rigidWater=True, 
                    ewaldErrorTolerance=0.0005)

                system.addForce(mm.MonteCarloBarostat(1*unit.bar, temperature, 100)) #Apply Monte Carlo Pressure changes in 100 timesteps
                integrator = mm.LangevinIntegrator(temperature, 1.0/unit.picoseconds, 
                    timestep)
                integrator.setConstraintTolerance(0.00001)

                platform = mm.Platform.getPlatformByName('CPU')
                simulation = app.Simulation(ala2_model.topology, system, integrator, platform)
                simulation.context.setPositions(ala2_model.positions)

                simulation.context.setVelocitiesToTemperature(temperature)

                simulation.reporters.append(app.PDBReporter(production, save_frequency))
                simulation.reporters.append(app.StateDataReporter('stateReporter_constantPressure.txt', 1000, step=True, 
                   totalEnergy=True, temperature=True, volume=True, progress=True, remainingTime=True, 
                    speed=True, totalSteps=production_steps, separator='\t'))

                print('Running Production...')
                simulation.step(production_steps)
                print('Done!')
                
                import mdtraj as md
                trj = md.load(production)
                return trj
                

        def runSimulation(self, R=3,N=1,s=200, method='RL'):
                global n_ec
                import numpy as np
                
                init = 'ala2_1stFrame.pdb' #pdb name
                inits = init
                #inits = [init for i in range(N)]
                n_ec = 2
                W_0 = [1/n_ec for i in range(n_ec)] # no direction
                #W_0 = [[1/(2*n_ec), 1/(2*n_ec)] for i in range(n_ec)] # consider direction
                Ws = []
                trj1 = self.run(production_steps = s, start=inits, production='trj_R_0.pdb') # return mdtraj object
                comb_trj1 = trj1 # single trajectory
                trjs = comb_trj1
                trj1_theta = self.map(trj1)
                trj1_Ps_theta, index = self.PreSamp(trj1_theta) # pre analysis (least count)
                
                #newPoints = self.findStarting(trj1_Ps_theta, trj1_Ps, W_0, starting_n = N , method = 'RL')
                newPoints_theta, newPoints_index_orig = self.findStarting(trj1_Ps_theta, index, W_0, starting_n = N , method = 'RL')
                newPoints = [trj1[i] for i in newPoints_index_orig] # extract a frame ?!!!!
                
                trjs_theta = trj1_theta
                trjs_Ps_theta = trj1_Ps_theta
                
                count = 1
                for round in range(R):
                        self.updateStat(trjs_theta) # based on all trajectories
                        W_1 = self.updateW(trjs_Ps_theta, W_0)
                        W_0 = W_1
                        Ws.append(W_0)
                        
                        trj1 = self.run(production_steps = s, start=newPoints, production='trj_R_'+str(count)+'.pdb') # return mdtraj object

                        #trj1 = np.concatenate(trj1) # 1 x n_all_frames  # single trj
        
                        com_trjs = np.concatenate((trjs, trj1)) # revise!!!! use mdtraj
                        #trjs = np.array(com_trjs)
                        trjs = com_trjs
                        trjs_theta = np.array(self.map(trjs))
                        #trjs_theta = np.array(self.map(trjs))
                        trjs_Ps_theta, index = self.PreSamp(trjs_theta)
                        
                        newPoints, newPoints_index_orig = self.findStarting(trjs_Ps_theta, trjs_Ps, W_1, starting_n = N , method = 'RL')
                        count = count + 1
                        
                np.save('w_'+'r'+str(int(R))+'N'+str(N)+'s'+str(s), Ws)
                return 
                        

        def multiSim_timeCal_script(self, method='RL'):

                T_n = range(10,1010,10) # number of trajectories
                T_len = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] # lenght of trajectories
                N = 10
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
                                myfile = open('run_'+'r'+str(int(r))+'N'+str(N)+'s'+str(s)+'.py','w')
                                myfile.write('import pickle \n')
                                myfile.write('import RLSim as rl \n')
                                myfile.write('import numpy as np \n')
                                myfile.write('msm =  pickle.load(open(\'MSM.pkl\',\'rb\'), encoding=\'latin1\') \n')
                                myfile.write('my_sim = rl.mockSimulation() \n')
                                myfile.write('my_sim.msm = msm \n')
                                myfile.write('my_sim.runSimulation(s='+str(int(s))+', R='+ str(int(r)) +', N='+ str(N)+') \n')
                                myRun = open('Run_'+str(count),'w')
                                myRun.write('ipython run_'+'r'+str(int(r))+'N'+str(N)+'s'+str(int(s))+'.py')
                                myRun.close()
                                count = count + 1
                                myfile.close()
                return



       
####################################

        def updateW_withDir(self, trj_Sp_theta, W_0):
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
        
        
        def reward_state_withDir(self, S, theta_mean, theta_std, W_):
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
        
