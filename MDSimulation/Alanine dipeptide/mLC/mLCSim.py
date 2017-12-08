
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


        def PreSamp(self, trj, starting_n=1, myn_clusters = 40): 
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
                #comb_trj_xy = np.array([[comb_trj[0][i], comb_trj[1][i]] for i in range(len(comb_trj[0]))]) ### Needs to be changed for diffretn directions
                comb_trj_xy = np.array([[comb_trj[0][i], comb_trj[1][i], comb_trj[2][i], comb_trj[3][i]] for i in range(len(comb_trj[0]))])
                trj_phi_sin = np.sin(np.array(comb_trj[0]))
                trj_phi_cos = np.cos(np.array(comb_trj[0]))
                trj_psi_sin = np.sin(np.array(comb_trj[1]))
                trj_psi_cos = np.cos(np.array(comb_trj[1]))                  
                trj_theta_sin = np.sin(np.array(comb_trj[2]))
                trj_theta_cos = np.cos(np.array(comb_trj[2]))
                trj_ksi_sin = np.sin(np.array(comb_trj[3]))
                trj_ksi_cos = np.cos(np.array(comb_trj[3]))
                
                #comb_trj_sincos = np.array([[trj_phi_sin[i], trj_phi_cos[i], trj_psi_sin[i], trj_psi_cos[i]] for i in range(len(trj_phi_sin))])
                comb_trj_sincos = np.array([[trj_phi_sin[i], trj_phi_cos[i], trj_psi_sin[i], trj_psi_cos[i], trj_theta_sin[i], trj_theta_cos[i], trj_ksi_sin[i], trj_ksi_cos[i]] for i in range(len(trj_phi_sin))])
                cluster = KMeans(n_clusters=myn_clusters)
                cluster.fit(comb_trj_sincos)
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
                #init_trj = [[init_trj_xy[i][0] for i in range(len(init_trj_xy))], [init_trj_xy[i][1] for i in range(len(init_trj_xy))]]
                init_trj = [[init_trj_xy[i][0] for i in range(len(init_trj_xy))], [init_trj_xy[i][1] for i in range(len(init_trj_xy))], [init_trj_xy[i][2] for i in range(len(init_trj_xy))],[init_trj_xy[i][3] for i in range(len(init_trj_xy))]]
                
                trj_Sp = init_trj

                # if number of states with least count is less than desired number of starting states
                while len(trj_Sp[0])<starting_n:
                        print('trj_Sp<starting_n')
                        print(len(trj_Sp[0]), starting_n)
                        trj_Sp = np.array([np.concatenate([trj_Sp[0], trj_Sp[0]]), np.concatenate([trj_Sp[1], trj_Sp[1]]), np.concatenate([trj_Sp[2], trj_Sp[2]]), np.concatenate([trj_Sp[3], trj_Sp[3]])])
                        #trj_Sp = np.array([np.concatenate([trj_Sp[0], trj_Sp[0]]), np.concatenate([trj_Sp[1], trj_Sp[1]])])
                return trj_Sp, init_index

        def map_angles(self, trj):
                """
                trj:
                      mdtraj pbject
                output:
                      n_ec x n_frames
                """
                # map coordinate space to reaction coorinates space
                import mdtraj as md
                import numpy as np
                
                phi = md.compute_phi(trj)[1]
                z_phi = np.array([phi[i][0] for i in range(len(phi))]) # in rad
                psi = md.compute_psi(trj)[1]
                z_psi = np.array([psi[i][0] for i in range(len(psi))]) # in rad
                # (name CA or name N and resname ALA) or ((name C or name O) and resname ACE)
                #atom_indix_theta = trj.topology.select('name O or name C or name N or name CA')
                atom_indix_theta = trj.topology.select('(name O and resname ACE) or (name C and resname ACE) or (name N and resname ALA) or (name CA and resname ALA)')
                theta = md.compute_dihedrals(trj, [atom_indix_theta])
                z_theta = np.array([theta[i][0] for i in range(len(theta))])
                
                # (name CA) or ((name C and resname ALA) or ((name N or name H) and resname NME))
                #atom_indix_ksi = trj.topology.select('name CA or name C or name N or name H')
                atom_indix_ksi = trj.topology.select('(name CA and resname ALA) or (name C and resname ALA) or (name N and resname NME) or (name H and resname NME)')
                ksi = md.compute_dihedrals(trj, [atom_indix_ksi])
                z_ksi = np.array([ksi[i][0] for i in range(len(ksi))])
               
                trj_theta2 = []
                trj_theta2.append(z_phi)
                trj_theta2.append(z_psi)
                trj_theta2.append(z_theta)
                trj_theta2.append(z_ksi)
                return trj_theta2

        def map(self, trj):
                """
                trj:
                      mdtraj pbject
                output:
                      n_ec x n_frames
                """
                
                # map coordinate space to reaction coorinates space
                import mdtraj as md
                import numpy as np
                
                phi = md.compute_phi(trj)[1]
                z_phi = np.array([phi[i][0] for i in range(len(phi))])
                psi = md.compute_psi(trj)[1]
                z_psi = np.array([psi[i][0] for i in range(len(psi))])

                trj_theta = []
                trj_theta.append(np.sin(z_phi)) # sin's input is in Radian
                trj_theta.append(np.cos(z_phi))
                trj_theta.append(np.sin(z_psi))
                trj_theta.append(np.cos(z_psi))
                return trj_theta

 
        def reward_state(self, S, theta_mean, theta_std, W_):
                # no direction
                r_s = 0
                import numpy as np
                for k in range(len(W_)):
                    if theta_std[k]==0:
                        print(theta_std[k])
                        theta_std[k]=1
                    dist = np.min([abs(S[k] - theta_mean[k]), abs(S[k] - theta_mean[k]-2*np.pi), abs(S[k] - theta_mean[k]+2*np.pi)])
                    if (S[k] - theta_mean[k]) < 0:  # direstional
                         r_s = r_s + W_[k][0]*(dist/theta_std[k]) 
                        #r_s = r_s + W_[k][0]*(abs(S[k] - theta_mean[k])/theta_std[k]) # W_[k][0] is weight for W_ negetive direction
                    else:
                        r_s = r_s + W_[k][1]*(dist/theta_std[k])
                        #r_s = r_s + W_[k][1]*(abs(S[k] - theta_mean[k])/theta_std[k]) # W_[k][1] is weight for W_ positive direction
                return r_s

        def findStarting(self, trj_Ps_theta, index_orig, starting_n=1 , method = 'LC'):
                """
                trj_Ps_theta: 
                         size n_theta x n_frames
                trj_Ps:
                """
                # get new starting points (in theta domain) using new reward function based on updated weigths (W_1)
                
                # calculate stat for the pre-sampled trj
                import numpy as np               

                newPoints_index = range(starting_n)   
                 
                newPoints_index_orig = [index_orig[int(i)] for i in newPoints_index]
                return newPoints_index_orig

                
                
        def run(self, production_steps = 200, start='ala2_1stFrame.pdb', production='ala2_production.pdb'): #### ?!!!!!!!!!!!!!!!!
                #from __future__ import print_function
                from simtk.openmm import app
                import simtk.openmm as mm
                from simtk import unit
                from sys import stdout
                 
                nonbondedCutoff = 1.0*unit.nanometers
                timestep = 2.0*unit.femtoseconds
                temperature = 300*unit.kelvin
                #save_frequency = 100 #Every 100 steps, save the trajectory
                save_frequency = 10 #Every 1 steps, save the trajectory

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
                #simulation.reporters.append(app.StateDataReporter('stateReporter_constantPressure.txt', 1000, step=True, 
                #   totalEnergy=True, temperature=True, volume=True, progress=True, remainingTime=True, 
                #    speed=True, totalSteps=production_steps, separator='\t'))

                print('Running Production...')
                simulation.step(production_steps)
                print('Done!')
                
                import mdtraj as md
                trj = md.load(production)
                return trj
                

        def runSimulation(self, R=5000, N=1,s=1000, method='RL'):
                """
                theta is the set of sine and cosine of the angles
                theta2 is set of the angles
                """
                global n_ec
                import numpy as np
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                matplotlib.pyplot.switch_backend('agg')

                plt.rcParams.update({'font.size':20})
                plt.rc('xtick', labelsize=20)
                plt.rc('ytick', labelsize=20)
                # step = 2 fs
                # each round is 2 fs * 1000 = 2 ps

                init = 'ala2_1stFrame.pdb' #pdb name
                inits = init
                n_ec = 2 # angles
                n_ec = 4 # four angles
                count = 1
                newPoints_name = 'start_r_'+str(count)+'.pdb'
                
                trj1 = self.run(production_steps = s, start=inits, production='trj_R_0.pdb') # return mdtraj object
                comb_trj1 = trj1 # single trajectory
                trjs = comb_trj1
                trj1_theta = self.map_angles(trj1) # changed for angles to display
                print('trj1_theta', len(trj1_theta), len(trj1_theta[0]))
                trj1_Ps_theta, index = self.PreSamp(trj1_theta, myn_clusters = 10) # pre analysis (least count)

                newPoints_index_orig = self.findStarting(trj1_Ps_theta, index, starting_n = N , method = 'LC') #need change
                newPoints = trj1[newPoints_index_orig[0]]
                newPoints.save_pdb(newPoints_name)
                
                print('trj1_theta[0]',trj1_theta[0])
                plt.scatter(trj1_theta[0], trj1_theta[1], color='dodgerblue', s=5, alpha=0.2)
                plt.xlim([-np.pi, np.pi])
                plt.ylim([-np.pi, np.pi])
                newPoints_theta_x = trj1_theta[0][newPoints_index_orig[0]]
                newPoints_theta_y = trj1_theta[1][newPoints_index_orig[0]]
                plt.scatter(newPoints_theta_x, newPoints_theta_y, color='red', s=50)
                plt.xlabel(r'$\phi$')
                plt.ylabel(r'$\psi$')
                plt.savefig('fig_'+str(count))
                plt.close()
                
                trjs_theta = trj1_theta
                trjs_Ps_theta = trj1_Ps_theta
                for round in range(R):
                        s = 1000
                        trj1 = self.run(production_steps = s, start=newPoints_name, production='trj_R_'+str(count)+'.pdb') # return mdtraj object
                        com_trjs = trjs.join(trj1) 
                        trjs = com_trjs
                        
                        trjs_theta = np.array(self.map_angles(trjs)) 
                        trjs_Ps_theta, index = self.PreSamp(trjs_theta, myn_clusters = 100)


                        newPoints_index_orig = self.findStarting(trjs_Ps_theta, index,starting_n = N , method = 'LC')
                        newPoints = trjs[newPoints_index_orig[0]] 
                        
                        count = count + 1
                        newPoints_name = 'start_r_'+str(count)+'.pdb'
                        newPoints.save_pdb(newPoints_name)
                        if round % 20 ==0:
                                plt.scatter(trjs_theta[0], trjs_theta[1], color='dodgerblue', s=5, alpha=0.2)
                                plt.xlim([-np.pi, np.pi])
                                plt.ylim([-np.pi, np.pi])
                                newPoints_theta_x = trjs_theta[0][newPoints_index_orig[0]]
                                newPoints_theta_y = trjs_theta[1][newPoints_index_orig[0]]
                                plt.scatter(newPoints_theta_x, newPoints_theta_y, color='red', s=50)
                                plt.xlabel(r'$\phi$')
                                plt.ylabel(r'$\psi$')
                                plt.savefig('fig_'+str(count))
                                plt.close()

                np.save('trjs_theta', trjs_theta)
                return 
                        


