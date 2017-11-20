
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
                trj_theta2 = []
                trj_theta2.append(z_phi)
                trj_theta2.append(z_psi)
                return trj_theta2
        
        def run(self, production_steps = 200, start='ala2_1stFrame.pdb', production='ala2_production.pdb'): ##
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
                simulation.reporters.append(app.StateDataReporter('stateReporter_constantPressure.txt', 1000, step=True, 
                   totalEnergy=True, temperature=True, volume=True, progress=True, remainingTime=True, 
                    speed=True, totalSteps=production_steps, separator='\t'))

                print('Running Production...')
                simulation.step(production_steps)
                print('Done!')
                
                import mdtraj as md
                trj = md.load(production)
                return trj
                

        def runSimulation_single(self, N=1,s=1000, method='RL'):
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
                trj1 = self.run(production_steps = s, start=inits, production='trj_R_0.pdb') # return mdtraj object
                comb_trj1 = trj1 # single trajectory
                trjs = comb_trj1
                trj1_theta = self.map_angles(trj1) # changed for angles to display
                print('trj1_theta', len(trj1_theta), len(trj1_theta[0]))

                newPoints = trj1[newPoints_index_orig[0]]
                newPoints.save_pdb(newPoints_name)
                
                print('trj1_theta[0]',trj1_theta[0])
                plt.scatter(trj1_theta[0], trj1_theta[1], color='dodgerblue', s=5, alpha=0.2)
                plt.xlim([-np.pi, np.pi])
                plt.ylim([-np.pi, np.pi])

                plt.xlabel(r'$\phi$')
                plt.ylabel(r'$\psi$')
                plt.savefig('fig_'+str(count))
                plt.close()
                trjs_theta = trj1_theta
                np.save('trjs_theta', trjs_theta)
                return 

        
        def runSimulation(self, R=1000, N=1,s=1000, method='RL'):
                global n_ec
                import numpy as np
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                matplotlib.pyplot.switch_backend('agg')
                # step = 2 fs
                # each round is 2 fs * 1000 = 2 ps

                init = 'ala2_1stFrame.pdb' #pdb name
                inits = init
                
                count = 1
                newPoints_name = 'start_r_'+str(count)+'.pdb'
                trj1 = self.run(production_steps = s, start=inits, production='trj_R_0.pdb') # return mdtraj object
                comb_trj1 = trj1 # single trajectory
                trjs = comb_trj1
                trj1_theta = self.map_angles(trj1)
                
                newPoints_index_orig = -1
                newPoints = trj1[newPoints_index_orig]
                
                newPoints.save_pdb(newPoints_name)
                trjs_theta = trj1_theta
                trjs_Ps_theta = trj1_Ps_theta

                for round in range(R):
                        s = 1000
                        trj1 = self.run(production_steps = s, start=newPoints_name, production='trj_R_'+str(count)+'.pdb') # return mdtraj object

                        com_trjs = trjs.join(trj1) 
                        trjs = com_trjs
                        
                        trjs_theta = np.array(self.map_angles(trjs))
                        newPoints_index_orig = -1
                        newPoints = trjs[newPoints_index_orig] 
                        
                        count = count + 1
                        newPoints_name = 'start_r_'+str(count)+'.pdb'
                        newPoints.save_pdb(newPoints_name)
  
                np.save('trjs_theta', trjs_theta)
                return 
       
