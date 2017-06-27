
from __future__ import print_function
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from sys import stdout


nonbondedCutoff = 1.0*unit.nanometers
timestep = 2.0*unit.femtoseconds
temperature = 300*unit.kelvin

production_steps = 200
save_frequency = 100 #Every 100 steps, save the trajectory


pdb = app.PDBFile('ala2_1stFrame.pdb')
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

simulation.reporters.append(app.PDBReporter('ala2_production.pdb', save_frequency))
simulation.reporters.append(app.StateDataReporter('stateReporter_constantPressure.txt', 1000, step=True, 
   totalEnergy=True, temperature=True, volume=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=production_steps, separator='\t'))

print('Running Production...')
simulation.step(production_steps)
print('Done!')
