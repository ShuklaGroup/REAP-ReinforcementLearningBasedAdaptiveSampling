## Run simulations using RL 

> import RLSim as rl

> z = rl.mockSimulation()

> z.runSimulation(R=1000, N=1,s=1000, method='RL')

Run single trajectory for 1000 round, each 1000 step (=2 fs*1000 = 2 ps).

## Animate the images

ffmpeg -f image2 -r 10 -i fig_%d.png -vcodec mpeg4 -y movie.mp4

##
