 import RLSim
 z=RLSim.mockSimulation.run([1,0.5,1.4],[1,1.2,1.7])
 
 # Animate
ffmpeg -framerate 1 -f image2 -i figs-Init/fig_I%d.png -vcodec mpeg4 -r 1 I_L.mp4
