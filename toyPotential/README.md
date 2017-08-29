 import RLSim
 z=RLSim.mockSimulation.run([1,0.5,1.4],[1,1.2,1.7])
 
 # Animate
 ffmpeg -f image2 -r 2 -i w/w50_%d.png -vcodec mpeg4 -y movie_w.mp4
