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
                #init = np.array(init)
                #init = init.astype(int)
        
                counter = 0
                init_index = []
                init_trj_xy = []
                for i in range(len(cl_trjs)):
                        if cl_trjs[i] in init_cl:
                                #print(cl_trjs[i])
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

             
        def findStarting(self, trj_Sp_theta, trj_Sp, starting_n=10 , method = 'LC'):
                # get new starting points (in theta domain) using new reward function based on updated weigths (W_1)
                import numpy as np
                
                n_coord = 2
                newPoints_index = range(starting_n)                   
                newPoints = []
                for coord in range(n_coord):
                          newPoints.append([trj_Sp[coord][int(i)] for i in newPoints_index])                                
          
                return newPoints
        
        
        def creatPotentioal(self):
                return True


        def run_noPlt(self, inits, nstepmax = 10):
                import numpy as np
                import time 
                from scipy.interpolate import interp1d
                #import matplotlib.pyplot as plt
                inits_x = inits[0]
                inits_y = inits[1]                    
                #plt.ion()
                # max number of iterations
                
                # frequency of plotting
                nstepplot = 1e1
                # plot string every nstepplot if flag1 = 1 
                flag1 = 1

                # temperature ###?!!!
                mu=9
                mu = 3
                # parameter used as stopping criterion
                tol1 = 1e-7

                # number of images during prerelaxation
                n2 = 1e1;
                # number of images along the string (try from  n1 = 3 up to n1 = 1e4)
                n1 = 25
                n1 = len(inits_x)
                # time-step (limited by the ODE step on line 83 & 84 but independent of n1)
                h = 1e-4
                #h = 5e-5
                # end points of the initial string
                # notice that they do NOT have to be at minima of V -- the method finds
                # those automatically

                # initialization
                
                x = np.array(inits_x)
                y = np.array(inits_y)
                dx = x-np.roll(x, 1)
                dy = y-np.roll(y, 1)
                dx[0] = 0
                dy[0] = 0
                xi = x
                yi = y

                                # parameters in Mueller potential

                aa = [-2, -20, -20, -20, -20] # inverse radius in x
                bb = [0, 0, 0, 0, 0] # radius in xy
                cc = [-20, -20, -2, -20, -20] # inverse radius in y
                AA = 30*[-200, -120, -200, -80, -80] # strength

                XX = [1, 0, 0, 0, 0.4] # center_x
                YY = [0, 0, 1, 0.4, 0] # center_y

                zxx = np.mgrid[-1:2.51:0.01]
                zyy = np.mgrid[-1:2.51:0.01]
                xx, yy = np.meshgrid(zxx, zyy)


                V1 = AA[0]*np.exp(aa[0] * np.square(xx-XX[0]) + bb[0] * (xx-XX[0]) * (yy-YY[0]) +cc[0]*np.square(yy-YY[0]))
                for j in range(1,5):
                        V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(xx-XX[j]) + bb[j]*(xx-XX[j])*(yy-YY[j]) + cc[j]*np.square(yy-YY[j]))

                #fig = plt.figure()
                #ax = fig.add_subplot(111)
                
                #plt.axvline(x=self.theta_mean[0])
                #plt.axhline(y=self.theta_mean[1])

                #plt.xlabel('x')
                #plt.ylabel('y')

##### Main loop

                trj_x = []
                trj_y = []
                for nstep in range(int(nstepmax)):
                        
                        # calculation of the x and y-components of the force, dVx and dVy respectively
                        #ax.contourf(xx,yy,np.minimum(V1,200), 40)
                        ee = AA[0]*np.exp(aa[0]*np.square(x-XX[0])+bb[0]*(x-XX[0])*(y-YY[0])+cc[0]*np.square(y-YY[0]))
                        #dVx = (2*aa[0]*(x-XX[0])+bb[0]*(y-YY[0]))*ee
                        #dVy = (bb[0]*(x-XX[0])+2*cc[0]*(y-YY[0]))*ee

                        dVx = (aa[0]*(x-XX[0])+bb[0]*(y-YY[0]))*ee
                        dVy = (bb[0]*(x-XX[0])+cc[0]*(y-YY[0]))*ee

                        for j in range(1,5):
                                ee = AA[j]*np.exp(aa[j]*np.square(x-XX[j])+bb[j]*(x-XX[j])*(y-YY[j])+cc[j]*np.square(y-YY[j]))
                                dVx = dVx + (aa[j]*(x-XX[j])+bb[j]*(y-YY[j]))*ee
                                dVy = dVy + (bb[j]*(x-XX[j])+cc[j]*(y-YY[j]))*ee

                                #dVx = dVx + (2*aa[j]*(x-XX[j])+bb[j]*(y-YY[j]))*ee
                                #dVy = dVy + (bb[j]*(x-XX[j])+2*cc[j]*(y-YY[j]))*ee
                        x0 = x
                        y0 = y
                        x = x - h*dVx + np.sqrt(h*mu)*np.random.randn(1,n1)
                        y = y - h*dVy + np.sqrt(h*mu)*np.random.randn(1,n1)
                        
                        #x = x - h*dVx + np.sqrt(2*h*mu)*np.random.randn(1,n1)
                        #y = y - h*dVy + np.sqrt(2*h*mu)*np.random.randn(1,n1) 
                        trj_x.append(x) 
                        trj_y.append(y)
                        #for j in range(len(trj_x)):
                        #        ax.plot(trj_x[j], trj_y[j], 'o', color='w')
                                
                        #ax.plot(x,y, 'o', color='r')
                        #fig.canvas.draw()
                        
                return trj_x, trj_y   



        def run(self, inits, nstepmax = 10):
                import numpy as np
                import time 
                from scipy.interpolate import interp1d
                import matplotlib.pyplot as plt
                inits_x = inits[0]
                inits_y = inits[1]                    
                plt.ion()
                # max number of iterations
                
                # frequency of plotting
                nstepplot = 1e1
                # plot string every nstepplot if flag1 = 1 
                flag1 = 1

                # temperature ###?!!!
                mu=9
                mu =3
                # parameter used as stopping criterion
                tol1 = 1e-7

                # number of images during prerelaxation
                n2 = 1e1;
                # number of images along the string (try from  n1 = 3 up to n1 = 1e4)
                n1 = 25
                n1 = len(inits_x)
                # time-step (limited by the ODE step on line 83 & 84 but independent of n1)
                h = 1e-4
                #h = 5e-5
                # end points of the initial string
                # notice that they do NOT have to be at minima of V -- the method finds
                # those automatically

                # initialization
                
                x = np.array(inits_x)
                y = np.array(inits_y)
                dx = x-np.roll(x, 1)
                dy = y-np.roll(y, 1)
                dx[0] = 0
                dy[0] = 0
                xi = x
                yi = y

                                # parameters in Mueller potential

                aa = [-2, -20, -20, -20, -20] # inverse radius in x
                bb = [0, 0, 0, 0, 0] # radius in xy
                cc = [-20, -20, -2, -20, -20] # inverse radius in y
                AA = 30*[-200, -120, -200, -80, -80] # strength


                XX = [1, 0, 0, 0, 0.4] # center_x
                YY = [0, 0, 1, 0.4, 0] # center_y

                zxx = np.mgrid[-1:2.51:0.01]
                zyy = np.mgrid[-1:2.51:0.01]
                xx, yy = np.meshgrid(zxx, zyy)


                V1 = AA[0]*np.exp(aa[0] * np.square(xx-XX[0]) + bb[0] * (xx-XX[0]) * (yy-YY[0]) +cc[0]*np.square(yy-YY[0]))
                for j in range(1,5):
                        V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(xx-XX[j]) + bb[j]*(xx-XX[j])*(yy-YY[j]) + cc[j]*np.square(yy-YY[j]))

                fig = plt.figure()
                ax = fig.add_subplot(111)
                
                plt.axvline(x=self.theta_mean[0])
                plt.axhline(y=self.theta_mean[1])

                plt.xlabel('x')
                plt.ylabel('y')


##### Main loop

                trj_x = []
                trj_y = []
                for nstep in range(int(nstepmax)):
                        
                        # calculation of the x and y-components of the force, dVx and dVy respectively
                        ax.contourf(xx,yy,np.minimum(V1,200), 40)
                        ee = AA[0]*np.exp(aa[0]*np.square(x-XX[0])+bb[0]*(x-XX[0])*(y-YY[0])+cc[0]*np.square(y-YY[0]))
                        dVx = (aa[0]*(x-XX[0])+bb[0]*(y-YY[0]))*ee
                        dVy = (bb[0]*(x-XX[0])+cc[0]*(y-YY[0]))*ee
                        for j in range(1,5):
                                ee = AA[j]*np.exp(aa[j]*np.square(x-XX[j])+bb[j]*(x-XX[j])*(y-YY[j])+cc[j]*np.square(y-YY[j]))
                                dVx = dVx + (aa[j]*(x-XX[j])+bb[j]*(y-YY[j]))*ee
                                dVy = dVy + (bb[j]*(x-XX[j])+cc[j]*(y-YY[j]))*ee
                        x0 = x
                        y0 = y
                        x = x - h*dVx + np.sqrt(h*mu)*np.random.randn(1,n1)
                        y = y - h*dVy + np.sqrt(h*mu)*np.random.randn(1,n1) 
                        trj_x.append(x) 
                        trj_y.append(y)
                        for j in range(len(trj_x)):
                                ax.plot(trj_x[j], trj_y[j], 'o', color='w')
                                
                        ax.plot(x,y, 'o', color='r')
                        fig.canvas.draw()
                        
                return trj_x, trj_y          


        def pltFinalPoints(self, trjs_theta):
                import numpy as np
                import matplotlib.pyplot as plt
                x = np.array(trjs_theta[0])
                y = np.array(trjs_theta[1])

                                # parameters in Mueller potential

                aa = [-2, -20, -20, -20, -20] # inverse radius in x
                bb = [0, 0, 0, 0, 0] # radius in xy
                cc = [-20, -20, -2, -20, -20] # inverse radius in y
                AA = 30*[-200, -120, -200, -80, -80] # strength

                XX = [1, 0, 0, 0, 0.4] # center_x
                YY = [0, 0, 1, 0.4, 0] # center_y

                zxx = np.mgrid[-1:2.51:0.01]
                zyy = np.mgrid[-1:2.51:0.01]
                xx, yy = np.meshgrid(zxx, zyy)


                V1 = AA[0]*np.exp(aa[0] * np.square(xx-XX[0]) + bb[0] * (xx-XX[0]) * (yy-YY[0]) +cc[0]*np.square(yy-YY[0]))
                for j in range(1,5):
                        V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(xx-XX[j]) + bb[j]*(xx-XX[j])*(yy-YY[j]) + cc[j]*np.square(yy-YY[j]))

                fig = plt.figure()
                ax = fig.add_subplot(111)
                
                ax.contourf(xx,yy,np.minimum(V1,200), 40)

                plt.xlabel('x')
                plt.ylabel('y')
                plt.plot(x, y, 'o')
                plt.savefig('fig_all.png')
# output size :
# 2 * simu length * number of parallel sims



