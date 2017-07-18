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
                alpha = 0.01
                alpha = 0.02
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
                
        def run_noPlt(self, inits, nstepmax = 10):
                import numpy as np
                import time 
                from scipy.interpolate import interp1d
                inits_x = inits[0]
                inits_y = inits[1]                    
                #plt.ion()
                # max number of iterations
                
                # frequency of plotting
                #nstepplot = 1e1
                # plot string every nstepplot if flag1 = 1 
                #flag1 = 1

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
                #h = 1e-5
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
                #AA = [-200, -120, -200, -80, -80] # strength


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
                        #x = x - h*dVx + np.sqrt(2*h*mu)*np.random.randn(1,n1)
                        #y = y - h*dVy + np.sqrt(2*h*mu)*np.random.randn(1,n1)
                        x = x - h*dVx + np.sqrt(h*mu)*np.random.randn(1,n1)
                        y = y - h*dVy + np.sqrt(h*mu)*np.random.randn(1,n1)
                        trj_x.append(x) 
                        trj_y.append(y)
                        #for j in range(len(trj_x)):
                        #        ax.plot(trj_x[j], trj_y[j], 'o', color='w')
                                
                        #ax.plot(x,y, 'o', color='r')
                        #fig.canvas.draw()
                        #print('zz')
                        
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
                # parameter used as stopping criterion
                tol1 = 1e-7

                # number of images during prerelaxation
                n2 = 1e1;
                # number of images along the string (try from  n1 = 3 up to n1 = 1e4)
                n1 = 25
                n1 = len(inits_x)
                # time-step (limited by the ODE step on line 83 & 84 but independent of n1)
                #h = 1e-4
                h = 5e-5
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
                AA = 3*[-200, -120, -200, -80, -80] # strength
                #AA = [-200, -120, -200, -80, -80] # strength


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
                        dVx = (2*aa[0]*(x-XX[0])+bb[0]*(y-YY[0]))*ee
                        dVy = (bb[0]*(x-XX[0])+2*cc[0]*(y-YY[0]))*ee
                        for j in range(1,5):
                                ee = AA[j]*np.exp(aa[j]*np.square(x-XX[j])+bb[j]*(x-XX[j])*(y-YY[j])+cc[j]*np.square(y-YY[j]))
                                dVx = dVx + (2*aa[j]*(x-XX[j])+bb[j]*(y-YY[j]))*ee
                                dVy = dVy + (bb[j]*(x-XX[j])+2*cc[j]*(y-YY[j]))*ee
                        x0 = x
                        y0 = y
                        x = x - h*dVx + np.sqrt(2*h*mu)*np.random.randn(1,n1)
                        y = y - h*dVy + np.sqrt(2*h*mu)*np.random.randn(1,n1) 
                        trj_x.append(x) 
                        trj_y.append(y)
                        for j in range(len(trj_x)):
                                ax.plot(trj_x[j], trj_y[j], 'o', color='w')
                                
                        ax.plot(x,y, 'o', color='r')
                        fig.canvas.draw()
                        #print('zz')
                        
                return trj_x, trj_y          

# output size :
# 2 * simu length * number of parallel sims


        def pltFinalPoints(self, trjs_theta):
                import numpy as np
                import matplotlib.pyplot as plt
                plt.rcParams.update({'font.size':20})
                plt.rc('xtick', labelsize=20)
                plt.rc('ytick', labelsize=20)
                x = np.array(trjs_theta[0])
                y = np.array(trjs_theta[1])

                                # parameters in Mueller potential

                aa = [-2, -20, -20, -20, -20] # inverse radius in x
                bb = [0, 0, 0, 0, 0] # radius in xy
                cc = [-20, -20, -2, -20, -20] # inverse radius in y
                AA = 3*[-200, -120, -200, -80, -80] # strength

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
                plt.plot(x, y, 'o', color='white', alpha=0.2, mec="black")
                plt.savefig('fig_all.png', dpi =500)



