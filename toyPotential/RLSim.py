class mockSimulation:
        r = 1#number of rounds
        s = 1# length of simulations
        N = 1# number of parallel simulations
        ## public
        
        def run_multipleSim():
                return True
        def runNxtRound():
                return True
                
        
        ## private
        def PreSamp(trj):
                """
                Pre-Sampling:
                        choose states with minimum counts or newly discovered states
                """
                trj_Sp = trj # pick all
                return trj_Sp
                
        def map(trj_Sp):
                # map coordinate space to reaction coorinates space
                trj_Sp_theta = trj_Sp
                return trj_Sp_theta
        
        def updateW(trj_Sp_theta, prior_weigths = W_0):
                """
                update weigths 
                """
                alpha = 0.05
                r_0 = reward_trj(trj_Sp_theta, weigths = W_0)
                
                W_a = [W_0[0]-alpha, W_0[1]+alpha]
                r_a = reward_trj(trj_Sp_theta, weigths = W_a)
                
                W_b = [W_0[0]+alpha, W_0[1]-alpha]
                r_b = reward_trj(trj_Sp_theta, weigths = W_b)
                
                max_r = np.max(r_0, r_a, a_b)
                
                if max_r == r_0:
                        W_1 = W_0
                        
                elif max_r == r_a:
                        W_1 = W_a
                        
                elif max_r == r_b:
                        W_1 = W_b
                        
                return W_1  
        
        def reward_trj(trj_Sp_theta, W_):
                """
                
                """
                import numpy as np
                theta_mean = []
                theta_std = []
                for theta in range(len(W_)):
                        theta_mean.append(np.mean(trj_Sp_theta[theta]))
                        theta_std.append(np.std(trj_Sp_theta[theta]))
                
                r = []
                # for over all dicovered states
                for state_index in range(len(trj_Sp_theta)):
                        state_theta = trj_Sp_theta[:][state_index]
                        r_s = reward(state_theta, theta_mean, theta_std, W_)
                        
                        #for k in range(len(W_)):
                        #        r_s = r_s + W_[k]*((trj_Sp_theta[k][s] - theta_mean[k])/theta_std[k])
                        r.append(r_s)
                        
                R = np.sum(np.array(r))
                return R
                
                
        def reward_state( S, theta_mean, theta_std, W_):
                
                r_s = 0
                for k in range(len(W_)):
                        r_s = r_s + W_[k]*((S[k] - theta_mean[k])/theta_std[k])
                return r_s
                
                
                
        def findStarting(trj_Sp_theta, W_1, starting_n = 10 , method = 'RL'):
                # get new starting points (in theta domain) using new reward function based on updated weigths (W_1)
                
                theta_mean = []
                theta_std = []
                for theta in range(len(W_)):
                        theta_mean.append(np.mean(trj_Sp_theta[theta]))
                        theta_std.append(np.std(trj_Sp_theta[theta]))
                        
                # self.theta_mean
                # self.theta_std
                
                ranks = {}
                for state_index in range(len(trj_Sp_theta)):
                        state_theta = trj_Sp_theta[:][state_index]
                        r = reward_state( state_theta, theta_mean, theta_std, W_1)
                        rank[state_index] = r
                
                
                return newPoints_index
        
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        def creatPotentioal():
                return True
                
        def run(inits_x, inits_y):
                import numpy as np
                import time 
                from scipy.interpolate import interp1d
                import matplotlib.pyplot as plt
                plt.ion()
                # max number of iterations
                nstepmax = 20
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
                h = 1e-4

                # end points of the initial string
                # notice that they do NOT have to be at minima of V -- the method finds
                # those automatically

                # initialization
                #xa = 1.5
                #ya = 0.3

                #xb = 1.7
                #yb = 0.3                
                #g1 = np.linspace(0,0.5,n1)
                #x = (xb-xa)*g1+xa
                #y = (x-xa)*(yb-ya)/(xb-xa)+ya

                #lxy = np.cumsum(np.sqrt(np.square(dx)+np.square(dy)))
                #lxy = lxy/lxy[n1-1]

                #set_interp1 = interp1d(lxy, x, kind='linear')
                #x = set_interp1(g1)
                #set_interp2 = interp1d(lxy, y, kind='linear')
                #y = set_interp2(g1)
                
                x = np.array(inits_x)
                y = np.array(inits_y)
                dx = x-np.roll(x, 1)
                dy = y-np.roll(y, 1)
                dx[0] = 0
                dy[0] = 0
                xi = x
                yi = y

                                # parameters in Mueller potential

                aa = [-0.8, -8, -10] # inverse radius in x
                bb = [0, 0, 0] # radius in xy
                cc = [-10, -8, -0.8] # inverse radius in y
                AA = [-100, -50, -100] # strength


                XX = [1.5, 0, 0] # center_x
                YY = [0, 0, 1.5] # center_y

                zxx = np.mgrid[-1:3.01:0.01]
                zyy = np.mgrid[-1:3.01:0.01]
                xx, yy = np.meshgrid(zxx, zyy)


                V1 = AA[0]*np.exp(aa[0] * np.square(xx-XX[0]) + bb[0] * (xx-XX[0]) * (yy-YY[0]) +cc[0]*np.square(yy-YY[0]))
                for j in range(1,3):
                        V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(xx-XX[j]) + bb[j]*(xx-XX[j])*(yy-YY[j]) + cc[j]*np.square(yy-YY[j]))

                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.xlabel('x')
                plt.ylabel('y')


####

## Main loop

                trj_x = []
                trj_y = []
                for nstep in range(int(nstepmax)):
                        
                        # calculation of the x and y-components of the force, dVx and dVy respectively
                        ax.contourf(xx,yy,np.minimum(V1,200), 40)
                        ee = AA[0]*np.exp(aa[0]*np.square(x-XX[0])+bb[0]*(x-XX[0])*(y-YY[0])+cc[0]*np.square(y-YY[0]))
                        dVx = (2*aa[0]*(x-XX[0])+bb[0]*(y-YY[0]))*ee
                        dVy = (bb[0]*(x-XX[0])+2*cc[0]*(y-YY[0]))*ee
                        for j in range(1,3):
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


