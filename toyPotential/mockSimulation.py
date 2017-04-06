class mockSimulation:
        r = #number of rounds
        s = # length of simulations
        N = # number of parallel simulations
        ## public
        def runNxtRound():
                
                
        
        ## private
        def adaptiveSampling():
                
                
        def creatPotentioal():
                ## 1st: Mueller potential
                import numpy as np
                from scipy.interpolate import interp1d
                # temperature ###?!!!
                mu=9
                # max number of iterations
                nstepmax = 2e3
                nstepmax = 10
                # frequency of plotting
                nstepplot = 1e1

                # plot string every nstepplot if flag1 = 1 
                flag1 = 1

                # parameter used as stopping criterion
                tol1 = 1e-7


                # number of images along the string (try from  n1 = 3 up to n1 = 1e4)
                n1 = 25

                # time-step (limited by the ODE step on line 83 & 84 but independent of n1)
                h = 1e-4

                # end points of the initial string
                # notice that they do NOT have to be at minima of V -- the method finds
                # those automatically
                xa = -1
                ya = 0.5

                xb = 0.7
                yb = 0.5

                # initialization
                g1 = np.linspace(0,1,n1)
                x = (xb-xa)*g1+xa
                y = (x-xa)*(yb-ya)/(xb-xa)+ya

                #dx = x-circshift(x,[0 1])
                #dy = y-circshift(y,[0 1])

                dx = x-np.roll(x, 1)
                dy = y-np.roll(y, 1)

                #dx(1) = 0
                #dy(1) = 0
                dx[0] = 0
                dy[0] = 0

                #lxy = cumsum(sqrt(dx.^2+dy.^2))
                #lxy = lxy/lxy(n1)
                lxy = np.cumsum(np.sqrt(np.square(dx)+np.square(dy)))
                lxy = lxy/lxy[n1-1]

                #x = interp1(lxy,x,g1)
                #y = interp1(lxy,y,g1)
                set_interp1 = interp1d(lxy, x, kind='linear')
                x = set_interp1(g1)
                set_interp2 = interp1d(lxy, y, kind='linear')
                y = set_interp2(g1)

                xi = x
                yi = y

                # parameters in Mueller potential
                aa = [-1, -1, -6.5, 0.7]
                bb = [0, 0, 11, 0.6]
                cc = [-10, -10, -6.5, 0.7]
                AA = [-200, -100, -170, 15]

                XX = [1, 0, -0.5, -1]
                YY = [0, 0.5, 1.5, 1]

                #[xx,yy] = meshgrid(-1.5:0.01:1.2,-0.2:0.01:2)
                xx = np.mgrid[-1.5:1.21:0.01]
                yy_1 = np.mgrid[-0.2:2.01:0.01]
                a = len(xx)-len(yy_1)
                yy_2 = np.mgrid[-0.2:(-0.2+a*0.01):0.01]
                yy = np.append(yy_1, yy_2)

                #V1 = AA(1)*exp(aa(1)*(xx-XX(1)).^2+bb(1)*(xx-XX(1)).*(yy-YY(1))+cc(1)*(yy-YY(1)).^2)
                #for j=2:4
                #    V1 =  V1 + AA(j)*exp(aa(j)*(xx-XX(j)).^2+bb(j)*(xx-XX(j)).*(yy-YY(j))+cc(j)*(yy-YY(j)).^2)
                #end
                V1 = AA[0]*np.exp(aa[0] * np.square(xx-XX[0]) + bb[0] * (xx-XX[0]) * (yy-YY[0]) +cc[0]*np.square(yy-YY[0]))
                for j in range(1,4):
                        V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(xx-XX[j]) + bb[j]*(xx-XX[j])*(yy-YY[j]) + cc[j]*np.square(yy-YY[j]))
                
                #figure(1);
                #clf;
                #contourf(xx,yy,min(V1,200),40);
                #hold on
                #plot(xi,yi,'.-w','MarkerSize',14)
                #set(gca,'XTick',-1.5:.5:1,'YTick',0:.5:2);
                #xlabel('x','FontAngle','italic');
                #ylabel('y','FontAngle','italic');
                #title('Initial string');
                #drawnow

                plt.contourf(xx,yy,V1, 40)
                plt.plot(xi, yi, '-')
                plt.xlabel('x')
                plt.ylabel('y')
                for nstep in range(int(nstepmax)):
                 # calculation of the x and y-components of the force, dVx and dVy respectively
                        ee = AA[0]*np.exp(aa[0]*np.square(x-XX[0])+bb[0]*(x-XX[0])*(y-YY[0])+cc[0]*np.square(y-YY[0]))
                        dVx = (2*aa[0]*(x-XX[0])+bb[0]*(y-YY[0]))*ee
                        dVy = (bb[0]*(x-XX[0])+2*cc[0]*(y-YY[0]))*ee
                        for j in range(1,4):
                                ee = AA[j]*np.exp(aa[j]*np.square(x-XX[j])+bb[j]*(x-XX[j])*(y-YY[j])+cc[j]*np.square(y-YY[j]))
                                dVx = dVx + (2*aa[j]*(x-XX[j])+bb[j]*(y-YY[j]))*ee
                                dVy = dVy + (bb[j]*(x-XX[j])+2*cc[j]*(y-YY[j]))*ee
                        x0 = x
                        y0 = y
                        x = x - h*dVx + np.sqrt(2*h*mu)*np.random.randn(1,n1)
                        y = y - h*dVy + np.sqrt(2*h*mu)*np.random.randn(1,n1) ## ?!!!! n2?!!!
                        plt.contourf(xx,yy,V1, 40)
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.plot(x,y, 'o', color='w')
                        plt.show()
                
