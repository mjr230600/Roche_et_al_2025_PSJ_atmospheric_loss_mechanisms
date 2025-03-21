#SJL 8/17
#script to calculate the Hugoniot from an EOS 
#function returns p, rho, E, up, T
#EOS needs a function calcpT_rho_E that takes rho and E point and returns p and T and optionally a flag

import numpy as np
import scipy as sp
import sys
import os
from scipy import constants as const

import random as rand_calc_hug
rand_calc_hug.seed(1)


##############
#need to identify which computer we are on
if sys.platform== 'darwin':

    #get the serial number
    import subprocess
    cmd = "system_profiler SPHardwareDataType | awk '/Serial Number/ {print $4}'"
    result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, check=True)
    serial_number = result.stdout.strip().decode('utf-8')
    
    #if uncle bulgaria:
    if serial_number=='D25M61Y9F8JC':
        path_db='/Users/simonlock/Dropbox'
        path_ody='/Volumes/Lock_onsite_active_memory_extension/Odyssey_transfer_31_3_2019'
    #or if either of the Bristol computers
    elif (serial_number=='C02H91W8PN7C'): #iMac
        path_db='/Users/vq21447/Dropbox'
        path_ody='/Volumes/Lock_office_active_backup_and_memext/Odyssey_transfer_31_3_2019'
    elif(serial_number=='C02H91W8PN7C'):
        print("CHECK THE LOCATION OF ODYSSEY BACKUP")
        path_db='/Users/vq21447/Dropbox'
        path_ody='/Volumes/Lock_office_active_backup_and_memext/Odyssey_transfer_31_3_2019'

elif (sys.platform== 'win32') | (sys.platform== 'win64'):
    path_db="C:\\Users\sjl49\Dropbox"
    path_ody="Lock_office_active_memory_extension:\\Odyssey_transfer_31_3_2019"
#############

from gradients import gradient2

#sys.path.insert(0, path_db+'/Research/Code_repository/EOS')
#sys.path.insert(0, '/Research/Code_repository/EOS')
#from Gadget_EOS_structure2 import *
#from GSesame_EOS import *

from scipy.optimize import fsolve
from scipy.integrate import quad

###########################################################
###########################################################
###########################################################
#class for the Hugoniot
class Hugoniot_from_EOS(object):
    def __init__(self, EOS):
  
        #EOS to use
        self.EOS=EOS

        #initial properties
        self.rho0=0.0
        self.E0=0.0
        self.p0=0.0
        self.T0=0.0
        self.s0=0.0

        #for release only
        self.up0=0.0

        #given and calculated points
        self.us=0.0
        self.p=0.0
        self.rho=0.0
        self.E=0.0
        self.up=0.0
        self.T=0.0
        self.s=0.0
        self.c=0.0

        #parameters for ideal gas EOS
        gamma=1.4
        ma=28E-3
        

    ##########################################################
    #function to calculate the hugiont from the EOS
    def calc_hugoniot(self, Us):

        print('NOT FINISHED FUNCTION')
        sys.exit()
        
        #record initial properties
        self.Us=Us

        #initialise output arrays
        flag=0
        self.p=np.empty(np.size(self.Us))
        self.rho=np.empty(np.size(self.Us))
        self.E=np.empty(np.size(self.Us))
        self.Up=np.empty(np.size(self.Us))
        self.T=np.empty(np.size(self.Us))
        
        #look at each Us seperately
        pinit=self.p0*1E10
        for i in np.arange(np.size(self.Us)):
            print('\t',self.Us[i]/1E3)
            #minimize the misfit in the pressure 
            temp=fsolve(self.zero_p, pinit/self.p0, args=(self.Us[i]))#,method='Nelder-Mead')
            self.p[i]=temp
            print('\t \t', temp, (self.p[i]-self.p0)/self.p0, self.p0, self.Us[i])
             
            #use this to calculate the other properties from RH eqns
            self.rho[i]=((self.rho0*self.Us[i])**2.0)/(self.rho0*(self.Us[i]**2.0)-(self.p[i]-self.p0)) 
            self.E[i]=self.E0+0.5*(self.p[i]+self.p0)*((1.0/self.rho0)-(1.0/self.rho[i]))
            self.Up[i]=np.sqrt(2.0*(self.E[i]-self.E0))

            print('\t \t',self.p[i], self.rho[i], self.E[i], self.Up[i])

            #then calculate the T from the EOS
            temp=self.EOS.calcpT_rho_E(self.rho[i],self.E[i])
            self.T[i]=temp[1]

            #check to see that the routine successfully converged
            Uscalc=(self.p[i]-self.p0)/(self.rho0*self.Up[i])

            #absolute
            if (Uscalc-self.Us[i])>1E-12:
                #fractional
                if (Uscalc-self.Us[i])/(1.0+self.Us[i])>1E-12:
                    print('Hugoniot calculation not converged')
                    print(self.Us[i], Uscalc, (Uscalc-self.Us[i]), (Uscalc-self.Us[i])/(1+self.Us[i]))
                    flag=-1

            #redefine the initial to the most recent value
            pinit=self.p[i]
            sys.exit()

        return flag

    ######### Function to use in finding the solution
    def zero_p(self, p, Us): #for equations see Hugoniot notes
        p=p*self.p0
        if p<self.p0:
            return 1.0

        #use the Rankine Hugoniot equations to calc the rho and E
        rho=((self.rho0*Us)**2.0)/(self.rho0*(Us**2)-(p-self.p0)) 
        print(self.rho0*Us, self.rho0*(Us**2), p-self.p0)
        if rho<self.rho0:
            sys.exit()
        E=self.E0+0.5*(p+self.p0)*((1.0/self.rho0)-(1.0/rho))
        Up=np.sqrt(2.0*(E-self.E0))
        
        #use these and the EOS to recalculate the pressure 
        temp=self.EOS.calcpT_rho_E(rho,E)
        pcalc=temp[0]

        return (p-pcalc)/(p)

    ##########################################################
    #function to calculate the hugiont from the EOS
    def calc_hugoniot_up(self, up, prop0, toll, input_init_flag=0, flag_release=0, max_itter=50000, init_guess=np.nan, flag_version=0, step_transitions=None, steps=None,verbosity=0):

        if input_init_flag==0:
            #initial properties
            self.p0=prop0[0]
            self.T0=prop0[1]

            #calculate other initial properties
            temp=self.EOS.find_PT(self.p0, self.T0)
            self.rho0=temp[0][0]
            self.E0=temp[1][0]

        #record initial properties
        self.up=up

        #
        if (flag_version==1)&(step_transitions==None):
            step_transitions=[1E-4,1E-6,1E-8,1E-10,1E-12]

        if (flag_version==1)&(steps==None):
            steps=[0.001,0.005,0.01,0.05,0.1,0.2]

        if (flag_version==1)&(np.size(step_transitions)!=(np.size(steps)-1)):
            raise ValueError('# of step_transitions not compatible with # of given step sizes')

        #initialise output arrays
        flag=0
        self.p=np.zeros(np.size(self.up))
        self.rho=np.zeros(np.size(self.up))
        self.E=np.zeros(np.size(self.up))
        self.us=np.zeros(np.size(self.up))
        self.T=np.zeros(np.size(self.up))
        self.c=np.zeros(np.size(self.up))

        #make a guess for the first point
        if np.any(np.isnan(init_guess)):
            if flag_release==0:
                p=1.1*self.p0
                rho=1.1*self.rho0
                E=1.001*self.E0
                us=1.1*up[0]
            else:
                p=0.9*self.p0
                rho=0.9*self.rho0
                E=0.95*self.E0
                us=0.9*up[0]

                p=0.9999*self.p0
                rho=0.99999*self.rho0
                E=0.9999*self.E0
                us=0.9999*up[0]
        else:
            p=init_guess[0]
            rho=init_guess[1]
            E=init_guess[2]
            us=init_guess[3]
        #print([p,rho,E,us])

        

        #look at each up seperately
        for i in np.arange(np.size(self.up)):
            #print(i,self.up[i])

            plast=0.999*p
            rholast=0.999*rho
            Elast=0.999*E
            uslast=0.999*us
            
            
            if (i==0)|(flag_version==0):
                plast=p
                rholast=rho
                Elast=E
                uslast=us
            else:
                if i==1:
                    p=plast
                elif i==2:
                    dpdu=gradient2(up[0:2],self.p[0:2], flag_calc=[1,0], flag_order=[1,2])
                    p=self.p[i-1]+(up[i]-up[i-1])*dpdu[-1]
                    #drhodu=gradient2(up[0:2],self.rho[0:2], flag_calc=[1,0], flag_order=[1,2])
                    #rho=self.rho[i-1]+(up[i]-up[i-1])*drhodu[-1]
                    #dEdu=gradient2(up[0:2],self.E[0:2], flag_calc=[1,0], flag_order=[1,2])
                    #E=self.E[i-1]+(up[i]-up[i-1])*dEdu[-1]
                    #dusdu=gradient2(up[0:2],self.us[0:2], flag_calc=[1,0], flag_order=[1,2])
                    #us=self.p[i-1]+(up[i]-up[i-1])*dusdu[-1]
                else:
                    dpdu, d2pdu2=gradient2(up[i-3:i],self.p[i-3:i], flag_calc=[1,1], flag_order=[2,2])
                    p=self.p[i-1]+(up[i]-up[i-1])*dpdu[-1]+((up[i]-up[i-1])**2)*d2pdu2[-1]/2.
                    drhodu, d2rhodu2=gradient2(up[i-3:i],self.rho[i-3:i], flag_calc=[1,1], flag_order=[2,2])
                    rho=self.rho[i-1]+(up[i]-up[i-1])*drhodu[-1]+((up[i]-up[i-1])**2)*d2rhodu2[-1]/2.
                    dEdu, d2Edu2=gradient2(up[i-3:i],self.E[i-3:i], flag_calc=[1,1], flag_order=[2,2])
                    E=self.E[i-1]+(up[i]-up[i-1])*dEdu[-1]+((up[i]-up[i-1])**2)*d2Edu2[-1]/2.
                    dusdu, d2usdu2=gradient2(up[i-3:i],self.us[i-3:i], flag_calc=[1,1], flag_order=[2,2])
                    us=self.us[i-1]+(up[i]-up[i-1])*dusdu[-1]+((up[i]-up[i-1])**2)*d2usdu2[-1]/2.
                    
                plast=p
                rholast=rho
                Elast=E
                uslast=us

            test=1E99
            testlast=2
            flag_conv=0
            if flag_version==0:
                frac=0.01
            else:
                frac=steps[0]
            count=-1
            min_test=1.0
            while test>toll:
                count+=1
                #if you have gone too long then fail
                if count>max_itter:
                    print(i, up[i], p, rho, E, us)
                    print(flag_conv)
                    print(test, min_test)
                    raise RuntimeError('Hugoniot calculation not converged')

                if (i<=2)|(flag_version==0):
                    #find us from momentum conservation
                    usnew=(p-self.p0)/self.rho0/up[i]
                    
                    #find density from mass conservation
                    rhonew=self.rho0*us/(us-up[i])
                    
                    #find the energy from energy conservation
                    Enew=self.E0+(p+self.p0)*(1.0/self.rho0-1.0/rho)/2

                    #now recalculate the pressure from the EOS given rho and E
                    temp=self.EOS.calcpT_rho_E(rho,E)
                    #temp=self.EOS.calcpT_rho_E(rho,E)
                    if (temp[2]!=0):
                        #print('oh oh')
                        pnew=p
                    else:
                        #print(temp)
                        pnew=temp[0][0]
                        #print(temp)

                else:
                    #find us from momentum conservation
                    usnew=(p-self.p0)/self.rho0/up[i]
                    
                    #find density from mass conservation
                    rhonew=self.rho0*usnew/(usnew-up[i])
                    
                    #find the energy from energy conservation
                    Enew=self.E0+(p+self.p0)*(1.0/self.rho0-1.0/rhonew)/2

                    #now recalculate the pressure from the EOS given rho and E
                    temp=self.EOS.calcpT_rho_E(rhonew,Enew)
                    #temp=self.EOS.calcpT_rho_E(rho,E)
                    if (temp[2]!=0):
                        #print('oh oh')
                        pnew=p
                    else:
                        #print(temp)
                        pnew=temp[0][0]
                        #print(temp)
                    

                #change the conversion requirements
                if flag_version==0:
                    if (flag_conv==0)&(test<np.amax([toll*10,1E-6])):
                        flag_conv=1

                    if flag_conv==0:
                        frac=0.01
                    elif flag_conv==1:
                        frac=0.1
                    else:
                        frac=0

                elif flag_version==1:
                
                    for test_ind, step_trans in enumerate(step_transitions):
                        if (test<step_trans)&(testlast<step_trans):
                            flag_conv=test_ind+1
                            frac=steps[test_ind+1]

                p=(frac*pnew+(1.-frac)*plast)
                rho=(frac*rhonew+(1.-frac)*rholast)
                E=(frac*Enew+(1.-frac)*Elast)
                us=(frac*usnew+(1.-frac)*uslast)

                #check that the updated value is an acceptable value
                if p<=self.p0:
                    #print('oh oh')
                    p=plast
                if rho<=self.rho0:
                    #print('oh oh')
                    rho=rholast
                if E<=self.E0:
                    #print('oh oh')
                    E=Elast
                if us<=up[i]:
                    #print('oh oh')
                    us=up[i]*rho/self.rho0

                #determine convergence
                #test=np.sum(np.asarray([((rho-rholast)/rho), ((E-Elast)/E), ((p-plast)/p), ((us-uslast)/us)])**2)
                #test=np.sum(np.asarray([((rhonew-rholast)/rholast), ((Enew-Elast)/Elast), ((pnew-plast)/plast), ((usnew-uslast)/uslast)])**2)
                #
                testlast=test*1.0
                test=np.amax([np.absolute((us-(p-self.p0)/self.rho0/up[i])/us), np.absolute((rho-self.rho0*us/(us-up[i]))/rho), np.absolute((E-(self.E0+(p+self.p0)*(1.0/self.rho0-1.0/rho)/2))/E)])
                #if up[i]>1E3:
                #    print(test, (rhonew-rholast)/rholast, ((Enew-Elast)/Elast), ((pnew-plast)/plast), ((usnew-uslast)/uslast))
                # print(test, (us-(p-self.p0)/self.rho0/up[i])/us, (rho-self.rho0*us/(us-up[i]))/rho, (E-(self.E0+(p+self.p0)*(1.0/self.rho0-1.0/rho)/2))/E,i)
                if test<min_test:
                    min_test=test

                # print(p/1E9,rho/1E3,E,us/1E3,frac,test)

                #update the previous values
                plast=p
                rholast=rho
                Elast=E
                uslast=us

                #if count==0:
                #    print(test)

            # print(test,count)

            #record converged values
            self.p[i]=p
            self.rho[i]=rho
            self.E[i]=E
            self.us[i]=us

            #print(rho/1E3,E)
            temp=self.EOS.calcpT_rho_E(rho,E)
            #print(temp)
            self.T[i]=temp[1][0]
            self.c[i]=temp[4][0]

            if verbosity!=0:
                print(self.up[i], p, rho, E, us,temp[1][0], count)
            #print(dkfjhdf)

        return flag

    
    ##########################################################
    #function to calculate the hugiont from the EOS
    def calc_hugoniot_rho(self, rho, prop0, toll, input_init_flag=0, flag_release=0, max_itter=10000):

        if input_init_flag==0:
            #initial properties
            self.p0=prop0[0]
            self.T0=prop0[1]

            #calculate other initial properties
            temp=self.EOS.find_PT(self.p0, self.T0)
            self.rho0=temp[0][0]
            self.E0=temp[1][0]


        #record initial properties
        self.rho=rho

        #initialise output arrays
        flag=0
        self.up=np.empty(np.size(self.rho))
        self.p=np.empty(np.size(self.rho))
        self.E=np.empty(np.size(self.rho))
        self.us=np.empty(np.size(self.rho))
        self.T=np.empty(np.size(self.rho))
        self.c=np.empty(np.size(self.rho))

        #make a guess for the first point
        if flag_release==0:
            p=1.1*self.p0
            E=1.001*self.E0
            up=1E2#np.sqrt(0.1*self.p0/self.rho0/1.1)
            us=1.1*up
        else:
            p=0.9*self.p0
            E=0.95*self.E0
            up=1E2#np.sqrt(0.1*self.p0/self.rho0/1.1)
            us=0.9*up

            p=0.99999*self.p0
            E=0.9999*self.E0
            up=1E2#np.sqrt(0.0001*self.p0/self.rho0/0.00001)
            us=0.9999*up
        #print([p,rho,E,us])
        
        #look at each up seperately
        for i in np.arange(np.size(self.rho)):
            #print(self.up[i])
            print(i,self.rho[i])

            
            uplast=0.999*up
            plast=0.999*p
            Elast=0.999*E
            uslast=0.999*us

            test=1
            flag_conv=0
            count=-1
            while test>toll:
                #print('\t',test,up,p,E,us)
                
                count+=1
                #if you have gone too long then fail
                if count>max_itter:
                    print('Hugoniot calculation not converged')
                    print(i, self.rho[i], up, p, E, us)
                    print(test)
                    print(dfkjhdf)
                    
                #find us from momentum conservation
                usnew=(p-self.p0)/self.rho0/up


                #find density from mass conservation
                upnew=us*(1.0-self.rho0/self.rho[i])
                

                #find the energy from energy conservation
                Enew=self.E0+(p+self.p0)*(1.0/self.rho0-1.0/self.rho[i])/2

                

                #now recalculate the pressure from the EOS given rho and E
                temp=self.EOS.calcpT_rho_E(self.rho[i],E)
                #temp=self.EOS.calcpT_rho_E(rho,E)
                if (temp[2]!=0):
                    #print('oh oh')
                    pnew=p
                else:
                    #print(temp)
                    pnew=temp[0][0]
                #print(temp)


                #change the conversion requirements
                if (flag_conv==0)&(test<np.amax([toll*10,1E-6])):
                    flag_conv=1

                #update the parameter value
                if flag_conv==0:
                    p=(0.01*pnew+0.99*plast)
                    up=(0.01*upnew+0.99*uplast)
                    E=(0.01*Enew+0.99*Elast)
                    us=(0.01*usnew+0.99*uslast)
                elif flag_conv==1:
                    p=(0.1*pnew+0.9*plast)
                    up=(0.1*upnew+0.9*uplast)
                    E=(0.1*Enew+0.9*Elast)
                    us=(0.1*usnew+0.9*uslast)
                else:
                    p=pnew
                    up=upnew
                    E=Enew
                    us=usnew

                #check that the updated value is an acceptable value
                if p<=self.p0:
                    #print('oh oh')
                    p=plast
                if E<=self.E0:
                    #print('oh oh')
                    E=Elast
                if us<=up:
                    #print('oh oh')
                    us=up*self.rho[i]/self.rho0

                #determine convergence
                #test=np.sum(np.asarray([((up-uplast)/up), ((E-Elast)/E), ((p-plast)/p), ((us-uslast)/us)])**2)
                test=np.sum(np.asarray([((upnew-uplast)/uplast), ((Enew-Elast)/Elast), ((pnew-plast)/plast), ((usnew-uslast)/uslast)])**2)

                #print(p/1E9,rho/1E3,E,us/1E3,test)

                #update the previous values
                plast=p
                uplast=up
                Elast=E
                uslast=us

            #print(test,count)

            #record converged values
            self.p[i]=p
            self.up[i]=up
            self.E[i]=E
            self.us[i]=us

            #print(rho/1E3,E)
            temp=self.EOS.calcpT_rho_E(self.rho[i],E)
            #print(temp)
            self.T[i]=temp[1][0]
            self.c[i]=temp[4][0]

            #print(p, rho, E, us,temp[1][0])
            #print(dkfjhdf)

        return flag

     ##########################################################
    #function to calculate the hugiont from the EOS
    def calc_hugoniot_p2(self, p, prop0, toll, input_init_flag=0, flag_release=0, max_itter=10000,init_guess=np.nan,slow_toll=[1E-6,1E-30]):

        if input_init_flag==0:
            #initial properties
            self.p0=prop0[0]
            self.T0=prop0[1]

            #calculate other initial properties
            temp=self.EOS.find_PT(self.p0, self.T0)
            self.rho0=temp[0][0]
            self.E0=temp[1][0]


        #record initial properties
        self.p=p

        #initialise output arrays
        flag=0
        self.up=np.empty(np.size(self.p))
        self.rho=np.empty(np.size(self.p))
        self.E=np.empty(np.size(self.p))
        self.us=np.empty(np.size(self.p))
        self.T=np.empty(np.size(self.p))
        self.c=np.empty(np.size(self.p))

        
        #make a guess for the first point
        if np.any(np.isnan(init_guess)):
        
            #make a guess for the first point
            if flag_release==0:
                rho=1.1*self.rho0
                E=1.001*self.E0
                up=1E2#np.sqrt(0.1*self.p0/self.rho0/1.1)
                us=1.1*up

                rho=1.1*self.rho0
                temp=self.EOS.find_dp(rho, 2*self.p[0])
                print(temp)
                E=temp[2][0]
                #E=1.0001*self.E0
                up=1E2#np.sqrt(0.1*self.p0/self.rho0/1.1)
                us=1.1*up
            else:
                rho=0.9*self.rho0
                E=0.95*self.E0
                up=1E2#np.sqrt(0.1*self.p0/self.rho0/1.1)
                us=0.9*up

                rho=0.99999*self.rho0
                E=0.9999*self.E0
                up=1E2#np.sqrt(0.0001*self.p0/self.rho0/0.00001)
                us=0.9999*up

        else:
            up=init_guess[0]
            rho=init_guess[1]
            E=init_guess[2]
            us=init_guess[3]
        #print([p,rho,E,us])
        
        #look at each up seperately
        for i in np.arange(np.size(self.p)):
            #print(self.up[i])
            print('\t',i,self.p[i])

            uplast=up
            rholast=rho
            Elast=E
            uslast=us

            test=1
            flag_conv=0
            count=-1
            while test>toll:
                print('\t',test,up,rho,E,us)
                
                count+=1
                #if you have gone too long then fail
                if count>max_itter:
                    print('Hugoniot calculation not converged')
                    print(i, self.p[i], up, rho, E, us)
                    print(((upnew-uplast)/uplast), ((Enew-Elast)/Elast), ((rhonew-rholast)/rholast), ((usnew-uslast)/uslast))
                    print(test)
                    print(dfkjhdf)
                    
                #find us from momentum conservation
                usnew=(self.p[i]-self.p0)/self.rho0/up


                #find density from mass conservation
                upnew=us*(1.0-self.rho0/rho)
                

                #find the energy from energy conservation
                rhonew=1.0/((1.0/self.rho0)-2*(E-self.E0)/(self.p[i]+self.p0))

                if rhonew<=self.rho0:
                    print('oh oh rho',rho,self.rho0)
                    rhonew=self.rho0



                #now calculate the energy from density and pressure
                temp=self.EOS.find_dp(rho,self.p[i])
                print(temp)
                if (temp[6]!=0):
                    #print('oh oh')
                    Enew=E
                else:
                    #print(temp)
                    #print(temp[2][0])
                    #print(dlfkjd)
                    Enew=temp[2][0]

                
                if Enew<=self.E0:
                    print('oh oh E',Enew,self.E0)
                    #Enew=self.E0

                
                print(us, usnew, up, upnew, rho, rhonew, E, Enew)


                #change the conversion requirements
                if (flag_conv==0)&(test<slow_toll[0]):
                    flag_conv=1
                elif (flag_conv==1)&(test<slow_toll[1]):
                    flag_conv=2

                #update the parameter value
                if flag_conv==0:
                    rho=(0.01*rhonew+0.99*rholast)
                    up=(0.01*upnew+0.99*uplast)
                    E=(0.01*Enew+0.99*Elast)
                    us=(0.01*usnew+0.99*uslast)
                elif flag_conv==1:
                    rho=(0.1*rhonew+0.9*rholast)
                    up=(0.1*upnew+0.9*uplast)
                    E=(0.1*Enew+0.9*Elast)
                    us=(0.1*usnew+0.9*uslast)
                else:
                    rho=rhonew
                    up=upnew
                    E=Enew
                    us=usnew


                #check that the updated value is an acceptable value
                if rho<=self.rho0:
                    print('oh oh rho',rho,self.rho0)
                    rho=self.rho0
               
                #determine convergence
                #test=np.sum(np.asarray([((up-uplast)/up), ((E-Elast)/E), ((rho-rholast)/rho), ((us-uslast)/us),((T-Tlast)/T)])**2)
                test=np.sum(np.asarray([((upnew-uplast)/uplast), ((Enew-Elast)/Elast), ((rhonew-rholast)/rholast), ((usnew-uslast)/uslast)])**2)
                #print(test,((upnew-uplast)/uplast), ((Enew-Elast)/Elast), ((rhonew-rholast)/rholast), ((usnew-uslast)/uslast),((Tnew-Tlast)/Tlast))
                #print(test, (us-(self.p[i]-self.p0)/self.rho0/up)/us, (up-us*(1.0-self.rho0/rho))/up, (E-(self.E0+(self.p[i]+self.p0)*(1.0/self.rho0-1.0/rho)/2))/E)

                
                #print(p/1E9,rho/1E3,E,us/1E3,test)

                #update the previous values
                rholast=rho
                uplast=up
                Elast=E
                uslast=us

            #print(test,count)

            #record converged values
            self.rho[i]=rho
            self.up[i]=up
            self.E[i]=E
            self.us[i]=us

            #print(rho/1E3,E)
            temp=self.EOS.calcpT_rho_E(rho,E)
            #print(temp)
            self.T[i]=temp[1][0]
            self.c[i]=temp[4][0]

            #print(p, rho, E, us,temp[1][0])
            #print(dkfjhdf)

        return flag


    
     ##########################################################
    #function to calculate the hugiont from the EOS
    def calc_hugoniot_p3(self, p, prop0, toll, input_init_flag=0, flag_release=0, max_itter=50000,init_guess=np.nan,slow_toll=[1E-6,1E-30]):

        if input_init_flag==0:
            #initial properties
            self.p0=prop0[0]
            self.T0=prop0[1]

            #calculate other initial properties
            temp=self.EOS.find_PT(self.p0, self.T0)
            self.rho0=temp[0][0]
            self.E0=temp[1][0]


        #record initial properties
        self.p=p

        #initialise output arrays
        flag=0
        self.up=np.empty(np.size(self.p))
        self.rho=np.empty(np.size(self.p))
        self.E=np.empty(np.size(self.p))
        self.us=np.empty(np.size(self.p))
        self.T=np.empty(np.size(self.p))
        self.c=np.empty(np.size(self.p))

        
        #make a guess for the first point
        if np.any(np.isnan(init_guess)):
        
            #make a guess for the first point
            if flag_release==0:
                rho=1.1*self.rho0
                E=1.001*self.E0
                up=1E2#np.sqrt(0.1*self.p0/self.rho0/1.1)
                us=1.1*up
                T=1.1*self.T0
            else:
                rho=0.9*self.rho0
                E=0.95*self.E0
                up=1E2#np.sqrt(0.1*self.p0/self.rho0/1.1)
                us=0.9*up
                T=0.9*self.T0

                rho=0.99999*self.rho0
                E=0.9999*self.E0
                up=1E2#np.sqrt(0.0001*self.p0/self.rho0/0.00001)
                us=0.9999*up
                T=0.9999*self.T0

        else:
            up=init_guess[0]
            rho=init_guess[1]
            E=init_guess[2]
            us=init_guess[3]
            T=init_guess[4]
        #print([p,rho,E,us])
        
        #look at each up seperately
        for i in np.arange(np.size(self.p)):
            print('\t',i,self.p[i])

            
            #uplast=0.999*up
            #rholast=0.999*rho
            #Elast=0.999*E
            #uslast=0.999*us
            #Tlast=0.999*T

            uplast=up
            rholast=rho
            Elast=E
            uslast=us
            Tlast=T

            test=1
            flag_conv=0
            count=-1
            while test>toll:
                #print('\t',test,up,rho,E,us)
                flag_error_test=0.0
                
                count+=1
                #if you have gone too long then fail
                if count>max_itter:
                    print('Hugoniot calculation not converged')
                    print(i, self.p[i], up, rho, E, us,T)
                    print(test)
                    print(dfkjhdf)
                    
                #find us from momentum conservation
                usnew=(self.p[i]-self.p0)/self.rho0/up


                #find density from mass conservation
                upnew=us*(1.0-self.rho0/rho)
                

                #find the energy from energy conservation
                Enew=self.E0+(self.p[i]+self.p0)*(1.0/self.rho0-1.0/rho)/2

                

                #now recalculate the pressure from the EOS given rho and E
                temp=self.EOS.calcpT_rho_E(rho,E)
                #temp=self.EOS.calcpT_rho_E(rho,E)
                if (temp[2]!=0):
                    #print('oh oh')
                    Tnew=T*(1E-2*rand_calc_hug.random()+1)
                    flag_error_test=1.0
                else:
                    #print(temp)
                    Tnew=temp[1][0]
                #print(temp)

                ##now recalculate the pressure from the EOS given rho and E
                #temp=self.EOS.find_dp(rho,self.p[i])
                ##temp=self.EOS.calcpT_rho_E(rho,E)
                #if (temp[6]!=0):
                #    print('oh oh T', count)
                #    Tnew=T
                #    flag_error_test=1.0
                #else:
                #    #print(temp)
                #    Tnew=temp[0][0]
                #    T=Tnew
                ##print(temp)

                temp=self.EOS.find_PT(self.p[i],T)
                if (temp[6]!=0):
                    print('oh oh rho', count)
                    rhonew=rho*(1E-2*rand_calc_hug.random()+1)
                    flag_error_test=1.0
                else:
                    #print(temp)
                    rhonew=temp[0][0]


                #change the conversion requirements
                if (flag_conv==0)&(test<slow_toll[0]):
                    flag_conv=1
                elif (flag_conv==1)&(test<slow_toll[1]):
                    flag_conv=2

                #update the parameter value
                if flag_conv==0:
                    rho=(0.01*rhonew+0.99*rholast)
                    up=(0.01*upnew+0.99*uplast)
                    E=(0.01*Enew+0.99*Elast)
                    us=(0.01*usnew+0.99*uslast)
                    #T=(0.01*Tnew+0.99*Tlast)
                elif flag_conv==1:
                    rho=(0.1*rhonew+0.9*rholast)
                    up=(0.1*upnew+0.9*uplast)
                    E=(0.1*Enew+0.9*Elast)
                    us=(0.1*usnew+0.9*uslast)
                    #T=(0.1*Tnew+0.9*Tlast)
                else:
                    rho=rhonew
                    up=upnew
                    E=Enew
                    us=usnew
                    #T=Tnew

                #check that the updated value is an acceptable value
                if rho<=self.rho0:
                    print('oh oh rho2', count)
                    rho=self.rho0+rand_calc_hug.random()*100
                    flag_error_test=1.0
                if E<=self.E0:
                    print('oh oh E2', count)
                    E=self.E0+rand_calc_hug.random()*1000
                    flag_error_test=1.0
                if us<=up:
                    print('oh oh us2', count)
                    us=up+rand_calc_hug.random()*1000
                    flag_error_test=1.0
                if T<=self.T0:
                    print('oh oh T2', count)
                    T=self.T0+rand_calc_hug.random()*1000
                    flag_error_test=1.0

                #determine convergence
                test=np.sum(np.asarray([((up-uplast)/up), ((E-Elast)/E), ((rho-rholast)/rho), ((us-uslast)/us),((T-Tlast)/T)])**2)
                test=flag_error_test+np.amax([np.absolute((us-(self.p[i]-self.p0)/self.rho0/up)/us), np.absolute((up-us*(1.0-self.rho0/rho))/up), np.absolute((E-(self.E0+(self.p[i]+self.p0)*(1.0/self.rho0-1.0/rho)/2))/E)])
                #print(test,((upnew-uplast)/uplast), ((Enew-Elast)/Elast), ((rhonew-rholast)/rholast), ((usnew-uslast)/uslast),((Tnew-Tlast)/Tlast))
                #print(test,(us-(self.p[i]-self.p0)/self.rho0/up)/us, (up-us*(1.0-self.rho0/rho))/up, (E-(self.E0+(self.p[i]+self.p0)*(1.0/self.rho0-1.0/rho)/2))/E,i)

                #print(p/1E9,rho/1E3,E,us/1E3,test)

                #update the previous values
                rholast=rho
                uplast=up
                Elast=E
                uslast=us
                Tlast=T

            print(test,count)

            #record converged values
            self.rho[i]=rho
            self.up[i]=up
            self.E[i]=E
            self.us[i]=us

            #print(rho/1E3,E)
            temp=self.EOS.calcpT_rho_E(rho,E)
            #print(temp)
            self.T[i]=temp[1][0]
            self.c[i]=temp[4][0]

            #print(p, rho, E, us,temp[1][0])
            #print(dkfjhdf)

        return flag

         ##########################################################
    #function to calculate the hugiont from the EOS
    def calc_hugoniot_p(self, p, prop0, toll, toll_fail=1E-4, input_init_flag=0, flag_release=0, max_itter=100000,init_guess=np.nan,frac_slow=0.5,Nslow=100,min_frac_conv=1E-4,itter_slow=1000,flag_version=0,flag_silent=1):

        #print('calc hug p')
        if input_init_flag==0:
            #initial properties
            self.p0=prop0[0]
            self.T0=prop0[1]

            #calculate other initial properties
            temp=self.EOS.find_PT(self.p0, self.T0)
            self.rho0=temp[0][0]
            self.E0=temp[1][0]


        #record initial properties
        self.p=p

        #initialise output arrays
        flag=0
        self.up=np.empty(np.size(self.p))
        self.rho=np.empty(np.size(self.p))
        self.E=np.empty(np.size(self.p))
        self.us=np.empty(np.size(self.p))
        self.T=np.empty(np.size(self.p))
        self.c=np.empty(np.size(self.p))

        
        #make a guess for the first point
        if np.any(np.isnan(init_guess)):
        
            #make a guess for the first point
            if flag_release==0:
                rho=1.1*self.rho0
                E=1.001*self.E0
                up=1E2#np.sqrt(0.1*self.p0/self.rho0/1.1)
                us=1.1*up
                T=1.1*self.T0
            else:
                rho=0.9*self.rho0
                E=0.95*self.E0
                up=1E2#np.sqrt(0.1*self.p0/self.rho0/1.1)
                us=0.9*up
                T=0.9*self.T0

                rho=0.99999*self.rho0
                E=0.9999*self.E0
                up=1E2#np.sqrt(0.0001*self.p0/self.rho0/0.00001)
                us=0.9999*up
                T=0.9999*self.T0

        else:
            up=init_guess[0]
            rho=init_guess[1]
            E=init_guess[2]
            us=init_guess[3]
            T=init_guess[4]
        #print([p,rho,E,us])

        #print('lets do this')
        
        #look at each up seperately
        for i in np.arange(np.size(self.p)):
            #print(self.up[i])
            #print('\t',i,self.p[i])

            if (i==0)|(flag_version==0):
                uplast=up
                rholast=rho
                Elast=E
                uslast=us
                Tlast=T

            else:
                if i==2:
                    dupdp=gradient2(self.p[0:2],self.up[0:2], flag_calc=[1,0], flag_order=[1,2])
                    up=self.up[i-1]+(self.p[i]-self.p[i-1])*dupdp[-1]

                    drhodp=gradient2(self.p[0:2],self.rho[0:2], flag_calc=[1,0], flag_order=[1,2])
                    rho=self.rho[i-1]+(self.p[i]-self.p[i-1])*drhodp[-1]

                    dEdp=gradient2(self.p[0:2],self.E[0:2], flag_calc=[1,0], flag_order=[1,2])
                    E=self.E[i-1]+(self.p[i]-self.p[i-1])*dEdp[-1]

                    dusdp=gradient2(self.p[0:2],self.us[0:2], flag_calc=[1,0], flag_order=[1,2])
                    us=self.us[i-1]+(self.p[i]-self.p[i-1])*dusdp[-1]

                    dTdp=gradient2(self.p[0:2],self.T[0:2], flag_calc=[1,0], flag_order=[1,2])
                    T=self.T[i-1]+(self.p[i]-self.p[i-1])*dTdp[-1]

                elif i>2:
                    dupdp, d2updp2=gradient2(self.p[i-3:i],self.up[i-3:i], flag_calc=[1,1], flag_order=[2,2])
                    up=self.up[i-1]+(self.p[i]-self.p[i-1])*dupdp[-1]+((self.p[i]-self.p[i-1])**2)*d2updp2[-1]/2.

                    drhodp, d2rhodp2=gradient2(self.p[i-3:i],self.rho[i-3:i], flag_calc=[1,1], flag_order=[2,2])
                    rho=self.rho[i-1]+(self.p[i]-self.p[i-1])*drhodp[-1]+((self.p[i]-self.p[i-1])**2)*d2rhodp2[-1]/2.

                    dEdp, d2Edp2=gradient2(self.p[i-3:i],self.E[i-3:i], flag_calc=[1,1], flag_order=[2,2])
                    E=self.E[i-1]+(self.p[i]-self.p[i-1])*dEdp[-1]+((self.p[i]-self.p[i-1])**2)*d2Edp2[-1]/2.

                    dusdp, d2usdp2=gradient2(self.p[i-3:i],self.us[i-3:i], flag_calc=[1,1], flag_order=[2,2])
                    us=self.us[i-1]+(self.p[i]-self.p[i-1])*dusdp[-1]+((self.p[i]-self.p[i-1])**2)*d2usdp2[-1]/2.

                    dTdp, d2Tdp2=gradient2(self.p[i-3:i],self.T[i-3:i], flag_calc=[1,1], flag_order=[2,2])
                    T=self.T[i-1]+(self.p[i]-self.p[i-1])*dTdp[-1]+((self.p[i]-self.p[i-1])**2)*d2Tdp2[-1]/2.

                #check that the updated value is an acceptable value
                #if rho<=self.rho0:
                #    #print('oh oh rho2', count)
                #    rho=rholast
                #    flag_error_test=1.0
                #if E<=self.E0:
                #    #print('oh oh E2', count)
                #    E=Elast
                #    flag_error_test=1.0
                #if us<=up:
                #    #print('oh oh us2', count)
                #    us=up*rho/self.rho0
                #    flag_error_test=1.0
                #if T<=self.T0:
                #    #print('oh oh T2', count)
                #    T=Tlast
                #    flag_error_test=1.0

                uplast=up
                rholast=rho
                Elast=E
                uslast=us
                Tlast=T

                    

            test=1
            flag_conv=0
            frac_conv=0.1
            
            count=-1
            count_tot=-1
            while test>toll:
                #print('\t',test,up,rho,E,us)

                flag_error_test=0
                
                count+=1
                count_tot+=1

                if (i<=2)|(flag_version==0):
                    #find us from momentum conservation
                    usnew=(self.p[i]-self.p0)/self.rho0/up
                    
                    
                    #find density from mass conservation
                    upnew=us*(1.0-self.rho0/rho)
                    
                    #find the energy from energy conservation
                    Enew=self.E0+(self.p[i]+self.p0)*(1.0/self.rho0-1.0/rho)/2
                    
                    
                    
                    #now recalculate the pressure from the EOS given rho and E
                    temp=self.EOS.calcpT_rho_E(rho,E)
                    #temp=self.EOS.calcpT_rho_E(rho,E)
                    if (temp[2]!=0):
                        print('oh oh T', count)
                        Tnew=T
                    else:
                        #print(temp)
                        Tnew=temp[1][0]
                        T=Tnew


                    temp=self.EOS.find_PT(self.p[i],T)
                    if np.isnan(temp[0]).any():#(temp[2]!=0):
                        rhonew=rho
                    else:
                        #print(temp)
                        #temp2=temp[0]
                        #print(temp[0],np.size(temp[0]))
                        #print(temp)
                        rhonew=temp[0][0]

                else:
                     #find us from momentum conservation
                    usnew=(self.p[i]-self.p0)/self.rho0/up
                    
                    
                    #find density from mass conservation
                    upnew=usnew*(1.0-self.rho0/rho)
                    
                    #find the energy from energy conservation
                    Enew=self.E0+(self.p[i]+self.p0)*(1.0/self.rho0-1.0/rho)/2
                    
                    
                    
                    #now recalculate the pressure from the EOS given rho and E
                    temp=self.EOS.calcpT_rho_E(rho,Enew)
                    #temp=self.EOS.calcpT_rho_E(rho,E)
                    if (temp[2]!=0):
                        print('oh oh T', count)
                        Tnew=T
                    else:
                        #print(temp)
                        Tnew=temp[1][0]
                        T=Tnew


                    temp=self.EOS.find_PT(self.p[i],Tnew)
                    if np.isnan(temp[0]).any():#(temp[2]!=0):
                        rhonew=rho
                    else:
                        rhonew=temp[0][0]
                    

                if 1==0:
                    #now recalculate the pressure from the EOS given rho and E
                    temp=self.EOS.find_dp(rho,self.p[i])
                    ##temp=self.EOS.calcpT_rho_E(rho,E)
                    if (temp[6]!=0):
                        #print('oh oh T', count)
                        Tnew=T
                        flag_error_test=1.0
                    else:
                        #print(temp)
                        Tnew=temp[0][0]
                        #T=Tnew
                        ##print(temp)

                    
                ##change the conversion requirements
                #if (flag_conv==0)&(test<slow_toll[0]):
                #    flag_conv=1
                #elif (flag_conv==1)&(test<slow_toll[1]):
                #    flag_conv=2

                
                
                rho=(frac_conv*rhonew+(1-frac_conv)*rholast)
                up=(frac_conv*upnew+(1-frac_conv)*uplast)
                E=(frac_conv*Enew+(1-frac_conv)*Elast)
                us=(frac_conv*usnew+(1-frac_conv)*uslast)
                #T=Tnew
                #T=(frac_conv*Tnew+(1-frac_conv)*Tlast)
               

                #check that the updated value is an acceptable value
                if rho<=self.rho0:
                    #print('oh oh rho2', count)
                    rho=rholast
                    flag_error_test=1.0
                if E<=self.E0:
                    #print('oh oh E2', count)
                    E=Elast
                    flag_error_test=1.0
                if us<=up:
                    #print('oh oh us2', count)
                    us=up*self.rho[i]/self.rho0
                    flag_error_test=1.0
                if T<=self.T0:
                    #print('oh oh T2', count)
                    T=Tlast
                    flag_error_test=1.0

                #determine convergence
                #test=np.sum(np.asarray([((upnew-uplast)/up), ((Enew-Elast)/E), ((rhonew-rholast)/rho), ((usnew-uslast)/us),((Tnew-Tlast)/T)])**2)
                test=flag_error_test+np.amax([np.absolute((us-(self.p[i]-self.p0)/self.rho0/up)/us), np.absolute((up-us*(1.0-self.rho0/rho))/up), np.absolute((E-(self.E0+(self.p[i]+self.p0)*(1.0/self.rho0-1.0/rho)/2))/E)])
                    
                #print(p/1E9,rho/1E3,E,us/1E3,test)

                #update the previous values
                rholast=rho
                uplast=up
                Elast=E
                uslast=us
                Tlast=T

                #if you have gone too long then fail
                if count>itter_slow:
                    if (count_tot>max_itter):
                        if flag_silent==0:
                            print('Hugoniot calculation not converged')
                            print(i, self.p[i], up, rho, E, us,T)
                            print(test,count_tot, flag_conv, count)
                        if (test<toll_fail)&(flag_error_test==0):
                            if flag_silent==0:
                                print('Convergence above failure limits', toll_fail)
                                print('Continuing')
                            test=0
                        elif (flag_error_test==1):
                            if flag_silent==0:
                                print('Error in EOS function', flag_error_test)
                                print('ABORTING')
                            #print('failure')
                            #print(i, self.p[0],self.p[i], up, rho, E, us,T)
                            raise RuntimeError('Error in EOS function and too many itterations')
                        else:
                            if flag_silent==0:
                                print('Convergence below failure limits', toll_fail)
                                print('ABORTING')
                            print(dfkjhdf)
                    elif (flag_conv<Nslow):

                        flag_conv+=1
                        if (frac_conv>toll):
                            frac_conv=frac_conv*frac_slow
                        if frac_conv<min_frac_conv: #make sure we don't go too small
                            frac_conv=min_frac_conv
                        count=0
                        if flag_silent==0:
                            print('Hugoniot calculation not converged',test,self.p[i],up)
                            print('Reducing fractional step',flag_conv, frac_conv)
                            print(count_tot,test,(us-(self.p[i]-self.p0)/self.rho0/up)/us, (up-us*(1.0-self.rho0/rho))/up, (E-(self.E0+(self.p[i]+self.p0)*(1.0/self.rho0-1.0/rho)/2))/E,i)

            #print('x')
            #print(i, count,test,((up-uplast)/up), ((E-Elast)/E), ((rho-rholast)/rho), ((us-uslast)/us),((T-Tlast)/T))
            #print(i, count,test,((upnew-uplast)/uplast), ((Enew-Elast)/Elast), ((rhonew-rholast)/rholast), ((usnew-uslast)/uslast),((Tnew-Tlast)/Tlast))
            #print(i, count_tot,test,(us-(self.p[i]-self.p0)/self.rho0/up)/us, (up-us*(1.0-self.rho0/rho))/up, (E-(self.E0+(self.p[i]+self.p0)*(1.0/self.rho0-1.0/rho)/2))/E,i)

            #record converged values
            self.rho[i]=rho
            self.up[i]=up
            self.E[i]=E
            self.us[i]=us

            #print(rho/1E3,E)
            temp=self.EOS.calcpT_rho_E(rho,E)
            #print(temp)
            self.T[i]=temp[1][0]
            self.c[i]=temp[4][0]

            #print(p, rho, E, us,temp[1][0])
            #print(dkfjhdf)

            

        return flag


    ##########################################################
    #function to calculate the release from a shocked state
    #var_flag: 0= density, 1=pressure
    def calc_release(self, var, up0, prop0, toll, input_init_flag=0,Nint_steps=3,var_flag=0):

        flag=0

        
        #record initial states
        self.up0=up0

        if input_init_flag==0:
            #initial properties
            self.p0=prop0[0]
            self.T0=prop0[1]

            #calculate other initial properties
            temp=self.EOS.find_PT(self.p0, self.T0)
            self.rho0=temp[0][0]
            self.E0=temp[1][0]
            self.s0=temp[2][0]

        #find the thermodynamic arrays depending on the variable flag
        self.rho=np.empty(np.size(var))
        self.p=np.empty(np.size(var))
        self.up=np.empty(np.size(var))
        self.E=np.empty(np.size(var))
        self.us=np.empty(np.size(var))
        self.T=np.empty(np.size(var))
        self.c=np.empty(np.size(var))
        self.s=np.ones(np.size(var))*self.s0
        if var_flag==0:
            self.rho=var

            for i in np.arange(np.size(var)):
                temp=self.EOS.find_ds(self.rho[i],self.s0)
                self.T[i]=temp[0][0]
                self.p[i]=temp[1][0]
                self.E[i]=temp[2][0]
                self.c[i]=temp[3][0]
        
        elif var_flag==1:
            self.p=var

            #calc initial step
            temp=self.EOS.find_ds(self.rho0,self.s0)
            self.rho[0]=self.rho0
            self.T[0]=temp[0][0]
            self.p[0]=temp[1][0]
            self.E[0]=temp[2][0]
            self.c[0]=temp[3][0]

            #then need to find subsequent steps itteratively
            for i in np.arange(1,np.size(var)):

                temp=fsolve(self.func_calc_isentrope_p, [self.rho[i-1]], args=(self.p[i],self.s0),xtol=1E-8)

                self.rho[i]=temp

                temp=self.EOS.find_ds(self.rho[i],self.s0)

                self.T[i]=temp[0][0]
                #self.p[i]=temp[1][0]
                self.E[i]=temp[2][0]
                self.c[i]=temp[3][0]

    

        #loop over each rho point to find the up values
        for i in np.arange(np.size(var)):
            #calculate the Reiman integral
            self.up[i]=self.up0

            #potential stages integration
            #fac=(self.rho0/self.rho[i])**(1.0/Nint_steps)
            #print(fac)
            #rhostart=self.rho0
            #rhoend=rhostart/fac
            #for j in np.arange(0,Nint_steps):
            #    print(self.rho0,self.rho[i],rhostart,rhoend)
            #    temp=quad(self.func_integrate_release,rhostart,rhoend,args=(self.s0),limit=500)
            #    rhostart=rhostart/fac
            #    rhoend=rhoend/fac
            #    self.up[i]=self.up0-temp[0]

            
            #rhostart=np.linspace(self.rho0,self.rho[i],Nint_steps+1)
            #for j in np.arange(Nint_steps):
            #    temp=quad(self.func_integrate_release,rhostart[j],rhostart[j+1],args=(self.s0),limit=5000)
            #    self.up[i]=self.up0-temp[0]

            
            #print(self.rho[i])    
            #potential log integration
            #temp=quad(self.func_integrate_release_log,np.log10(self.rho0),np.log10(self.rho[i]),args=(self.s0),limit=5000)


            #if density is the variable just do a simple integration
            if i==0:
                temp=quad(self.func_integrate_release,self.rho0,self.rho[i],args=(self.s0),epsabs=toll, epsrel=toll, limit=50000)
                self.up[i]=self.up0-temp[0]
            else:
                temp=quad(self.func_integrate_release,self.rho[i-1],self.rho[i],args=(self.s0),epsabs=toll, epsrel=toll, limit=50000)
                self.up[i]=self.up[i-1]-temp[0]
                    


        return flag
        

    #######################
    ##integration function for the Reimann integral
    #def func_integrate_release_log(self,logrho,s):
    #    rho=10**logrho
    #    temp=self.EOS.find_ds(rho,s)
    #    #print('ll')
    #    #print(rho,s)
    #    #print(temp)
    #    if np.isnan(temp[3])==False:
    #        return temp[3][0]
    #    else:
    #        return 0

    ######################
    #integration function for the Reimann integral
    def func_integrate_release(self,rho,s):
        temp=self.EOS.find_ds(rho,s)
        #print('ll')
        #print(rho,s)
        #print(temp)
        if np.isnan(temp[3])==False:
            return temp[3][0]/rho
        else:
            return 0

    
    ######################
    #Function to find the density corresponding to a given pressure on an isentrope            
    def func_calc_isentrope_p(self,rho,p,s):
        temp=self.EOS.find_ds(rho,s)

        return p-temp[1][0]

        

    ##########################################################
    #function to calculate the hugiont of an ideal gas
    def calc_hugoniot_up_ideal_gas(self, up, prop0, gamma, ma, input_init_flag=0):

        #record gas properties
        self.gamma=gamma
        self.ma=ma

        #process initial state
        if input_init_flag==0:
            #initial properties
            self.p0=prop0[0]
            self.T0=prop0[1]

            self.rho0=self.ma*self.p0/(const.R*self.T0)
        elif input_init_flag==1:
            #initial properties
            self.rho0=prop0[0]
            self.T0=prop0[1]

            self.p0=(self.rho0*const.R*self.T0)/self.ma

        #record up array
        self.up=up

        #initialise output arrays
        flag=0
        self.p=np.empty(np.size(self.up))
        self.rho=np.empty(np.size(self.up))
        self.E=np.empty(np.size(self.up))
        self.us=np.empty(np.size(self.up))
        self.T=np.empty(np.size(self.up))

        #find the terms for the quadratic equation
        a=1.0/(self.gamma-1)
        b=-self.rho0*self.up**2*(1.0/(self.gamma-1)+0.5)-2*self.p0/(self.gamma-1)
        c=self.p0**2/(self.gamma-1)-self.rho0*self.up**2*self.p0/2

        #find the pressure
        self.p=(-b+np.sqrt(b**2-4*a*c))/2/a

        #find the other properties
        self.us=(self.p-self.p0)/(self.rho0*self.up)
        self.rho=self.rho0*self.us/(self.us-self.up)
        self.T=self.ma*self.p/(const.R*self.rho)
        self.E=self.p/self.rho/(self.gamma-1)

        return flag

###############################################################
###############################################################
#Functions to calculate the impedance matching
###############################################################
###############################################################
#find the impedance match using the reverse hugoniot
def find_impedance_match_reverve_hug_ideal_gas(H,gamma,ma,p0,T0,upmax):

    temp=fsolve(func_find_impedance_match_reverve_hug_ideal_gas, [0.1*upmax/1E3], args=(H,gamma,ma,p0,T0,upmax),xtol=1E-8)

    return temp*1E3


def func_find_impedance_match_reverve_hug_ideal_gas(up,H,gamma,ma,p0,T0,upmax):
    
    up=up*1E3
    rho0_gas=ma*p0/(const.R*T0)
    
    a=1.0/(gamma-1)
    b=-rho0_gas*up**2*(1.0/(gamma-1)+0.5)-2*p0/(gamma-1)
    c=p0**2/(gamma-1)-rho0_gas*up**2*p0/2

    
    pgas=(-b+np.sqrt(b**2-4*a*c))/2/a

    H.calc_hugoniot_up(2*upmax-up,[p0,T0],1E-15)
    
    return (H.p-pgas)

#find the impedance match using the reverse hugoniot in the strong shock regime
def find_impedance_match_reverve_hug_ideal_gas_strong_shock(H,gamma,ma,p0,T0,upmax):

    temp=fsolve(func_find_impedance_match_reverve_hug_ideal_gas_strong_shock, [0.1*upmax/1E3], args=(H,gamma,ma,p0,T0,upmax),xtol=1E-8)

    return temp*1E3

def func_find_impedance_match_reverve_hug_ideal_gas_strong_shock(up,H,gamma,ma,p0,T0,upmax):
    
    up=up*1E3
    rho0_gas=ma*p0/(const.R*T0)
    
    us_gas=up*(gamma+1)/2
    
    pgas=p0+rho0_gas*up*us_gas
    
    H.calc_hugoniot_up(2*upmax-up,[p0,T0],1E-15)
    
    return (H.p-pgas)/pgas

#find the impedance match using an adiabatic release
def find_impedance_match_ideal_gas(rel,H_IG,gamma,ma,p0,T0,upmax,pmax,Tmax, input_init_flag=0,initial_guess_fac=0.1):

    if input_init_flag==0:
        temp=rel.EOS.find_PT(pmax, Tmax)
        rhomax=temp[0][0]

    temp=fsolve(func_find_impedance_match_ideal_gas, np.asarray([rhomax*initial_guess_fac]), args=(rel,H_IG,gamma,ma,p0,T0,upmax,pmax,Tmax,rhomax),xtol=1E-15)
    

    #print(func_find_impedance_match_ideal_gas(temp,rel,H_IG,gamma,ma,p0,T0,upmax,pmax,Tmax,rhomax))

    return temp

def func_find_impedance_match_ideal_gas(rho,rel,H_IG,gamma,ma,p0,T0,upmax,pmax,Tmax,rhomax):

    if rho>rhomax:
        rho=np.asarray([rhomax])

    if rho<0.0:
        return rho-1.0

    temp=np.linspace(rhomax, rho,10)
    rel.calc_release(temp,upmax,[pmax,Tmax],1.0E-15)
    
    H_IG.calc_hugoniot_up_ideal_gas(rel.up[-1], [p0,T0], gamma, ma)

    #print(rho,rel.p[-1],H_IG.p,(rel.p[-1]-H_IG.p)/rel.p[-1])

    return (rel.p[-1]-H_IG.p)/rel.p[-1] #(rel.rho[-1]-H_IG.rho)/rel.rho[-1]


#############
#find the impedance match using an adiabatic release
def find_impedance_match(rel,Hug_match,p0,T0,upmax,pmax,Tmax, input_init_flag=0,initial_guess_fac=0.1, max_itter=10000,toll=1E-14,Ncalc=10, toll_fail_Hug=1E-4, max_itter_Hug=100000, frac_slow_Hug=0.5,Nslow_Hug=100,min_frac_conv_Hug=1E-4,itter_slow_Hug=1000, flag_silent=1, flag_Hugp_calc_version=0):


    if input_init_flag==0:
        temp=rel.EOS.find_PT(pmax, Tmax)
        rhomax=temp[0][0]

    temp=Hug_match.EOS.find_PT(p0, T0)
    rho0_match=temp[0][0]

    temp=fsolve(func_find_impedance_match, np.asarray([rhomax*initial_guess_fac]), args=(rel,Hug_match,p0,T0,rho0_match,upmax,pmax,Tmax,rhomax,max_itter,toll,Ncalc,toll_fail_Hug, max_itter_Hug, frac_slow_Hug,Nslow_Hug,min_frac_conv_Hug,itter_slow_Hug,flag_silent,flag_Hugp_calc_version),xtol=toll)

    #print('Do we alreadt gave it?', Hug_match.up[-1])

    #print(func_find_impedance_match_ideal_gas(temp,rel,H_IG,gamma,ma,p0,T0,upmax,pmax,Tmax,rhomax))

    return temp

def func_find_impedance_match(rho,rel,Hug_match,p0,T0,rho0_match,upmax,pmax,Tmax,rhomax,max_itter,toll,Ncalc,toll_fail_Hug, max_itter_Hug, frac_slow_Hug,Nslow_Hug,min_frac_conv_Hug,itter_slow_Hug,flag_silent,flag_Hugp_calc_version):

    if flag_silent==0:
        print('come again', rho,rhomax,rho0_match)

    if rho>rhomax:
        rho=np.asarray([rhomax])
        return rho-rhomax+1

    if rho<0.0:
        return rho-1.0

    temp=np.linspace(rhomax, rho,Ncalc)
    rel.calc_release(temp,upmax,[pmax,Tmax],toll)

    if flag_silent==0:
        print(rel.up[-1])
        print(rel.p[-1],p0)
        print(rel.rho[-1])
        print('\t albitros')
        
    if np.any(Hug_match.p==0.0):
        if flag_silent==0:
            print('start it easy')
        init_guess=np.nan
        pcalc=np.linspace(np.amin([2*p0,rel.p[-1]*0.9]),rel.p[-1],Ncalc) #2
        #pcalc=np.logspace(np.log10(np.amin([2*p0,rel.p[-1]*0.9])),np.log10(rel.p[-1]),Ncalc) #2
    elif (np.absolute((rel.up[-1]-Hug_match.up[-1])>1.0)):
        if flag_silent==0:
            print('still some way to go')
            
        #init_guess=[Hug_match.up[0],Hug_match.rho[0],Hug_match.E[0],\
        #            Hug_match.us[0],Hug_match.T[0]]
        #if (np.isnan(init_guess).any)| (np.isinf(init_guess).any):
        #    init_guess=np.nan
        init_guess=np.nan
        pcalc=np.linspace(np.amin([2*p0,rel.p[-1]*0.9]),rel.p[-1],Ncalc)
        #pcalc=np.logspace(np.log10(np.amin([2*p0,rel.p[-1]*0.9])),np.log10(rel.p[-1]),Ncalc) #2
    else:
        init_guess=[Hug_match.up[-1],Hug_match.rho[-1],Hug_match.E[-1],\
                    Hug_match.us[-1],Hug_match.T[-1]]
        #if (np.isnan(init_guess).any)| (np.isinf(init_guess).any):
        #    init_guess=np.nan
        pcalc=[rel.p[-1]]

        if flag_silent==0:
            print('we know what is happening', pcalc, init_guess)

    
    #print(pcalc)
    #Hug_match.calc_hugoniot_p(pcalc, [p0,T0], toll,max_itter=max_itter,init_guess=init_guess)
    
    Hug_match.calc_hugoniot_p(pcalc, [p0,T0], toll,toll_fail=toll_fail_Hug, max_itter=max_itter_Hug,init_guess=init_guess,frac_slow=frac_slow_Hug,Nslow=Nslow_Hug,min_frac_conv=min_frac_conv_Hug,itter_slow=itter_slow_Hug,flag_version=flag_Hugp_calc_version)

    if flag_silent==0:
        print('\t',rel.up[-1],Hug_match.up[-1],rel.p[-1],rel.rho[-1])

    return (rel.up[-1]-Hug_match.up[-1])/rel.up[-1]


    #if Hug_match.p==0.0:
    #    print('start it easy')
    #    init_guess=np.nan
    #    up_calc=np.linspace(np.amin([100,rel.up[-1]]),rel.up[-1],Ncalc)
    #    print(up_calc)
    #else:
    #    init_guess=[Hug_match.p[-1],Hug_match.rho[-1],Hug_match.E[-1],Hug_match.us[-1]]
    #    up_calc=[rel.up[-1]]
    #    print('we know what is happening', up_calc, init_guess)
    #Hug_match.calc_hugoniot_up(up_calc, [p0,T0], toll,max_itter=max_itter,init_guess=init_guess)

    #print('\t',rel.up[-1],rel.p[-1], Hug_match.p[-1],(rel.p[-1]-Hug_match.p[-1])/rel.p[-1])

    #print(rho,rel.p[-1],H_IG.p,(rel.p[-1]-H_IG.p)/rel.p[-1])

    #return (rel.p[-1]-Hug_match.p[-1])/rel.p[-1] #(rel.rho[-1]-H_IG.rho)/rel.rho[-1]


##find the impedance match using an adiabatic release
#def find_impedance_match(rel,H,p0,T0,upmax,pmax,Tmax, input_init_flag=0,NHug_calc=10,up_calc_min=0.1E3,toll=1E-10):
#
#    if input_init_flag==0:
#        temp=rel.EOS.find_PT(pmax, Tmax)
#        rhomax=temp[0][0]
#
#    temp=fsolve(func_find_impedance_match, np.asarray([rhomax*0.99]), \
#                args=(rel,H,p0,T0,upmax,pmax,Tmax,rhomax,NHug_calc,up_calc_min,toll),xtol=1E-14)

#    #print(func_find_impedance_match_ideal_gas(temp,rel,H_IG,gamma,ma,p0,T0,upmax,pmax,Tmax,rhomax))

#    return temp

#def func_find_impedance_match(rho,rel,H_IG,p0,T0,upmax,pmax,Tmax,rhomax,NHug_calc=10,up_calc_min=0.1E3, toll=1E-10):

#    if rho>rhomax:
#        rho=np.asarray([rhomax])
#        
#    rel.calc_release(rho,upmax,[pmax,Tmax],1.0)
#    
#    calc_up=np.linspace(up_calc_min,rel.up[-1],NHug)
#    H.calc_hugoniot_up(calc_up, [p0,T0],toll)
#    
#    return (rel.rho[-1]-H.rho[-1])/rel.rho

