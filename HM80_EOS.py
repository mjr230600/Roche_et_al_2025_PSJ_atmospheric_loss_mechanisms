#SJL 7/22
#Implementation of EOS from Hubbard & MacFarlane 1980
#In the same format and with some of the same functions of GSESAME_EOS to allow use with other functions
#currently on the He-H solar mixture EOS. Potentially update later

import numpy as np
from scipy import constants as const
import scipy as sp
import os
import sys
from scipy import integrate
from scipy.optimize import fsolve

from solve_cubic import solve_cubic

###########################################################
###########################################################
###########################################################
#define a class object that is a sesame EOS referenced in density, temperature space
#flag_material: 0, H2-He mixture
#flag_silent: 0 (print) or 1 (don't print non essential output)
#flag_calc_cv: -1 (integrate first along isotherm then isochore), 0 (integrate first along isochore then isotherm), 1 (for H2-He mixture use CvT as used in Kegerreis et al. 2019)
class HM80_EOS(object):
    def __init__(self, flag_material=0, flag_silent=0, flag_calc_cv=0):
        #standard GSESAME_EOS parameters
        self.filename=''
        self.darr=np.empty(0)    #1D array of densities
        self.tarr=np.empty(0)    #1D array of temperatures
        self.dijarr=np.empty(0)*np.nan  #2D array of densities
        self.tijarr=np.empty(0)*np.nan  #2D array of temperatures
        self.parr=np.empty(0)    #2D array of pressures
        self.earr=np.empty(0)    #2D array of internal energies
        self.sarr=np.empty(0)*np.nan    #2D array of entropies
        self.farr=np.empty(0)*np.nan    #2D array of helmholtz free energies
        self.carr=np.empty(0)    #2D array of entropies
        self.KPAarr=np.empty(0)*np.nan    #2D array of ANEOS KPA flags
        self.dsize=0    #number of points in density
        self.tsize=0    #number of points in temperature
        self.fmn=0   #mean atomic number
        self.fmw=0   #mean atomic weight
        self.rho0_ref=0   #reference density
        self.k0_ref=0     #reference ??
        self.t0_ref=0     #reference temperature
        self.cvarr=np.empty(0)*np.nan  #2D array of cv
        self.cparr=np.empty(0)*np.nan  #2D array fo cp
        self.Dome=np.empty(0)  #array for the dome data if required
        self.Dome_an=np.nan #function for an analytical vapor dome

        self.unitstxt=None #text to identify units
        self.unit_conversion=1 #1 for convert to SI if unitstxt=None
        
        #extra flags
        self.flag_material=flag_material
        self.flag_silent=flag_silent
        self.flag_calc_cv=flag_calc_cv
        
        #extra parameters from HM80 for different materials
        if flag_material==0: #H-He solar mixture with sound speed determination from Kegerreis et al. 2018
            # if flag_silent==0: 
                # print('He-H EOS selected')
            
            self.coeff_b=np.asarray([0.328471, 0.0286529, -0.00139609, -0.0231158, 0.0579055, 0.0454488])
            self.coeff_u=np.asarray([-16.05895, 1.22808, -0.0217930, 0.141021, 0.147156, 0.277708, 0.0455347, -0.0558596])
            self.coeff_c=np.asarray([2.3638, -4.9842E-5, 1.1788E-8, -3.8101E-4, 2.6182, 0.45053])
            self.rho0_ref=0.005*1E3
            self.fmw=2.285731445911016E-3 #To be consistent with Kegerreis et al. 2018 -> assuming a H2 content of nebula of ~0.76
            
            self.TS=300.
            self.pS=1E5
            self.rhoS=self.find_PT(np.asarray([self.pS]), np.asarray([self.TS]), flag_calcS=0)[0][0]
            
        else:
            print('WARNING: EOS not coded')
            print('EXITING')
            print(dlkf)
            
    ###########################################################
    #place holder for file read if attempted
    def read_file(self):
        if flag_silent==0: 
            print('No file read necessary')
        return 
    
    ###########################################################
    #function to read the dome data
    def read_Dome(self, Dome_file):
        fDome=open(Dome_file, 'r')
        self.Dome=np.loadtxt(fDome,skiprows=6)
        #convert to base SI
        self.Dome[:,3]=1.0E9*self.Dome[:,3]
        self.Dome[:,4]=1.0E9*self.Dome[:,4]

        return
    
    ###########################################################
    #function to determine properties give d and t
    def interp_all(self, d, t, flag_calcS=0):
        d=np.asarray(d)
        t=np.asarray(t)
        x=np.log(d/self.rho0_ref)
        y=np.log(t)
        
        if self.flag_material==0:
            p=1E11*np.exp(self.coeff_u[0]+self.coeff_u[1]*y+self.coeff_u[2]*(y**2)+self.coeff_u[3]*x*y+\
                     self.coeff_u[4]*x+self.coeff_u[5]*(x**2)+self.coeff_u[6]*(x**3)+self.coeff_u[7]*y*(x**2))

            cv=const.R/self.fmw*(self.coeff_c[0]+self.coeff_c[1]*t+self.coeff_c[2]*(t**2)+\
                        self.coeff_c[3]*(d/1E3)*t+self.coeff_c[4]*(d/1E3)+self.coeff_c[5]*((d/1E3)**2))
            
            if self.flag_calc_cv==0:
                e=const.R/self.fmw*((self.coeff_c[0]+self.coeff_c[4]*(d/1E3)+self.coeff_c[5]*((d/1E3)**2))*t+\
                        0.5*(self.coeff_c[1]+self.coeff_c[3]*(d/1E3))*(t**2)+self.coeff_c[2]/3.0*(t**3))
            elif self.flag_calc_cv==-1:
                e=np.nan*np.ones(np.size(d))
                # print(d,e,t,np.size(d))
                for i,d_temp in enumerate(d):
                    
                    temp=integrate.quad(self.func_integrate_E,self.rho0_ref,d_temp,limit=50000000,epsrel=1E-30, epsabs=1E-30, args=(t[i]))
                    e[i]=temp[0]+const.R/self.fmw*((self.coeff_c[0]+self.coeff_c[4]*(self.rho0_ref/1E3)+self.coeff_c[5]*((self.rho0_ref/1E3)**2))*t[i]+\
                        0.5*(self.coeff_c[1]+self.coeff_c[3]*(self.rho0_ref/1E3))*(t[i]**2)+self.coeff_c[2]/3.0*(t[i]**3))
            else:
                e=cv*t
            
            gamma=self.coeff_b[0]+self.coeff_b[1]*y+self.coeff_b[2]*y**2+\
                    self.coeff_b[3]*x*y+self.coeff_b[4]*x+self.coeff_b[5]*x**2
            c=np.sqrt(gamma*p/d)
        
            #other parameters are just nans
            s=np.nan*np.ones(np.size(x))
            cp=np.nan*np.ones(np.size(x))
            
            if flag_calcS==1:
                for i,d_temp in enumerate(d):
                    temp=integrate.quad(self.func_integrate_S,self.rhoS,d_temp,limit=50000000,epsrel=1E-30, epsabs=1E-30)
                    
                    s[i]=temp[0]+const.R/self.fmw*((self.coeff_c[0]+self.coeff_c[4]*(d_temp/1E3)+self.coeff_c[5]*(d_temp/1E3)**2)*np.log(t[i]/self.TS)+\
                                                    (self.coeff_c[1]+self.coeff_c[3]*(d_temp/1E3))*(t[i]-self.TS)+self.coeff_c[2]*(t[i]**2-self.TS**2)/2.)
                    
                    # print(temp[0],const.R/self.fmw*((self.coeff_c[0]+self.coeff_c[4]*(d_temp/1E3)+self.coeff_c[5]*(d_temp/1E3)**2)*np.log(t[i]/self.TS)+\
                    #                                 (self.coeff_c[1]+self.coeff_c[3]*(d_temp/1E3))*(t[i]-self.TS)+self.coeff_c[2]*(t[i]**2-self.TS**2)/2.),s[i])
                    
            elif flag_calcS==2:
                for i,d_temp in enumerate(d):
                    temp=integrate.quad(self.func_integrate_S2,self.rhoS,d_temp,limit=50000000,epsrel=1E-30, epsabs=1E-30, args=(t[i]))
                    
                    s[i]=temp[0]+const.R/self.fmw*((self.coeff_c[0]+self.coeff_c[4]*(self.rho0_ref/1E3)+self.coeff_c[5]*(self.rho0_ref/1E3)**2)*np.log(t[i]/self.TS)+\
                                                    (self.coeff_c[1]+self.coeff_c[3]*(self.rho0_ref/1E3))*(t[i]-self.TS)+self.coeff_c[2]*(t[i]**2-self.TS**2)/2.)
        
        flag=0
        return p,e,s,c,cv,cp,flag
    
    def func_integrate_E(self,rho,T): #Function for integrating energy when required
        x=np.log(rho/self.rho0_ref)
        yS=np.log(T)
        ptemp=1E11*np.exp(self.coeff_u[0]+self.coeff_u[1]*yS+self.coeff_u[2]*(yS**2)+self.coeff_u[3]*x*yS+\
                     self.coeff_u[4]*x+self.coeff_u[5]*(x**2)+self.coeff_u[6]*(x**3)+self.coeff_u[7]*yS*(x**2))
        
        # p=1E11*np.exp(self.coeff_u[0]+self.coeff_u[1]*y+self.coeff_u[2]*(y**2)+self.coeff_u[3]*x*y+\
        #              self.coeff_u[4]*x+self.coeff_u[5]*(x**2)+self.coeff_u[6]*(x**3)+self.coeff_u[7]*y*(x**2))
        dSdrho=-(self.coeff_u[1]+2.*self.coeff_u[2]*(yS)+self.coeff_u[3]*x+\
                     self.coeff_u[7]*(x**2))*ptemp/T/(rho**2)
        
        dEdrho=ptemp/(rho**2)+T*dSdrho
        
        return dEdrho
        
    
    def func_integrate_S(self,rho):
        x=np.log(rho/self.rho0_ref)
        yS=np.log(self.TS)
        ptemp=1E11*np.exp(self.coeff_u[0]+self.coeff_u[1]*yS+self.coeff_u[2]*(yS**2)+self.coeff_u[3]*x*yS+\
                     self.coeff_u[4]*x+self.coeff_u[5]*(x**2)+self.coeff_u[6]*(x**3)+self.coeff_u[7]*yS*(x**2))
        
        # p=1E11*np.exp(self.coeff_u[0]+self.coeff_u[1]*y+self.coeff_u[2]*(y**2)+self.coeff_u[3]*x*y+\
        #              self.coeff_u[4]*x+self.coeff_u[5]*(x**2)+self.coeff_u[6]*(x**3)+self.coeff_u[7]*y*(x**2))
        dSdrho=-(self.coeff_u[1]+2.*self.coeff_u[2]*(yS)+self.coeff_u[3]*x+\
                     self.coeff_u[7]*(x**2))*ptemp/self.TS/(rho**2)
        
        return dSdrho
    
    def func_integrate_S2(self,rho,T):
        x=np.log(rho/self.rho0_ref)
        yS=np.log(T)
        ptemp=1E11*np.exp(self.coeff_u[0]+self.coeff_u[1]*yS+self.coeff_u[2]*(yS**2)+self.coeff_u[3]*x*yS+\
                     self.coeff_u[4]*x+self.coeff_u[5]*(x**2)+self.coeff_u[6]*(x**3)+self.coeff_u[7]*yS*(x**2))
        
        # p=1E11*np.exp(self.coeff_u[0]+self.coeff_u[1]*y+self.coeff_u[2]*(y**2)+self.coeff_u[3]*x*y+\
        #              self.coeff_u[4]*x+self.coeff_u[5]*(x**2)+self.coeff_u[6]*(x**3)+self.coeff_u[7]*y*(x**2))
        dSdrho=-(self.coeff_u[1]+2.*self.coeff_u[2]*(yS)+self.coeff_u[3]*x+\
                     self.coeff_u[7]*(x**2))*ptemp/T/(rho**2)
        
        return dSdrho
    
    ###########################################################
    #use linear interpolation to find e,s,d for a p-T point
    #returns the points and a flag: -1 if out of range, 0 is for succesful
    def find_PT(self, p, t, flag_calcS=0):
        # print(p,t)
        # p=np.asarray(p)
        # t=np.asarray(t)
        y=np.log(t)

        # print(np.size(p))
        
        if self.flag_material==0:
            #first need to find density as the solution of the cubic
            a3=self.coeff_u[6]
            a2=(self.coeff_u[5]+self.coeff_u[7]*y)
            a1=self.coeff_u[3]*y+self.coeff_u[4]
            a0=self.coeff_u[0]+self.coeff_u[1]*y+self.coeff_u[2]*y**2-np.log(p/1E11)
            # print(a3,a2,a1,a0)
            
            if (type(p)==float)|(type(p)==int)|(type(p)==np.float64)|(type(p)==np.int64):
                (sol)=solve_cubic(a3,a2,a1,a0)[0]
                if np.size(sol)>1:
                    if self.flag_silent==0:
                        print('More than one solution in HM80_EOS.find_PT', sol, p, t, self.rho0_ref*np.exp(sol))
                        print('choosing minium')
                    x=np.asarray([np.amin(sol)])
                else:
                    x=np.asarray([sol])
            else:
                x=np.ones(np.size(y))*np.nan
                for i in np.arange(np.size(y)):
                    (sol)=solve_cubic(a3,a2[i],a1[i],a0[i])

                    if np.size(sol)>1:
                        if self.flag_silent==0:
                            print('More than one solution in HM80_EOS.find_PT', sol, p[i], t[i], self.rho0_ref*np.exp(sol))
                            print('choosing minium')
                        x[i]=np.amin(sol)
                    else:
                        x[i]=sol
                    
            # print(x)
            d=self.rho0_ref*np.exp(x)
                  
            #now find the other properties
            t=np.ones(np.shape(d))*t
            (p,e,s,c,cv,cp,flag)=self.interp_all(d, t, flag_calcS=flag_calcS)
            
        flag=0
        # print(p)
        return d,e,s,c,cv,cp,flag
    
    ###########################################################
    #use linear interpolation to find properties from for a rho-E point
    def calcpT_rho_E(self, d, e):
        # d=np.asarray(d)
        # e=np.asarray(e)
        x=np.log(d/self.rho0_ref)
        
        
        if self.flag_material==0:
            
#             if (self.flag_calc_cv==-1):
#                 d=np.asarray([d])
#                 t=np.ones(np.size(d))*np.nan
#                 #if we are using the second path for calculating energy then we need to solve the energy equation numerically
#                 # print('Need to calculate the root of the energy equation')
                
#                 if (type(d)==float)|(type(d)==int)|(type(d)==np.float64)|(type(d)==np.int64):
#                     # temp=fsolve(self.func_calc_Eroot,t,args=(d,e))#,xtol=1E-14)
#                     temp=fsolve(self.func_calc_Eroot,500.,args=(d,e))#,xtol=1E-14)
#                     t[0]=temp[0]
#                 elif np.size(d)==1:
#                     # temp=fsolve(self.func_calc_Eroot,t[0],args=(d[0],e))#,xtol=1E-14)
#                     temp=fsolve(self.func_calc_Eroot,500.,args=(d[0],e))#,xtol=1E-14)
#                     t[0]=temp[0]*np.ones(np.shape(d))
#                 else:
#                     for i,d_temp in enumerate(d):
#                         # temp=fsolve(self.func_calc_Eroot,t[i],args=(d_temp,e[i]))#,xtol=1E-14)
#                         temp=fsolve(self.func_calc_Eroot,500.,args=(d_temp,e[i]))#,xtol=1E-14)
#                         t[i]=temp[0]
                        
#                 #now find the other properties
#                 (p,e,s,c,cv,cp,flag)=self.interp_all(d, t)
            
#                 # print(p,e,s,c,cv,cp,flag)
        
#                 flag=0
#                 return p,t,flag,s,c,cv,cp

            if (self.flag_calc_cv==0)|(self.flag_calc_cv==-1):
                #first need to find temperature as the solution of the cubic (for flag=-1 this is just a first guess)
                a3=self.coeff_c[2]/3.0
                a2=(self.coeff_c[1]+self.coeff_c[3]*(d/1E3))/2.0
                a1=self.coeff_c[5]*(d/1E3)**2+self.coeff_c[4]*(d/1E3)+self.coeff_c[0]
                a0=-e/const.R*self.fmw

            else:
                a3=self.coeff_c[2]
                a2=(self.coeff_c[1]+self.coeff_c[3]*(d/1E3))
                a1=self.coeff_c[5]*(d/1E3)**2+self.coeff_c[4]*(d/1E3)+self.coeff_c[0]
                a0=-e/const.R*self.fmw

            if (type(d)==float)|(type(d)==int)|(type(d)==np.float64)|(type(d)==np.int64):
                (sol)=solve_cubic(a3,a2,a1,a0)[0]
                if np.size(sol)>1:
                    if self.flag_silent==0:
                        print('More than one solution in HM80_EOS.find_PT', sol, d, e)
                        print('choosing minium')
                    t=np.asarray([np.amin(sol)])
                else:
                    t=np.asarray([sol])
                d=np.asarray([d])
            else:
                t=np.ones(np.size(d))*np.nan
                for i in np.arange(np.size(d)):
                    (sol)=solve_cubic(a3,a2[i],a1[i],a0[i])

                    if np.size(sol)>1:
                        if self.flag_silent==0:
                            print('More than one solution in HM80_EOS.find_PT', sol, d[i], e[i])
                            print('choosing minium')
                        t[i]=np.amin(sol)
                    else:
                        t[i]=sol
                            
            if (self.flag_calc_cv==-1):
                #if we are using the second path for calculating energy then we need to solve the energy equation numerically
                # print('Need to calculate the root of the energy equation')
                
                if (type(d)==float)|(type(d)==int)|(type(d)==np.float64)|(type(d)==np.int64):
                    temp=fsolve(self.func_calc_Eroot,t,args=(d,e))#,xtol=1E-14)
                    # temp=fsolve(self.func_calc_Eroot,500.,args=(d,e))#,xtol=1E-14)
                    t=temp[0]
                elif np.size(d)==1:
                    temp=fsolve(self.func_calc_Eroot,t[0],args=(d[0],e))#,xtol=1E-14)
                    # temp=fsolve(self.func_calc_Eroot,500.,args=(d[0],e))#,xtol=1E-14)
                    t=temp[0]*np.ones(np.shape(d))
                else:
                    for i,d_temp in enumerate(d):
                        temp=fsolve(self.func_calc_Eroot,t[i],args=(d_temp,e[i]))#,xtol=1E-14)
                        # temp=fsolve(self.func_calc_Eroot,500.,args=(d_temp,e[i]))#,xtol=1E-14)
                        t[i]=temp[0]
                        
                # print(d,t,e)
                        
            #now find the other properties
            (p,e,s,c,cv,cp,flag)=self.interp_all(d, t)
            
            # print(p,e,s,c,cv,cp,flag)
        
        flag=0
        return p,t,flag,s,c,cv,cp
    
    
    #Function required to find the root of the energy equation
    def func_calc_Eroot(self,t,d,Ereq):
        (p,e,s,c,cv,cp,flag)=self.interp_all([d], [t])
    
        return (e-Ereq)/Ereq
    
    ###############################################################
    #function to calculate an adiabat in rho-T space
    def calc_adiabat_rhoT(self,T0,rho_calc):
        
        
        temp=integrate.solve_ivp(self.func_calc_adiabat_rhoT,[rho_calc[0],rho_calc[-1]],[T0],t_eval=rho_calc,method='DOP853', atol=1E-30, rtol=3E-14)#, method='DOP853')#, first_step=1E-10, max_step=1E-3)
        # print(temp)
        
        return temp.y[0]
        
    def func_calc_adiabat_rhoT(self, d, t):
        d=np.asarray(d)
        t=np.asarray(t)
        x=np.log(d/self.rho0_ref)
        y=np.log(t)
        
        if self.flag_material==0:
            
            temp=self.interp_all(d, t)

            p=temp[0][0]
            cv=temp[4][0]
            
            dTdrho=p*(self.coeff_u[1]+2.*self.coeff_u[2]*(y)+self.coeff_u[3]*x+\
                     self.coeff_u[7]*(x**2))/(d**2)/cv
            
        return dTdrho
            

                    
    ###############################################################
    #function to calculate an adiabat in rho-T space using a brute force minimization of entropy
    def calc_adiabat_rhoT_rf(self,T0,rho_calc):
        
        # TS_init=self.TS
        # rhoS_init=self.rhoS
        
        T=np.ones(np.size(rho_calc))*np.nan
        
        T[0]=T0
        
        Sad=self.interp_all([rho_calc[0]],[T0],flag_calcS=1)[2][0]
        
        for i in np.arange(1,np.size(rho_calc)):
            
            temp=fsolve(self.func_calc_adiabat_rhoT_rf,T[i-1],args=(rho_calc[i],T[0],rho_calc[0],Sad))#,xtol=1E-14)
            
            T[i]=temp[0]
            
        # self.TS=TS_init
        # self.rhoS=rhoS_init
            
        return T
    
    def func_calc_adiabat_rhoT_rf(self,T,rho,TS,rhoS,Sad):

        
        # self.TS=TS
        # self.rhoS=rhoS
        
        temp=self.interp_all([rho],[T],flag_calcS=1)[2][0]
        
        
        
        return temp-Sad