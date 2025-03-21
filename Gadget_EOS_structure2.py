#SJL 15/11/16
#Structure to hold a gadget EOS
#Updated to include the developments done for HERCULES inclusing log-linear interpolation

import numpy as np
from scipy import constants as const
import scipy as sp
import os
import sys

from scipy.optimize import minimize
from scipy.optimize import fsolve

###########################################################
###########################################################
###########################################################
#define a class object that is a gadget altered sesame EOS referenced in density, S space
class Gadget_EOS(object):
    def __init__(self):
        self.filename=''  #file name containing the EOS
        self.darr=np.empty(0)    #1D array of densities
        self.sarr=np.empty(0)    #1D array of temperatures
        self.dijarr=np.empty(0)  #2D array of densities
        self.sijarr=np.empty(0)  #2D array of temperatures
        self.parr=np.empty(0)    #2D array of pressures
        self.earr=np.empty(0)    #2D array of internal energies
        self.tarr=np.empty(0)    #2D array of entropies
        self.carr=np.empty(0)    #2D array of entropies
        self.dsize=0    #number of points in density
        self.ssize=0    #number of points in temperature
        self.Dome=np.empty(0)  #array for the dome data if required

    ###########################################################
    #function to read in the EOS data from a Gadget Sesame EOS file
    def read_file(self, EOS_file):
        # adapated from /n/gstore/Labs/Stewart_Lab/sstewart/cthruns/from_scratch2/blowoff/init/readgisen.pro
        self.filename=EOS_file

        f=open(EOS_file,'r')

        #read in the number of density and entropy points
        line=f.readline()
        line1=np.asarray(list(map(int,line.strip().split()[0:2])))

        self.dsize=line1[0]
        self.ssize=line1[1]
        ntot=2+self.dsize+self.ssize+4*self.ssize*self.dsize
        #print(self.dsize, self.ssize)

        #read the rest of the first line
        line1=np.asarray(list(map(float,line.strip().split()[2:])))
        allarr=line1
        
        #loop over all remaining rows and make an array of values
        line=f.readline()
        i=0
        #loop until hit the END line
        for i in np.arange(0,ntot/5+1):
            i=i+1
            line1=np.asarray(list(map(float,line.strip().split())))
            allarr=np.append(allarr,line1)
            line=f.readline()

        #now define the arrays and assign the values
        self.darr=np.empty(self.dsize) #density read in as g/cm3 but converted to kg/m^3
        self.sarr=np.empty(self.ssize) #entropy in units??? (1E4 * SI) converted to SI
        self.parr=np.empty((self.ssize,self.dsize)) #pressure in units of dyne/cm^2  but converted to Pa
        self.earr=np.empty((self.ssize,self.dsize)) #internal energy read in as erg/g  but converted to J/kg
        self.tarr=np.empty((self.ssize,self.dsize)) #temperature in K
        self.carr=np.empty((self.ssize,self.dsize)) #sound speed read in as cm/s converted to m/s

        #define d and s including conversion to SI
        #inext is the counter for the point in the all array
        inext=0
        self.darr=allarr[inext:(inext+self.dsize)]*1.0E3 
        inext+=self.dsize
        self.sarr=allarr[inext:(inext+self.ssize)]/1.0E4
        inext+=self.ssize

        #loop over the pressure to put into array
        for i in np.arange(self.ssize):
            self.parr[i,:]=allarr[inext:(inext+self.dsize)]/10.0
            inext+=self.dsize

        #same for temperature
        for i in np.arange(self.ssize):
            self.tarr[i,:]=allarr[inext:(inext+self.dsize)]
            inext+=self.dsize
        
        #same for energy
        for i in np.arange(self.ssize):
            self.earr[i,:]=allarr[inext:(inext+self.dsize)]*1.0E-4
            inext+=self.dsize

        #same for sound speed
        for i in np.arange(self.ssize):
            self.carr[i,:]=allarr[inext:(inext+self.dsize)]/100.0
            inext+=self.dsize

        #print(self.darr[-1], self.sarr[0])
        #print(self.sarr[-1], self.parr[0,0])
        #print(self.parr[-1,-1], self.tarr[0,0])
        #print(self.tarr[-1,-1], self.earr[0,0])
        #print(self.earr[-1,-1], self.carr[0,0])

        #close the file
        f.close()

        #define arrays for i,j T and d and loop to populate
        self.dijarr=np.empty((self.ssize,self.dsize))
        self.sijarr=np.empty((self.ssize,self.dsize))
        for i in np.arange(self.dsize):
            self.sijarr[:,i]=self.sarr
        for i in np.arange(self.ssize):
            self.dijarr[i,:]=self.darr
        
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
    #interpolate the rho-S grid to find the p,t,e,c points
    #returns the points and a flag: -1 if out of range, 0 is for succesful
    def interp_all(self, d, s):
        #search to find the location of the lower bound on t and p
        #if out of bounds return and exit
        temp=np.where(self.darr>=d)
        if np.size(temp[0])==0:
            flag=-1
            p=np.nan
            e=np.nan
            t=np.nan
            c=np.nan
            return p,e,t,c,flag
        else:
            indd=temp[0][0]
        temp=np.where(self.sarr>=s)
        if np.size(temp[0])==0:
            flag=-1
            p=np.nan
            e=np.nan
            t=np.nan
            c=np.nan
            return p,e,t,c,flag
        else:
            inds=temp[0][0]

        #now linearly interpolate
        fracs=(s-self.sarr[inds-1])/(self.sarr[inds]-self.sarr[inds-1])
        fracd=(d-self.darr[indd-1])/(self.darr[indd]-self.darr[indd-1])

        #for p
        temp1=self.parr[inds-1,indd-1]*(1-fracs)+self.parr[inds,indd-1]*fracs
        temp2=self.parr[inds-1,indd]*(1-fracs)+self.parr[inds,indd]*fracs
        p=temp1*(1-fracd)+temp2*fracd

        #for e
        temp1=self.earr[inds-1,indd-1]*(1-fracs)+self.earr[inds,indd-1]*fracs
        temp2=self.earr[inds-1,indd]*(1-fracs)+self.earr[inds,indd]*fracs
        e=temp1*(1-fracd)+temp2*fracd

        #for t
        temp1=self.tarr[inds-1,indd-1]*(1-fracs)+self.tarr[inds,indd-1]*fracs
        temp2=self.tarr[inds-1,indd]*(1-fracs)+self.tarr[inds,indd]*fracs
        t=temp1*(1-fracd)+temp2*fracd

        #for c
        temp1=self.carr[inds-1,indd-1]*(1-fracs)+self.carr[inds,indd-1]*fracs
        temp2=self.carr[inds-1,indd]*(1-fracs)+self.carr[inds,indd]*fracs
        c=temp1*(1-fracd)+temp2*fracd

        flag=0
        return p,e,t,c,flag


    ###########################################################
    #interpolate the rho-S grid to find the p,t,e,c points
    #returns the points and a flag: -1 if out of range, 0 is for succesful
    #This version interpolates in log space for rho<2g/cc
    def interp_all_log(self, d, s):
        rho_log=2E3 #1E99
        #search to find the location of the lower bound on t and p
        #if out of bounds return and exit
        temp=np.where(self.darr>=d)
        if np.size(temp[0])==0:
            flag=-1
            p=np.nan
            e=np.nan
            t=np.nan
            c=np.nan
            return p,e,t,c,flag
        else:
            indd=temp[0][0]

        temp=np.where(self.sarr>=s)
        if np.size(temp[0])==0:
            flag=-1
            p=np.nan
            e=np.nan
            t=np.nan
            c=np.nan
            return p,e,t,c,flag
        else:
            inds=temp[0][0]

        #if density is high enough linearly interpolate
        if (d>rho_log):
            #now linear interpolate
            fracs=(s-self.sarr[inds-1])/(self.sarr[inds]-self.sarr[inds-1])
            fracd=(d-self.darr[indd-1])/(self.darr[indd]-self.darr[indd-1])
            
            #for p
            temp1=self.parr[inds-1,indd-1]*(1-fracs)+self.parr[inds,indd-1]*fracs
            temp2=self.parr[inds-1,indd]*(1-fracs)+self.parr[inds,indd]*fracs
            p=temp1*(1-fracd)+temp2*fracd
            
            #for e
            temp1=self.earr[inds-1,indd-1]*(1-fracs)+self.earr[inds,indd-1]*fracs
            temp2=self.earr[inds-1,indd]*(1-fracs)+self.earr[inds,indd]*fracs
            e=temp1*(1-fracd)+temp2*fracd
            
            #for t
            temp1=self.tarr[inds-1,indd-1]*(1-fracs)+self.tarr[inds,indd-1]*fracs
            temp2=self.tarr[inds-1,indd]*(1-fracs)+self.tarr[inds,indd]*fracs
            t=temp1*(1-fracd)+temp2*fracd
            
            #for c
            temp1=self.carr[inds-1,indd-1]*(1-fracs)+self.carr[inds,indd-1]*fracs
            temp2=self.carr[inds-1,indd]*(1-fracs)+self.carr[inds,indd]*fracs
            c=temp1*(1-fracd)+temp2*fracd

        #for lower densities use log-linear interpolation
        else:
            #now linear interpolate
            fracs=(np.log10(s)-np.log10(self.sarr[inds-1]))/(np.log10(self.sarr[inds])-np.log10(self.sarr[inds-1]))
            fracd=(np.log10(d)-np.log10(self.darr[indd-1]))/(np.log10(self.darr[indd])-np.log10(self.darr[indd-1]))
            
            #for p
            temp1=np.log10(self.parr[inds-1,indd-1])*(1-fracs)+np.log10(self.parr[inds,indd-1])*fracs
            temp2=np.log10(self.parr[inds-1,indd])*(1-fracs)+np.log10(self.parr[inds,indd])*fracs
            p=10**(temp1*(1-fracd)+temp2*fracd)
            
            #for e
            temp1=np.log10(self.earr[inds-1,indd-1])*(1-fracs)+np.log10(self.earr[inds,indd-1])*fracs
            temp2=np.log10(self.earr[inds-1,indd])*(1-fracs)+np.log10(self.earr[inds,indd])*fracs
            e=10**(temp1*(1-fracd)+temp2*fracd)
            
            #for t
            temp1=np.log10(self.tarr[inds-1,indd-1])*(1-fracs)+np.log10(self.tarr[inds,indd-1])*fracs
            temp2=np.log10(self.tarr[inds-1,indd])*(1-fracs)+np.log10(self.tarr[inds,indd])*fracs
            t=10**(temp1*(1-fracd)+temp2*fracd)
            
            #for c
            temp1=np.log10(self.carr[inds-1,indd-1])*(1-fracs)+np.log10(self.carr[inds,indd-1])*fracs
            temp2=np.log10(self.carr[inds-1,indd])*(1-fracs)+np.log10(self.carr[inds,indd])*fracs
            c=10**(temp1*(1-fracd)+temp2*fracd)
            
            
        flag=0
        return p,e,t,c,flag



    ###########################################################
    #find the T, rho at a given S,p point
    #returns the points and a flag: -1 if out of range, 0 is for succesful
    def find_PS(self, p,s):
        #search to find the location of the lower bound on S
        #if out of bounds return and exit
        temp=np.where(self.sarr>=s)
        if np.size(temp[0])==0:
            flag=-1
            d=np.nan
            t=np.nan
            return d,t,flag
        else:
            inds=temp[0][0]

        #now define the array of presure linearly interpolated between the s points
        fracs=(s-self.sarr[inds-1])/(self.sarr[inds]-self.sarr[inds-1])
        temp1=self.parr[inds-1,:]*(1-fracs)+self.parr[inds,:]*fracs

        #find the density points that correspond to this pressure
        #note we are able to pick up degenerate values
        #initialise the ind array and 
        indd=np.asarray([],dtype=int)
        test_stop=0
        test_ind=0
        while test_stop!=1:
            #find the next value based on the current value of pressure 
            if temp1[test_ind]<=p:
                temp=np.where(temp1[test_ind:-1]>=p)
            else:
                temp=np.where(temp1[test_ind:-1]<=p)
            #then test to see if we are at the end or out of range
            if np.size(temp[0])==0:
                if test_ind==0:
                    flag=-1
                    d=np.nan
                    t=np.nan
                    return d,t,flag
                else:
                    test_stop=1
            else:
                #append value and move on
                indd=np.append(indd,test_ind+temp[0][0])
                test_ind=test_ind+temp[0][0]+1

        #now can find array of possible densities
        d=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indd[i]-1])/(temp1[indd[i]]-temp1[indd[i]-1])
            d[i]=self.darr[indd[i]-1]*(1-fracp)+self.darr[indd[i]]*fracp

        #do the same for the other paratmers
        #e
        #temp2=self.earr[inds-1,:]*(1-fracs)+self.earr[inds,:]*fracs
        #e=np.empty(np.size(indd))
        #for i in np.arange(np.size(indd)):
        #    fracp=(p-temp1[indd[i]-1])/(temp1[indd[i]]-temp1[indd[i]-1])
        #    e[i]=temp2[indd[i]-1]*(1-fracp)+temp2[indd[i]]*fracp

        #t
        temp2=self.tarr[inds-1,:]*(1-fracs)+self.tarr[inds,:]*fracs
        t=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indd[i]-1])/(temp1[indd[i]]-temp1[indd[i]-1])
            t[i]=temp2[indd[i]-1]*(1-fracp)+temp2[indd[i]]*fracp

        #c
        #temp2=self.carr[inds-1,:]*(1-fracs)+self.carr[inds,:]*fracs
        #c=np.empty(np.size(indd))
        #for i in np.arange(np.size(indd)):
        #    fracp=(p-temp1[indd[i]-1])/(temp1[indd[i]]-temp1[indd[i]-1])
        #    c[i]=temp2[indd[i]-1]*(1-fracp)+temp2[indd[i]]*fracp


        flag=0
        return d,t,flag

    
    ###########################################################
    #find the S, T, E for a given p, rho point
    #returns the points and a flag: -1 if out of range, 0 is for succesful
    def find_Pd(self, p,d):
        #search to find the location of the lower bound on d
        #if out of bounds return and exit
        temp=np.where(self.darr>=d)
        if np.size(temp[0])==0:
            flag=-1
            s=np.nan
            t=np.nan
            e=np.nan
            return s,t,e,flag
        else:
            inds=temp[0][0]

        #now define the array of presure linearly interpolated between the d points
        fracd=(d-self.darr[inds-1])/(self.darr[inds]-self.darr[inds-1])
        temp1=self.parr[inds-1,:]*(1-fracd)+self.parr[inds,:]*fracd

        #find the entropy points that correspond to this pressure
        #note we are able to pick up degenerate values
        #initialise the ind array and 
        inds=np.asarray([])
        test_stop=0
        test_ind=0
        while test_stop!=1:
            #find the next value based on the current value of pressure 
            if temp1[test_ind]<=p:
                temp=np.where(temp1[test_ind:-1]>=p)
            else:
                temp=np.where(temp1[test_ind:-1]<=p)
            #then test to see if we are at the end or out of range
            if np.size(temp[0])==0:
                if test_ind==0:
                    flag=-1
                    s=np.nan
                    t=np.nan
                    e=np.nan
                    return s,t,e,flag
                else:
                    test_stop=1
            else:
                #append value and move on
                inds=np.append(inds,test_ind+temp[0][0])
                test_ind=test_ind+temp[0][0]+1

        #now can find array of possible entropies
        s=np.empty(np.size(inds))
        for i in np.arange(np.size(inds)):
            fracp=(p-temp1[inds[i]-1])/(temp1[inds[i]]-temp1[inds[i]-1])
            s[i]=self.sarr[inds[i]-1]*(1-fracp)+self.sarr[inds[i]]*fracp

        #do the same for the other paratmers
        #e
        temp2=self.earr[inds-1,:]*(1-fracd)+self.earr[inds,:]*fracd
        e=np.empty(np.size(inds))
        for i in np.arange(np.size(inds)):
            fracp=(p-temp1[inds[i]-1])/(temp1[inds[i]]-temp1[inds[i]-1])
            e[i]=temp2[inds[i]-1]*(1-fracp)+temp2[inds[i]]*fracp

        #t
        temp2=self.tarr[inds-1,:]*(1-fracd)+self.tarr[inds,:]*fracd
        t=np.empty(np.size(inds))
        for i in np.arange(np.size(inds)):
            fracp=(p-temp1[inds[i]-1])/(temp1[inds[i]]-temp1[inds[i]-1])
            t[i]=temp2[inds[i]-1]*(1-fracp)+temp2[inds[i]]*fracp

        #c
        #temp2=self.carr[inds-1,:]*(1-fracd)+self.carr[inds,:]*fracd
        #c=np.empty(np.size(inds))
        #for i in np.arange(np.size(inds)):
        #    fracp=(p-temp1[inds[i]-1])/(temp1[inds[i]]-temp1[inds[i]-1])
        #    c[i]=temp2[inds[i]-1]*(1-fracp)+temp2[inds[i]]*fracp


        flag=0
        return s,t,e,flag

    ###############################################################
    #find the properties corresponding to a given p-T point
    #uses the other interpolation regimes and does not interpolate itself
    #flag is 0 for successful, -1 for out of range
    def find_PT(self, p,T):
        flag=0

        #find entropy
        #temp=minimize(self.zero_S_p, 5.0E3, args=(p, T),method='Nelder-Mead')
        #s=temp.x
        temp=fsolve(self.zero_S_p, 5.0E3, args=(p, T))#,method='Nelder-Mead')
        s=temp

        #find the other parameters
        temp=self.find_PS(p, s)

        return s, temp[0], temp[2]

    #needed for optimization
    def zero_S_p(self, s, p, T):
        temp=self.find_PS(p, s)
        return np.absolute((temp[1]-T))

    ###############################################################
    #find the properties corresponding to a given rho-E point
    #uses the other interpolation regimes and does not interpolate itself
    #flag is 0 for successful, -1 for out of range
    def calcpT_rho_E(self, d,e):
        flag=0

        #find entropy
        temp=fsolve(self.zero_S_d, 5.0E3, args=(d, e))#,method='Nelder-Mead')
        s=temp

        #find the other parameters
        temp=self.interp_all_log(d, s)
        p=temp[0]
        t=temp[2]
        flag=temp[4]

        return p,t,flag

    #needed for optimization
    def zero_S_d(self, s, d, e):
        temp=self.interp_all_log(d, s)
        return np.absolute((temp[1]-e))


    ###########################################################
    #function to find the pressure, entropy and temperature on the vapour dome using the ANEOS vapour dome
    #Originally developed for the ANEOS_Dunite_VapDome
    #Returns: The pressure on the dome at required entropy, flag (0 if in dome, 1 if above dome on vapour side, 2 if above dome on liquid side), 
    #S_liq, rho_liq, T_liq, S_vap, rho_vap, T_vap
    #note that T_liq and T_vap are the same by definition
    #Note that this is an inherently imprecise process due to the fact that rho is non unique in conversion to p 
    def calc_Dome(self, S, press):
        #first check to see if the entropy point is physical (i.e. not zeroed by the interpolation)
        if S>0:
            #check to see whether the entropy point is on the vapour or liquid side of the dome
            ind_vap=np.argmax(self.Dome[:,8]>=S)
            if ind_vap==0:

                #The entropy puts it below the critical point so we use the liq arm
                ind_liq=np.argmax(self.Dome[:,7]<=S)
                #at the critical point
                if ind_liq==0:
                    p_dome=self.Dome[0,3]
                    T_dome=self.Dome[0,0]
                #linearly interpolate to get the pressure on the dome for this entropy
                else:
                    p_dome=self.Dome[ind_liq,3]+(S-self.Dome[ind_liq,7])*\
                        (self.Dome[ind_liq-1,3]-self.Dome[ind_liq,3])/\
                        (self.Dome[ind_liq-1,7]-self.Dome[ind_liq,7])
                    T_dome=self.Dome[ind_liq,0]+(S-self.Dome[ind_liq,7])*\
                        (self.Dome[ind_liq-1,0]-self.Dome[ind_liq,0])/\
                        (self.Dome[ind_liq-1,7]-self.Dome[ind_liq,7])

            #else we are on the vapour side so again linearly interpolate
            else:
                 p_dome=self.Dome[ind_vap-1,4]+(S-self.Dome[ind_vap-1,8])*\
                     (self.Dome[ind_vap-1,4]-self.Dome[ind_vap,4])/\
                     (self.Dome[ind_vap-1,8]-self.Dome[ind_vap,8])
                 T_dome=self.Dome[ind_vap-1,0]+(S-self.Dome[ind_vap-1,8])*\
                     (self.Dome[ind_vap-1,0]-self.Dome[ind_vap,0])/\
                     (self.Dome[ind_vap-1,8]-self.Dome[ind_vap,8])


            #Now we need to test if the material is actually below the dome
            #The material is not in the dome
            if press>p_dome:
                if S>self.Dome[0,7]:
                    #The pressure is above the dome on the vapour side
                    dome_flag=1 
                    S_liq=0.0
                    rho_liq=0.0
                    T_liq=0.0
                    S_vap=0.0
                    rho_vap=0.0
                    #record the temperature that would be on the dome at that entropy
                    T_vap=T_dome
                else:
                    #The pressure is above the dome but on liquid side
                    dome_flag=2
                    S_liq=0.0
                    rho_liq=0.0
                    T_liq=T_dome
                    S_vap=0.0
                    rho_vap=0.0
                    T_vap=0.0
            else:
                dome_flag=0 #the material is in the dome
                #find the average properties either side of the dome based on pressure
                ind_liq=np.argmax(self.Dome[:,3]<=press)
                ind_vap=np.argmax(self.Dome[:,4]<=press)

                #linearly interpolate to find the entropy and density on the liquid and vapour branches
                S_liq=self.Dome[ind_liq,7]+(press-self.Dome[ind_liq,3])*\
                    (self.Dome[ind_liq-1,7]-self.Dome[ind_liq,7])/\
                    (self.Dome[ind_liq-1,3]-self.Dome[ind_liq,3])
                rho_liq=self.Dome[ind_liq,1]+(press-self.Dome[ind_liq,3])*\
                    (self.Dome[ind_liq-1,1]-self.Dome[ind_liq,1])/\
                    (self.Dome[ind_liq-1,3]-self.Dome[ind_liq,3])

                S_vap=self.Dome[ind_vap,8]+(press-self.Dome[ind_vap,4])*\
                    (self.Dome[ind_vap-1,8]-self.Dome[ind_vap,8])/\
                    (self.Dome[ind_vap-1,4]-self.Dome[ind_vap,4])
                rho_vap=self.Dome[ind_vap,2]+(press-self.Dome[ind_vap,4])*\
                    (self.Dome[ind_vap-1,2]-self.Dome[ind_vap,2])/\
                    (self.Dome[ind_vap-1,4]-self.Dome[ind_vap,4])

                #also find the temperature
                T_liq=self.Dome[ind_liq,0]+(press-self.Dome[ind_liq,3])*\
                    (self.Dome[ind_liq-1,0]-self.Dome[ind_liq,0])/\
                    (self.Dome[ind_liq-1,3]-self.Dome[ind_liq,3])

                T_vap=T_liq

        else:
            dome_flag=-1  #for unphysical points
            p_dome=0.0
            S_liq=0.0
            rho_liq=0.0
            T_liq=0.0
            S_vap=0.0
            rho_vap=0.0
            T_vap=0.0


        return (p_dome, dome_flag, S_liq, rho_liq, T_liq, S_vap, rho_vap, T_vap)


    ###########################################################
    #function to find the pressure, entropy and temperature on the vapour dome using the ANEOS vapour dome
    #Originally developed for the ANEOS_Dunite_VapDome
    #Returns: The pressure on the dome at required entropy, flag (0 if in dome, 1 if above dome on vapour side, 2 if above dome on liquid side), 
    #S_liq, rho_liq, T_liq, S_vap, rho_vap, T_vap
    #note that T_liq and T_vap are the same by definition
    #Note that this is an inherently imprecise process due to the fact that rho is non unique in conversion to p 
    #SJL 9_3_16: Inside the dome we do not give values for parameters. Just return dome values.
    def calc_Dome_rho_log(self, S, rho):
        rho_log=2E3

        #first check to see if the entropy point is physical (i.e. not zeroed by the interpolation)
        if (S>0):
            #check to see whether the entropy point is on the vapour or liquid side of the dome
            ind_vap=np.argmax(self.Dome[:,8]>=S)
            if (ind_vap==0):
                #The entropy puts it below the critical point so we use the liq arm
                ind_liq=np.argmax(self.Dome[:,7]<=S)
                #at the critical point
                if (ind_liq==0):
                    rho_dome=self.Dome[0,2]
                #interpolate to get the pressure on the dome for this entropy
                else:
                    #if at low density then log interp
                    if (self.Dome[ind_liq,1]<rho_log):
                        rho_dome=10**(np.log10(self.Dome[ind_liq,1])+(np.log10(S)-np.log10(self.Dome[ind_liq,7]))*\
                                      (np.log10(self.Dome[ind_liq-1,1])-np.log10(self.Dome[ind_liq,1]))/\
                                      (np.log10(self.Dome[ind_liq-1,7])-np.log10(self.Dome[ind_liq,7])))
                        T_dome=10**(np.log10(self.Dome[ind_liq,0])+(np.log10(S)-np.log10(self.Dome[ind_liq,7]))*\
                                    (np.log10(self.Dome[ind_liq-1,0])-np.log10(self.Dome[ind_liq,0]))/\
                                    (np.log10(self.Dome[ind_liq-1,7])-np.log10(self.Dome[ind_liq,7])))
                        p_dome=10**(np.log10(self.Dome[ind_liq,3])+(np.log10(S)-np.log10(self.Dome[ind_liq,7]))*\
                                      (np.log10(self.Dome[ind_liq-1,3])-np.log10(self.Dome[ind_liq,3]))/\
                                      (np.log10(self.Dome[ind_liq-1,7])-np.log10(self.Dome[ind_liq,7])))
                        #approximation of dome entopy on other side
                        S_opposite=10**(np.log10(self.Dome[ind_liq,8])+(np.log10(S)-np.log10(self.Dome[ind_liq,7]))*\
                                        (np.log10(self.Dome[ind_liq-1,8])-np.log10(self.Dome[ind_liq,8]))/\
                                        (np.log10(self.Dome[ind_liq-1,7])-np.log10(self.Dome[ind_liq,7])))
                        
                    else:
                        rho_dome=self.Dome[ind_liq,1]+(S-self.Dome[ind_liq,7])*\
                                  (self.Dome[ind_liq-1,1]-self.Dome[ind_liq,1])/\
                                  (self.Dome[ind_liq-1,7]-self.Dome[ind_liq,7])
                        T_dome=self.Dome[ind_liq,0]+(S-self.Dome[ind_liq,7])*\
                                (self.Dome[ind_liq-1,0]-self.Dome[ind_liq,0])/\
                                (self.Dome[ind_liq-1,7]-self.Dome[ind_liq,7])
                        p_dome=self.Dome[ind_liq,3]+(S-self.Dome[ind_liq,7])*\
                                  (self.Dome[ind_liq-1,3]-self.Dome[ind_liq,3])/\
                                  (self.Dome[ind_liq-1,7]-self.Dome[ind_liq,7])
                        #approximation of dome entopy on other side
                        S_opposite=self.Dome[ind_liq,8]+(S-self.Dome[ind_liq,7])*\
                                    (self.Dome[ind_liq-1,8]-self.Dome[ind_liq,8])/\
                                    (self.Dome[ind_liq-1,7]-self.Dome[ind_liq,7])
                        
            #else we are on the vapour side so again interpolate
            else:
                #if at low density then log interp
                if (self.Dome[ind_vap-1, 2]<rho_log):
                    rho_dome=10**(np.log10(self.Dome[ind_vap-1,2])+(np.log10(S)-np.log10(self.Dome[ind_vap-1,8]))*\
                                  (np.log10(self.Dome[ind_vap-1,2])-np.log10(self.Dome[ind_vap,2]))/\
                                  (np.log10(self.Dome[ind_vap-1,8])-np.log10(self.Dome[ind_vap,8])))
                    T_dome=10**(np.log10(self.Dome[ind_vap-1,0])+(np.log10(S)-np.log10(self.Dome[ind_vap-1,8]))*\
                                (np.log10(self.Dome[ind_vap-1,0])-np.log10(self.Dome[ind_vap,0]))/\
                                (np.log10(self.Dome[ind_vap-1,8])-np.log10(self.Dome[ind_vap,8])))
                    p_dome=10**(np.log10(self.Dome[ind_vap-1,4])+(np.log10(S)-np.log10(self.Dome[ind_vap-1,8]))*\
                                  (np.log10(self.Dome[ind_vap-1,4])-np.log10(self.Dome[ind_vap,4]))/\
                                  (np.log10(self.Dome[ind_vap-1,8])-np.log10(self.Dome[ind_vap,8])))
                    #approximation of dome entopy on other side
                    S_opposite=10**(np.log10(self.Dome[ind_vap-1,7])+(np.log10(S)-np.log10(self.Dome[ind_vap-1,8]))*\
                                     (np.log10(self.Dome[ind_vap-1,7])-np.log10(self.Dome[ind_vap,7]))/\
                                     (np.log10(self.Dome[ind_vap-1,8])-np.log10(self.Dome[ind_vap,8])))
                    #else use linear interp
                else:
                    rho_dome=self.Dome[ind_vap-1,2]+(S-self.Dome[ind_vap-1,8])*\
                              (self.Dome[ind_vap-1,2]-self.Dome[ind_vap,2])/\
                              (self.Dome[ind_vap-1,8]-self.Dome[ind_vap,8])
                    T_dome=self.Dome[ind_vap-1,0]+(S-self.Dome[ind_vap-1,8])*\
                            (self.Dome[ind_vap-1,0]-self.Dome[ind_vap,0])/\
                            (self.Dome[ind_vap-1,8]-self.Dome[ind_vap,8])
                    p_dome=self.Dome[ind_vap-1,4]+(S-self.Dome[ind_vap-1,8])*\
                              (self.Dome[ind_vap-1,4]-self.Dome[ind_vap,4])/\
                              (self.Dome[ind_vap-1,8]-self.Dome[ind_vap,8])
                    #approximation of dome entopy on other side
                    S_opposite=self.Dome[ind_vap-1,7]+(S-self.Dome[ind_vap-1,8])*\
                              (self.Dome[ind_vap-1,7]-self.Dome[ind_vap,7])/\
                              (self.Dome[ind_vap-1,8]-self.Dome[ind_vap,8])


            #The space for rho-S is degenerate. If you are on the vapour side of the dome and the density requested is lower than the vapour dome density then that cannot be satisfied for this material as a mixed phase. On the liquid side it may be possible. At the moment we do not need to overcome these problems and stop at the density of the dome returning the fact that the material is in the dome and leaving it at that.
            #The material is not in the dome
            if rho>rho_dome:
                if S>self.Dome[0,7]:
                    #The pressure is above the dome on the vapour side
                    dome_flag=1 
                    S_liq=0.0
                    rho_liq=0.0
                    T_liq=0.0
                    p_liq=0.0
                    S_vap=0.0
                    rho_vap=0.0
                    p_vap=0.0
                    #record the temperature that would be on the dome at that entropy
                    T_vap=T_dome
                else:
                    #The pressure is above the dome but on liquid side
                    dome_flag=2
                    S_liq=0.0
                    rho_liq=0.0
                    T_liq=T_dome
                    p_liq=0.0
                    S_vap=0.0
                    rho_vap=0.0
                    T_vap=0.0
                    p_vap=0.0
                    
            #material MAY be in the dome. Just record the dome pressure and entropy
            else:
                dome_flag=0
                S_liq=S_opposite
                rho_liq=0
                T_liq=T_dome
                p_liq=p_dome
                S_vap=S_opposite
                rho_vap=0
                p_vap=p_dome
                T_vap=T_dome
                

        else:
            dome_flag=-1  #for unphysical points
            p_dome=0.0
            S_liq=0.0
            rho_liq=0.0
            T_liq=0.0
            S_vap=0.0
            rho_vap=0.0
            T_vap=0.0


        return (rho_dome, dome_flag, S_liq, rho_liq, T_liq, p_liq, S_vap, rho_vap, T_vap, p_vap)
