#SJL
#Structures for a sesame EOS with potential vapor dome
#SJL: 7/20: Added analytical vapor dome option and coded NIST water EOS dome
#SJL: 7/20: Altered old table input as it did not seem to be giving correct values
import numpy as np
from scipy import constants as const
import scipy as sp
import os
import sys

from scipy import interpolate


###########################################################
###########################################################
###########################################################
#define some general functions
#Function to calculate the numerical gradient of unevenly spaced points to 2nd order
def gradient2(x,y):
    #define the difference between all the points
    dx=np.diff(x)
    #define the output array
    dydx=np.empty(np.size(y))
    #use centered difference for the middle points
    dydx[1:-1]=(y[2:]*(dx[0:-1]**2.0)-y[0:-2]*(dx[1:]**2.0)-\
                              y[1:-1]*((dx[0:-1]**2.0)-(dx[1:]**2.0)))/\
                             (dx[0:-1]*dx[1:]*(dx[1:]+dx[0:-1]))
    #use forward diff for first point
    dydx[0]=(y[2]*(dx[0]**2)-y[1]*((dx[1]+dx[0])**2)-y[0]*((dx[0]**2)-((dx[1]+dx[0])**2)))/\
        (dx[0]*(dx[0]+dx[1])*-dx[1])
    #use backwards difference for the last point
    dydx[-1]=(y[-3]*(dx[-1]**2)-y[-2]*((dx[-1]+dx[-2])**2)-y[-1]*((dx[-1]**2)-((dx[-1]+dx[-2])**2)))/\
        (dx[-1]*(dx[-1]+dx[-2])*dx[-2])
    return dydx


###########################################################
###########################################################
###########################################################
#define a class for analytical dome classes
#added SJL 7/20
class Analytical_dome(object):
    def __init__(self):
        self.Tc=0.0
        self.pc=0.0
        self.rhoc=0.0

        self.Tdome=np.empty(0)
        self.rho_l=np.empty(0)
        self.rho_v=np.empty(0)
        self.pdome=np.empty(0)


    ###########################################################
    #function to calculate the points on the dome for a given Tdome
    #A function to calculate the rho, p points on the vapour dome for a given set of T using the IAPWS 95 steam EOS (Wagner 2002)
    def NIST_water_EOS_calc_domeT(self, Tdome=0, allocate_values=1):

        #define the constants from the Wagner paper
        self.Tc=647.096 #temperature at critical point
        self.pc=22.064E6 #pressure at critical point
        self.rhoc=322 #density at critical point

        #if passed T values then assign
        if np.size(Tdome)==1:
            if Tdome==0:
                Tdome=self.Tdome

        temp=np.where(Tdome>=self.Tc)
        if np.size(temp)>0:
            Tdome[temp[0][0]]=np.nan

        #parameters for pressure equation
        a=[-7.85951783,
            1.84408259,
            -11.7866497,
            22.6807411,
            -15.9618719,
            1.80122502]

        #parameters for liquid density
        b=[1.99274064,
            1.09965342,
            -0.510839303,
            -1.75493479,
            -45.5170352,
            -6.74694450E5]

        #parameters for vapor density
        c=[-2.03150240,
            -2.68302940,
            -5.38626492,
            -17.2991605,
            -44.7586581,
            -63.9201063]

        ###################################
        #calculate the parameters
        theta=(1-Tdome/self.Tc)

        pdome=self.pc*np.exp((self.Tc/Tdome)*(a[0]*theta + a[1]*theta**(1.5) +\
            a[2]*theta**3 + a[3]*theta**(3.5) + a[4]*theta**4 +\
            a[5]*theta**(7.5)))

        rho_l=self.rhoc*(1.0+b[0]*theta**(1.0/3.0) + b[1]*theta**(2.0/3.0) +\
            b[2]*theta**(5.0/3) + b[3]*theta**(16.0/3) + b[4]*theta**(43.0/3) +\
            b[5]*theta**(110.0/3))

        rho_v=self.rhoc*np.exp(c[0]*theta**(2.0/6) + c[1]*theta**(4.0/6) +\
            c[2]*theta**(8.0/6) + c[3]*theta**(18.0/6) + c[4]*theta**(37.0/6) +\
            c[5]*theta**(71.0/6))

        if allocate_values==1:
            self.Tdome=Tdome
            self.pdome=pdome
            self.rho_l=rho_l
            self.rho_v=rho_v

        else:
            return Tdome, pdome, rho_l, rho_v

    ###########################################################
    #function to check whether an array of given p or T and rho point is in the dome
    def NIST_water_EOS_dome_query(self, var, rho, interp_var='T'):
        
        flag_dome=np.zeros(np.size(var),dtype=int)

        if interp_var=='T':
            (Tdome, pdome, rho_l, rho_v)=self.NIST_water_EOS_calc_domeT(Tdome=var,allocate_values=0)

            #now find where it is in the dome
            temp=np.where((rho<=rho_l)&(rho>=rho_v))
            if np.size(temp)>0:
                flag_dome[temp[0]]=1
            
        
        if interp_var=='p':
            #need to call calc_dome to make sure critical values are programed
            temp=self.NIST_water_EOS_calc_domeT(Tdome=1.0,allocate_values=0)

            #find the p on the dome and set up an interpolation hull
            Tdome=np.linspace(0.1,self.Tc,1000)
            (Tdome, pdome, rho_l, rho_v)=self.NIST_water_EOS_calc_domeT(Tdome=Tdome,allocate_values=0)
            interp = interpolate.interp1d(pdome, Tdome, kind = "linear")

            #initialize arrays
            Tdome=np.ones(np.size(var))*np.nan
            pdome=np.ones(np.size(var))*np.nan
            rho_l=np.ones(np.size(var))*np.nan
            rho_v=np.ones(np.size(var))*np.nan

            #only do interpolation for cases at pressures below pc
            ind=np.where(var<=self.pc)[0]
            if np.size(ind)>0:
                Tdome[ind]=interp(var[ind])
                (Tdome[ind], pdome[ind], rho_l[ind], rho_v[ind])=self.NIST_water_EOS_calc_domeT(Tdome=Tdome[ind],allocate_values=0)

                #now find where it is in the dome
                temp=np.where((rho[ind]<=rho_l[ind])&(rho[ind]>=rho_v[ind]))
                if np.size(temp)>0:
                    flag_dome[ind[temp[0]]]=1

            
        
        return flag_dome, Tdome,pdome, rho_l, rho_v
        


###########################################################
###########################################################
###########################################################
#define a class object that is a sesame EOS referenced in density, temperature space
class GSesame_EOS(object):
    def __init__(self):
        self.filename=''  #file name containing the EOS
        self.darr=np.empty(0)    #1D array of densities
        self.tarr=np.empty(0)    #1D array of temperatures
        self.dijarr=np.empty(0)  #2D array of densities
        self.tijarr=np.empty(0)  #2D array of temperatures
        self.parr=np.empty(0)    #2D array of pressures
        self.earr=np.empty(0)    #2D array of internal energies
        self.sarr=np.empty(0)    #2D array of entropies
        self.farr=np.empty(0)    #2D array of helmholtz free energies
        self.carr=np.empty(0)    #2D array of entropies
        self.KPAarr=np.empty(0)    #2D array of ANEOS KPA flags
        self.dsize=0    #number of points in density
        self.tsize=0    #number of points in temperature
        self.fmn=0   #mean atomic number
        self.fmw=0   #mean atomic weight
        self.rho0_ref=0   #reference density
        self.k0_ref=0     #reference ??
        self.t0_ref=0     #reference temperature
        self.cvarr=np.empty(0)  #2D array of cv
        self.cparr=np.empty(0)  #2D array fo cp
        self.Dome=np.empty(0)  #array for the dome data if required
        self.Dome_an=Analytical_dome() #function for an analytical vapor dome

        self.unitstxt=None #text to identify units
        self.unit_conversion=1 #1 for convert to SI if unitstxt=None


    ###########################################################
    #Wrapper function to read in the EOS data from a Gadget Sesame EOS file
    #flag_read=0: in the format of Senft & Stewart
    #flag_read=1: Standard sesame table
    #flag_read=2: Extended sesame table
    #flag_read=3: Both a standard and extended sesame table (extended file must be given as EOS_file2)
    def read_file(self, EOS_file, flag_read=0, EOS_file2=None, unitstxt=None, unit_conversion=1):
        if flag_read==0:
            self.read_file_Senft_Stewart(EOS_file)
        elif flag_read==1:
            self.loadstdsesame(EOS_file, unitstxt=unitstxt, unit_conversion=unit_conversion)
        elif flag_read==2:
            self.loadextsesame(EOS_file, unitstxt=unitstxt, unit_conversion=unit_conversion)
        elif flag_read==3:
            if EOS_file2 is None:
                print('Extended SESAME file not given to GSESAME_EOS.read_file')
                print('ABORTING')
            else:
                self.loadstdsesame(EOS_file,unitstxt=unitstxt, unit_conversion=unit_conversion)
                self.loadextsesame(EOS_file2,unitstxt=unitstxt, unit_conversion=unit_conversion)
        else:
            print('Unknown flag_read given to GSESAME_EOS.read_file')
            print('ABORTING')

        return

    
    ##########################################
    #function for reading in SESAME EOS tables modified from Stewart et al. 2019 (Zenodo)
    def loadstdsesame(self, fname, unitstxt=None, unit_conversion=1):
        """Function for loading STD SESAME-STYLE EOS table output from ANEOS"""
        data = ([])
        if unitstxt is None:
            self.units = 'Units: rho g/cm3, T K, P GPa, U MJ/kg, A MJ/kg, S MJ/K/kg, cs cm/s, cv MJ/K/kg, KPA flag. 2D arrays are (NT,ND).'
            if unit_conversion==1:
                unit_conversion=np.asarray([1E3,1,1E9,1E6,1E6,1E6,1E-2,1E6,1E6,1])
        else:
            self.units = unitstxt
            self.unit_conversion=unit_conversion
        sesamefile = open(fname,"r")  
        sesamedata=sesamefile.readlines()
        sesamefile.close()
        nskip = 6 # skip standard header to get to the content of the 301 table
        # num.density, num. temps
        tmp = sesamedata[nskip][0:16]
        dlen = float(tmp)
        tmp = sesamedata[nskip][16:32]
        tlen = float(tmp)
        if (np.mod((dlen*tlen*3.0+dlen+tlen+2.0),5.0) == 0):
            neos = int((dlen*tlen*3.0+dlen+tlen+2.0)/5.0) 
        else:
            neos = int((dlen*tlen*3.0+dlen+tlen+2.0)/5.0) +1
        #print(dlen,tlen,neos,len(sesamedata))
        data = np.zeros((neos,5),dtype=float)
        for j in range(nskip,neos+nskip):
            tmp3 = sesamedata[j]
            tmp4 = list(tmp3.split())
            if len(tmp4) < 5:
                lentmp4 = len(tmp4)
                data[j-nskip,0:lentmp4] = np.asarray(tmp4[0:lentmp4])
            else:
                data[j-nskip,:] = np.asarray(tmp4)
            #print(j,eosarr[j,:])
        #print(data.shape)
        data=np.resize(data,(neos*5))
        #print(data.shape)
        self.dsize  = data[0].astype(int)  # now fill the extEOStable class
        self.tsize  = data[1].astype(int)
        self.darr = data[2:2+self.dsize]*unit_conversion[0]
        self.tarr   = data[2+self.dsize : 2+self.dsize+self.tsize]*unit_conversion[1]
        self.parr   = data[2+self.dsize+self.tsize : 2+self.dsize+self.tsize+self.dsize*self.tsize
                            ].reshape(self.tsize,self.dsize)*unit_conversion[2]
        self.earr   = data[2+self.dsize+self.tsize+self.dsize*self.tsize
                            : 2+self.dsize+self.tsize+2*self.dsize*self.tsize
                            ].reshape(self.tsize,self.dsize)*unit_conversion[3]
        self.farr   = data[2+self.dsize+self.tsize+2*self.dsize*self.tsize
                            : 2+self.dsize+self.tsize+3*self.dsize*self.tsize
                            ].reshape(self.tsize,self.dsize)*unit_conversion[4]
        #self.sarr   = data[2+self.dsize+self.tsize+3*self.dsize*self.tsize
        #                    : 2+self.dsize+self.tsize+4*self.dsize*self.tsize
        #                    ].reshape(self.tsize,self.dsize)*unit_conversion[5]
        #self.carr  = data[2+self.dsize+self.tsize+4*self.dsize*self.tsize
        #                    : 2+self.dsize+self.tsize+5*self.dsize*self.tsize
        #                    ].reshape(self.tsize,self.dsize)*unit_conversion[6]
        #self.cvarr  = data[2+self.dsize+self.tsize+5*self.dsize*self.tsize
        #                    : 2+self.dsize+self.tsize+6*self.dsize*self.tsize
        #                    ].reshape(self.tsize,self.dsize)*unit_conversion[7]
        #self.KPA = data[2+self.dsize+self.tsize+6*self.dsize*self.tsize
        #                    : 2+self.dsize+self.tsize+7*self.dsize*self.tsize
        #                    ].reshape(self.tsize,self.dsize)*unit_conversion[8]


        
        #define arrays for i,j T and d and loop to populate
        self.dijarr=np.empty((self.tsize,self.dsize))
        self.tijarr=np.empty((self.tsize,self.dsize))
        for i in np.arange(self.dsize):
            self.tijarr[:,i]=self.tarr
        for i in np.arange(self.tsize):
            self.dijarr[i,:]=self.darr

        #also initialize other arrays that are not used
        if np.size(self.sarr)<1:
            self.sarr=np.empty((self.tsize,self.dsize))
        if np.size(self.carr)<1:
            self.carr=np.empty((self.tsize,self.dsize))
        if np.size(self.cvarr)<1:
            self.cvarr=np.empty((self.tsize,self.dsize))
        if np.size(self.cparr)<1:
            self.cparr=np.empty((self.tsize,self.dsize))
        if np.size(self.KPAarr)<1:
            self.KPAarr=np.empty((self.tsize,self.dsize))

        #print(self.dsize, self.tsize)
        #print(self.darr[600:620])
        #print(self.tarr[600:620])
        #print(self.dijarr[500,600:620])
        #print(self.tijarr[500,600:620])
        #print(self.parr[500,600:620])
        #print(self.earr[500,600:620])
        #print(self.farr[500,600:620])
        #print(self.sarr[500,600:620])
        #print(self.carr[500,600:620])
        #print(self.cvarr[500,600:620])
        #print(self.KPAarr[500,600:620])
        #print(self.cparr[500,600:620])

        return

    ##########################################
    #function for reading in extended SESAME EOS tables modified from Stewart et al. 2019 (Zenodo)
    def loadextsesame(self, fname, unitstxt=None, unit_conversion=1):
        """Function for loading EXTENDED SESAME-STYLE EOS table output from ANEOS"""
        data = ([])
        if unitstxt is None:
            self.units = 'Units: rho g/cm3, T K, P GPa, U MJ/kg, A MJ/kg, S MJ/K/kg, cs cm/s, cv MJ/K/kg, KPA flag. 2D arrays are (NT,ND).'
            if unit_conversion==1:
                unit_conversion=np.asarray([1E3,1,1E9,1E6,1E6,1E6,1E-2,1E6,1E6,1])
        else:
            self.units = unitstxt
            self.unit_conversion=unit_conversion
            
        sesamefile = open(fname,"r")  
        sesamedata=sesamefile.readlines()
        sesamefile.close()
        nskip = 6 # skip standard header to get to the content of the 301 table
        # num.density, num. temps
        tmp = sesamedata[nskip][0:16]
        dlen = float(tmp)
        tmp = sesamedata[nskip][16:32]
        tlen = float(tmp)
        if (np.mod((dlen*tlen*4.0+dlen+tlen+2.0),5.0) == 0):
            neos = int((dlen*tlen*4.0+dlen+tlen+2.0)/5.0)
        else:
            neos = int((dlen*tlen*4.0+dlen+tlen+2.0)/5.0) +1
        #print(dlen,tlen,neos,len(sesamedata))
        data = np.zeros((neos,5),dtype=float)
        for j in range(nskip,neos+nskip):
            tmp3 = sesamedata[j]
            tmp4 = list(tmp3.split())
            if len(tmp4) < 5:
                lentmp4 = len(tmp4)
                data[j-nskip,0:lentmp4] = np.asarray(tmp4[0:lentmp4])
            else:
                data[j-nskip,:] = np.asarray(tmp4)        
            #print(j,eosarr[j,:])
        #print(data.shape)
        data=np.resize(data,(neos*5))
        #print(data.shape)
        self.dsize  = data[0].astype(int)  # now fill the extEOStable class
        self.tsize  = data[1].astype(int)
        self.darr = data[2:2+self.dsize]*unit_conversion[0]
        self.tarr   = data[2+self.dsize : 2+self.dsize+self.tsize]*unit_conversion[1]
        #self.parr   = data[2+self.dsize+self.tsize : 2+self.dsize+self.tsize+self.dsize*self.tsize
        #                    ].reshape(self.tsize,self.dsize)
        #self.earr   = data[2+self.dsize+self.tsize+self.dsize*self.tsize
        #                    : 2+self.dsize+self.tsize+2*self.dsize*self.tsize
        #                    ].reshape(self.tsize,self.dsize)
        #self.farr   = data[2+self.dsize+self.tsize+2*self.dsize*self.tsize
        #                    : 2+self.dsize+self.tsize+3*self.dsize*self.tsize
        #                    ].reshape(self.tsize,self.dsize)
        self.sarr   = data[2+self.dsize+self.tsize+0*self.dsize*self.tsize
                            : 2+self.dsize+self.tsize+1*self.dsize*self.tsize
                            ].reshape(self.tsize,self.dsize)*unit_conversion[5]
        self.carr  = data[2+self.dsize+self.tsize+1*self.dsize*self.tsize
                            : 2+self.dsize+self.tsize+2*self.dsize*self.tsize
                            ].reshape(self.tsize,self.dsize)*unit_conversion[6]
        self.cvarr  = data[2+self.dsize+self.tsize+2*self.dsize*self.tsize
                            : 2+self.dsize+self.tsize+3*self.dsize*self.tsize
                            ].reshape(self.tsize,self.dsize)*unit_conversion[7]
        self.KPAarr = data[2+self.dsize+self.tsize+3*self.dsize*self.tsize
                            : 2+self.dsize+self.tsize+4*self.dsize*self.tsize
                            ].reshape(self.tsize,self.dsize)*unit_conversion[8]

        
        #define arrays for i,j T and d and loop to populate
        self.dijarr=np.empty((self.tsize,self.dsize))
        self.tijarr=np.empty((self.tsize,self.dsize))
        for i in np.arange(self.dsize):
            self.tijarr[:,i]=self.tarr
        for i in np.arange(self.tsize):
            self.dijarr[i,:]=self.darr

        #heat capacities at constant pressure
        if np.size(self.parr)>1:
            self.cparr=np.empty((self.tsize,self.dsize)) 
            dpdrho_T=np.empty((self.tsize,self.dsize))
            for i in np.arange(self.tsize):
                dpdrho_T[i,:]=gradient2(self.darr,self.parr[i,:])
            dpdT_rho=np.empty((self.tsize,self.dsize))
            for i in np.arange(self.dsize):
                dpdT_rho[:,i]=gradient2(self.tarr,self.parr[:,i])

            for i in np.arange(self.tsize):
                for j in np.arange(self.dsize):
                    self.cparr[i,j]=self.cvarr[i,j]+self.tarr[i]*(dpdT_rho[i,j]**2)/(self.darr[j]**2)/dpdrho_T[i,j]

        #also initialize other arrays that are not used
        if np.size(self.parr)<1:
            self.parr=np.empty((self.tsize,self.dsize))
        if np.size(self.earr)<1:
            self.earr=np.empty((self.tsize,self.dsize))
        if np.size(self.farr)<1:
            self.farr=np.empty((self.tsize,self.dsize))
        if np.size(self.cparr)<1:
            self.cparr=np.empty((self.tsize,self.dsize))

        
        #print(self.dsize, self.tsize)
        #print(self.darr[600:620])
        #print(self.tarr[600:620])
        #print(self.dijarr[500,600:620])
        #print(self.tijarr[500,600:620])
        #print(self.parr[500,600:620])
        #print(self.earr[500,600:620])
        #print(self.farr[500,600:620])
        #print(self.sarr[500,600:620])
        #print(self.carr[500,600:620])
        #print(self.cvarr[500,600:620])
        #print(self.KPAarr[500,600:620])
        #print(self.cparr[500,600:620])
            

        return


    ##########################################
    def read_file_Senft_Stewart(self,EOS_file):
        self.filename=EOS_file

        f=open(EOS_file,'r')

        #loop until all the needed cards have been read
        test=0
        while test==0:
            line=f.readline()

            #index card
            if 'INDEX' in line:
                line=f.readline()
                line1=np.asarray(line.strip().split(),dtype='float')

            #Record card 201
            if ('RECORD' in line)&('TYPE =  201' in line):
                line=f.readline()
                line1=np.asarray(line.strip().split(),dtype='float')
                self.fmn=line1[0]
                self.fmw=line1[1]
                self.rho_ref=line1[2]
                self.k0_ref=line1[3]
                self.t0_ref=line1[4]

            #record card 301
            if ('RECORD' in line)&('TYPE =  301' in line):

                #read in first line and assign values before looping
                line=f.readline()
                line1=np.asarray(line.strip().split(),dtype='float')
                self.dsize=int(line1[0])
                self.tsize=int(line1[1])
        
                #loop over all remaining rows and make an array of values
                allarr=line1[2:]
                line=f.readline()
                i=0
                
                #loop until hit the END line
                while (line.strip()!='END')&(len(line.strip())!=0):
                    i=i+1
                    line1=np.asarray(line.strip().split(),dtype='float')
                    allarr=np.append(allarr,line1)
                    line=f.readline()

                test=1

        Nprop=(np.size(allarr)-self.tsize-self.dsize)/(self.tsize*self.dsize)

        #now define the arrays and assign the values
        self.darr=np.empty(self.dsize) #density read in as g/cm3 but converted to kg/m^3
        self.tarr=np.empty(self.tsize) #temperature in K
        self.parr=np.empty((self.tsize,self.dsize)) #pressure read in as GPa but converted to Pa
        self.earr=np.empty((self.tsize,self.dsize)) #internal energy read in as MJ/kg but converted to J/kg
        self.sarr=np.empty((self.tsize,self.dsize)) #entropy read in as MJ/kg/K but converted to J/kg/K
        self.carr=np.empty((self.tsize,self.dsize)) #sound speed read in a cm/s and recorded as m/s
        

        #define d and t including conversion to SI
        #inext is the counter for the point in the all array
        inext=0
        self.darr=allarr[inext:(inext+self.dsize)]*1.0E3 
        inext+=self.dsize
        self.tarr=allarr[inext:(inext+self.tsize)]
        inext+=self.tsize

        #loop over the pressure to put into array
        for i in np.arange(self.tsize):
            self.parr[i,:]=allarr[inext:(inext+self.dsize)]*1.0E9
            inext+=self.dsize

        #loop over other parameters
        #same for energy
        for i in np.arange(self.tsize):
            self.earr[i,:]=allarr[inext:(inext+self.dsize)]*1.0E6
            inext+=self.dsize

        #same for entropy
        for i in np.arange(self.tsize):
            self.sarr[i,:]=allarr[inext:(inext+self.dsize)]*1.0E6
            inext+=self.dsize

        if Nprop>3:

            #same for sound speed
            for i in np.arange(self.tsize):
                self.carr[i,:]=allarr[inext:(inext+self.dsize)]/100.0
                inext+=self.dsize

        #close the file
        f.close()

        #define arrays for i,j T and d and loop to populate
        self.dijarr=np.empty((self.tsize,self.dsize))
        self.tijarr=np.empty((self.tsize,self.dsize))
        for i in np.arange(self.dsize):
            self.tijarr[:,i]=self.tarr
        for i in np.arange(self.tsize):
            self.dijarr[i,:]=self.darr

        #heat capacities
        self.cvarr=np.empty((self.tsize,self.dsize)) 
        self.cparr=np.empty((self.tsize,self.dsize))
        #find the heat capacity at constant volume
        for i in np.arange(self.dsize):
            self.cvarr[:,i]=self.tarr*gradient2(self.tarr,self.sarr[:,i])

        #more complicated to find the heat capacity at constant pressure
        dpdrho_T=np.empty((self.tsize,self.dsize))
        for i in np.arange(self.tsize):
            dpdrho_T[i,:]=gradient2(self.darr,self.parr[i,:])
        dpdT_rho=np.empty((self.tsize,self.dsize))
        for i in np.arange(self.dsize):
            dpdT_rho[:,i]=gradient2(self.tarr,self.parr[:,i])

        for i in np.arange(self.tsize):
            for j in np.arange(self.dsize):
                self.cparr[i,j]=self.cvarr[i,j]+self.tarr[i]*(dpdT_rho[i,j]**2)/(self.darr[j]**2)/dpdrho_T[i,j]

        #can also calculate the sound speed
        if Nprop<=3:
            for i in np.arange(self.tsize):
                for j in np.arange(self.dsize):
                    self.carr[i,j]=np.sqrt(dpdrho_T[i,j]+self.tarr[i]*(dpdT_rho[i,j]**2)/(self.darr[j]**2)/self.cvarr[i,j])
            self.carr=np.sqrt(self.cparr/self.cvarr*dpdrho_T)
        

        ##find the heat capacity at constant pressure assuming an ideal gas
        #ind_nonzero=np.where((self.dijarr!=0)&(self.tijarr!=0.0))
        #self.cparr[ind_nonzero]=self.cvarr[ind_nonzero]+np.divide(self.parr[ind_nonzero],((self.dijarr[ind_nonzero]*self.tijarr[ind_nonzero])))
        #ind_zero=np.where(np.any((self.dijarr==0,self.tijarr==0)))
        #self.cparr[ind_zero]=0.0
        
        return

    ###########################################################
    #SJL 7/20 replaced with new version as this version seemed to fail. Kept just in case
    #function to read in the EOS data from a Gadget Sesame EOS file
    def read_file_old(self, EOS_file):
        self.filename=EOS_file

        f=open(EOS_file,'r')

        #read in index card
        line=f.readline()
        line=f.readline()
        line1=np.asarray(map(float,line.strip().split()))
        line=f.readline()
        line1=np.asarray(map(float,line.strip().split()))
        
        #read in card 201 and assign values to variables
        line=f.readline()
        line=f.readline()
        line1=np.asarray(map(float,line.strip().split()))
        self.fmn=line1[0]
        self.fmw=line1[1]
        self.rho_ref=line1[2]
        self.k0_ref=line1[3]
        self.t0_ref=line1[4]

        #read in the main table (cards 301-305)
        line=f.readline()
        #read in first line and assign values before looping
        line=f.readline()
        line1=np.asarray(map(float,line.strip().split()))
        self.dsize=int(line1[0])
        self.tsize=int(line1[1])
        
        #loop over all remaining rows and make an array of values
        allarr=line1[2:]
        line=f.readline()
        i=0
        #loop until hit the END line
        while line.strip()!='END':
            i=i+1
            line1=np.asarray(map(float,line.strip().split()))
            allarr=np.append(allarr,line1)
            line=f.readline()

        #now define the arrays and assign the values
        self.darr=np.empty(self.dsize) #density read in as g/cm3 but converted to kg/m^3
        self.tarr=np.empty(self.tsize) #temperature in K
        self.parr=np.empty((self.tsize,self.dsize)) #pressure read in as GPa but converted to Pa
        self.earr=np.empty((self.tsize,self.dsize)) #internal energy read in as MJ/kg but converted to J/kg
        self.sarr=np.empty((self.tsize,self.dsize)) #entropy read in as MJ/kg/K but converted to J/kg/K
        self.carr=np.empty((self.tsize,self.dsize)) #sound speed read in a cm/s and recorded as m/s

        #define d and t including conversion to SI
        #inext is the counter for the point in the all array
        inext=0
        self.darr=allarr[inext:(inext+self.dsize)]*1.0E3 
        inext+=self.dsize
        self.tarr=allarr[inext:(inext+self.tsize)]
        inext+=self.tsize

        #loop over the pressure to put into array
        for i in np.arange(self.tsize):
            self.parr[i,:]=allarr[inext:(inext+self.dsize)]*1.0E9
            inext+=self.dsize
        
        #same for energy
        for i in np.arange(self.tsize):
            self.earr[i,:]=allarr[inext:(inext+self.dsize)]*1.0E6
            inext+=self.dsize

        #same for entropy
        for i in np.arange(self.tsize):
            self.sarr[i,:]=allarr[inext:(inext+self.dsize)]*1.0E6
            inext+=self.dsize

        #same for sound speed
        for i in np.arange(self.tsize):
            self.carr[i,:]=allarr[inext:(inext+self.dsize)]/100.0
            inext+=self.dsize

        #close the file
        f.close()

        #define arrays for i,j T and d and loop to populate
        self.dijarr=np.empty((self.tsize,self.dsize))
        self.tijarr=np.empty((self.tsize,self.dsize))
        for i in np.arange(self.dsize):
            self.tijarr[:,i]=self.tarr
        for i in np.arange(self.tsize):
            self.dijarr[i,:]=self.darr

        #heat capacities
        self.cvarr=np.empty((self.tsize,self.dsize)) 
        self.cparr=np.empty((self.tsize,self.dsize)) 
        #find the heat capacity at constnat volume
        for i in np.arange(self.dsize):
            self.cvarr[:,i]=self.tarr*gradient2(self.tarr,self.sarr[:,i])

        #find the heat capacity at constant pressure assuming an ideal gas
        ind_nonzero=np.where((self.dijarr!=0)&(self.tijarr!=0.0))
        self.cparr[ind_nonzero]=self.cvarr[ind_nonzero]+np.divide(self.parr[ind_nonzero],((self.dijarr[ind_nonzero]*self.tijarr[ind_nonzero])))
        ind_zero=np.where(np.any((self.dijarr==0,self.tijarr==0)))
        self.cparr[ind_zero]=0.0
        
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
    #interpolate the rho-T grid to find the p,e,s points
    #returns the points and a flag: -1 if out of range, 0 is for succesful
    def interp_all(self, d, t):
        #search to find the location of the lower bound on t and p
        #if out of bounds return and exit
        temp=np.where(self.darr>=d)
        if np.size(temp[0])==0:
            flag=-1
            p=np.nan
            e=np.nan
            s=np.nan
            return p,e,s,flag
        else:
            indd=temp[0][0]
        temp=np.where(self.tarr>=t)
        if np.size(temp[0])==0:
            flag=-1
            p=np.nan
            e=np.nan
            s=np.nan
            return p,e,s,flag
        else:
            indt=temp[0][0]

        #now linearly interpolate
        fract=(t-self.tarr[indt-1])/(self.tarr[indt]-self.tarr[indt-1])
        fracd=(d-self.darr[indd-1])/(self.darr[indd]-self.darr[indd-1])

        #for p
        temp1=self.parr[indt-1,indd-1]*(1-fract)+self.parr[indt,indd-1]*fract
        temp2=self.parr[indt-1,indd]*(1-fract)+self.parr[indt,indd]*fract
        p=temp1*(1-fracd)+temp2*fracd

        #for e
        temp1=self.earr[indt-1,indd-1]*(1-fract)+self.earr[indt,indd-1]*fract
        temp2=self.earr[indt-1,indd]*(1-fract)+self.earr[indt,indd]*fract
        e=temp1*(1-fracd)+temp2*fracd

        #for s
        temp1=self.sarr[indt-1,indd-1]*(1-fract)+self.sarr[indt,indd-1]*fract
        temp2=self.sarr[indt-1,indd]*(1-fract)+self.sarr[indt,indd]*fract
        s=temp1*(1-fracd)+temp2*fracd

        #for c
        temp1=self.carr[indt-1,indd-1]*(1-fract)+self.carr[indt,indd-1]*fract
        temp2=self.carr[indt-1,indd]*(1-fract)+self.carr[indt,indd]*fract
        c=temp1*(1-fracd)+temp2*fracd

        #for cv
        temp1=self.cvarr[indt-1,indd-1]*(1-fract)+self.cvarr[indt,indd-1]*fract
        temp2=self.cvarr[indt-1,indd]*(1-fract)+self.cvarr[indt,indd]*fract
        cv=temp1*(1-fracd)+temp2*fracd

        #for cp
        temp1=self.cparr[indt-1,indd-1]*(1-fract)+self.cparr[indt,indd-1]*fract
        temp2=self.cparr[indt-1,indd]*(1-fract)+self.cparr[indt,indd]*fract
        cp=temp1*(1-fracd)+temp2*fracd

        flag=0
        return p,e,s,c,cv,cp,flag


    ###########################################################
    #use linear interpolation to find e,s,d for a p-T point
    #returns the points and a flag: -1 if out of range, 0 is for succesful
    def find_PT(self, p, t):
        #search to find the location of the lower bound on t 
        #if out of bounds return and exit
        temp=np.where(self.tarr>=t)
        if np.size(temp[0])==0:
            flag=-1
            p=np.nan
            e=np.nan
            s=np.nan
            return p,e,s,flag
        else:
            indt=temp[0][0]

        #now define the array of presure linearly interpolated between the t points
        fract=(t-self.tarr[indt-1])/(self.tarr[indt]-self.tarr[indt-1])
        temp1=self.parr[indt-1,:]*(1-fract)+self.parr[indt,:]*fract


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
                    e=np.nan
                    s=np.nan
                    c=np.nan
                    cv=np.nan
                    cp=np.nan
                    return d,e,s,c,cv,cp,flag
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
        temp2=self.earr[indt-1,:]*(1-fract)+self.earr[indt,:]*fract
        e=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indd[i]-1])/(temp1[indd[i]]-temp1[indd[i]-1])
            e[i]=temp2[indd[i]-1]*(1-fracp)+temp2[indd[i]]*fracp

        #s
        temp2=self.sarr[indt-1,:]*(1-fract)+self.sarr[indt,:]*fract
        s=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indd[i]-1])/(temp1[indd[i]]-temp1[indd[i]-1])
            s[i]=temp2[indd[i]-1]*(1-fracp)+temp2[indd[i]]*fracp

        #c
        temp2=self.carr[indt-1,:]*(1-fract)+self.carr[indt,:]*fract
        c=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indd[i]-1])/(temp1[indd[i]]-temp1[indd[i]-1])
            c[i]=temp2[indd[i]-1]*(1-fracp)+temp2[indd[i]]*fracp

        #cv
        temp2=self.cvarr[indt-1,:]*(1-fract)+self.cvarr[indt,:]*fract
        cv=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indd[i]-1])/(temp1[indd[i]]-temp1[indd[i]-1])
            cv[i]=temp2[indd[i]-1]*(1-fracp)+temp2[indd[i]]*fracp

        #c
        temp2=self.cparr[indt-1,:]*(1-fract)+self.cparr[indt,:]*fract
        cp=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indd[i]-1])/(temp1[indd[i]]-temp1[indd[i]-1])
            cp[i]=temp2[indd[i]-1]*(1-fracp)+temp2[indd[i]]*fracp

        flag=0
        return d,e,s,c,cv,cp,flag


    
    ###########################################################
    #use linear interpolation to find p,T etc. for a d-e points
    #returns the points and a flag: -1 if out of range, 0 is for succesful
    def calcpT_rho_E(self, d, e):
        #search to find the location of the lower bound on t 
        #if out of bounds return and exit
        temp=np.where(self.darr>=d)
        if np.size(temp[0])==0:
            flag=-1
            T=np.nan
            p=np.nan
            s=np.nan
            c=np.nan
            cv=np.nan
            cp=np.nan
            return p,T,flag,s
        else:
            indd=temp[0][0]

        #now define the array of energy linearly interpolated between the d points
        fracd=(d-self.darr[indd-1])/(self.darr[indd]-self.darr[indd-1])
        temp1=self.earr[:,indd-1]*(1.0-fracd)+self.earr[:,indd]*fracd

        #find the temperature points that correspond to this energy
        #note we are able to pick up degenerate values
        #initialise the ind array and 
        indt=np.asarray([],dtype=int)
        test_stop=0
        test_ind=0

        while test_stop!=1:
 
            #find the next value based on the current value of e
            if temp1[test_ind]<=e:
                temp=np.where(temp1[test_ind:-1]>=e)
            else:
                temp=np.where(temp1[test_ind:-1]<=e)
            #print(temp)
            
            #then test to see if we are at the end or out of range
            if np.size(temp[0])==0:
                if test_ind==0:
                    flag=-2
                    T=np.nan
                    p=np.nan
                    s=np.nan
                    c=np.nan
                    cv=np.nan
                    cp=np.nan
                    return p,T,flag,s
                else:
                    test_stop=1
            else:
                #append value and move on
                indt=np.append(indt,test_ind+temp[0][0])
                test_ind=test_ind+temp[0][0]+1

        #now can find array of possible temperatures
        t=np.empty(np.size(indt))
        for i in np.arange(np.size(indt)):
            frace=(e-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            t[i]=self.tarr[indt[i]-1]*(1-frace)+self.tarr[indt[i]]*frace
        

        #do the same for the other paratmers
        #pressure
        temp2=self.parr[:,indd-1]*(1-fracd)+self.parr[:,indd]*fracd
        p=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            frace=(e-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            p[i]=temp2[indt[i]-1]*(1-frace)+temp2[indt[i]]*frace

        
        #s
        temp2=self.sarr[:,indd-1]*(1-fracd)+self.sarr[:,indd]*fracd
        s=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            frace=(e-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            s[i]=temp2[indt[i]-1]*(1-frace)+temp2[indt[i]]*frace
            
        
        #c
        temp2=self.carr[:,indd-1]*(1-fracd)+self.carr[:,indd]*fracd
        c=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            frace=(e-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            c[i]=temp2[indt[i]-1]*(1-frace)+temp2[indt[i]]*frace

        #cv
        temp2=self.cvarr[:,indd-1]*(1-fracd)+self.cvarr[:,indd]*fracd
        cv=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            frace=(e-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            cv[i]=temp2[indt[i]-1]*(1-frace)+temp2[indt[i]]*frace

        #cp
        temp2=self.cparr[:,indd-1]*(1-fracd)+self.cparr[:,indd]*fracd
        cp=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            frace=(e-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            cp[i]=temp2[indt[i]-1]*(1-frace)+temp2[indt[i]]*frace
    


        flag=0
        return p,t,flag,s,c,cv,cp

    ###########################################################
    #use linear interpolation to find p,T etc. for given d-s points
    #returns the points and a flag: -1 if out of range, 0 is for succesful
    def find_ds(self, d, s):
        #search to find the location of the lower bound on t 
        #if out of bounds return and exit
        temp=np.where(self.darr>=d)
        if np.size(temp[0])==0:
            flag=-1
            t=np.nan
            p=np.nan
            e=np.nan
            c=np.nan
            cv=np.nan
            cp=np.nan
            return t,p,e,c,cv,cp,flag
        else:
            indd=temp[0][0]

        #now define the array of energy linearly interpolated between the d points
        fracd=(d-self.darr[indd-1])/(self.darr[indd]-self.darr[indd-1])
        temp1=self.sarr[:,indd-1]*(1.0-fracd)+self.sarr[:,indd]*fracd

        #find the temperature points that correspond to this energy
        #note we are able to pick up degenerate values
        #initialise the ind array and 
        indt=np.asarray([],dtype=int)
        test_stop=0
        test_ind=0

        while test_stop!=1:
 
            #find the next value based on the current value of e
            if temp1[test_ind]<=s:
                temp=np.where(temp1[test_ind:-1]>=s)
            else:
                temp=np.where(temp1[test_ind:-1]<=s)
            #print(temp)
            
            #then test to see if we are at the end or out of range
            if np.size(temp[0])==0:
                if test_ind==0:
                    flag=-2
                    t=np.nan
                    p=np.nan
                    e=np.nan
                    c=np.nan
                    cv=np.nan
                    cp=np.nan
                    return t,p,e,c,cv,cp,flag
                else:
                    test_stop=1
            else:
                #append value and move on
                indt=np.append(indt,test_ind+temp[0][0])
                test_ind=test_ind+temp[0][0]+1

        #now can find array of possible temperatures
        t=np.empty(np.size(indt))
        for i in np.arange(np.size(indt)):
            fracs=(s-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            t[i]=self.tarr[indt[i]-1]*(1-fracs)+self.tarr[indt[i]]*fracs

        #do the same for the other paratmers
        #pressure
        temp2=self.parr[:,indd-1]*(1-fracd)+self.parr[:,indd]*fracd
        p=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracs=(s-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            p[i]=temp2[indt[i]-1]*(1-fracs)+temp2[indt[i]]*fracs

        
        #e
        temp2=self.earr[:,indd-1]*(1-fracd)+self.earr[:,indd]*fracd
        e=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracs=(s-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            e[i]=temp2[indt[i]-1]*(1-fracs)+temp2[indt[i]]*fracs

        #c
        temp2=self.carr[:,indd-1]*(1-fracd)+self.carr[:,indd]*fracd
        c=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracs=(s-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            c[i]=temp2[indt[i]-1]*(1-fracs)+temp2[indt[i]]*fracs

        #cv
        temp2=self.cvarr[:,indd-1]*(1-fracd)+self.cvarr[:,indd]*fracd
        cv=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracs=(s-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            cv[i]=temp2[indt[i]-1]*(1-fracs)+temp2[indt[i]]*fracs

        #cp
        temp2=self.cparr[:,indd-1]*(1-fracd)+self.cparr[:,indd]*fracd
        cp=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracs=(s-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            cp[i]=temp2[indt[i]-1]*(1-fracs)+temp2[indt[i]]*fracs

        


        flag=0
        return t,p,e,c,cv,cp,flag


    ###########################################################
    #use linear interpolation to find e,T etc. for given d-p points
    #returns the points and a flag: -1 if out of range, 0 is for succesful
    def find_dp(self, d, p):
        #search to find the location of the lower bound on t 
        #if out of bounds return and exit
        temp=np.where(self.darr>=d)
        if np.size(temp[0])==0:
            flag=-1
            t=np.nan
            s=np.nan
            e=np.nan
            c=np.nan
            cv=np.nan
            cp=np.nan
            return t,s,e,c,cv,cp,flag
        else:
            indd=temp[0][0]

        #now define the array of energy linearly interpolated between the d points
        fracd=(d-self.darr[indd-1])/(self.darr[indd]-self.darr[indd-1])
        temp1=self.parr[:,indd-1]*(1.0-fracd)+self.parr[:,indd]*fracd

        #find the temperature points that correspond to this energy
        #note we are able to pick up degenerate values
        #initialise the ind array and 
        indt=np.asarray([],dtype=int)
        test_stop=0
        test_ind=0

        while test_stop!=1:
 
            #find the next value based on the current value of p
            if temp1[test_ind]<=p:
                temp=np.where(temp1[test_ind:-1]>=p)
            else:
                temp=np.where(temp1[test_ind:-1]<=p)
            #print(temp)
            
            #then test to see if we are at the end or out of range
            if np.size(temp[0])==0:
                if test_ind==0:
                    flag=-2
                    t=np.nan
                    s=np.nan
                    e=np.nan
                    c=np.nan
                    cv=np.nan
                    cp=np.nan
                    return t,s,e,c,cv,cp,flag
                else:
                    test_stop=1
            else:
                #append value and move on
                indt=np.append(indt,test_ind+temp[0][0])
                test_ind=test_ind+temp[0][0]+1

        #now can find array of possible temperatures
        t=np.empty(np.size(indt))
        for i in np.arange(np.size(indt)):
            fracp=(p-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            t[i]=self.tarr[indt[i]-1]*(1-fracp)+self.tarr[indt[i]]*fracp

        #do the same for the other paratmers
        #entropy
        temp2=self.sarr[:,indd-1]*(1-fracd)+self.sarr[:,indd]*fracd
        s=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            s[i]=temp2[indt[i]-1]*(1-fracp)+temp2[indt[i]]*fracp

        
        #e
        temp2=self.earr[:,indd-1]*(1-fracd)+self.earr[:,indd]*fracd
        e=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            e[i]=temp2[indt[i]-1]*(1-fracp)+temp2[indt[i]]*fracp

        #c
        temp2=self.carr[:,indd-1]*(1-fracd)+self.carr[:,indd]*fracd
        c=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            c[i]=temp2[indt[i]-1]*(1-fracp)+temp2[indt[i]]*fracp

        #cv
        temp2=self.cvarr[:,indd-1]*(1-fracd)+self.cvarr[:,indd]*fracd
        cv=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            cv[i]=temp2[indt[i]-1]*(1-fracp)+temp2[indt[i]]*fracp

        #cp
        temp2=self.cparr[:,indd-1]*(1-fracd)+self.cparr[:,indd]*fracd
        cp=np.empty(np.size(indd))
        for i in np.arange(np.size(indd)):
            fracp=(p-temp1[indt[i]-1])/(temp1[indt[i]]-temp1[indt[i]-1])
            cp[i]=temp2[indt[i]-1]*(1-fracp)+temp2[indt[i]]*fracp

        


        flag=0
        return t,s,e,c,cv,cp,flag

    

    #function to find the pressure, entropy and temperature on the vapour dome using the ANEOS vapour dome
    #Dome must be passed as a numpy array
    #Originally developed for the ANEOS_Dunite_VapDome
    #Returns: The pressure on the dome at required entropy, flag (0 if in dome, 1 if above dome on vapour side, 2 if above dome on liquid side), 
    #S_liq, rho_liq, T_liq, S_vap, rho_vap, T_vap
    #note that T_liq and T_vap are the same by definition
    #Note that this is an inherently imprecise process due to the fact that rho is non unique in conversion to p 
    def Dome_query(self, S, press):
        Dome=self.Dome
        #first check to see if the entropy point is physical (i.e. not zeroed by the interpolation)
        if S>0:
            #check to see whether the entropy point is on the vapour or liquid side of the dome
            ind_vap=np.argmax(Dome[:,8]>=S)
            if ind_vap==0:

                #The entropy puts it below the critical point so we use the liq arm
                ind_liq=np.argmax(Dome[:,7]<=S)
                #at the critical point
                if ind_liq==0:
                    p_dome=Dome[0,3]
                    T_dome=Dome[0,0]
                #linearly interpolate to get the pressure on the dome for this entropy
                else:
                    p_dome=Dome[ind_liq,3]+(S-Dome[ind_liq,7])*\
                        (Dome[ind_liq-1,3]-Dome[ind_liq,3])/\
                        (Dome[ind_liq-1,7]-Dome[ind_liq,7])
                    T_dome=Dome[ind_liq,0]+(S-Dome[ind_liq,7])*\
                        (Dome[ind_liq-1,0]-Dome[ind_liq,0])/\
                        (Dome[ind_liq-1,7]-Dome[ind_liq,7])

            #else we are on the vapour side so again linearly interpolate
            else:
                 p_dome=Dome[ind_vap-1,4]+(S-Dome[ind_vap-1,8])*\
                     (Dome[ind_vap-1,4]-Dome[ind_vap,4])/\
                     (Dome[ind_vap-1,8]-Dome[ind_vap,8])
                 T_dome=Dome[ind_vap-1,0]+(S-Dome[ind_vap-1,8])*\
                     (Dome[ind_vap-1,0]-Dome[ind_vap,0])/\
                     (Dome[ind_vap-1,8]-Dome[ind_vap,8])


            #Now we need to test if the material is actually below the dome
            #The material is not in the dome
            if press>p_dome:
                if S>Dome[0,7]:
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
                ind_liq=np.argmax(Dome[:,3]<=press)
                ind_vap=np.argmax(Dome[:,4]<=press)

                #linearly interpolate to find the entropy and density on the liquid and vapour branches
                S_liq=Dome[ind_liq,7]+(press-Dome[ind_liq,3])*\
                    (Dome[ind_liq-1,7]-Dome[ind_liq,7])/\
                    (Dome[ind_liq-1,3]-Dome[ind_liq,3])
                rho_liq=Dome[ind_liq,1]+(press-Dome[ind_liq,3])*\
                    (Dome[ind_liq-1,1]-Dome[ind_liq,1])/\
                    (Dome[ind_liq-1,3]-Dome[ind_liq,3])

                S_vap=Dome[ind_vap,8]+(press-Dome[ind_vap,4])*\
                    (Dome[ind_vap-1,8]-Dome[ind_vap,8])/\
                    (Dome[ind_vap-1,4]-Dome[ind_vap,4])
                rho_vap=Dome[ind_vap,2]+(press-Dome[ind_vap,4])*\
                    (Dome[ind_vap-1,2]-Dome[ind_vap,2])/\
                    (Dome[ind_vap-1,4]-Dome[ind_vap,4])

                #also find the temperature
                T_liq=Dome[ind_liq,0]+(press-Dome[ind_liq,3])*\
                    (Dome[ind_liq-1,0]-Dome[ind_liq,0])/\
                    (Dome[ind_liq-1,3]-Dome[ind_liq,3])

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
