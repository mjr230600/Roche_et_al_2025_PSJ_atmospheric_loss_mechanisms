# SJL 10/2020
# Function to calculate the loss due to a given ground velocity, surface conditions and planetary mass

####################################################
# Dependent function
import numpy as np
#from scipy import special
from scipy.special import lambertw
vesc_Earth = 11.2E3

####################################################
# No ocean loss function:
def loss_func_NA_forced(params, vesc, ug):

    params = [0.0, params[0], -1.0, params[1]]
    
    params[0] = 1.0/(((1 + np.exp(params[1] + params[2]))**(params[3])) - ((1 + np.exp(params[2]))**(params[3])))
    alpha5 = -params[0] * ((1 + np.exp(params[2]))**(params[3]))
    calc_loss = params[0] * (1 + np.exp(params[1] * (ug / vesc)+params[2]))**(params[3]) + alpha5
    
    return calc_loss

####################################################
# Function for velocity dependence of alpha 2:
def alpha2(params, vesc, ug):

    params_temp = np.empty(4)
    for mm in np.arange(4):
        params_temp[mm] = params[2*mm] + params[2*mm + 1] * vesc/vesc_Earth

    alpha2 = params_temp[0] + params_temp[1] * (ug/vesc) + params_temp[2] * (ug/vesc)**2 + params_temp[3] * (ug/vesc)**3
    
    return alpha2

####################################################
# Function for velocity dependence of alpha 3:
def alpha3(params, vesc, ug):

    alpha3 = params[0] + params[1] * (ug/vesc) + params[2] * (ug/vesc)**2 + params[3] * (ug/vesc)**3
        
    return alpha3

####################################################
# Function for velocity and planetary mass dependence of alpha 4:

def alpha4(params, vesc, ug):

    params_temp = np.empty(4)
    for mm in np.arange(4):
        params_temp[mm] = params[2*mm] + params[2*mm + 1] * vesc/vesc_Earth

    alpha4 = params_temp[0] * np.exp(np.sin(ug/vesc * np.pi) * (params_temp[1] * ug/vesc + params_temp[2] * (ug/vesc)**2 + params_temp[3] * (ug/vesc)**3))

    return alpha4

####################################################
# Function for velocity and planetary mass dependence of alpha 5:
# More generally can be used to describe a loss curve
def alpha5(params, vesc, ug):

    # Assign mass despendence:
    params_temp = np.empty(5)
    for mm in np.arange(5):
        params_temp[mm] = params[2*mm] + params[2*mm + 1] * vesc/vesc_Earth

    params_low = params_temp[0:3]
    params_high = params_temp[3:]
    
    # Find the point at which total loss is reached in the lower regime:
    temp1 = lambertw((1.0/params_low[0])**(1.0/params_low[2]) * params_low[1]/params_low[2], 0) * params_low[2]/params_low[1]
    temp2 = lambertw((1.0/params_low[0])**(1.0/params_low[2]) * params_low[1]/params_low[2], -1) * params_low[2]/params_low[1]
    
    vtotal_loss = np.amin([np.real(temp1), np.real(temp2)]) * vesc
    
    # Find the two limits of loss:
    loss_low = params_low[0] * np.exp(params_low[1] * ug/vesc) * ((ug/vesc)**params_low[2])
    loss_high = 2 + params_high[0] * ((ug - vtotal_loss) / (vesc - vtotal_loss) - 1) +\
                params_high[1] * (((ug - vtotal_loss) / (vesc - vtotal_loss))**2 - 1)
 
    # Now deed to decide when to use each bit of loss:
    loss = loss_high

    # If we are below total loss use the lower function:
    ind1 = np.where(ug <= vtotal_loss)

    if np.size(ind1) > 0:
        if np.size(loss) == 1:
            loss = loss_low
        else:
            loss[ind1] = loss_low[ind1]

    # If we are above the total loss but the higher limit is not yet above one then it is 1:
    ind2 = np.where((ug > vtotal_loss) & (loss < 1))

    if np.size(ind2) > 0:
        if np.size(loss) == 1:
            loss = 1
        else:
            loss[ind2] = 1
   
    # Limiting cases in terms of velocity:
    temp = np.where((ug/vesc > 1))

    if np.size(temp) > 0:
        if np.size(loss) == 1:
            loss = 2
        else:
            loss[temp] = 2

    temp = np.where((ug/vesc < 0))

    if np.size(temp) > 0:
        if np.size(loss) == 1:
            loss = 0
        else:
            loss[temp] = 0
        
    # Make sure we still have no spurious points:
    temp = np.where((loss > 2))

    if np.size(temp) > 0:
        if np.size(loss) == 1:
            loss = 2
        else:
            loss[temp] = 2

    temp = np.where((loss < 0))

    if np.size(temp) > 0:
        if np.size(loss) == 1:
            loss = 0
        else:
            loss[temp] = 0

    return loss

####################################################
# Function for calculating loss with an ocean
def loss_func_R(params, vesc, ug, R, NA_params = 0):

    Nparam_alpha = [8, 4, 8, 10]
    Nparam_alpha_cum = np.cumsum(Nparam_alpha)
    
    alpha = np.zeros(5)
    
    alpha[1] = alpha2(params[0:Nparam_alpha_cum[0]], vesc,ug)
    alpha[2] = alpha3(params[Nparam_alpha_cum[0]:Nparam_alpha_cum[1]], vesc,ug)
    alpha[3] = alpha4(params[Nparam_alpha_cum[1]:Nparam_alpha_cum[2]], vesc,ug)
    alpha[4] = alpha5(params[Nparam_alpha_cum[2]:], vesc,ug)

    #if alpha[1]>0:
    #    alpha[1]=0.0

    # Unless given other parameters use the fit from Lock & Stewart:
    if np.size(NA_params) ==1 :
        NA_params = [-4.321342002881395139e+00,	-3.774520283718452163e+01]

    loss = loss_func_NA_forced(NA_params, vesc,ug)
    alpha[0]=loss-alpha[4]
    
    calc_loss = alpha[0] * (1 + np.exp(alpha[1] * np.log10(R) + alpha[2]))**(alpha[3]) + alpha[4]
    
    if np.size(calc_loss) == 1:
        if calc_loss < loss:
            calc_loss = loss
        elif calc_loss > 2:
            calc_loss = 2.0
            
        if ug/vesc <= 0:
            calc_loss = 0.
        elif ug/vesc >= 1:
            calc_loss = 2.
            
    else:
        temp = np.where((calc_loss >= 2))
        if (np.size(temp) > 0):
            calc_loss[temp] = 2.

        temp = np.where((calc_loss <= loss))
        if (np.size(temp) > 0):
            calc_loss[temp] = loss
            
        temp = np.where((ug/vesc <= 0))
        if (np.size(temp) > 0):
            calc_loss[temp] = 0.

        temp = np.where((ug/vesc >= 1))
        if (np.size(temp) > 0):
            calc_loss[temp] = 2.

    return calc_loss

####################################################
# Function for individual Mp and ug:
def calculate_loss(ug, vesc = vesc_Earth, R = 0, params = 0, NA_params = 0):

    # If no ocean use:
    if R == 0:

        # Unless given other parameters use the fit from Lock & Stewart:
        if np.size(NA_params) == 1:
            NA_params = [-4.321342002881395139e+00,	-3.774520283718452163e+01]

        # Check whether velocity are within limits:
        if ug <= 0.0:
            return 0, 0
        elif ug >= vesc:
            return 1, 0

        calc_loss = loss_func_NA_forced(NA_params, vesc, ug)

        if calc_loss < 0:
            calc_loss = 0.0
        elif calc_loss > 1:
            calc_loss = 1.0

        return calc_loss, 0

    else:

        # Unless given other parameters use the fit from Lock & Stewart:
        if np.size(params) == 1:
            #print('Use default params')
            params = [-1.62584384e+01,  1.42828078e+01,  7.15029708e+01, -8.64465281e+01,
       -1.17862430e+02,  1.64871407e+02,  6.01395645e+01, -9.56142191e+01,
       -7.66878923e+00,  4.39524753e+01, -1.05616622e+02,  7.74557563e+01,
       -7.67532594e-01, -2.63620386e-01, -1.20443695e+01, -1.15024085e+01,
        4.47091911e+01,  3.98696345e+01, -3.88346991e+01, -2.87008315e+01,
        1.06601668e+05,  1.38979517e+04, -1.29490638e+01,  6.40100268e-01,
        7.35306571e+00, -3.84763832e-01,  2.12776033e+00, -3.24256357e-01,
       -1.14723912e+00,  4.18100341e-01]

        # Check whether velocity are within limits:
        if ug <= 0.0:
            return 0, 0
        elif ug >= vesc:
            return 1, 1

        calc_loss = loss_func_R(params, vesc, ug, R, NA_params = NA_params)
        
        if calc_loss < 0:
            calc_loss = 0.0
        elif calc_loss > 2:
            calc_loss = 2.0

        if calc_loss < 1:
            atmo_loss = calc_loss
            oc_loss = 0.0
        else:
            atmo_loss = 1.0
            oc_loss = calc_loss - 1.0

        return atmo_loss, oc_loss