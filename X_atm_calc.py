import numpy as np

def Q_R_prime_calc(R_t = None,
            R_i = None,
            M_t = None,
            M_i = None,
            b = None,
            v_c = None
            ):
    
    """
    Calculate QR'.

    Parameters:
    R_t : float
        Target planet radius (in m).
    R_i : float
        Impactor planet radius (in m).
    M_t : float
        Target planet mass (in kg).
    M_i : float
        Impactor planet mass (in kg).
    b : float
        Impact parameter.
    v_c : float
        Impact velocity (in m/s).
    

    Returns:
    Q_R_prime : float
        Centre of mass modified specific impact energy (in J/kg. Convert to MJ/kg for use in the X_atm_calc function).

        
    --------------------------------------
    M.J.Roche       11/11/24        
    --------------------------------------
    """
    
    B = (R_t + R_i) * b

    condition = B + R_i <= R_t
    l = np.where(condition, 2 * R_i, R_t + R_i - B) # Projected length of the projectile overlapping the target
    alpha = np.where(condition, 1.0, (3 * R_i * l**2 - l**3) / (4 * R_i**3))
    
    M_tot = M_t + M_i # Total system mass
    mu = (M_t * M_i) / M_tot # Reduced mass
    mu_alpha = (alpha * M_t * M_i) / ((alpha * M_i) + M_t) # Reduced mass for the overlapping projectile
    Q_R = (mu * v_c**2) / (2 * M_tot) # Unmodified centre of mass specific impact energy
    Q_R_prime = (mu_alpha * Q_R) / mu

    return Q_R_prime

###############################################################################################################

def X_atm_calc(b,
               gamma,
               vc,
               M_t, 
               M_i,
               Q_R_prime,
               fit_params = None):
    
    """
    Calculate near-, far-, and total atmospheric loss fractions (X_NF, X_FF, X_atm).

    Parameters:
    b : float
        Impact parameter.
    gamma : float
        Impactor–total system mass ratio.
    vc : float
        Impact velocity (in multiples of the mutual escape velocity).
    M_t : float
        Target planet mass (in Earth masses), including atmosphere.
    M_i : float
        Impactor planet mass (in Earth masses).
    Q_R_prime : float
        Centre of mass modified specific impact energy (in MJ kg^-1).
    fit_params : array-like, optional
        User-defined fitting parameters. Uses default values if not provided.

    If passing an array of values, pass as a numpy array.

    Returns:
    X_NF : float
        Near-field atmospheric loss fraction.
    X_FF : float
        Far-field atmospheric loss fraction.
    X_atm : float
        Total atmospheric loss fraction.

    
    --------------------------------------
    M.J.Roche       11/11/24        
    --------------------------------------
    """

    # Fitting parameters:
    if fit_params is None:
        k11 = -642.791857924034	
        k12 = 0.927477898498122
        k13 = -0.241148775689502
        k14 = 642.833442674385
        k15 = 0.000182752531197892
        k16 = -0.00836533821561896	
        k21 = -816.782086261419
        k22 = 0.121617779506277
        k23 = -0.105605535270046
        k24 = 816.784045030932
        k25 = 0.000565572490323047
        k31 = -0.467620825461448
        k32 = 1.24840607083957
        k33 = -0.515623557791712
        k34 = 2.47613037112204
        k35 = -4.63761084282108

        s11 = -8150.04628078906
        s12 = 7894.35070127486
        s13 = 0.00016689298129267
        s14 = 255.871531705373
        s15 = -0.0000468291252580014
        s21 = 1.78886747425527
        s22 = -1.79025757238811
        s23 = 0.000132077646943565
        s24 = 0.00709170453170774
        s25 = -0.484402121687731
        s31 = 928.125714756011
        s32 = -928.416993790275
        s33 = 0.00143945748780139
        s41 = -74.9998290321775
        s42 = -4.2536032406539
        s43 = 1.69066832469392
        s44 = 75.9178537500984
        s45 = 5.62560794607606E-06
    else:
        fit_params_NF = fit_params[0]
        k11, k12, k13, k14, k15, k16, k21, k22, k23, k24, k25, k31, k32, k33, k34, k35 = fit_params_NF.iloc[0].values

        fit_params_FF = fit_params[1]
        s11, s12, s13, s14, s15, s21, s22, s23, s24, s25, s31, s32, s33, s41, s42, s43, s44, s45 = fit_params_FF.iloc[0].values

    ########################################################################

    # Near-field:
    xi1 = k11 + (k12 * gamma) + (k13 * gamma**2) + (k14 * vc**(k15)) + (k16 * M_t)
    xi2 = k21 + (k22 * gamma) + (k23 * gamma**2) + (k24 * vc**(k25))
    xi3 = k31 + (k32 * gamma) + (k33 * gamma**2) + (k34 * vc**(k35))
    
    X_NF = xi1 - xi2 * (b + xi3)**2

    xi1 = k11 + (k12 * gamma) + (k13 * gamma**2) + (k14) + (k16 * M_t)
    xi2 = k21 + (k22 * gamma) + (k23 * gamma**2) + (k24)
    xi3 = k31 + (k32 * gamma) + (k33 * gamma**2) + (k34)

    X_NF_vesc = xi1 - xi2 * (b + xi3)**2

    if np.isscalar(X_NF):
        X_NF = np.clip(X_NF, max(0, X_NF_vesc), 1)
    else:
        X_NF = np.clip(np.array(X_NF), np.maximum(0, np.array(X_NF_vesc)), 1)

    ########################################################################

    # Far-field:
    psi1 = s11 + (s12 * gamma**s13) + (s14 * M_t**s15)
    psi2 = s21 + (s22 * gamma**s23) + (s24 * M_t**s25)
    psi3 = s31 + (s32 * gamma**s33)
    psi4 = s41 + (s42 * gamma**s43) + (s44 * M_t**s45)

    X_FF = psi1 * np.exp(-psi2 * (Q_R_prime * (1 + (M_i / M_t)) * (1 - b)**psi4)) + psi3
    X_FF = np.clip(X_FF, 0, 1)

    ########################################################################

    # Total:
    X_atm = X_NF + X_FF
    X_atm = np.clip(X_atm, 0, 1)

    return X_NF, X_FF, X_atm
