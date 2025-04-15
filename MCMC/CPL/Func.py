import os
import numpy as np
import math
import scipy.special as special
from scipy import interpolate as spi
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from colossus.cosmology import cosmology
cosmo=cosmology.setCosmology('planck18')

h = cosmo.H0/100
c = 3e+5    # km/s

############################## cosmology ##############################
def Ez(z,omegam,w0,wa):
    em = omegam*(1+z)**3
    ede = (1.-omegam) * (1.+z)**(3.*(1.+w0+wa)) * np.exp(-3.*wa*z/(1.+z))
    return np.sqrt(em+ede)

def Ea(a,omegam,w0,wa):
    atoz = 1./a -1.
    return Ez(atoz,omegam,w0,wa)



############################## growth factor D ###################################

def D(sigma8,Om0,w0,wa,z):
    """
    the growth factou D(a), the range of z is [1e-3, 1]
    with the Initial condition D(a=1e-3)=1e-3, D'(a=1e-3)=1
    """
    def model(y,x,Om0=Om0,w0=w0,wa=wa):
        dydx = y[1]
        c = Om0/x**4 + (1.-Om0) * np.exp(-3.*wa*(1.-x)) * ((1.+w0+wa)*x**(-3.*(1.+w0+wa)-1.) - wa*x**(-3.*(1.+w0+wa)))
        A = 3./2.*c/Ea(x,Om0,w0,wa)**2 - 3./x
        B = 3./2. * Om0/x**5 / Ea(x,Om0,w0,wa)**2
        d2ydx2 = A*y[1] + B*y[0]
        return [dydx, d2ydx2]

    x = np.linspace(1e-3, 1, 500)    # x:a
    y0 = [1e-3,1e-3]   #初始条件
    y = odeint(model, y0, x)    # D(a)
    func = interp1d(x,y[:,0]/y[-1,0],kind='linear')

    return func(1./(1.+z))

def growthrate(Om0,w0,wa,z):
    """
    growth rate f = dlnD/dlna, the range of z is [1e-3, 1]
    with the Initial condition D(a=1e-3)=1e-3, D'(a=1e-3)=1 
    """
    def model(y,x,Om0=Om0,w0=w0,wa=wa):
        dydx = y[1]
        c = Om0/x**4 + (1.-Om0) * np.exp(-3.*wa*(1.-x)) * ((1.+w0+wa)*x**(-3.*(1.+w0+wa)-1.) - wa*x**(-3.*(1.+w0+wa)))
        A = 3./2.*c/Ea(x,Om0,w0,wa)**2 - 3./x
        B = 3./2. * Om0/x**5 / Ea(x,Om0,w0,wa)**2
        d2ydx2 = A*y[1] + B*y[0]
        return [dydx, d2ydx2]

    x = np.linspace(1e-3, 1., 500)    # x:a
    y0 = [1e-3,1.]   #初始条件
    y = odeint(model, y0, x)    # D(a)
    func = interp1d(x,y[:,0]/y[-1,0],kind='cubic')  # D(a)

    interp_D = func(x)
    lnD = np.log(interp_D)
    dlnDdlna = x * np.gradient(lnD, x)

    func_f = interp1d(x, dlnDdlna,kind='cubic')

    return func_f(1./(1.+z))

def sigma(sigma8,Om0,w0,wa,z):
    """
    sigma_8, the range of z [1e-3, 1]
    input: sigma8, Om0  is today value, z can be an array
    return: 
    """
    def model(y,x,Om0=Om0,w0=w0,wa=wa):
        dydx = y[1]
        c = Om0/x**4 + (1.-Om0) * np.exp(-3.*wa*(1.-x)) * ((1.+w0+wa)*x**(-3.*(1.+w0+wa)-1.) - wa*x**(-3.*(1.+w0+wa)))
        A = 3./2.*c/Ea(x,Om0,w0,wa)**2 - 3./x
        B = 3./2. * Om0/x**5 / Ea(x,Om0,w0,wa)**2
        d2ydx2 = A*y[1] + B*y[0]
        return [dydx, d2ydx2]

    x = np.linspace(1e-3, 1, 500)    # x:a
    y0 = [1e-3,1.]   #初始条件
    y = odeint(model, y0, x)    # D(a)
    func = interp1d(x,y[:,0]/y[-1,0],kind='linear')

    return func(1./(1.+z))*sigma8


############################## free electron density today ##############################

constants = {
    'thompson_SI': 6.6524e-29,
    'meter_to_megaparsec': 3.241e-23,
    'G_SI': 6.674e-11,
    'mProton_SI': 1.673e-27,
    'H100_SI': 3.241e-18
}

def chi(Yp,NHe):
    val = (1-Yp*(1-NHe/4.))/(1-Yp/2.)
    return val

def ne0_(Yp=0.24,NHe=0,me = 1.14,gasfrac = 0.9):
    '''
    Average electron density today
    Eq 3 of 1109.0553
    Units: 1/meter**3
    '''
    ombh2 = cosmo.Ombh2
    omgh2 = gasfrac* ombh2
    mu_e = 1.14 # mu_e*mass_proton = mean mass per electron
    ne0_SI = chi(Yp,NHe)*omgh2 * 3.*(constants['H100_SI']**2.)/constants['mProton_SI']/8./np.pi/constants['G_SI']/mu_e
    return ne0_SI



################################ survey  ################################

def Vs(fsky, z_min, z_max):
    """ 
    the volume of a shell between z_min and z_max.
    fsky is the survey overlap such as 0.x 
    retuen Vs in Mpc^3
    """
    omega = fsky*41253*(math.pi/180)**2
    d2 = cosmo.comovingDistance(0,z_min)
    d3 = cosmo.comovingDistance(0,z_max)   # Mpc/h
    return omega/3*(d3**3 - d2**3)/h**3  # (Mpc)^3

def get_kmin(v):
    """
    caculate kmin : 2*pi / v**(1/3)
    input v in Mpc^3
    return kmin in 1/Mpc
    """
    return 2*math.pi/v**(1/3)

def get_kmax(z):
    """
    caculate kmax : 0.1*D(0)/D(z) *h
    return kmax in 1/Mpc
    """
    sigma8 = cosmo.sigma8
    Om0 = cosmo.Om0
    return 0.1/D(sigma8,Om0,z)*h

def biasg(z):
    return 1.+0.84*z



########################## fisher toolkit #########################

def Add_fisher_gets(f_ori, f_add, row_ind, col_ind):
    """
    f_ori is a 3*3 metric, f_add is a number(bv)
    return sigma(fs8)
    """
    metric_add = np.zeros(np.shape(f_ori))
    metric_add[row_ind,col_ind] += f_add
    cov = np.linalg.inv(f_ori+metric_add)
    return cov[1,1]**0.5

############################### save path  ################################

def paths(zg):
    # set path to save
    current_folder = os.getcwd()
    if zg==0.15:
        data_folder = os.path.join(current_folder, "z15/data")
        fig_folder = os.path.join(current_folder, "z15/fig")
        return data_folder,fig_folder
    elif zg==0.45:
        data_folder = os.path.join(current_folder, "z45/data")
        fig_folder = os.path.join(current_folder, "z45/fig")
        return data_folder,fig_folder
    elif zg==0.75:
        data_folder = os.path.join(current_folder, "z75/data")
        fig_folder = os.path.join(current_folder, "z75/fig")
        return data_folder,fig_folder
    else:
        print("Unsupported value of zg")


