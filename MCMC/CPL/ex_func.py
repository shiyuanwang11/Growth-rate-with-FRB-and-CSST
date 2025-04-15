import Func
import Main
import os
import numpy as np
import matplotlib.pyplot as plt
from cobaya.run import run
from getdist import loadMCSamples
from colossus.cosmology import cosmology
cosmo=cosmology.setCosmology('planck18')

def add_matrix_elements(a, b):
    # 创建矩阵a的副本
    result = np.copy(a)
    result[6:10, 6:10] += b
    return result

def rsd_err(zg):
    if zg==0.15:
        return 0.037622781639503125
    elif zg==0.45:
        return 0.01302791847528607
    elif zg==0.75:
        return 0.007837485697296847

def rsd_fish(zg):
    if zg==0.15:
        return np.array([[33144.45316437,  4680.02535599],[ 4680.02535599,  1367.30088841]])
    elif zg==0.45:
        return np.array([[279365.83868817 , 39260.49386234],[ 39260.49386234,  11409.2739193 ]])
    elif zg==0.75:
        return np.array([[857124.21518862, 115036.58644071],[115036.58644071 , 31719.02437426]])

def ksz_fish(zg):
    if zg==0.15:
        return np.array([[ 129991.54061874, -126909.54563653,  -57836.57645322],[-126909.54563653,  235743.87508634,  107435.72192638],[ -57836.57645322,  107435.72192638,   48961.75708326]])
    elif zg==0.45:
        return np.array([[ 702868.8967015,  -426831.1479537 , -202333.2591985 ],[-426831.1479537  , 797468.43532155 , 378028.61482835],[-202333.2591985  , 378028.61482835 , 179199.10970698]])
    elif zg==0.75:
        return np.array([[1633837.84443803, -500223.07606521, -228450.23265919],[-500223.07606521 , 990055.10699039,  452154.90919872],[-228450.23265919  ,452154.90919872,  206497.65903838]])


def get_err(zg):
    """
    assume at a redshift
    input: fisher of RSD, kSZ, bv100 is by reading data-file
    return: ksztomog_RSD100, ksztomog_RSD300, ksztomog_RSD1000
    """
    data_folder,fig_folder = Func.paths(zg)
    bv100 = np.loadtxt(os.path.join(data_folder, "bv100.txt"), delimiter=' ', dtype='str').astype(float) 
    bv300 = np.loadtxt(os.path.join(data_folder, "bv300.txt"), delimiter=' ', dtype='str').astype(float) 
    bv1000 = np.loadtxt(os.path.join(data_folder, "bv1000.txt"), delimiter=' ', dtype='str').astype(float) 

    Ff_bv_100 = 1/bv100**2
    Ff_bv_300 = 1/bv300**2
    Ff_bv_1000 = 1/bv1000**2

    Ff_ksz = ksz_fish(zg)
    Ff_RSD_fs8 = rsd_fish(zg)
    
    ksztomog_RSD100 = np.zeros(len(bv100))
    for i in range(len(bv100)):
        ksztomog_RSD100[i] = Main.Add_2_fisher_get_fs8(Ff_ksz, Ff_bv_100[i], Ff_RSD_fs8)

    ksztomog_RSD300 = np.zeros(len(bv300))
    for i in range(len(bv300)):
        ksztomog_RSD300[i] = Main.Add_2_fisher_get_fs8(Ff_ksz, Ff_bv_300[i], Ff_RSD_fs8)

    ksztomog_RSD1000 = np.zeros(len(bv1000))
    for i in range(len(bv1000)):
        ksztomog_RSD1000[i] = Main.Add_2_fisher_get_fs8(Ff_ksz, Ff_bv_1000[i], Ff_RSD_fs8)

    return ksztomog_RSD100, ksztomog_RSD300, ksztomog_RSD1000

def da(z_array, omegam, w0, wa):
    results = []
    for z in z_array:
        int_z = np.linspace(0, z, 50)
        E = Func.Ez(int_z, omegam, w0, wa)
        integral = np.trapz(1./E, int_z)
        results.append(integral)
    return np.array(results)


def run_mcmc(err_rsd_ksz):
    """
    仅ksz, 输入三个红移处的ksz给的误差, 输出的是fisher矩阵
    """
    # array([1.e+02, 1.e+04, 1.e+06, 1.e+08]), 
    # .take([0,2,6,8,10,12,14,16])
    Nfrb = np.logspace(1, 9, num=17, endpoint=True, base=10.0, dtype=None) 

    zz = np.array([0.15,0.45,0.75])

    w0_fid = -0.57779661
    wa_fid = -1.31630327
    omegam_fid = 0.34166733
    sigma8_fid = 0.79547449
    fs8 = Func.growthrate(Om0=omegam_fid,w0=w0_fid, wa=wa_fid,z=zz) * Func.sigma(sigma8=sigma8_fid, Om0=omegam_fid,w0=w0_fid, wa=wa_fid,z=zz)
    ref=[omegam_fid,sigma8_fid,w0_fid,wa_fid]

    def log_like(omegam,sigma8,w0,wa):

        E = Func.Ez(zz,omegam,w0,wa)
        E_fid = Func.Ez(zz,omegam_fid,w0_fid,wa_fid)

        D = da(zz, omegam, w0, wa)
        D_fid = da(zz, omegam_fid, w0_fid, wa_fid)

        f_th = Func.growthrate(Om0=omegam,w0=w0, wa=wa,z=zz)
        sigma8_th = Func.sigma(sigma8=sigma8, Om0=omegam,w0=w0, wa=wa,z=zz)

        Chi_1 = (fs8 - (E*D)/(E_fid*D_fid)*sigma8_th*f_th)

        Chi_2 = err_rsd_ksz**2
        Chi_C = Chi_1**2/Chi_2
        Chi = np.sum(Chi_C)

        return -0.5*Chi

    info = {"likelihood": {"loglike": log_like}, \
        "params": {"omegam": {"prior": {"min": 0.0, "max": 0.6},'ref': ref[0],"latex": r'\Omega_m',"proposal": 0.001}, \
                    "sigma8": {"prior": {"min": 0.5, "max": 1.0},'ref': ref[1],"latex": r'\sigma8',"proposal": 0.001}, \
                    "w0": {"prior": {"min": -2., "max": 2.0},'ref': ref[2],"latex": r'w_0',"proposal": 0.001}, \
                    "wa": {"prior": {"min": -3., "max": 1.0},'ref': ref[3],"latex": r'w_a',"proposal": 0.001}
                    }, \
            "sampler": {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 10000},},\
            "output": "chains/joint"
            }
    updated_info, sampler = run(info,force=True)

    readsamps = loadMCSamples('chains/joint')
    cov = readsamps.cov(['w0','wa','omegam','sigma8'])

    return np.linalg.inv(cov)

def get_para(err_rsd_ksz):
    """
    仅针对一种情况下的误差, 给
    """

    rsd_ksz_fish = run_mcmc(err_rsd_ksz)

    #  ['omegabh2','omegach2','theta','tau','logA','ns','w0','wa','omegam','sigma8','S8']

    #read
    planck_fish = np.array([[ 5.03576557e+09, -4.53801685e+09,  1.38435468e+09,
         1.08165304e+06, -1.74929630e+08, -1.27170156e+08,
        -3.79040213e+07, -7.49865402e+06,  1.10722339e+08,
         6.80782403e+07,  3.47954327e+08],
       [-4.53801685e+09,  4.15636360e+09, -1.27161263e+09,
        -4.53460404e+05,  1.59676880e+08,  1.16402617e+08,
         3.47140253e+07,  6.87068175e+06, -1.01321257e+08,
        -6.21643749e+07, -3.18300102e+08],
       [ 1.38435468e+09, -1.27161263e+09,  4.03301111e+08,
         2.65883771e+05, -4.87225761e+07, -3.55535428e+07,
        -1.08437338e+07, -2.15850901e+06,  3.13279882e+07,
         1.84652682e+07,  9.74425321e+07],
       [ 1.08165304e+06, -4.53460404e+05,  2.65883771e+05,
         1.63559051e+05, -9.82614401e+04, -3.34119914e+04,
        -4.94433270e+03, -9.85908861e+02,  1.37198134e+04,
         1.04390158e+04,  4.90641711e+04],
       [-1.74929630e+08,  1.59676880e+08, -4.87225761e+07,
        -9.82614401e+04,  6.21326271e+06,  4.49563108e+06,
         1.30399454e+06,  2.56379465e+05, -3.84577388e+06,
        -2.46732892e+06, -1.22445510e+07],
       [-1.27170156e+08,  1.16402617e+08, -3.55535428e+07,
        -3.34119914e+04,  4.49563108e+06,  3.38006299e+06,
         9.49048886e+05,  1.86638138e+05, -2.80400306e+06,
        -1.79717650e+06, -8.89876060e+06],
       [-3.79040213e+07,  3.47140253e+07, -1.08437338e+07,
        -4.94433270e+03,  1.30399454e+06,  9.49048886e+05,
         3.19193192e+05,  6.47751323e+04, -8.92135694e+05,
        -4.45176189e+05, -2.65678672e+06],
       [-7.49865402e+06,  6.87068175e+06, -2.15850901e+06,
        -9.85908861e+02,  2.56379465e+05,  1.86638138e+05,
         6.47751323e+04,  1.32316484e+04, -1.79046476e+05,
        -8.41324963e+04, -5.25536141e+05],
       [ 1.10722339e+08, -1.01321257e+08,  3.13279882e+07,
         1.37198134e+04, -3.84577388e+06, -2.80400306e+06,
        -8.92135694e+05, -1.79046476e+05,  2.94866397e+06,
         1.74423740e+06,  7.43289352e+06],
       [ 6.80782403e+07, -6.21643749e+07,  1.84652682e+07,
         1.04390158e+04, -2.46732892e+06, -1.79717650e+06,
        -4.45176189e+05, -8.41324963e+04,  1.74423740e+06,
         1.41409110e+06,  4.48986737e+06],
       [ 3.47954327e+08, -3.18300102e+08,  9.74425321e+07,
         4.90641711e+04, -1.22445510e+07, -8.89876060e+06,
        -2.65678672e+06, -5.25536141e+05,  7.43289352e+06,
         4.48986737e+06,  2.46510267e+07]])

    
    planck_rsd_ksz_fish = add_matrix_elements(planck_fish, rsd_ksz_fish)
    planck_rsd_ksz_cov = np.linalg.inv(planck_rsd_ksz_fish)

    w0 = planck_rsd_ksz_cov[6,6]
    wa = planck_rsd_ksz_cov[7,7]
    omegam = planck_rsd_ksz_cov[8,8]
    sigma8 = planck_rsd_ksz_cov[9,9]
    S8 = planck_rsd_ksz_cov[10,10]

    return w0, wa, omegam, sigma8, S8