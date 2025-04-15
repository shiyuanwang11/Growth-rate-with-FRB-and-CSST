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
    result[6:9, 6:9] += b
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

    # array([1.e+02, 1.e+04, 1.e+06, 1.e+08])
    Nfrb = np.logspace(1, 9, num=17, endpoint=True, base=10.0, dtype=None)#.take([2,6,10,14])

    data_folder,fig_folder = Func.paths(zg)
    bv100 = np.loadtxt(os.path.join(data_folder, "bv100.txt"), delimiter=' ', dtype='str').astype(float)#.take([2,6,10,14])
    bv300 = np.loadtxt(os.path.join(data_folder, "bv300.txt"), delimiter=' ', dtype='str').astype(float)#.take([2,6,10,14])
    bv1000 = np.loadtxt(os.path.join(data_folder, "bv1000.txt"), delimiter=' ', dtype='str').astype(float)#.take([2,6,10,14])

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

def da(z_array, omegam, w):
    results = []
    for z in z_array:
        int_z = np.linspace(0, z, 50)
        E = Func.Ez(int_z, omegam, w)
        integral = np.trapz(1./E, int_z)
        results.append(integral)
    return np.array(results)


def run_mcmc(err_rsd_ksz):
    """
    仅ksz, 输入三个红移处的ksz给的误差, 输出的是fisher矩阵
    """
    # array([1.e+02, 1.e+04, 1.e+06, 1.e+08])
    Nfrb = np.logspace(1, 9, num=17, endpoint=True, base=10.0, dtype=None)#.take([2,6,10,14])

    zz = np.array([0.15,0.45,0.75])

    w_fid = -1.58491121
    omegam_fid = 0.19765297
    sigma8_fid = 0.97234554

    fs8 = Func.growthrate(Om0=omegam_fid,w=w_fid,z=zz) * Func.sigma(sigma8=sigma8_fid, Om0=omegam_fid,w=w_fid,z=zz)
    ref=[omegam_fid,sigma8_fid,w_fid]

    def log_like(omegam,sigma8,w):

        E = Func.Ez(zz, omegam, w)
        E_fid = Func.Ez(zz, omegam_fid, w_fid)

        D = da(zz, omegam, w)
        D_fid = da(zz, omegam_fid, w_fid)

        f_th = Func.growthrate(Om0=omegam,w=w, z=zz)
        sigma8_th = Func.sigma(sigma8=sigma8, Om0=omegam, w=w, z=zz)

        Chi_1 = (fs8 - (E*D)/(E_fid*D_fid)*sigma8_th*f_th)

        Chi_2 = err_rsd_ksz**2
        Chi_C = Chi_1**2/Chi_2
        Chi = np.sum(Chi_C)

        return -0.5*Chi

    info = {"likelihood": {"loglike": log_like}, \
        "params": {"omegam": {"prior": {"min": 0.0, "max": 1.0},'ref': ref[0],"latex": r'\Omega_m'}, \
                    "sigma8": {"prior": {"min": 0.5, "max": 1.5},'ref': ref[1],"latex": r'\sigma8'}, \
                    "w": {"prior": {"min": -2., "max": 0.0},'ref': ref[2],"latex": r'w'}
                    }, \
            "sampler": {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 10000},},\
            "output": "chains/joint"
            }
    updated_info, sampler = run(info,force=True)

    readsamps = loadMCSamples('chains/joint')
    cov = readsamps.cov(['w','omegam','sigma8'])

    return np.linalg.inv(cov)

def get_para(err_rsd_ksz):
    """
    仅针对一种情况下的误差, 给
    """

    rsd_ksz_fish = run_mcmc(err_rsd_ksz)

    #  ['omegabh2','omegach2','theta','tau','logA','ns','w','omegam','sigma8','S8']

    #read
    planck_fish = np.array([[ 2.17072062e+09, -1.90598633e+09,  5.34704389e+08,
         8.30179107e+05, -7.66609692e+07, -7.20267526e+07,
        -9.64590415e+06,  2.74596215e+07,  3.45062621e+07,
         1.52993246e+08],
       [-1.90598633e+09,  1.74335394e+09, -4.94290055e+08,
        -1.98445746e+05,  6.93889378e+07,  6.56360101e+07,
         8.92883793e+06, -2.50054924e+07, -3.09952706e+07,
        -1.39598805e+08],
       [ 5.34704389e+08, -4.94290055e+08,  1.57305184e+08,
         1.60040206e+05, -1.91317119e+07, -1.81557008e+07,
        -2.98559737e+06,  7.07019502e+06,  7.25502267e+06,
         3.99655572e+07],
       [ 8.30179107e+05, -1.98445746e+05,  1.60040206e+05,
         1.62588492e+05, -8.90857077e+04, -3.05915258e+04,
        -1.40775681e+03,  5.34823610e+03,  8.64867873e+03,
         3.05024774e+04],
       [-7.66609692e+07,  6.93889378e+07, -1.91317119e+07,
        -8.90857077e+04,  2.89821376e+06,  2.69953249e+06,
         2.91711802e+05, -1.00150400e+06, -1.45786252e+06,
        -5.52224868e+06],
       [-7.20267526e+07,  6.56360101e+07, -1.81557008e+07,
        -3.05915258e+04,  2.69953249e+06,  2.65038156e+06,
         2.74877217e+05, -9.43139274e+05, -1.37193637e+06,
        -5.19925401e+06],
       [-9.64590415e+06,  8.92883793e+06, -2.98559737e+06,
        -1.40775681e+03,  2.91711802e+05,  2.74877217e+05,
         9.18530226e+04, -1.25227898e+05, -4.98835436e+02,
        -7.48668490e+05],
       [ 2.74596215e+07, -2.50054924e+07,  7.07019502e+06,
         5.34823610e+03, -1.00150400e+06, -9.43139274e+05,
        -1.25227898e+05,  3.75945432e+05,  4.62620786e+05,
         1.99254823e+06],
       [ 3.45062621e+07, -3.09952706e+07,  7.25502267e+06,
         8.64867873e+03, -1.45786252e+06, -1.37193637e+06,
        -4.98835436e+02,  4.62620786e+05,  1.09946720e+06,
         2.36629718e+06],
       [ 1.52993246e+08, -1.39598805e+08,  3.99655572e+07,
         3.05024774e+04, -5.52224868e+06, -5.19925401e+06,
        -7.48668490e+05,  1.99254823e+06,  2.36629718e+06,
         1.12174236e+07]])

    
    planck_rsd_ksz_fish = add_matrix_elements(planck_fish, rsd_ksz_fish)
    planck_rsd_ksz_cov = np.linalg.inv(planck_rsd_ksz_fish)

    w = planck_rsd_ksz_cov[6,6]
    omegam = planck_rsd_ksz_cov[7,7]
    sigma8 = planck_rsd_ksz_cov[8,8]
    S8 = planck_rsd_ksz_cov[9,9]

    return w, omegam, sigma8, S8