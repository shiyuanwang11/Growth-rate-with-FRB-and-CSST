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
    result[6:8, 6:8] += b
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

def da(z_array, omegam):
    results = []
    for z in z_array:
        int_z = np.linspace(0, z, 50)
        E = Func.Ez(int_z, omegam)
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

    omegam_fid = 0.31655132
    sigma8_fid = 0.8119776
    ref=[omegam_fid,sigma8_fid]

    fs8 = Func.growthrate(Om0=omegam_fid, z=zz) * Func.sigma(sigma8=sigma8_fid, Om0=omegam_fid, z=zz)

    def log_like(omegam,sigma8):

        E = Func.Ez(zz, omegam)
        E_fid = Func.Ez(zz, omegam_fid)

        D = da(zz, omegam)
        D_fid = da(zz, omegam_fid)

        f_th = Func.growthrate(Om0=omegam,z=zz)
        sigma8_th = Func.sigma(sigma8=sigma8, Om0=omegam,z=zz)

        Chi_1 = (fs8 - (E*D)/(E_fid*D_fid)*sigma8_th*f_th)

        Chi_2 = err_rsd_ksz**2
        Chi_C = Chi_1**2/Chi_2
        Chi = np.sum(Chi_C)
        return -0.5*Chi

    info = {"likelihood": {"loglike": log_like}, \
            "params": {"omegam": {"prior": {"min": 0.0, "max": 1.0},'ref': ref[0],"latex": r'\Omega_m'}, \
                        "sigma8": {"prior": {"min": 0.0, "max": 1.0},'ref': ref[1],"latex": r'\sigma8'}}, \
            "sampler": {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 10000},},\
            "output": "chains/joint"
            }
    updated_info, sampler = run(info,force=True)

    readsamps = loadMCSamples('chains/joint')
    cov = readsamps.cov(['omegam','sigma8'])

    return np.linalg.inv(cov)

def get_para(err_rsd_ksz):
    """
    仅针对一种情况下的误差, 给
    """

    rsd_ksz_fish = run_mcmc(err_rsd_ksz)

    # ['omegabh2','omegach2','theta','tau','logA','ns','omegam','sigma8','S8','H0']

    #read
    planck_fish = np.array([[ 1.92435890e+11, -1.04891492e+11,  8.29653928e+10,
         2.06339312e+06, -2.16599914e+08, -1.64686954e+08,
         7.04935309e+09,  3.77970645e+08,  1.49955563e+08,
        -1.81392615e+08],
       [-1.04891492e+11,  6.15884691e+10, -4.22896826e+10,
        -7.28757505e+05,  6.73827426e+08,  5.09918726e+08,
        -4.85926004e+09, -1.80297763e+09,  1.39367453e+08,
         8.71262777e+07],
       [ 8.29653928e+10, -4.22896826e+10,  3.82426891e+10,
         9.07497654e+05,  3.40549404e+08,  2.56482177e+08,
         2.25991618e+09, -1.19052827e+09,  3.41510496e+08,
        -8.64386798e+07],
       [ 2.06339312e+06, -7.28757505e+05,  9.07497655e+05,
         1.54616173e+05, -5.74782828e+04, -4.24042500e+03,
        -1.16037917e+05, -1.78943785e+05,  1.45363926e+05,
        -1.69858173e+03],
       [-2.16599914e+08,  6.73827426e+08,  3.40549404e+08,
        -5.74782828e+04,  7.85565570e+07,  5.93035221e+07,
        -1.49265156e+08, -2.39319971e+08,  4.46288532e+07,
        -1.32397123e+06],
       [-1.64686954e+08,  5.09918726e+08,  2.56482177e+08,
        -4.24042500e+03,  5.93035221e+07,  4.48993777e+07,
        -1.12766369e+08, -1.80708466e+08,  3.36784176e+07,
        -9.98908921e+05],
       [ 7.04935309e+09, -4.85926004e+09,  2.25991618e+09,
        -1.16037917e+05, -1.49265156e+08, -1.12766369e+08,
         6.84226529e+08,  5.75386237e+08, -2.02101209e+08,
        -3.86156714e+06],
       [ 3.77970645e+08, -1.80297763e+09, -1.19052827e+09,
        -1.78943785e+05, -2.39319971e+08, -1.80708466e+08,
         5.75386237e+08,  8.55994452e+08, -2.59202703e+08,
         4.09603526e+06],
       [ 1.49955563e+08,  1.39367453e+08,  3.41510496e+08,
         1.45363926e+05,  4.46288532e+07,  3.36784176e+07,
        -2.02101209e+08, -2.59202703e+08,  1.45233472e+08,
        -8.10811929e+05],
       [-1.81392615e+08,  8.71262777e+07, -8.64386798e+07,
        -1.69858173e+03, -1.32397123e+06, -9.98908921e+05,
        -3.86156714e+06,  4.09603526e+06, -8.10811929e+05,
         2.02870478e+05]])

    
    planck_rsd_ksz_fish = add_matrix_elements(planck_fish, rsd_ksz_fish)
    planck_rsd_ksz_cov = np.linalg.inv(planck_rsd_ksz_fish)

    omegam = planck_rsd_ksz_cov[6,6]
    sigma8 = planck_rsd_ksz_cov[7,7]
    S8 = planck_rsd_ksz_cov[8,8]
    h0 = planck_rsd_ksz_cov[9,9]

    return omegam, sigma8, S8, h0