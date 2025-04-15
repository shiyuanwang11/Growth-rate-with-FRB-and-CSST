"""
nohup python baseline.py >out75.out 2>&1 &
FRBs at different redshifts
changes: zg, ngal, 
        get_power(zfrb) , small scale ---- if needs
"""


import os
import Main
from Main import PowerSpec, Analysis
import numpy as np
import matplotlib.pyplot as plt
import Func
from colossus.cosmology import cosmology
cosmo=cosmology.setCosmology('planck18')
h=cosmo.H0/100.

def rgba(cc):
    return cc/255.
c1 = rgba(np.array([41.,157.,143.,255.]))
c2 = rgba(np.array([233.,196.,106.,255.]))
c3 = rgba(np.array([216.,118.,89.,255.]))


##### set
dz=0.15
zg=0.15
ngal=3.4e-2     # h/Mpc ^3
ngal_h=ngal*h**3     # 1/Mpc ^3
fsky=17500./41253.
bg = Func.biasg(zg)
bv=1.

# set path to save
data_folder,fig_folder = Func.paths(zg)


# calculate power spectrum of small scales, which are needed to calculate σ(bv) and Nvv
# CMB-S4
get_power = PowerSpec(fsky, zg, dz, ngal_h, zfrb=zg+0.25, small_scale=False)
analysis = Analysis(fsky, zg, dz, ngal_h,clt_beam=1.5, clt_RMS=1.8, small_scale=False)

#get_power.get_pk()
#get_power.get_cl()
#
#analysis.get_snr()   # small_scale=False
#analysis.get_pge_err()    # small_scale=False
# 
# analysis.get_bv_err()     #small_scale=True
# nvv = analysis.Nvv_no_mu()

# nvv 75 45 15 : 0.034226697218877215, 0.02045758484862518, 0.01337198775965809

nvv=0.01337198775965809
# 
# file_path = os.path.join(data_folder,'bv1000.txt')
# if os.path.exists(file_path):
#     print("File exists! Continue running the script...")
# else:
#     print("Error: File does not exist.")
#     exit(1)
# 
# fisher forecast of RSD 
Ff_RSD_fs8, err_rsd = Main.fisher_RSD(zg,dz,fsky,ngal)

# fisher forecast of kSZ tomography
Ff_ksz = Main.fisher_ksz(zg, fsky, dz, bg, bv, ngal, nvv)


bv100 = np.loadtxt(os.path.join(data_folder, "bv100.txt"), delimiter=' ', dtype='str').astype(float)
bv300 = np.loadtxt(os.path.join(data_folder, "bv300.txt"), delimiter=' ', dtype='str').astype(float)
bv1000 = np.loadtxt(os.path.join(data_folder, "bv1000.txt"), delimiter=' ', dtype='str').astype(float)
Ff_bv_100 = 1/bv100**2
Ff_bv_300 = 1/bv300**2
Ff_bv_1000 = 1/bv1000**2



# add fisher matrix
ksz_tomog100 = np.zeros(len(bv100))
ksztomog_RSD100 = np.zeros(len(bv100))
for i in range(len(ksz_tomog100)):
    ksz_tomog100[i] = Main.Add_fisher_gets_fs8(Ff_ksz,Ff_bv_100[i])
    ksztomog_RSD100[i] = Main.Add_2_fisher_get_fs8(Ff_ksz, Ff_bv_100[i], Ff_RSD_fs8)

ksz_tomog300 = np.zeros(len(bv300))
ksztomog_RSD300 = np.zeros(len(bv300))
for i in range(len(ksz_tomog300)):
    ksz_tomog300[i] = Main.Add_fisher_gets_fs8(Ff_ksz,Ff_bv_300[i])
    ksztomog_RSD300[i] = Main.Add_2_fisher_get_fs8(Ff_ksz, Ff_bv_300[i], Ff_RSD_fs8)

ksz_tomog1000 = np.zeros(len(bv1000))
ksztomog_RSD1000 = np.zeros(len(bv1000))
for i in range(len(ksz_tomog1000)):
    ksz_tomog1000[i] = Main.Add_fisher_gets_fs8(Ff_ksz,Ff_bv_1000[i])
    ksztomog_RSD1000[i] = Main.Add_2_fisher_get_fs8(Ff_ksz, Ff_bv_1000[i], Ff_RSD_fs8)

np.savetxt(os.path.join(data_folder, "ksz_tomog100.txt"), ksz_tomog100, fmt='%.10e',comments='')
np.savetxt(os.path.join(data_folder, "ksztomog_RSD100.txt"), ksztomog_RSD100, fmt='%.10e',comments='')
np.savetxt(os.path.join(data_folder, "ksz_tomog300.txt"), ksz_tomog300, fmt='%.10e',comments='')
np.savetxt(os.path.join(data_folder, "ksztomog_RSD300.txt"), ksztomog_RSD300, fmt='%.10e',comments='')
np.savetxt(os.path.join(data_folder, "ksz_tomog1000.txt"), ksz_tomog1000, fmt='%.10e',comments='')
np.savetxt(os.path.join(data_folder, "ksztomog_RSD1000.txt"), ksztomog_RSD1000, fmt='%.10e',comments='')

# plot constraint results
#Nfrb = np.logspace(1, 9, num=17, endpoint=True, base=10.0, dtype=None)
#plt.clf()
#plt.figure(figsize=(10, 5))
#plt.plot(Nfrb,ksz_tomog100,c=c1,linestyle='-',label='ksz tomography')
#plt.plot(Nfrb,ksz_tomog300,c=c2,linestyle='-')
#plt.plot(Nfrb,ksz_tomog1000,c=c3,linestyle='-')
#plt.plot(Nfrb,ksztomog_RSD100,c=c1,linestyle=':',label='ksz tomography + RSD')
#plt.plot(Nfrb,ksztomog_RSD300,c=c2,linestyle=':')
#plt.plot(Nfrb,ksztomog_RSD1000,c=c3,linestyle=':')
#plt.axhline(y=err_rsd,label='RSD',c='grey', linestyle='--')
#plt.xlabel(r'$ N_{FRB} $',size=13)
#plt.ylabel(r'$ \sigma(f \sigma_8) $',size=13)
#plt.legend(fontsize=13)
##plt.xlim([1e+2,1e+9])
#plt.xscale('log')
#plt.yscale('log')
#plt.savefig(os.path.join(fig_folder,'fisher_add.png'))
#print('save figure successfully!')
#
#print(Nfrb[10], 'sigma_DM=300')
#print('ksz tomography: ',ksz_tomog300[10])
#print('RSD+ksz tomography: ', ksztomog_RSD300[10])
