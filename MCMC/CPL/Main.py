import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology
cosmo=cosmology.setCosmology('planck18')
import camb

from scipy.integrate import quad, dblquad
from functools import partial

import hmvec as hm
import Func

c = 3e+5  #km/s
c_speed = c
h = cosmo.H0/100
Nfrb = np.logspace(1, 9, num=17, endpoint=True, base=10.0, dtype=None)
sigmaD = [100,300,1000]
ne0 = Func.ne0_()


class PowerSpec:
    def __init__(self, fsky, zg, dz, ngal, zfrb, small_scale=False, data_folder=None, fig_folder=None):
        self.fsky = fsky
        self.zg = zg
        self.dz = dz   # half
        self.ngal = ngal   # Mpc^-3
        self.zfrb = zfrb
        self.small_scale = small_scale

        self.V = Func.Vs(self.fsky, self.zg - self.dz, self.zg + self.dz)
        self.kmin = Func.get_kmin(self.V)
        self.kmax = 10.      #Func.get_kmax(self.dz)
        self.bg = Func.biasg(self.zg)


        self.ksmax = 10.      # /Mpc
        self.ksmin = 0.1
        self.chig = cosmo.comovingDistance(0.,self.zg)/h    # Mpc
        self.dchig = cosmo.comovingDistance(z_min=self.zg-self.dz,z_max=self.zg+self.dz)/h

        self.zs = np.linspace(0.,1.,101)
        self.ms = np.geomspace(2e10,1e17,200)
#        self.kk = np.geomspace(self.kmin,self.kmax,200)
#        self.ks = np.geomspace(self.ksmin,self.ksmax,200)

        self.data_folder, _ = Func.paths(self.zg)
        print(self.data_folder)
        

        if not small_scale:
            self.kk = np.geomspace(self.kmin,self.kmax,200)
            np.savetxt(os.path.join(self.data_folder, "kk.txt"), self.kk, fmt='%.10e',comments='')
            self.ell = self.chig*self.kk
            np.savetxt(os.path.join(self.data_folder, "ell.txt"), self.ell, fmt='%.10e',comments='')
        else:
            self.kk = np.geomspace(self.ksmin,self.ksmax,200)
            np.savetxt(os.path.join(self.data_folder, "ks.txt"), self.kk, fmt='%.10e',comments='')
            self.ell = self.chig*self.kk
            np.savetxt(os.path.join(self.data_folder, "ell_s.txt"), self.ell, fmt='%.10e',comments='')


        
        self.index = np.where(self.zs==self.zg)[0].astype(int)
        

    ####### Pk
    def get_pge(self):
        hcos1 = hm.HaloModel(self.zs,self.kk,ms=self.ms)
        hcos1.add_hod("g",ngal=self.ngal+self.zs*0.,corr="max")
        hcos1.add_battaglia_profile("electron",family="AGN",xmax=20,nxs=5000)

        pge_1h = hcos1.get_power_1halo(name="g", name2="electron")[self.index,:][0,:]
        pge_2h = hcos1.get_power_2halo(name="g", name2="electron")[self.index,:][0,:]
        pge = pge_1h + pge_2h

        if not self.small_scale:
            np.savetxt(os.path.join(self.data_folder, "pge.txt"), pge, fmt='%.10e',comments='')
        else:
            np.savetxt(os.path.join(self.data_folder, "pge_s.txt"), pge, fmt='%.10e',comments='')
        return pge


    def get_pgg(self):
        hcos1 = hm.HaloModel(self.zs,self.kk,ms=self.ms)
        hcos1.add_hod("g",ngal=self.ngal+self.zs*0.,corr="max")

        pgg_1h = hcos1.get_power_1halo(name="g")[self.index,:][0,:]
        pgg_2h = hcos1.get_power_2halo(name="g")[self.index,:][0,:]
        pgg = pgg_1h + pgg_2h
        if not self.small_scale:
            np.savetxt(os.path.join(self.data_folder, "pgg.txt"), pgg, fmt='%.10e',comments='')
        else:
            np.savetxt(os.path.join(self.data_folder, "pgg_s.txt"), pgg, fmt='%.10e',comments='')
        return pgg
     

    def get_pee(self):
        hcos1 = hm.HaloModel(self.zs,self.kk,ms=self.ms)
        hcos1.add_battaglia_profile("electron",family="AGN",xmax=20,nxs=5000)

        pee_1h = hcos1.get_power_1halo(name="electron")[self.index,:][0,:]
        pee_2h = hcos1.get_power_2halo(name="electron")[self.index,:][0,:]
        pee = pee_1h + pee_2h
        if not self.small_scale:
            np.savetxt(os.path.join(self.data_folder, "pee.txt"), pee, fmt='%.10e',comments='')
        else:
            np.savetxt(os.path.join(self.data_folder, "pee_s.txt"), pee, fmt='%.10e',comments='')
        return pee


    ####### Cl
    def nncldd_int(self, el):
        """
        caculate integral part of Cldd, get Cldd in (pc/cm^3)^2 after multiply ne0(m^-3)
        ell is a given value
        """

        z_int = np.linspace(0.,self.zfrb,20)  # len = 20
        comoving_d = cosmo.comovingDistance(z_min=0.,z_max=z_int[1:])/h   #len = 19
        kl = np.flip(el/comoving_d)    #len = 19
        dzint = np.diff(z_int)[5]

        hcos = hm.HaloModel(z_int,kl,ms=self.ms)
        hcos.add_battaglia_profile("electron",family="AGN",xmax=20,nxs=5000)
        pee_full = hcos.get_power("electron","electron",verbose=False )[1:,:]    #shape:(20,19) changes to (19,19)
        pee_diag = np.diagonal(pee_full)
        pee = pee_diag.reshape(-1, 1)

        z_chi_h = (1+z_int[1:])**2/comoving_d**2*c/cosmo.Hz(z_int[1:])
        zchih = z_chi_h#[np.newaxis, :]

        result = 0
        for i in range(len(z_int)-1):
            result = result+zchih[i]*pee[i]*dzint

        return result


    def get_cldd(self):
        cldd = np.array([self.nncldd_int(l)*ne0**2 for l in self.ell])
        if not self.small_scale:
            np.savetxt(os.path.join(self.data_folder, "cldd.txt"), cldd, fmt='%.10e',comments='')
        else:
            np.savetxt(os.path.join(self.data_folder, "cldd_s.txt"), cldd, fmt='%.10e',comments='')
        return cldd


    def get_clgg(self):
        pgg = self.get_pgg()
        clgg = pgg/self.chig**2/self.dchig
        if not self.small_scale:
            np.savetxt(os.path.join(self.data_folder, "clgg.txt"), clgg, fmt='%.10e',comments='')
        else:
            np.savetxt(os.path.join(self.data_folder, "clgg_s.txt"), clgg, fmt='%.10e',comments='')
        return clgg


    def get_cldg(self):
        pge = self.get_pge()
        cldg = ne0*(1.+self.zg)/self.chig**2*pge
        if not self.small_scale:
            np.savetxt(os.path.join(self.data_folder, "cldg.txt"), cldg, fmt='%.10e',comments='')
        else:
            np.savetxt(os.path.join(self.data_folder, "cldg_s.txt"), cldg, fmt='%.10e',comments='')
        return cldg


    ###### get results
    def get_pk(self):
        print('now is calculating pk')
        self.get_pgg()
        self.get_pge()
        self.get_pee()

    def get_cl(self):
        print('now is calculating cl')
        self.get_cldd()
        self.get_clgg()
        self.get_cldg()        



class Analysis:
    def __init__(self, fsky, zg, dz, ngal, clt_RMS, clt_beam, small_scale=False, clgg=None, cldd=None, cldg=None, pge=None, pgg=None, data_folder=None):
        self.fsky = fsky
        self.zg = zg
        self.dz = dz
        self.ngal = ngal
        self.clt_RMS = clt_RMS
        self.clt_beam = clt_beam
        self.small_scale = small_scale

        self.V = Func.Vs(self.fsky, self.zg - self.dz, self.zg + self.dz)
        self.kmin = Func.get_kmin(self.V)
        self.kmax = 10.     #Func.get_kmax(self.dz)
        self.bg = Func.biasg(self.zg)
        self.chig = cosmo.comovingDistance(0.,self.zg)/h
        self.dchig = cosmo.comovingDistance(z_min=self.zg-self.dz,z_max=self.zg+self.dz)/h

        self.ksmax = 10.      # /Mpc
        self.ksmin = 0.1

        if not small_scale:
            self.kk = np.geomspace(self.kmin,self.kmax,200)
        else:
            self.kk = np.geomspace(self.ksmin,self.ksmax,200)

        self.ell = self.chig*self.kk

        self.data_folder, _ = Func.paths(self.zg)


    def load_data(self):
        if not self.small_scale:
            self.cldg = np.loadtxt(os.path.join(self.data_folder, "cldg.txt"), delimiter=' ', dtype='str').astype(float)
            self.clgg = np.loadtxt(os.path.join(self.data_folder, "clgg.txt"), delimiter=' ', dtype='str').astype(float)
            self.cldd = np.loadtxt(os.path.join(self.data_folder, "cldd.txt"), delimiter=' ', dtype='str').astype(float)
            self.pge = np.loadtxt(os.path.join(self.data_folder, "pge.txt"), delimiter=' ', dtype='str').astype(float)
            self.pgg = np.loadtxt(os.path.join(self.data_folder, "pgg.txt"), delimiter=' ', dtype='str').astype(float)
        else:
            self.cldg = np.loadtxt(os.path.join(self.data_folder, "cldg_s.txt"), delimiter=' ', dtype='str').astype(float)
            self.clgg = np.loadtxt(os.path.join(self.data_folder, "clgg_s.txt"), delimiter=' ', dtype='str').astype(float)
            self.cldd = np.loadtxt(os.path.join(self.data_folder, "cldd_s.txt"), delimiter=' ', dtype='str').astype(float)
            self.pge = np.loadtxt(os.path.join(self.data_folder, "pge_s.txt"), delimiter=' ', dtype='str').astype(float)
            self.pgg = np.loadtxt(os.path.join(self.data_folder, "pgg_s.txt"), delimiter=' ', dtype='str').astype(float)
    
    #### noise
    def nldd(self, sigmad, N):
        """
        caculate DM noise power spectrum, sigmad can be 100, 300, 1000 pc/cm^3
        nf2d is the number density (per steradian) of FRBs
        return Nl in (pc/cm^3)^2
        """
        omega = 4.*np.pi*self.fsky
        nf2d = N/omega
        return sigmad**2/nf2d

    def nlgg(self):
        omega1 = 4.*np.pi*self.fsky
        ng2d = self.ngal*self.V/omega1
        return 1./ng2d

    def nldg2(self,nlgg,nldd):
        self.load_data()
        return (self.clgg+nlgg)*(self.cldd+nldd)


    ############ SNR
    def snr2(self, nldg2):
        self.load_data()
        omega = 4*np.pi*self.fsky
        dell = np.diff(self.ell)
        result = 0
        for k in range(len(self.ell)-1):
            result = result+self.ell[k]/2/np.pi*dell[k]*self.cldg[k]**2/nldg2[k]
        #result = np.trapz(self.ell/2./np.pi*self.cldg**2/nldg2, self.ell)
        return result/omega**(1/2)

    def get_snr(self):
        """
        'small scale' = False(as default)
        """ 
        SNR = np.zeros((len(sigmaD),len(Nfrb)))
        nl_gg = self.nlgg()
        for i in range(len(sigmaD)):
            for j in range(len(Nfrb)):
                nl_dd = self.nldd(sigmad=sigmaD[i],N=Nfrb[j])
                nl_dg2 = self.nldg2(nl_gg, nl_dd)
                SNR[i,j] = self.snr2(nl_dg2)
        
        snr100 = SNR[0,:]
        snr300 = SNR[1,:]
        snr1000 = SNR[2,:]
        np.savetxt(os.path.join(self.data_folder, "snr100.txt"), snr100, fmt='%.10e',comments='')
        np.savetxt(os.path.join(self.data_folder, "snr300.txt"), snr300, fmt='%.10e',comments='')
        np.savetxt(os.path.join(self.data_folder, "snr1000.txt"), snr1000, fmt='%.10e',comments='')

    ######## uncertainty of Pge

    def pge_err(self, kint, nldg2):
        """
        normal k
        return Mpc^3
        """
        omega = 4*np.pi*self.fsky
        per = self.chig/ne0/(1.+self.zg)/omega**(1/2)
        dkint = np.diff(kint)

        result = 0
        for i in range(len(kint)-1):
            result = result+kint[i]/2/np.pi*dkint[i]/nldg2[i]

        return per/result**(1/2)

    def get_pge_err(self):
        self.load_data()
        kbin_num = 40
        k_int = self.kk.reshape((kbin_num, int(len(self.kk)/kbin_num)))    #(20,10)
        row_diff = np.ptp(k_int, axis=-1)
        pge_y = self.pge.reshape((kbin_num, int(len(self.kk)/kbin_num)))
        nl_dd = self.nldd(sigmad=300,N=1e+4)
        nl_gg = self.nlgg()
        nl_dg2_full = self.nldg2(nl_gg, nl_dd)
        nl_dg2_int = nl_dg2_full.reshape((kbin_num, int(len(self.kk)/kbin_num)))
        dpge = np.zeros(kbin_num)
        for i in range(kbin_num):
            dpge[i] = self.pge_err(kint=k_int[i,:],nldg2=nl_dg2_int[i,:])
        np.savetxt(os.path.join(self.data_folder, "dpge.txt"), dpge, fmt='%.10e',comments='')


    ####### /sigma(bv)

    def get_clT(self):
        lllmin = (min(self.ell) // 100 ) * 100       # 取整
        lllmax = math.ceil(max(self.ell) / 1000) * 1000

        # planck 2018
        params = camb.CAMBparams()
        params.set_cosmology(H0=cosmo.H0, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)   
        params.InitPower.set_params(ns=0.965, r=0, pivot_scalar=0.05, pivot_tensor=0.05)
        params.set_for_lmax(lllmax, lens_potential_accuracy=0)

        el_l = np.linspace(lllmin,lllmax,300).astype(int)
        results = camb.get_results(params)
        tt_l = results.get_cmb_power_spectra(params,CMB_unit='muK', raw_cl=True)['total'][el_l, 0]
        tt_l = np.array(tt_l)
        Func._tt = interp1d(el_l, tt_l, kind='cubic')

        ell_value = self.ell
        ctt = Func._tt(ell_value)
        return ctt

    def clT_noise(self):
        return (self.clt_RMS*np.pi/10800.)**2 * np.exp(self.ell*(self.ell+1.)*(self.clt_beam*np.pi/10800.)**2/8./math.log(2)) 

    def clT_tot(self):
        return self.get_clT()+self.clT_noise()

    def Fks(self):
        """
        ALL NEED IN ks
        ks, pge, pgg, clt are lists, and have same length ==len(ks)
        return F(ks) is also a list
        """
        clt_tot = self.clT_tot()
        self.load_data()
        pgg_tot = self.pgg+1./self.ngal
        return self.kk*self.pge/pgg_tot/clt_tot

    def Gks(self,nldg2):
        """
        ALL NEED IN ks
        """
        omega = 4*np.pi*self.fsky
        per = (ne0*(1.+self.zg)/self.chig)**2*omega/2./np.pi
        return per*self.kk/nldg2

    def bv_err(self,nldg2):
        """
        ALL NEED IN ks
        """
        # self.load_data()
        # dks = np.diff(self.kk)
        # fks = self.Fks()
        # gks = self.Gks(nldg2)
        # #print(len(fks),len(gks))
        # b1 = np.sum(dks*fks[:-1]**2/gks[:-1])
        # b2 = np.sum(dks*fks[:-1]*self.pge[:-1])

        self.load_data()
        fks = self.Fks()
        gks = self.Gks(nldg2)
        b1 = np.trapz(fks**2/gks, self.kk)
        b2 = np.trapz(fks*self.pge, self.kk)

        return b1/b2**2

    def get_bv_err(self):
        print('now is calculating bv_err')
        sigma_bv = np.zeros((len(sigmaD),len(Nfrb)))

        for i in range(len(sigmaD)):
            for j in range(len(Nfrb)):
                nl_dd_ks = self.nldd(sigmad=sigmaD[i],N=Nfrb[j])
                nl_gg_ks = self.nlgg()
                nl_dg2_ks = self.nldg2(nl_gg_ks,nl_dd_ks)
                sigma_bv[i,j] = self.bv_err(nldg2=nl_dg2_ks)
        bv100 = np.sqrt(sigma_bv[0,:])
        bv300 = np.sqrt(sigma_bv[1,:])
        bv1000 = np.sqrt(sigma_bv[2,:])
        np.savetxt(os.path.join(self.data_folder, "bv100.txt"), bv100, fmt='%.10e',comments='')
        np.savetxt(os.path.join(self.data_folder, "bv300.txt"), bv300, fmt='%.10e',comments='')
        np.savetxt(os.path.join(self.data_folder, "bv1000.txt"), bv1000, fmt='%.10e',comments='')



    ##### Nvv
    def ksz_radial_function(self):
        """
        K(z) = - T_CMB sigma_T n_e0 x_e(z) exp(-tau(z)) (1+z)^2
        Eq 4 of 1810.13423
        """
        xe = 1.
        tau=0
        T_CMB_muk = 2.7e+6 # muK
        thompson_SI = 6.6524e-29           #constants['thompson_SI']
        meterToMegaparsec = 3.241e-23      #constants['meter_to_megaparsec']
        return T_CMB_muk*thompson_SI*ne0*(1.+self.zg)**2./meterToMegaparsec  * xe  *np.exp(-tau)

    def Nvv_no_mu(self):
        """
        Mpc^3
        """
        # self.load_data()
        # clt_tot = self.clT_tot()
        # pgg_tot = self.pgg+1./self.ngal
        # dkk = np.diff(self.kk)
        # nvvint = np.sum(self.kk[1:]*dkk/2/np.pi*self.pge[1:]**2/pgg_tot[1:]/clt_tot[:1:])

        self.load_data()
        clt_tot = self.clT_tot()
        pgg_tot = self.pgg+1./self.ngal
        nvvint = np.trapz(self.kk/2./np.pi*self.pge**2/pgg_tot/clt_tot, self.kk)
        nvvnomu = self.chig**2/self.ksz_radial_function()**2/nvvint
        print('nvv without mu is:', nvvnomu)
        return nvvnomu  
    








##### Fisher forecast for only one galaxy survey, and we can get almost same results from these two methods.

def fisher_RSD(zg,dz,fsky,ngal):
    """
    ngal : h/Mpc^3
    dz : zg-dz, zg+dz
    """

    bg=Func.biasg(zg)
    f = Func.growthrate(z=zg)
    s8 = Func.sigma(z=zg)
    v = Func.Vs_h(fsky, zg-dz, zg+dz)    # Mpc/h ^3
    kmin = Func.get_kmin_h(v)    # h/Mpc
    kmax = Func.get_kmax_h(zg)    # h/Mpc
    #print(kmin,kmax)

    def damp(k,mu):
        c = 3e+5
        sz = 0.002
        sd = c * sz*(1.+zg) / cosmo.Hz(zg) * h
        return np.exp(-k**2 * mu**2 * sd**2)

    def pgg(k, mu):
        return (bg + f*mu**2)**2 * cosmo.matterPowerSpectrum(k, zg) * damp(k,mu) + 1./ngal

    def npnp(k, mu):
        return (ngal*pgg(k,mu)/(1.+ngal*pgg(k,mu)))**2
    def dpdbs8(k,mu):    #1
        return (2.*bg*s8)/(bg*s8+f*s8*mu**2)
    def dpdfs8(k, mu):   #2
        return (2.*f*s8*mu**2)/(bg*s8+f*s8*mu**2)

    def f11(k,mu):
        return 0.5 * npnp(k,mu) * dpdbs8(k,mu)**2 * k**2
    def f22(k,mu):
        return 0.5 * npnp(k,mu) * dpdfs8(k,mu)**2 * k**2
    def f12(k,mu):
        return 0.5 * npnp(k,mu) * dpdbs8(k,mu)*dpdfs8(k,mu) * k**2


    k_range = (kmin, kmax)
    int_11, _ = quad(lambda k: quad(lambda mu: f11(k, mu), -1, 1)[0], *k_range)
    int_22, _ = quad(lambda k: quad(lambda mu: f22(k, mu), -1, 1)[0], *k_range)
    int_12, _ = quad(lambda k: quad(lambda mu: f12(k, mu), -1, 1)[0], *k_range)
     
    f11 = v/4./np.pi**2 * int_11
    f22 = v/4./np.pi**2 * int_22
    f12 = v/4./np.pi**2 * int_12
    fisher_matrix = np.array([[f11,f12],[f12,f22]])
    cov_matrix =[[a,b],[c,d]]=np.linalg.inv(fisher_matrix)
    print(zg)
    print('σ(σ8f)=',d**0.5)
    print('fisher matrix of RSD:is:', fisher_matrix)
    return fisher_matrix, d**0.5
    

##### 2 tracers
def fisher_ksz(zg, fsky, dz, bg, bv, ngal, nvv):

    c_speed = 3e+5    # km/s
    nlim = 1000
    h = cosmo.H0/100.

    v = Func.Vs_h(fsky, zg-dz, zg+dz)
    kmin = Func.get_kmin_h(v)
    kmax = Func.get_kmax_h(zg)

    f = Func.growthrate(z=zg)
    s8 = Func.sigma(z=zg)

    def pgg(k):
        return bg**2 * cosmo.matterPowerSpectrum(k,zg)

    def pvv(k):
        return bv**2 * (f/(1.+zg)*cosmo.Hz(zg)/c_speed/k/h)**2 * cosmo.matterPowerSpectrum(k,zg)

    def pgv(k):
        return bg * bv * (f/(1.+zg)*cosmo.Hz(zg)/c_speed/k/h) * cosmo.matterPowerSpectrum(k,zg)

    def get_c(k,mu):
        c_mat = np.zeros((2, 2))
        c_mat[0,0] = pgg(k) + 1./ngal
        c_mat[1,1] = pvv(k) + nvv/mu**2
        c_mat[0,1] = pgv(k)
        c_mat[1,0] = pgv(k)
        return np.linalg.inv(c_mat)

    def Dt(X,i,k):
        if X=='g':
            if i==1:   #bs8
                return 2. * bg*s8 * cosmo.matterPowerSpectrum(k,0.)/ cosmo.sigma8**2
            if i==2:   #fs8
                return 0.
            if i==3:   #bv
                return 0.
        if X=='v':
            if i==1:
                return 0.
            if i==2:
                return 2. * bv**2 * f*s8 * (1./(1.+zg)*cosmo.Hz(zg)/c_speed/k/h)**2 * cosmo.matterPowerSpectrum(k,0.) / cosmo.sigma8**2
            if i==3:
                return 2. * bv * (f*s8)**2 * (1./(1.+zg)*cosmo.Hz(zg)/c_speed/k/h)**2 * cosmo.matterPowerSpectrum(k,0.) / cosmo.sigma8**2
        if X=='gv':
            if i==1:
                return bv * f*s8 * (1./(1.+zg)*cosmo.Hz(zg)/c_speed/k/h) * cosmo.matterPowerSpectrum(k,0.) / cosmo.sigma8**2
            if i==2:
                return bv * bg*s8 * (1./(1.+zg)*cosmo.Hz(zg)/c_speed/k/h) * cosmo.matterPowerSpectrum(k,0.) / cosmo.sigma8**2 
            if i==3:
                return f*s8 * bg*s8 * (1./(1.+zg)*cosmo.Hz(zg)/c_speed/k/h) * cosmo.matterPowerSpectrum(k,0.) / cosmo.sigma8**2

    def dc(i,k):
        dc_mat = np.zeros((2, 2))   # call: dc_mat[0,0]
        if i == 1:
            dc_mat[0,0] = Dt('g',1,k)
            dc_mat[1,1] = Dt('v',1,k)
            dc_mat[0,1] = Dt('gv',1,k)
            dc_mat[1,0] = dc_mat[0,1]
            return dc_mat
        if i == 2:
            dc_mat[0,0] = Dt('g',2,k)
            dc_mat[1,1] = Dt('v',2,k)
            dc_mat[0,1] = Dt('gv',2,k)
            dc_mat[1,0] = dc_mat[0,1]
            return dc_mat
        if i == 3:
            dc_mat[0,0] = Dt('g',3,k)
            dc_mat[1,1] = Dt('v',3,k)
            dc_mat[0,1] = Dt('gv',3,k)
            dc_mat[1,0] = dc_mat[0,1]
            return dc_mat

    def get_int(k,mu,m,n):
        dot_mat = np.dot( np.dot( np.dot(dc(m,k), get_c(k,mu)), dc(n,k)), get_c(k,mu))
        return np.trace(dot_mat)* k**2

    def integrat(muint,int_i,int_j):
        return quad(partial(get_int, mu=muint, m=int_i, n=int_j),kmin,kmax,epsrel=0.0000001,epsabs=0,limit=nlim)[0]


    def get_f1():
        f11=v/(8.*np.pi**2)*quad(partial(integrat,int_i=1,int_j=1),-1., -1e-5,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f12=v/(8.*np.pi**2)*quad(partial(integrat,int_i=1,int_j=2),-1., -1e-5,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f13=v/(8.*np.pi**2)*quad(partial(integrat,int_i=1,int_j=3),-1., -1e-5,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f22=v/(8.*np.pi**2)*quad(partial(integrat,int_i=2,int_j=2),-1., -1e-5,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f23=v/(8.*np.pi**2)*quad(partial(integrat,int_i=2,int_j=3),-1., -1e-5,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f33=v/(8.*np.pi**2)*quad(partial(integrat,int_i=3,int_j=3),-1., -1e-5,epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        return np.array([[f11,f12,f13],[f12,f22,f23],[f13,f23,f33]])
    
    def get_f2():
        f11=v/(8.*np.pi**2)*quad(partial(integrat,int_i=1,int_j=1), 1e-5, 1., epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f12=v/(8.*np.pi**2)*quad(partial(integrat,int_i=1,int_j=2), 1e-5, 1., epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f13=v/(8.*np.pi**2)*quad(partial(integrat,int_i=1,int_j=3), 1e-5, 1., epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f22=v/(8.*np.pi**2)*quad(partial(integrat,int_i=2,int_j=2), 1e-5, 1., epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f23=v/(8.*np.pi**2)*quad(partial(integrat,int_i=2,int_j=3), 1e-5, 1., epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        f33=v/(8.*np.pi**2)*quad(partial(integrat,int_i=3,int_j=3), 1e-5, 1., epsrel=0.001,epsabs=0.0,limit=nlim)[0]
        return np.array([[f11,f12,f13],[f12,f22,f23],[f13,f23,f33]])
    
    fish_mat = get_f1()+get_f2()
    print('fisher matrix of ksz and FRB is:', fish_mat)
    return fish_mat



def Add_fisher_gets_fs8(f_ori, f_add):
    """
    f_ori is a 3*3 metrix
    f_add is a number(bv)
    return \sigma(fs8)
    """
    metric_add = np.copy(f_ori)
    metric_add[2,2] += f_add
    cov = np.linalg.inv(metric_add)
    return cov[1,1]**0.5


def Add_2_fisher_get_fs8(f_ori, f_add1, f_add2):
    """
    f_ori(ksz) is a 3*3 metric, [bs8, fs8, bv]
    f_add1 is a number(bv), [bv]
    f_add2(rsd) is a 2*2 metric, [bs8,fs8]
    return \sigma(fs8)
    """
    result = np.copy(f_ori)
    result[2,2] += f_add1
    result[:2,:2] += f_add2
    cov = np.linalg.inv(result)
    return cov[1,1]**0.5
