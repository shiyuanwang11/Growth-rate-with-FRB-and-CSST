import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from getdist import loadMCSamples


# Define likelihood
def my_like(
        # Parameters to sample over
        omegam=0.3,
        sigma8=0.8,
        # Cobaya instance
        _self=None):
    # Load Planck MCMC samples (replace path with actual file location)
    samples = loadMCSamples('/home/wangsy2/software/MGCobaya/chains_LCDM_t3/mugamma')
    omegam_values = samples.samples[:, samples.index['omegam']]
    sigma8_values = samples.samples[:, samples.index['sigma8']]
    sigma8_fid = samples.mean('sigma8')
    omegam_fid = samples.mean('omegam')

    # Observational data and errors
    zz = np.array([0.15, 0.45, 0.75])
    err_rsd = np.array([0.037622781639503125,0.013120288894571081,0.00801293154199452])
    #err_rsd_ksz = np.array([0.0048570460593981195, 0.0029804102835748773, 0.0028959660223023967])

    # Define helper functions
    def Ez(z, Om0):
        return np.sqrt(Om0 * (1 + z)**3 + 1 - Om0)

    def Ea(a, Om0):
        return np.sqrt(Om0 / a**3 + 1 - Om0)

    def growthrate(Om0, z):
        def model(y, x, Om0=Om0):
            dydx = y[1]
            c = Om0 / x**4
            A = 3. / 2. * c / Ea(x, Om0)**2 - 3. / x
            B = 3. / 2. * Om0 / x**5 / Ea(x, Om0)**2
            d2ydx2 = A * y[1] + B * y[0]
            return [dydx, d2ydx2]

        x = np.linspace(1e-3, 1., 500)  # x: a
        y0 = [1e-3, 1.]  # Initial conditions
        y = odeint(model, y0, x)  # D(a)
        func = interp1d(x, y[:, 0] / y[-1, 0], kind='cubic')  # D(a)
        interp_D = func(x)
        lnD = np.log(interp_D)
        dlnDdlna = x * np.gradient(lnD, x)
        func_f = interp1d(x, dlnDdlna, kind='cubic')
        return func_f(1. / (1. + z))

    def sigma(sigma80, Om0, z):
        def model(y, x, Om0=Om0):
            dydx = y[1]
            c = Om0 / x**4
            A = 3. / 2. * c / Ea(x, Om0)**2 - 3. / x
            B = 3. / 2. * Om0 / x**5 / Ea(x, Om0)**2
            d2ydx2 = A * y[1] + B * y[0]
            return [dydx, d2ydx2]

        x = np.linspace(1e-3, 1., 500)  # x: a
        y0 = [1e-3, 1.]  # Initial conditions
        y = odeint(model, y0, x)  # D(a)
        func = interp1d(x, y[:, 0] / y[-1, 0], kind='linear')
        return func(1. / (1. + z)) * sigma80

    def fsigma8(Om0, sigma80, z):
        return growthrate(Om0, z) * sigma(sigma80, Om0, z)

    def da(z_array, omegam):
        results = []
        for z in z_array:
            int_z = np.linspace(0, z, 50)
            E = Ez(int_z, omegam)
            integral = np.trapz(1. / E, int_z)
            results.append(integral)
        return np.array(results)

    def ratio(z, Om0):
        return Ez(z, Om0) * da(z, Om0) / (Ez(z, omegam_fid) * da(z, omegam_fid))

    # Compute fsigma8 data and theory
    fsigma8_data = fsigma8(omegam_values[-1], sigma8_values[-1], zz)
    fsigma8_theory = ratio(zz, omegam) * fsigma8(omegam, sigma8, zz)

    # Log-likelihood calculation
    logp = -0.5 * np.sum(((fsigma8_data - fsigma8_theory) / err_rsd) ** 2)



    return logp
