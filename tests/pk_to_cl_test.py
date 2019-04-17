from __future__ import print_function
import numpy as np
import pylab
import hankel
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d
from scipy.integrate import quad
import sys
sys.path.append("LOG_HT")
from LOG_HT import fft_log
import time

inv_sqrt2pi = 1./np.sqrt(2*np.pi)

def gaussian(x, mu, sigma):
    d = x-mu
    inv_sigma = 1./sigma
    return np.exp(-0.5*d*d*inv_sigma*inv_sigma) * inv_sqrt2pi * inv_sigma

def limber_integrand(chi, ell, kernel_interp, pk0_interp_loglog, growth_interp):
    if chi<1.e-9:
        return 0.
    kernel = kernel_interp(chi)
    nu = ell+0.5
    k = nu/chi
    pk0 = np.exp(pk0_interp_loglog(np.log(k)))
    g = growth_interp(chi)
    pk = pk0 * g * g
    return kernel * kernel * pk / chi / chi

def non_limber_integral(ell, chimin, chimax, nchi, fftlog_kernel_interp1, fftlog_kernel_interp2, pk0_interp_loglog, chi_pad_factor=1):
    """full integral is \int_0^\inf k^{-1} dk P(k,0) I_1(k) I_2(k)
    where I_1(k) = \int_0^{\inf} k dr_1 F_1(r_1) r^{-0.5} D(r_1) J_{l+0.5}(kr_1),
    and F_1(r_1) is the radial kernel for tracer 1.
    We want to use a fftlog for the I_1(k) calculation, so write it in the form 
    I_1(k) = \int_0^{\inf} k dr_1 f_1(r_1) J_{mu}(kr_1).
    So f_1(r_1_) = F_1(r_1) D(r_1) r^{-0.5}, this is what should be passed as the fftlog_kernel_interp1 spline.
    We actually do the integral in log(k), so calculate \int_0^\inf dlogk P(k,0) I_1(k) I_2(k).
    """
    q=0
    log_chimin, log_chimax = np.log(chimin), np.log(chimax)
    log_chimin_padded, log_chimax_padded = log_chimin-chi_pad_factor, log_chimax+chi_pad_factor
    log_chi_vals = np.linspace(log_chimin_padded, log_chimax_padded, nchi+2*chi_pad_factor)
    chi_vals = np.exp(log_chi_vals)
    kernel1_vals = fftlog_kernel_interp1(chi_vals)
    k_vals, I_1 = fft_log(chi_vals, kernel1_vals, q, ell+0.5)
    if fftlog_kernel_interp2 is fftlog_kernel_interp1:
        I_2 = I_1
    else:
        kernel2_vals = fftlog_kernel_interp2(chi_vals)
        _, I_2 = fft_log(chi_vals, kernel2_vals, q, ell+0.5)
    pk_vals = np.exp(pk0_interp_loglog(np.log(k_vals)))
    #Now we can compute the full integral \int_0^{\inf} k dk P(k,0) I_1(k) I_2(k)
    #We are values logspaced in k, so calculate as \int_0^{inf} k^2 dlog(k) P(k,0) I_1(k) I_2(k)
    #integrand_vals = k_vals * k_vals * pk_vals * I_1 * I_2
    integrand_vals = pk_vals * I_1 * I_2
    logk_vals = np.log(k_vals)
    integrand_interp = IUS(logk_vals, integrand_vals)
    integral = integrand_interp.integral(logk_vals.min(), logk_vals.max())
    return integral

def main():

    do_plot=True

    #Read in z, chi, k, P(k,0) and growth D(z)
    z_pk = np.loadtxt("test_output_pkhighres/matter_power_lin/z.txt")
    z_dist = np.loadtxt("test_output_pkhighres/distances/z.txt")
    assert np.allclose(z_pk,z_dist)
    chi = np.loadtxt("test_output_pkhighres/distances/d_m.txt")
    k = np.loadtxt("test_output_pkhighres/matter_power_lin/k_h.txt")
    logk = np.log(k)
    pk0 = np.loadtxt("test_output_pkhighres/matter_power_lin/p_k.txt")[0]
    growth = np.loadtxt("growth.dat")

    #We'll need an interpolator for Pk0
    #Do this in log-log
    pk0_interp_loglog = IUS(np.log(k), np.log(pk0))

    #And one for growth(chi) I guess
    growth_interp = interp1d(chi, growth, bounds_error=False, fill_value=0.)

    #define a Gaussian kernel in chi
    #e.g. corresponding to z_mean=0.5, sigma_z=0.1 and also sigma_z=0.05
    z_mean=0.5
    if do_plot:
        fig=pylab.figure()
        ax = fig.add_subplot(111)

    for sigma_z in [0.1, 0.05]:
        chi_of_z_interp = IUS(z_pk, chi)
        chi_mean = chi_of_z_interp(z_mean)
        sigma_chi = sigma_z * chi_of_z_interp.derivative()(z_mean)
        print("chi_mean, sigma_chi:")
        print(chi_mean, sigma_chi)
        #Set chi_min/max +/- 4 sigma from mean
        chi_min, chi_max = max(chi.min(), chi_mean-4*sigma_chi), min(chi.max(), chi_mean+4*sigma_chi)
        print("chi_min, chi_max:")
        print(chi_min, chi_max)

        #make interpolators for kernels
        raw_kernel = gaussian(chi, chi_mean, sigma_chi)
        raw_kernel_interp = interp1d(chi, raw_kernel, bounds_error=False, fill_value=0.)
        #fftlog kernel is raw_kernel * chi^{-1/2} * growth (see comments in non_limber_integral function)
        fftlog_kernel = raw_kernel * growth * chi**-0.5
        fftlog_kernel[0] = 0.
        fftlog_kernel_interp = interp1d(chi, fftlog_kernel, bounds_error=False, fill_value=0.)

        #loop through ells computing C(l) using Limber and full calculation
        ells = np.linspace(1,100,50)
        #numerical settings for full calculation:
        chi_pad_factor = 5
        nchi = 5000

        cls_limber = np.zeros_like(ells)
        cls_full = np.zeros_like(ells)
        for i,ell in enumerate(ells):
            print("ell:",ell)
            cl_limber,_ = quad(limber_integrand, chi_min, chi_max, 
                             args = (ell, raw_kernel_interp, pk0_interp_loglog, growth_interp))
            cls_limber[i] = cl_limber
            cl_full = non_limber_integral(ell, chi_min, chi_max, nchi, fftlog_kernel_interp, fftlog_kernel_interp, 
                                          pk0_interp_loglog, chi_pad_factor=chi_pad_factor)
            cls_full[i] = cl_full

        if do_plot:
            ax.plot(ells, cls_full/cls_limber-1, label="sigma_z = %.2f"%sigma_z)
    if do_plot:
        ax.legend()
        ax.set_xlabel(r"$l$")
        ax.set_ylabel(r"$C_l^{\mathrm{full}}/C_l^{\mathrm{limber}}-1$")

    fig.savefig("pk_to_cl_test1.png")











if __name__=="__main__":
    main()



