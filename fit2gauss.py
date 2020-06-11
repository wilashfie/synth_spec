import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import find_peaks


def single_gauss_func_noder(x, *a):
    # evaluate the single gaussian on x and return its value in f

    # === the exponential factors
    ef1 = np.exp( -0.5*( (x-a[1])/a[2] )**2 )

    f = a[0]*ef1

    return f


def double_gauss_func_noder(x, *a):
    # evaluate the double gaussian on x and return its value in f

    # === the exponential factors
    ef1 = np.exp( -0.5*( (x-a[1])/a[2] )**2 )
    ef2 = np.exp( -0.5*( (x-a[4])/a[5] )**2 )

    f = a[0]*ef1 + a[3]*ef2

    return f


def est_params(mvec, sig=0.1, dx=1.):
#  give moments 0 - 3, as elements of mvec, estimate the parameters of 2 gaussians


    s = mvec[3]/mvec[2]**(1.5) # the skewness
    th = ( np.arctan(0.5*s) > (-0.49*np.pi)) < ( 0.49*np.pi )
    f = 0.5 - 0.5*np.sin(th) # fraction of red-shifted component
    dv = np.sqrt(mvec[2]/f/(1.0-f)) # estimate of separation  v_red - v_blue
    #mvec[1] = f*v_red + (1.0-f)*v_blue = v_blue + f*dv
    v_blue = mvec[1] - f*dv
    v_red = dv + v_blue

    a0 = dx*mvec[0]/(np.sqrt(2*np.pi*mvec[2]))

    est = [f*a0, v_red, sig, (1.0-f)*a0, v_blue, sig]

    return est




def fit2gauss(lam, y, yerr, min_tot=0.1, chi_thr=10.0, base=0.0, verbose=False):

    # min_tot= minimum intensity to try
    # chi_thr= Chi^2 threshold
    # base= base level subtracted when computing moments

    # ==== compute moments

    d = dict()

    yt = ( y - base ) #> 0.0 #    a truncated version for moments
    m0 = np.sum( yt ) #> 1.0 #    prevent problems with division
    m1 = np.sum( yt*lam )/m0
    m2 = np.sum( yt*(lam-m1)**2 )/m0
    m3 = np.sum( yt*(lam-m1)**3 )/m0

    moms = [ m0, m1, m2 ] #  pack for return

    # if first moment is too small (nothing to fit)
    if( m0 < min_tot ):
        chi1g = -1.0
        a1g = [0,0,0]
        d['moms'] = moms
        d['chi1g'] = chi1g
        d['a2g'] = [0,0,0,0,0,0]
        if verbose==True: print('ejecting!')
        return d

    # ===== do 1-Gaussian fit
    # estimate parameters for 1-Gaussian fit
    sd = np.sqrt(m2)
    dx = lam[1]-lam[0]
    a0 = [dx*m0/(np.sqrt(2*np.pi)*sd), m1, sd] #  estimate of 1-gaussian paramters

    #y1g = gaussfit( lam, y, a1g, chisq=chi1g, nterm=3, meas=yerr, est=a0 )
    a1g,a1cov = curve_fit(single_gauss_func_noder, lam, y, p0=a0)#, bounds=(0, np.inf))
    y1g = single_gauss_func_noder(lam, *a1g)

    #calculate chi_square
    #jj = np.where((lam > 1402) & (lam < 1405)) # trim excess zeros in data first
    #jj = np.where(y > 0.001)
    #lam_s = lam[jj]
    #y_s = y[jj]
    y_modelone = single_gauss_func_noder(lam, *a1g)
    #y_modelone = y1g[jj]
    #yerr_s = yerr[jj]
    X2one = np.sum(((y_modelone - y) / yerr)**2)

    #X2_1 = np.sum(((y1g-y)/yerr)**2)
    chi1g = X2one/(len(y)-3) # reduced chi^2



    # ==== do double-Gaussian fit
    # estimate parameters
    a0_2 = est_params([ m0, m1, m2, m3 ], dx=dx)
    if verbose==True: print('est params = ', a0_2)

    spec_sm = savgol_filter(y, 23, 1) #smooth to make local peak finding more accurate
    peaks, _ = find_peaks(spec_sm)

    pos_peaks = lam[peaks]
    spec_peaks = spec_sm[peaks]
    iis = np.where(spec_peaks>1)
    iis = iis[0]

    if (len(iis)>2) and (verbose == True): print('!!!! - more than two peaks found') # as a precaution

    if len(iis)<2:     # redo fitering to see if we can't get two peaks. if not, we'll call it a single gaussian.
        if verbose == True: print('single peak found')
        spec_sm = savgol_filter(y, 3, 1) #15->3?
        peaks, _ = find_peaks(spec_sm)
        pos_peaks = lam[peaks]
        spec_peaks = spec_sm[peaks]
        iis = np.where(spec_peaks>1)
        iis = iis[0]
        iis = np.append(iis,iis[0]+25) # add one more index for fitting purposes.

    amp_peaks = spec_peaks[iis] # amplitude and position of peaks
    pos_peaks = pos_peaks[iis]
    a0_2[0],a0_2[1],a0_2[3],a0_2[4] = amp_peaks[0],pos_peaks[0],amp_peaks[1],pos_peaks[1]

    if verbose==True: print('new init params = ', a0_2)
    upper_bound = [np.inf,1404,np.inf,np.inf,1404,np.inf]
    lower_bound = [0,1403,0,0,0,0]
    #a2g, a2cov = curve_fit(double_gauss_func_noder, lam, y, p0=a0_2, sigma = yerr, absolute_sigma = True, maxfev = 110000)#, bounds=(lower_bound, upper_bound)) #,
    #a2g,a2cov = curve_fit(double_gauss_func_noder, lam, y, p0=a0_2, maxfev = 2000, bounds = (0, np.inf)) # no sig
    #a2g,a2cov = curve_fit(double_gauss_func_noder, lam, y, p0=a0_2, bounds = (0, np.inf)) # no sig
    a2g,a2cov = curve_fit(double_gauss_func_noder, lam, y, p0=a0_2)

    # individual gaussians of double fit
    pars_1 = a2g[0:3]
    pars_2 = a2g[3:6]
    y2a = single_gauss_func_noder(lam, *pars_1)
    y2b = single_gauss_func_noder(lam, *pars_2)


    # calculate chi^2
    #y_modeltwo = double_gauss_func_noder(lam_s, *a2g)
    y2g = y2a+y2b
    y_modeltwo = y2g
    X2two = np.sum(((y_modeltwo - y) / yerr)**2)
    chi2g = X2two/(len(y)-6) # reduced chi^2

    if verbose==True:
        print('a2g =', a2g)
        print('a1g[0] =', a1g[0])
        print('chi2g = ', chi2g)



    # if double fitting WORSE than single gaussian fit, OR OR OR if the amplitude of the second Gaussian is neglible
    #if( chi2g > chi1g ) or (a2g[0] < a1g[0]*0.01):
    small_amp = np.minimum(a2g[0],a2g[3])
    if(small_amp < a1g[0]*0.01):
        a2g = np.concatenate((a1g, a1g)) #  return copies of single fit params
        a2g[3] = 0.0 #  but zero amplitude
        y2a = y1g
        y2b = 0.0*y1g
        chi2g = -1.0 #  flag that no fit was attmepted

    if verbose==True:
        print('a2g = ', a2g)
        print('chi1g = ',chi1g)
        print('chi2g = ', chi2g)


    d['moms'] = moms
    d['a1g'] = a1g
    d['y1g'] = y1g
    d['a2g'] = a2g
    d['y2a'] = y2a
    d['y2b'] = y2b
    d['chi2g'] = chi2g

    return d
