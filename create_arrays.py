import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from scipy.interpolate import interp1d
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from matplotlib import gridspec

from fit2gauss import fit2gauss


# creates syntetic Si IV - 1402.77 ang. line from PREFT simulation


# inputs: tube, frac, logT, log10G, time (optional)
# tube - tarr array from PREFT sim, imported into python notebook using scipy.readsav()
# frac - ionization fraction for Si IV, created using IDL function on server
# log10T/log10G, log of contribution function G and corresponding log of temperature T, imported from dana's email

# set inital values
line = 1402.77
mass = 28.0*1.66054e-27
kb = 1.3807e-23
c = 300
h = 6.62607e-27
flux=1.0e3 #? why this val?
ll = np.arange(line-10,line+10,0.01)


class siiv:

    def __init__(self,tube,frac,log10T,log10G,time=55,verbose=False, interp = 'linear'):

        global TUBE,FRAC,LOG10T,LOG10G
        TUBE = tube
        FRAC = frac
        LOG10T = log10T
        LOG10G = log10G

        # define arrays from tube.tarr
        t = tube.tarr.t[time]
        n = tube.tarr.n[time]
        los_v = tube.tarr.v[time].T[0]
        sm_v = -los_v
        los_x = tube.tarr.x[time].T[0]
        n_e = tube.tarr.epamu[time]*tube.tarr.rho[time]/1.67e-8 # number density
        b = tube.tarr.b[time]
        dl_e = tube.tarr.dl_e[time]

        self.t_n = tube.tarr.shape[0] - 1 # len of tube in time (in steps of 0.1s)

        # interpolate our GofT data
        te = 10**log10T
        inter = interp1d(te,log10G,kind='cubic', bounds_error=False, fill_value=-10e6) #fill outide vals with large, small number
        temp = 1e6*tube.tarr.t[time]
        G = inter(temp)

        # set all log10G with low temp to large, small number (st 10^G~0)
        t_0 = tube.tarr.t[0]
        temp_0 = 1e6*np.round(t_0[0],4)
        G[temp<temp_0] = -10000

        # nei/eqi arrays at time =time
        f_nei=frac.arrs.f_nei[0]
        f_nei = f_nei[:,time]
        f_eqi=frac.arrs.f_eqi[0]
        f_eqi = f_eqi[:,time]

        temp_fac = f_nei/f_eqi
        np.nan_to_num(temp_fac, copy=False, nan=1); # replace inf values with 1 (due to zeros in eqi)
        i_half = int(n/2) #[0:i_half] = left half of tube
        test = f_nei[0:i_half]
        nei_jj = np.where(test > test[0])
        nei_jj=nei_jj[0]

        i_half = int(n/2) #[0:i_half] = left half of tube
        temp_fac = temp_fac[0:i_half]
        f_jj = np.where(temp_fac > temp_fac[0])
        f_jj=f_jj[0]

        if (len(f_jj) == 0): f_jj = nei_jj

        # interpolation arrays
        # define subregion
        #i_min,i_max = f_jj[0]-30,f_jj[-1]+30 # left half of tube where nei is significant
        #i_min,i_max = 350,1850 # fixed interval for averaging in time (for n=400)
        i_min,i_max = 250,i_half-2 # fixed interval for averaging in time (for n=200)

        t_s = t[i_min:i_max]
        n_s = len(t_s)
        los_v_s = los_v[i_min:i_max]
        sm_v_s = sm_v[i_min:i_max]
        los_x_s = los_x[i_min:i_max]
        n_e_s = n_e[i_min:i_max]
        b_s = b[i_min:i_max]
        dl_e_s = dl_e[i_min:i_max]
        G_s = G[i_min:i_max]
        f_nei_s = f_nei[i_min:i_max]
        f_eqi_s = f_eqi[i_min:i_max]

        # interpolate
        N=10*n_s
        i_s = np.arange(0,n_s)
        ii = np.arange(0,10*(n_s-1))*0.1
        i_length = len(ii)

        int_x = interp1d(i_s,los_x_s,kind=interp)#,fill_value="extrapolate")
        int_v = interp1d(i_s,sm_v_s,kind=interp)
        int_t = interp1d(i_s,t_s,kind=interp)
        int_ne = interp1d(i_s,n_e_s,kind=interp)
        int_b = interp1d(i_s,b_s,kind=interp)
        int_dl_e = interp1d(i_s,dl_e_s,kind=interp)
        int_G = interp1d(i_s,G_s,kind=interp)
        int_fnei = interp1d(i_s,f_nei_s,kind=interp)
        int_feqi = interp1d(i_s,f_eqi_s,kind=interp)

        # new, interpolated arrays from tarr/tube
        x = int_x(ii)
        v = int_v(ii)
        T = int_t(ii)
        ne = int_ne(ii)
        B = int_b(ii)
        Dl = int_dl_e(ii)
        g = int_G(ii)
        nei = int_fnei(ii)
        eqi = int_feqi(ii)

        # factor to multiply GofT by to get actual emission given NEI
        factor = nei/eqi
        np.nan_to_num(factor, copy=False, nan=1); # replace inf values with 1 (due to zeros in eqi)

        # process to create volume given per flux (per Maxwell)
        rad=np.sqrt(flux/B/np.pi)
        b_e=0.5*( np.roll(B, -1) + B )
        A1=rad
        A2=np.roll(rad,-1)
        A2[(len(B))-1]=A2[len(B)-2]
        #volume=(1./3.)*Dl*(A1+A2+np.sqrt(A1*A2))*1e24
        A = flux/B # replace by area of pixel in Mm (~0.01 Mm^2)
        A_pixel = 0.029 # Mm^2 -  pixel area: 0.33 x 0.167 arcsec (as seen on sun..)
        volume = Dl*A_pixel*1e24
        EM=volume*ne**2 #calcualte emission measure EM


        # calculate prefactor to turn intensity into counts
        photo_erg = h*c*1e6/line*1e10 #erg/photon
        pixel_size = 12.8e-3 # in ˚A [12.8 m˚A (FUV), 25.6 m˚A (NUV)]
        dim = 19e-2
        A_iris = 2.2e-4 # effective area FUV
        au = 1.49e11
        atn = A_iris/au**2
        exp_time = 4.0 # double checked, this is more or less right (3.999, or whatever.)
        photo_fac = pixel_size*atn*exp_time/photo_erg # converts EM*g/sig (erg/s/sr/˚A) -> photon count


        # determine line broadening given combination of thermal, non-thermal, and instrumental broadenings
        freq = c/line*1e3
        v_inst = 3.9 # instrumental broadening (km/s)
        sig_inst = v_inst/freq
        v_nt = 20.0 # non-thermal broadening (km/s) [De Pontieu et al, 2015]
        sig_nt = v_nt/freq
        sig_th = line*np.sqrt(kb*1e6*T/mass)/(c*1e6) # thermal broadening (in ˚A)
        sig = np.sqrt(sig_th**2+sig_nt**2+sig_inst**2) # total broadening

        # create emission array(s) for each fluid element
        nn=10*(n_s-1)
        emissNEI = np.empty([nn,len(ll)])
        emiss = emissNEI
        for i in range(nn):
            emissNEI[i,:] = photo_fac*EM[i]*factor[i]*10**g[i]/np.sqrt(2*np.pi)/sig[i]*np.exp(-(ll-line-line*v[i]/c)**2/(2*sig[i]**2))
            emiss[i,:] = photo_fac*EM[i]*10**g[i]/np.sqrt(2*np.pi)/sig[i]*np.exp(-(ll-line-line*v[i]/c)**2/(2*sig[i]**2))

        # add together all emissions along loop
        tot_emissNEI = np.sum(emissNEI,axis=0)
        tot_emiss = np.sum(emiss,axis=0)

        #generate and add noise
        yerr = np.sqrt(tot_emissNEI)
        noise = np.random.normal(0.0 ,size = len(tot_emissNEI)) # make gaussian noise instead.
        error = yerr*noise
        np.nan_to_num(error, copy=False, nan=0)
        tot_emissNEI += error
        meas_error = np.sqrt(tot_emissNEI) # error measured
        np.nan_to_num(meas_error, copy=False, nan=0)
        rando = np.random.randn(2000)*0.001*np.max(meas_error) # add small amount of noise to zero-ish values
        too_small = np.where(meas_error < 0.01*np.max(meas_error))
        meas_error[too_small] += rando[too_small]

        if verbose == True:
            print('A = ', A)
            print('A_pixel = ', A_pixel)
            print('ne = ', ne)
            print('vol = ', volume)
            print('area iris = ', A_iris)
            print('atn = ', atn)
            print('photo erg = ', photo_erg)

        self.wav = ll
        self.spec = tot_emissNEI
        self.error = meas_error
        self.EM = EM
        self.g = g
        self.fac = photo_fac*factor
        self.x = x
        self.v = v
        self.T = T
        self.ne = ne
        self.los_x = los_x
        self.los_v = sm_v
        self.i_length = i_length



    def master(self):

        SPEC = np.zeros((self.t_n,len(self.spec)))
        ERROR = np.zeros((self.t_n,len(self.error)))

        for i in range(0,self.t_n):

            arrs = siiv(TUBE,FRAC,LOG10T,LOG10G,time=i)

            SPEC[i,:] = arrs.spec
            ERROR[i,:] = arrs.error

        self.total_spec = SPEC
        self.total_error = ERROR


    def rebin(self,dt=0.2,int_time=5.):
        reshape = int(int_time/dt)
        print('reshape index = ',reshape)

        self.master()

        nu_spec = self.total_spec.reshape(-1,reshape,2000)
        nu_error = self.total_error.reshape(-1,reshape,2000)

        self.respec = np.mean(nu_spec,axis=1)
        self.reerror = np.mean(nu_error,axis=1)

        print('number of time elements after rebin: ',self.reerror.shape[0])

        self.time = np.arange(0,self.reerror.shape[0]*int_time,int_time)


    def fitspec(self):
        nt = len(self.time)
        print('nt = ',nt)

        self.vr = np.zeros(nt)
        self.vb = np.zeros(nt)
        self.wr = np.zeros(nt)
        self.wb = np.zeros(nt)
        self.amp = np.zeros(nt)

        for i in range(0,nt):

            t_i = self.time[i]


            dat = self.respec[i,:]
            err  = self.reerror[i,:] # need to run self.master() prior to running this..

            res = fit2gauss(ll,dat,err,chi_thr=100.)
            a2g = res["a2g"] # extract fit parameters
            a1g = res["a1g"]
            if a2g[1]!=a2g[4]:
                a2gB = np.min(a2g[1],a2g[4])
                a2gR = np.max(a2g[1],a2g[4])
            else:
                a2gB,a2gR = a2g[1],a2g[4]

            self.amp[i] = a1g[0]
            #calculate Doppler velocitiesand wavelengths
            line = 1402.77
            c = 300.
            freq = c/line*1e3
            self.vb[i] = (a2g[1]-line)/line*3e5 # in km/s
            self.vr[i] = (a2g[4]-line)/line*3e5


    def plotgauss(self,itime=0):
        spec_fit,error_fit = self.respec[itime,:],self.reerror[itime,:]
        res = fit2gauss(ll,spec_fit,error_fit,chi_thr=50.,verbose=False)
        # extract arrays
        y2a = res["y2a"]
        y2b = res["y2b"]
        y1g = res["y1g"]

        #plot
        fig = plt.figure(figsize=(14,8))
        gs = gridspec.GridSpec(1,1)
        ax1 = fig.add_subplot(gs[0])

        # spectra
        ax1.plot(ll, spec_fit)

        # peak 1
        ax1.plot(ll, y2a, "g")
        ax1.fill_between(ll, y2a.min(), y2a, facecolor="green", alpha=0.5)

        # peak 2
        ax1.plot(ll, y2b, "y")
        ax1.fill_between(ll, y2b.min(), y2b, facecolor="yellow", alpha=0.5)

        #both
        ax1.plot(ll,y2a+y2b, "r")

        # single fit
        ax1.plot(ll,y1g,'b--')

        plt.xlim(1402.5,1404)
        #plt.ylim(0,3000)
        ax1.set_xlabel("wavlength [$\AA$]",  fontsize=12)
        ax1.set_ylabel("normalized intensity",  fontsize=12)
        fig.tight_layout()
