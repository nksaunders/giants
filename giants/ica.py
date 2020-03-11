import giants as g
import numpy as np
import lightkurve as lk
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
from astropy.stats import BoxLeastSquares
import matplotlib.ticker as mtick

__all__ = ['get_ica_components']

def get_ica_components(tpf, n_components=20, return_components=True, plot=False):
    """
    """

    raw_lc = tpf.to_lightcurve(aperture_mask='all')

    ##Perform ICA

    X = np.ascontiguousarray(np.nan_to_num(tpf.flux), np.float64)
    X_flat = X.reshape(len(tpf.flux), -1) #turns three dimensional into two dimensional

    f1 = np.reshape(X_flat, (len(X), -1))
    X_pix = f1 / np.nansum(X_flat, axis=-1)[:, None]

    ica = FastICA(n_components=n_components) #define n_components
    S_ = ica.fit_transform(X_pix)
    A_ = ica.mixing_ #combine x_flat to get x

    a = np.dot(S_.T, S_)
    a[np.diag_indices_from(a)] += 1e-5
    b = np.dot(S_.T, raw_lc.flux)

    w = np.linalg.solve(a, b)

    comp_lcs = []
    blss = []
    max_powers = []

    for i,s in enumerate(S_.T):
        component_lc = s * w[i]
        comp_lcs.append(component_lc)
        # plt.plot(component_lc + i*1e5)

        model = BoxLeastSquares(tpf.time, component_lc)
        results = model.autopower(0.16, minimum_period=.5, maximum_period=24.)
        # model = transitleastsquares(tpf.time, component_lc)
        # results = model.power()
        period, power = results.period, results.power
        blss.append([period, power])
        # print(results.depth_snr[np.argmax(power)])
        if (np.std(component_lc) > 1e4) or (np.abs(period[np.argmax(power)] - 14) < 2) or (results.depth[np.argmax(power)]/np.median(component_lc) < 0):
            power = [0]

        max_powers.append(np.max(power))

    best_pers = blss[np.argmax(max_powers)][0]
    best_powers = blss[np.argmax(max_powers)][1]

    period = best_pers[np.argmax(best_powers)]

    transit_lc = lk.LightCurve(time=tpf.time, flux=comp_lcs[np.argmax(max_powers)])

    scale = np.median(raw_lc.flux) / 10

    if plot:
        fig, ax = plt.subplots(2, 3, figsize=(12, 7))
        fig.suptitle(f'{tpf.targetid}')

        for i,c in enumerate(comp_lcs):
            ax[0,0].plot(tpf.time, c + i*scale)
        ax[0,0].set_ylim(-scale, (n_components)*scale)
        ax[0,0].set_xlim(tpf.time[0], tpf.time[-1])
        ax[0,0].set_xlabel('Time')
        ax[0,0].set_ylabel('Flux')
        ax[0,0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
        ax[0,0].set_title('ICA Components')

        transit_lc.plot(ax=ax[0,1])
        ax[0,1].set_xlim(tpf.time[0], tpf.time[-1])
        ax[0,1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
        ax[0,1].set_title('ICA comp with max BLS power')

        transit_lc.remove_outliers(9).fold(period).scatter(ax=ax[0,2], c='k', label=f'Period={period:.2f}')
        transit_lc.remove_outliers(9).fold(period).bin(7).plot(ax=ax[0,2], c='r', lw=2, C='C1', label='Binned')
        ax[0,2].set_ylim(-5*np.std(transit_lc.flux), 2*np.std(transit_lc.flux))
        ax[0,2].set_xlim(-.5,.5)
        ax[0,2].set_title('Folded ICA Transit Component')

        A_useful = A_.reshape(tpf.shape[1],tpf.shape[2],n_components).T #reshape from 2d to 3d

        weighted_comp = A_useful[np.argmax(max_powers)].T * w[np.argmax(max_powers)]

        ax[1,0].imshow(weighted_comp, origin='lower')
        ax[1,1].imshow(tpf.flux[200], origin='lower')
        im = ax[1,2].imshow(weighted_comp / tpf.flux[200], origin='lower')

        ax[1,0].set_title('Weighted Transit Component')
        ax[1,1].set_title('TPF')
        ax[1,2].set_title('Model / Flux')

        plt.colorbar(im)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    if return_components:
        return np.array(comp_lcs).T
