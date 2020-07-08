import os
import healpy as hp
from numba import njit, types
from numba.typed import Dict
import numpy as np
from .template import Model
from .. import units as u
from .. import utils
from ..utils import trapz_step_inplace
import warnings


class InterpolatingComponent(Model):
    def __init__(
        self,
        path,
        input_units,
        nside,
        interpolation_kind="linear_origin",
        has_polarization=False,
        map_dist=None,
        verbose=False,
    ):
        """PySM component interpolating between precomputed maps

        In order to save memory, maps are converted to float32, if this is not acceptable, please
        open an issue on the PySM repository.
        When you create the model, PySM checks the folder of the templates and stores a list of
        available frequencies. Once you call `get_emission`, maps are read, ud_graded to the target
        nside and stored for future use. This is useful if you are running many channels
        with a similar bandpass.
        If not, you can call `cached_maps.clear()` to remove the cached maps.

        Parameters
        ----------
        path : str
            Path should contain maps named as the frequency in GHz
            e.g. 20.fits or 20.5.fits or 00100.fits
        input_units : str
            Any unit available in PySM3 e.g. "uK_RJ", "uK_CMB"
        nside : int
            HEALPix NSIDE of the output maps
        interpolation_kind : string
            Change alternate way of interpolating linearly as well as 
            logarithmic. Will compare computation time and quality to
            original pysm3 commit. 
            Supported: "linear", "log", "linear_origin"
        has_polarization : bool
            whether or not to simulate also polarization maps
        map_dist : pysm.MapDistribution
            Required for partial sky or MPI, see the PySM docs
        verbose : bool
            Control amount of output
        """

        super().__init__(nside=nside, map_dist=map_dist)
        self.maps = {}
        self.maps = self.get_filenames(path)

        # use a numba typed Dict so we can used in JIT compiled code
        self.cached_maps = Dict.empty(
            key_type=types.float32, value_type=types.float32[:, :]
        )

        self.freqs = np.array(list(self.maps.keys()))
        self.freqs.sort()
        self.input_units = input_units
        self.has_polarization = has_polarization
        self.interpolation_kind = interpolation_kind
        self.verbose = verbose

    def get_filenames(self, path):
        # Override this to implement name convention
        filenames = {}
        for f in os.listdir(path):
            if f.endswith(".fits"):
                freq = float(os.path.splitext(f)[0])
                filenames[freq] = os.path.join(path, f)
        return filenames

    @u.quantity_input
    def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:
        nu = utils.check_freq_input(freqs)
        weights = utils.normalize_weights(nu, weights)

        if len(nu) == 1:

            # special case: we request only 1 frequency and that is among the ones
            # available as input
            check_isclose = np.isclose(self.freqs, nu[0])
            if np.any(check_isclose):

                freq = self.freqs[check_isclose][0]
                out = self.read_map_by_frequency(freq)
                if self.has_polarization:
                    return out << u.uK_RJ
                else:
                    zeros = np.zeros_like(out)
                    return np.array([out, zeros, zeros]) << u.uK_RJ

        npix = hp.nside2npix(self.nside)
        if nu[0] < self.freqs[0]:
            warnings.warn(
                "Frequency not supported, requested {} Ghz < lower bound {} GHz".format(
                    nu[0], self.freqs[0]
                )
            )
            return np.zeros((3, npix)) << u.uK_RJ
        if nu[-1] > self.freqs[-1]:
            warnings.warn(
                "Frequency not supported, requested {} Ghz > upper bound {} GHz".format(
                    nu[-1], self.freqs[-1]
                )
            )
            return np.zeros((3, npix)) << u.uK_RJ

        first_freq_i, last_freq_i = np.searchsorted(self.freqs, [nu[0], nu[-1]])
        first_freq_i -= 1
        last_freq_i += 1

        freq_range = self.freqs[first_freq_i:last_freq_i]

        if self.verbose:
            print("Frequencies considered:", freq_range)

        for freq in freq_range:
            if freq not in self.cached_maps:
                m = self.read_map_by_frequency(freq)
                if not self.has_polarization:
                    m = m.reshape((1, -1))
                self.cached_maps[freq] = m.astype(np.float32)
                if self.verbose:
                    for i_pol, pol in enumerate(
                        "IQU" if self.has_polarization else "I"
                    ):
                        print(
                            "Mean emission at {} GHz in {}: {:.4g} uK_RJ".format(
                                freq, pol, self.cached_maps[freq][i_pol].mean()
                            )
                        )

        out = compute_interpolated_emission_numba(
            nu, weights, freq_range, 
            self.cached_maps, self.interpolation_kind
        )

        # the output of out is always 2D, (IQU, npix)
        return out << u.uK_RJ

    def read_map_by_frequency(self, freq):
        filename = self.maps[freq]
        return self.read_map_file(freq, filename)

    def read_map_file(self, freq, filename):
        if self.verbose:
            print("Reading map {}".format(filename))
        m = self.read_map(
            filename,
            field=(0, 1, 2) if self.has_polarization else 0,
            unit=self.input_units,
        )
        return m


#@njit(parallel=False)
def compute_interpolated_emission_numba(freqs, weights, freq_range, all_maps, kind):
    """
    Compute a single/bandpass emission from given cached frequency emissions. 
    Intended only for intensity (for now) until polarization are clearly needed.

    Parameters
    ----------
    freqs : np.array
        Array containing all frequencies to be integrated in a bandpass. Single value
        implies return one interpolated emission map at that frequency.
    weights : np.array
        Array containing the normalized corresponding weights of each frequency within
        the bandpass. If none, assume tophat integration. 
    freq_range : np.array
        Array containing the range of frequencies considered within the cached maps. The range
        extends on the lower and upper bound just enough to encapsulate all frequencies in the 
        given bandpass. Frequencies outside this range are not used for interpolation.
    all_maps : numbad typed Dictionary
        Contains key and value pairs of frequency and emission data. Retrieve emission
        from the dictionary by passing in the corresponding frequency.
    kind : string
        Specifies the kind of interpolation technique: "linear", or "log". Use "linear_origin"
        for the original linear interpolation by pysm3. 

    Returns
    ----------
    output : np.ndarray
        Array containing output emission. 
    """ 
   
    output = np.zeros(
        all_maps[freq_range[0]].shape, dtype=all_maps[freq_range[0]].dtype
    )
    index_range = np.arange(len(freq_range))
    if kind == 'linear_origin':
        print('linear_origin')
        for i in range(len(freqs)):
            interpolation_weight = np.interp(freqs[i], freq_range, index_range)
            int_interpolation_weight = int(interpolation_weight)
            m = (interpolation_weight - int_interpolation_weight) * all_maps[
                freq_range[int_interpolation_weight]
            ]
            m += (int_interpolation_weight + 1 - interpolation_weight) * all_maps[
                freq_range[int_interpolation_weight + 1]
            ]
            trapz_step_inplace(freqs, weights, i, m, output)
    else:
        # Loop through every frequency in freqs.
        for i in range(len(freqs)):
            if (freqs[i] in freq_range):
                m = all_maps[freqs[i]] # Case where desired frequency is cached.
            else:
                pos = np.interp(freqs[i], freq_range, index_range) # Find position this freq belongs
                # in available frequencies through interpolation. 
                int_pos = int(pos)
                # Upper and lower bounds of interpolation. 
                lower = all_maps[freq_range[int_pos]]
                upper = all_maps[freq_range[int_pos+1]]
                m = calc_interpolation(freqs[i], 
                                      freq_range[int_pos], freq_range[int_pos+1], 
                                      lower, upper, kind) # Calculate interpolated emission.
                trapz_step_inplace(freqs, weights, i, m, output)
    return output

def calc_interpolation(nu, nu_a, nu_b, ma, mb, kind):
    if kind == 'linear':
        # Linear interpolation formula.
        m = ((nu - nu_a) * (mb - ma) / (nu_b - nu_a)) + ma
    elif kind == 'log':
        # Logarithmic space interpolation formula.
        expo = np.log10(mb/ma) * np.log10(nu/nu_a) / np.log10(nu_b/nu_a)
        print(expo)
        m = ma * np.power(10.0, expo)
    return m
