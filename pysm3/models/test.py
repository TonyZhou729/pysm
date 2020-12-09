import pysm3 as pysm
import pysm3.units as u
import numpy as np
import time
import healpy as hp

path = '/mount/citadel1/zz1994/codes/fground/maps/interpolate_cache'
input_units = u.K_CMB
nside=2048
verbose=True


comp = pysm.models.interpolating.InterpolatingComponent(path, input_units, nside, 
                                                        interpolation_kind='log', 
                                                        verbose=verbose)
start = time.perf_counter()
freqs = np.array([502])
m = comp.get_emission(freqs * u.GHz)
end = time.perf_counter()

print(end - start)
print(m.shape)

hp.write_map('log.fits', m[0])

