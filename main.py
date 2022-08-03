from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy.optimize import curve_fit
import os

#################
### Functions ###
#################

def Data_analysis(initial_data):
    masked_data = ma.masked_invalid(initial_data)
    median = ma.median(masked_data)
    corrected_data = initial_data - median
    return corrected_data
    
def Gaussian(x, height, x_center, std, offset):
    return(height * (np.exp(-((x-x_center)**2 / (2*std**2)))) + offset)

def Gaussian_fit(x_axis, y_axis, initial_vals):
    fit_param, statis = curve_fit(Gaussian, x_axis, y_axis, initial_vals)
    model = Gaussian(x_axis, *fit_param) 
    return model, fit_param

def Surface_density(annulii):
    area = np.pi*(annulii[1][1:]**2 - annulii[1][:-1]**2)
    surf_dens = annulii[0]/area
    annulli_cent = np.convolve(annulii[1],np.array([0.5, 0.5]), mode= 'valid')
    errors = np.sqrt(annulii[0])/area
    return surf_dens, errors , annulli_cent

def King_model(r, n_0, alpha, gamma):
    return n_0*(1+(r/alpha)**2)**(-gamma/2)

def King_model_fit(radii, surf_dens, initial_vals, errors):
    fit_param, statis = curve_fit(King_model, radii, surf_dens, initial_vals, sigma=errors)
    model = King_model(radii, *fit_param) 
    return model, fit_param


#################
### Main Code ###
#################

# Read and plot the data
path = os.getcwd()

hdul = fits.open(path+'/data/ic2r02050_drz.fits')    
#hdul.info()                                     # This line is to help explore the data 
#hdul[1].header                                  # This line is to help explore the data
data = hdul[1].data
wcs = WCS(hdul[1].header)

fig= plt.figure()
plt.imshow(data, vmin = np.nanmin(data), vmax=2)
plt.xlabel('Number of x Pixels')
plt.ylabel('Number of y Pixels')
plt.title('Globular Cluster NGC 104')
plt.show()

# Clean the data ,detect stars and visualize their position

data[np.isnan(Data_analysis(data))] = 0
daofind = DAOStarFinder(threshold = 8,fwhm = 3) # I select a threshold of 8
sources = daofind(data)

for col in sources.colnames:
    sources[col].info.format = '%.8g'  # for consistent table output, I just round the values  


positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.)

fig = plt.figure()
plt.subplot(projection=wcs)
plt.imshow(data, vmin = np.nanmin(data), vmax=2)
apertures.plot(color='red', lw=1.5, alpha=0.5)
plt.ylabel('Declination')
plt.xlabel('Right Ascension')
plt.title('Star Positions in the Globular Cluster NGC 104')
plt.show()

# Analyse the data and model the distribution of stars with respect to x,y coordinates

x_star = np.histogram(sources['xcentroid'], bins = 100)
y_star = np.histogram(sources['ycentroid'], bins = 100)

x_star_cent = np.convolve(x_star[1], np.array([0.5, 0.5]), mode= 'valid')
y_star_cent = np.convolve(y_star[1], np.array([0.5, 0.5]), mode= 'valid')

model_x, param_x = Gaussian_fit(x_star_cent, x_star[0], [200,2100,900,10])
model_y, param_y = Gaussian_fit(y_star_cent, y_star[0], [200,2100,900, 10])

fig = plt.figure()
plt.plot(x_star_cent,model_x)
plt.hist(sources['xcentroid'], bins = 100)
plt.xlabel('Number of pixels')
plt.ylabel('Number of stars')
plt.title('Numerical Density of Stars with respect to x-coordinates')
plt.legend(['Fit', 'Data'])
plt.grid()
plt.show()

fig = plt.figure()
plt.plot(y_star_cent,model_y)
plt.hist(sources['ycentroid'], bins = 100)
plt.xlabel('Number of pixels')
plt.ylabel('Number of stars')
plt.title('Numerical Density of Stars with respect to y-coordinates')
plt.legend(['Fit', 'Data'])
plt.grid()
plt.show()

# Calculates the stellar density profile of the cluster

r_stars = np.sqrt((param_x[1]-sources['xcentroid'])**2 + (param_x[1]-sources['ycentroid'])**2)
r_stars_arc = r_stars * 0.04
r=2000*0.04
r_stars_arc=r_stars_arc[r_stars_arc<r]

annulii = np.histogram(r_stars_arc, bins=100)
surf_dens, numb_error, r_cent_annulii = Surface_density(annulii)

fig= plt.figure()
plt.errorbar(r_cent_annulii,surf_dens, numb_error)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Radius [arcsec]')
plt.ylabel(r'Surface Density [arcsec$^{-2}]$')
plt.title('Surface Density of the Globular Cluster NGC 104')
plt.grid()
plt.show()

# Calculate the 'core radius' of the cluster in arcsec

model, param = King_model_fit(r_cent_annulii, surf_dens, [3,8,1], numb_error)

fig = plt.figure()
plt.plot(r_cent_annulii,model)
plt.errorbar(r_cent_annulii,surf_dens, numb_error)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Radius [arcsec]')
plt.ylabel(r'Surface Density [arcsec$^{-2}]$')
plt.title('Surface Density of the Globular Cluster NGC 104')
plt.legend(['Fit', 'Data'])
plt.grid()
plt.show()

r_c = param[1]*np.sqrt((2**(2/param[2])-1))
print("The 'core radius' of the cluster in arcsec is {:.2f}".format(r_c))
