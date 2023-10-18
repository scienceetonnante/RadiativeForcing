import matplotlib.pyplot as plt
import numpy as np
from physipy import units, constants
from physipy.quantity.utils import qarange # equivalent of np.arange but for quantities

# physical constants
h = constants['h']
c = constants['c']
kB = constants['k']

# units
m = units['m']     # meter
km = units['km']   # kilometer
sr = units['sr']   # steradian
Pa = units['Pa']   # Pascal
K = units['K']     # Kelvin
W = units['W']     # Watt
mum = units['mum']

# ----------------------------------------------------------------------------------------------------------------------

# ===================
# BLACKBODY RADIATION
# ===================

def planck_function(lambda_wavelength, T):
    #h = 6.62607015e-34      # Planck's constant, J*s
    #c = 2.998e8             # Speed of light, m/s
    #kB = 1.380649e-23       # Boltzmann's constant, J/K
    term1 = (2 * h * c**2) / lambda_wavelength**5
    term2 = np.exp((h * c) / (lambda_wavelength * kB * T)) - 1
    return term1 / term2 / sr

# ----------------------------------------------------------------------------------------------------------------------

# ================
# ATMOSPHERE MODEL
# ================

def pressure(z):
    P0 = 101325 * Pa # Pressure at sea level in Pa
    H = 8500 * m     # Scale height in m
    return P0 * np.exp(-z / H)

def temperature_uniform(z):
    T0 = 288.2 * K # Temperature at sea level in K
    return T0 * np.ones_like(z)


def temperature_simple(z):
    T0 = 288.2 * K 
    z_trop = 11000 * m    # Tropopause height in m
    Gamma = -0.0065 * K/m # Temperature gradient in K/m
    T_trop = T0 + Gamma * z_trop
    return np.piecewise(z, [z < z_trop, z >= z_trop],
                        [lambda z: T0 + Gamma * z,
                         lambda z: T_trop])


def temperature_US1976(z):

    # Troposphere (0 to 11 km)
    T0 = 288.15 * K
    z_trop = 11 * km

    # Tropopause (11 to 20 km)
    T_tropopause = 216.65 * K
    z_tropopause = 20 * K

    # Stratosphere 1 (20 to 32 km)
    T_strat1 = T_tropopause
    z_strat1 = 32 * km

    # Stratosphere 2 (32 to 47 km)
    T_strat2 = 228.65 * K
    z_strat2 = 47 * km

    # Stratopause (47 to 51 km)
    T_stratopause = 270.65 * K
    z_stratopause = 51 * km

    # Mesosphere 1 (51 to 71 km)
    T_meso1 = T_stratopause
    z_meso1 = 71 * km

    # Mesosphere 2 (71 to ...)
    T_meso2 = 214.65 * K
    return np.piecewise(z,
                        [z < z_trop,
                         (z >= z_trop) & (z < z_tropopause),
                         (z >= z_tropopause) & (z < z_strat1),
                         (z >= z_strat1) & (z < z_strat2),
                         (z >= z_strat2) & (z < z_stratopause),
                         (z >= z_stratopause) & (z < z_meso1),
                         z >= z_meso1],
                        [lambda z: T0 - 6.5 * z,
                         lambda z: T_tropopause,
                         lambda z: T_strat1 + 1 * (z - z_tropopause),
                         lambda z: T_strat2 + 2.8 * (z - z_strat1),
                         lambda z: T_stratopause,
                         lambda z: T_meso1 - 2.8 * (z - z_stratopause),
                         lambda z: T_meso2 - 2 * (z - z_meso1)])


# ==> CHOOSE HERE THE TEMPERATURE MODEL
def temperature(z):
    return temperature_simple(z)

def air_number_density(z):
    return pressure(z) / (kB * temperature(z))

# ----------------------------------------------------------------------------------------------------------------------

# ==============
# CO2 ABSORPTION
# ==============
def cross_section_CO2(wavelength):
    LAMBDA_0 = 15.0e-6 * m # Band center in m
    exponent = -22.5 - 24 * np.abs((wavelength - LAMBDA_0) / LAMBDA_0)
    sigma = 10 ** exponent
    return sigma * m**2

# ----------------------------------------------------------------------------------------------------------------------

# =============================
# RADIATIVE TRANSFER SIMULATION
# =============================

# All wavelengths are treated in parallel using vectorization

def simulate_radiative_transfer(
        CO2_fraction,
        z_max = 80000 * m,
        delta_z = 10 * m,
        lambda_min = 0.1e-6 * m,
        lambda_max = 100e-6 * m,
        delta_lambda = 0.01e-6 * m
    ):

    # Altitude and wavelength grids
    z_range = qarange(0 * m, z_max, delta_z)
    lambda_range = qarange(lambda_min, lambda_max, delta_lambda)
    lambda_range.favunit = mum # set micron as "favorit unit" for display purpose

    # Boundary condition : Compute the outward vertical flux emitted by the Earth's surface for all wavelengths
    earth_flux = np.pi * sr * planck_function(lambda_range, temperature(0*m)) * delta_lambda
    print(f"Total earth surface flux in wavelength range: {earth_flux.into(W/m**2).sum():.2f}") # todo: remove hardcoded unit and add favunit

    # Initialize arrays
    upward_flux = np.zeros((len(z_range), len(lambda_range))) * W/m**2
    optical_thickness = np.zeros((len(z_range), len(lambda_range)))

    flux_in = earth_flux
    for i, z in enumerate(z_range):

        # Number density of CO2 molecules and absorption coefficient
        n_CO2 = air_number_density(z) * CO2_fraction
        kappa = cross_section_CO2(lambda_range) * n_CO2

        # Compute fluxes within the layer
        optical_thickness[i,:] = kappa * delta_z
        absorbed_flux = np.minimum(kappa * delta_z * flux_in , flux_in)
        emitted_flux = optical_thickness[i,:] * np.pi * sr * planck_function(lambda_range, temperature(z)) * delta_lambda
        upward_flux[i, :] = flux_in - absorbed_flux + emitted_flux

        # The flux leaving the layer becomes the flux entering the next layer
        flux_in = upward_flux[i, :]

    print(f"Total outgoing flux at the top of the atmosphere: {upward_flux[-1,:].into(W/m**2).sum():.2f}")

    return lambda_range, z_range, upward_flux, optical_thickness

# ----------------------------------------------------------------------------------------------------------------------

# MAIN

CO2_fraction = 280e-6
lambda_range, z_range, upward_flux, optical_thickness = simulate_radiative_transfer(CO2_fraction)
CO2_fraction *= 2
lambda_range, z_range, upward_flux2, optical_thickness2 = simulate_radiative_transfer(CO2_fraction)

from physipy import setup_matplotlib
setup_matplotlib()
# Plot top of atmosphere spectrum
plt.figure(figsize=(14, 9))
# Superimpose blackbody spectrum at Earth's surface temperature and 220K
plt.plot(lambda_range, (np.pi * sr * planck_function(lambda_range, temperature(0*m))).into(W/m**2/mum),'--k')
plt.plot(lambda_range, (np.pi * sr * planck_function(lambda_range, 216*K)).into(W/m**2/mum),'--k')

delta_lambda = lambda_range[1] - lambda_range[0]
plt.plot(lambda_range, (upward_flux[-1, :]/delta_lambda).into(W/m**2/mum),'-g')
plt.plot(lambda_range, (upward_flux2[-1, :]/delta_lambda).into(W/m**2/mum),'-r')
plt.fill_between(lambda_range, upward_flux[-1, :]/delta_lambda, upward_flux2[-1, :]/delta_lambda, color='yellow', alpha=0.9)
plt.xlabel("Longueur d'onde (μm)")
#plt.ylabel("Luminance spectrale (W/m²/μm/sr)")
plt.xlim(0*mum, 50*mum)
plt.ylim(0*W/m**2/mum, 30*W/m**2/mum)
plt.grid(True)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------