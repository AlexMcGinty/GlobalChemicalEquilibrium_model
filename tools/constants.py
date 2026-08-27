R = 8.314462618153  # J /(mol K)

# fitting coefficients for H2 silicate activity coefficient, from Gilmore & Stixrude (2025)
tau = 4800
ppi = -35.0
A = 622000
B = -4950

# Interaction parameter for H2O silicate activity coefficient, from Kovacevic et al. (2025)
W = 74826

G = 6.67430e-11  # gravitational constant [m^3 kg^-1 s^-2]
M_earth = 5.972e24  # kg
R_earth = 6.371e6  # m
log_to_ln = 2.302585093
ref_T = 298.15

# ΔIW range for Earth's core formation (Frost et al. 2008, Wood et al. 2006)
# Used for shading in delta_IW plots to indicate Earth-like redox conditions
EARTH_CORE_FORMATION_DELTA_IW_MIN = -3.1
EARTH_CORE_FORMATION_DELTA_IW_MAX = -1.8



