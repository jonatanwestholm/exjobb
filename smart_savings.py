# smart_savings.py

# basics
n_drives = 72000
drive_lifetime = 1200
array_size = 5 # number of drives in RAID

dependence_correction = 4 # empirical dependence correction factor

p_no_warning = 0.5
#p_no_warning *= 0.7 # with early warning system

# derived
n_arrays = n_drives / array_size
p_two_fails = (array_size/drive_lifetime)*((array_size-1)/drive_lifetime)*(p_no_warning**2)
p_two_fails *= dependence_correction
print(p_two_fails)
e_disaster = n_arrays*p_two_fails
print(e_disaster)

