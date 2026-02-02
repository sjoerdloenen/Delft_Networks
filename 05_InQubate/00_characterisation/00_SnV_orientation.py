import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Angular domain
theta_snv_1_rad = np.deg2rad( 1./2 * (180-109.5) )  # Fixed angle relative to the waveguide direction for SnV centers aligned parallel to the direction of the waveguide
theta_snv_3_4_rad = np.deg2rad( 90 )  # Fixed angle relative to the waveguide direction for SnV centers aligned perpendicular to the direction of the waveguide
theta = np.linspace(0, 2*np.pi, 1000)

# Dipole model
def dipole_excitation(theta, theta0):
    return np.cos(theta - theta0)**2

# Initial parameters
theta_sample_init_deg = 25                              # initial sample angle in degrees
theta_sample_init = np.deg2rad(theta_sample_init_deg)   # initial sample angle

# Initial curves
snv_1_polar_rad = np.arctan( np.sin(theta_sample_init) * np.tan(theta_snv_1_rad))
snv_3_4_polar_rad = np.arctan( np.sin(theta_sample_init) * np.tan(theta_snv_3_4_rad))

I_snv_1 = dipole_excitation(theta, snv_1_polar_rad)
I_snv_2 = dipole_excitation(theta, -snv_1_polar_rad)
I_snv_3_4 = dipole_excitation(theta, snv_3_4_polar_rad)

# Create figure
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)

line1, = ax.plot(theta, I_snv_1, 'g-', lw=2, label='SnV orient. 1')
line2, = ax.plot(theta, I_snv_2, 'b-', lw=2, label='SnV orient. 2')
line3, = ax.plot(theta, I_snv_3_4, 'r:', lw=2, label='SnV orient. 3-4')

ax.set_ylim(0, 1)
ax.set_title("Dipole excitation: $I(\\theta)=\\cos^2(\\theta-\\theta_0)$")
ax.legend(loc='upper right')

# Slider axis
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
theta_sample_slider = Slider(
    ax=ax_slider,
    label=r'$\theta_{sample}$ (deg)',
    valmin=0,
    valmax=90,
    valinit=theta_sample_init_deg
)

# Update function
def update(val):
    theta_sample_deg = theta_sample_slider.val
    theta_sample_rad = np.deg2rad(theta_sample_deg)
    theta_polar_rad = np.arctan( np.sin(theta_sample_rad) * np.tan(theta_snv_1_rad))
    line1.set_ydata(dipole_excitation(theta, theta_polar_rad))
    line2.set_ydata(dipole_excitation(theta, -theta_polar_rad))
    fig.canvas.draw_idle()


theta_sample_slider.on_changed(update)

plt.show()
