import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Angular domain
theta_SnV_rad = np.deg2rad( 1./2 * (180-109.5) )  # Fixed angle for SnV centers
theta = np.linspace(0, 2*np.pi, 1000)

# Dipole model
def dipole_excitation(theta, theta0):
    return np.cos(theta - theta0)**2

# Initial parameters
theta_snv_1 = 0           # variable dipole orientation
theta_snv_3_4 = np.pi/2           # fixed dipole orientation

# Initial curves
I_snv_1 = dipole_excitation(theta, theta_snv_1)
I_snv_2 = dipole_excitation(theta, -theta_snv_1)
I_snv_3_4 = dipole_excitation(theta, theta_snv_3_4)

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
    valinit=theta_snv_1*180/np.pi
)

# Update function
def update(val):
    theta_sample_deg = theta_sample_slider.val
    theta_sample_rad = np.deg2rad(theta_sample_deg)
    theta_polar_rad = np.arctan( np.sin(theta_sample_rad) * np.tan(theta_SnV_rad))
    line1.set_ydata(dipole_excitation(theta, theta_polar_rad))
    line2.set_ydata(dipole_excitation(theta, -theta_polar_rad))
    fig.canvas.draw_idle()


theta_sample_slider.on_changed(update)

plt.show()
