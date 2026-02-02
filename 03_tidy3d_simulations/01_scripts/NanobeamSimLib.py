import tidy3d as td
from tidy3d.constants import C_0
import tidy3d.web as web
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode.web import run as run_mode_solver
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import dataclass
from typing import Literal, List

@dataclass
class WaveguideParams:
    beam_size_x: float = 1.0
    beam_size_y: float = 1.0
    beam_size_z: float = 0.160
    beam_loc_x: float = 0.0
    beam_loc_y: float = 0.0
    beam_loc_z: float = 0.0
    beam_index: float = 2.41

@dataclass
class MonitorParams:
    mon_loc_x: float = 0
    mon_loc_y: float = 0
    mon_loc_z: float = 0
    mon_size_x: float = 0.0
    mon_size_y: float = 1.0
    mon_size_z: float = 1.0
    mon_num_modes: int = 5
    mon_name: str = "mode_mon"

@dataclass
class DipoleParams:
    dipole: Literal["trans", "axial", "uni"] = "trans"
    dip_loc_x: float = 0.0
    dip_loc_y: float = 0.0
    dip_loc_z: float = 0.0

@dataclass
class ModeSourceParams:
    mode_loc_x: float = 0
    mode_loc_y: float = 0
    mode_loc_z: float = 0
    mode_size_x: float = 0.0
    mode_size_y: float = 1.0
    mode_size_z: float = 1.0
    mode_index: int = 0
    mode_direction: Literal['+', '-'] = '+'
    mode_num_modes: int = 5
    mode_name: str = "mode_source"

@dataclass
class GeneralSimParams:
    size_x: float = 10.0
    size_y: float = 3.0
    size_z: float = 3.0
    wavelength: float = 0.619
    run_time: float = 0
    symmetry: List = (0, 0, 0)
    mesh_per_wvl: int = 15

class NanobeamSim:

    def __init__(self, params: GeneralSimParams):
        # variables to store structures, sources, and monitors etc
        self.structures = []
        self.sources = []
        self.monitors = []
        self.sim = None

        # setting up some used variables for later
        self.freq0 = td.C_0 / params.wavelength
        self.fwidth = self.freq0 / 10
        self.sim_size = (params.size_x, params.size_y, params.size_z)

        # other sim parameters
        self.run_time = params.run_time
        self.symmetry = params.symmetry
        self.mesh_per_wvl = params.mesh_per_wvl

    def add_waveguide_nanobeam(self, params: WaveguideParams):
        waveguide = td.Structure(
            geometry = td.Box(
                size = (params.beam_size_x, params.beam_size_y, params.beam_size_z),
                center = (params.beam_loc_x, params.beam_loc_y, params.beam_loc_z),
            ),
            medium = td.Medium(permittivity = params.beam_index**2)
        )
        self.structures.append(waveguide)

    def add_mode_monitor(self, params: MonitorParams):
        mode_spec = td.ModeSpec(num_modes = params.mon_num_modes)
        monitor = td.ModeMonitor(
            center = (params.mon_loc_x, params.mon_loc_y, params.mon_loc_z),
            size = (params.mon_size_x, params.mon_size_y, params.mon_size_z),
            freqs = [self.freq0],
            mode_spec = mode_spec,
            name = params.mon_name,
        )
        self.monitors.append(monitor)

    def add_flux_monitor(self, params: MonitorParams):
        monitor = td.FluxMonitor(
            center = (params.mon_loc_x, params.mon_loc_y, params.mon_loc_z),
            size = (params.mon_size_x, params.mon_size_y, params.mon_size_z),
            freqs = [self.freq0],
            name = params.mon_name,
        )
        self.monitors.append(monitor)

    def add_field_monitor(self, params: MonitorParams):
        monitor = td.FieldMonitor(
            # center = (params.mon_loc_x, params.mon_loc_y, params.mon_loc_z),
            # size = (params.mon_size_x, params.mon_size_y, params.mon_size_z),
            # # freqs = [self.freq0],
            # interval = 10,
            # start = 2 / self.fwidth,
            # name = params.mon_name,
            center = (params.mon_loc_x, params.mon_loc_y, params.mon_loc_z),
            size = (params.mon_size_x, params.mon_size_y, params.mon_size_z),
            freqs = [self.freq0],
            name = params.mon_name,
        )
        self.monitors.append(monitor)

    def add_dipole_source(self, params: DipoleParams):
        dipole_fields = ('Ex', 'Ey', 'Ez')
        if params.dipole == 'trans':
            dipole_vec = (0, np.sqrt(2), 1)
        elif params.dipole == 'axial':
            dipole_vec = (np.sqrt(2), 0, 1)
        elif params.dipole == 'uni':
            dipole_vec = (1, 1, 1)
        for amp, comp in zip(dipole_vec, dipole_fields):
            if amp != 0:
                dipole = td.PointDipole(
                    center = (params.dip_loc_x, params.dip_loc_y, params.dip_loc_z),
                    polarization = comp,
                    source_time = td.GaussianPulse(
                        freq0 = self.freq0,
                        fwidth = self.fwidth,
                        amplitude = amp,
                        phase = 0,
                    )
                )
                self.sources.append(dipole)

    def add_mode_source(self, params: ModeSourceParams):
        mode_spec = td.ModeSpec(num_modes = params.mode_num_modes)
        source = td.ModeSource(
            center = (params.mode_loc_x, params.mode_loc_y, params.mode_loc_z),
            size = (params.mode_size_x, params.mode_size_y, params.mode_size_z),
            source_time = td.GaussianPulse(
                    freq0 = self.freq0,
                    fwidth = self.fwidth,
            ),
            direction = params.mode_direction,
            mode_spec = mode_spec,
            mode_index = params.mode_index,
            name = params.mode_name,
        )
        self.sources.append(source)

    def generate_sim_object(self):
        grid_spec = td.GridSpec.auto(min_steps_per_wvl = self.mesh_per_wvl)
        boundary_spec = td.BoundarySpec.all_sides(boundary = td.PML())

        self.sim = td.Simulation(
            size = self.sim_size,
            grid_spec = grid_spec,
            structures = self.structures,
            sources = self.sources,
            monitors = self.monitors,
            run_time = 200/self.freq0,
            boundary_spec = boundary_spec,
        )
        print("Structure types:", [type(s) for s in self.structures])
        print("Source types:", [type(s) for s in self.sources])
        print("Monitor types:", [type(m) for m in self.monitors])



    def plot_sim(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.sim.plot(z=z)
        plt.show()
        self.sim.plot(x=x)
        plt.show()
        self.sim.plot(y=y)
        plt.show()


