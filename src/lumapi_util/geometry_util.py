from .general_util import *
from collections import OrderedDict, defaultdict
import numpy as np
from functools import partial
from dataclasses import dataclass
from typing import List

axis_type = {
	"x": "2D X Normal",
	"y": "2D Y Normal",
	"z": "2D Z Normal",
}

def find_axis(size):
    none_indices = [i for i, v in enumerate(size) if v is None]
    if len(none_indices) != 1:
        raise ValueError("Exactly one value in `size` must be None to infer the propagation axis.")

    return "xyz"[none_indices[0]]


@convert_length_units(["size", "center"])
def rect_properties(size, center, unit='um'):
    return [
        *[(s, v) for s, v in zip("xyz", center) if v is not None],
        *[(f"{s} span", v) for s, v in zip("xyz", size) if v is not None],
    ]


@convert_length_units(["size", "center"])
def define_mesh(sim, name="mesh", size=[5, 5, 5], center=[0, 0, 0], steps=None, wavelength=0.97, **kwargs):
    wl = wavelength
    if steps is None:
        steps = [wl / 12] * 3
    properties = OrderedDict([
                ("name", name),
                *rect_properties(size, center),
                *[(f"override {s} mesh", v is not None) for s, v in zip("xyz", steps)],
                *[(f"d{s}", v * 1e-6) for s, v in zip("xyz", steps) if v is not None],
            
    ])
    properties.update(kwargs)

    ensure_object(sim, "mesh", name, properties)
    # try:
    #     set_properties(sim, name, properties)
    # except:
    #     add_lumerical_object(sim, "mesh", properties)
        # sim.addmesh(properties=properties)
    # name = sim.get("name")
    return name
      

def add_rect(sim, size, center):
      sim.addrect(properties=rect_properties(size, center))



@convert_length_units(["size", "center"])
def add_layerstack_fromfile(sim, filename, size=[50, 50, None], center=[0, 0, 0], gds_center=None):
    filename = str(filename)
    sim.addlayerbuilder()
    sim.loadprocessfile(filename)
    properties = rect_properties(size, center)
    if gds_center is not None:
        properties.extend([
            ("gds position reference", "Centered at custom coordinate"),
            ("gds center x", gds_center[0] * 1e-6),
            ("gds center y", gds_center[1] * 1e-6),
        ])
    set_properties(sim, "layer group", OrderedDict(properties))


def rotate(points, angle_deg, origin=(0., 0.)):
    """
    Rotate a set of 2D points around a given origin.

    Parameters:
    - points: (N, 2) array-like of [x, y] coordinates
    - angle_deg: rotation angle in degrees (counterclockwise)
    - origin: tuple (x0, y0) to rotate around

    Returns:
    - (N, 2) NumPy array of rotated points
    """
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    points = np.asarray(points)
    origin = np.asarray(origin)

    # Translate points to origin, apply rotation, then translate back
    return (points - origin) @ rotation_matrix.T + origin


def rectangle(size, center):
    half_size = np.array(size) / 2
    center = np.array(center)
    # Define corners relative to origin, counterclockwise
    rel_corners = np.array([[-1, -1], [-1,  1], [ 1,  1], [ 1, -1]])
    return rel_corners * half_size + center  # translate to center and scale

def straight_waveguide(length, width, origin=(0., 0)):  
    """
    Simple waveguide in the x-direction forward
    """
    return rectangle([length, width], [origin[0] + length / 2, origin[1]])





@dataclass
class Section:
    name: str
    width: float
    offset: float  # center position relative to propagation axis

    @property
    def left(self) -> float:
        return self.offset - self.width / 2

    @property
    def right(self) -> float:
        return self.offset + self.width / 2


@dataclass
class CrossSection:
    z: float
    sections: List[Section]




def build_waveguide_polygons(cross_sections: List[CrossSection]):
    # Store edge points grouped by section name
    left_edges = defaultdict(list)
    right_edges = defaultdict(list)

    # Sort by z (propagation axis)
    for cs in sorted(cross_sections, key=lambda c: c.z):
        for sec in cs.sections:
            left_edges[sec.name].append((cs.z, sec.left))
            right_edges[sec.name].append((cs.z, sec.right))

    polygons = {}
    for name in left_edges:
        left = left_edges[name]                # forward order
        right = right_edges[name][::-1]        # reverse order
        polygon = left + right + [left[0]]                 # closed strip polygon
        polygons[name] = polygon

    return polygons



#this function assumes that the fiber is connected to the chip at its left (looking at x axis)
#for a SMF-28: add_ccsmf(x, 4.1, 1.44, 20, 1.434816, [y;z])
def add_ccsmf(sim, x, core_rad, core_index, clad_rad, clad_index, yz_center):
    sim.addobject("cc_fiber")
    sim.groupscope("::model::cc_fiber")
    sim.set("first axis", 'y')
    sim.set("rotation 1", 90)
    sim.set("radius core", core_rad * 1e-6)
    sim.set("index core", core_index)
    sim.set("radius cladding", clad_rad * 1e-6)
    sim.set("index cladding", clad_index)
    sim.set("z span", 15 *1e-6)
    sim.set("x", (x + 7.5) *1e-6)
    sim.set("y", yz_center[0] *1e-6)
    sim.set("z", yz_center[1] *1e-6)
    sim.groupscope("::model")




def ellipse(radii, center=(0., 0), n_points=50):
    t = np.linspace(0., 2 * np.pi, n_points)
    a, b = radii
    return np.column_stack((
        a * np.cos(2 * np.pi * t),
        b * np.sin(2 * np.pi * t),
    )) + np.array(center)


def circle(radius, center=(0., 0), n_points=50):
    return ellipse((radius, radius), center, n_points)


def polynomial_profile(wi, wf, ln, m=1, n_points=50):
	x = np.linspace(0, 1, n_points)
	y = (wi - wf) * (1 - x)**m + wf
	return np.column_stack((x * ln, y))


def inverse_polynomial_profile(wi, wf, ln, m=1, n_points=50):
	x = np.linspace(0, 1, n_points)
	y = (wi - wf) * (1 - x)**m + wf
	return np.column_stack((x * ln, y))


def inverse_exponential_profile(wi, wf, ln, m=5, n_points=50):
	x = np.linspace(0, 1, n_points)
	y = wi + (wf - wi) * np.exp(-m * (abs(1-x)))
	return np.column_stack((x * ln, y))


def sine_profile(wi, wf, ln, n_points=30):
	x = np.linspace(0, 1, n_points)
	y = 0.5 * (wf - wi) * np.sin(np.pi * (x - 0.5)) + 0.5 * (wf + wi)
	return np.column_stack((x * ln, y))



def build_taper(profile, extension=None):
    """
    Generates a symmetric taper polygon from a width profile.

    Parameters:
        profile (np.ndarray): Nx2 array of (x, width).
        extension (tuple): Optional (x_start, x_end) to add straight ends.

    Returns:
        np.ndarray: Polygon as (x, y) points.
    """
    x, w = profile[:, 0], profile[:, 1] / 2

    if extension:
        x_start, x_end = extension
        x = np.insert(x, 0, x_start)
        w = np.insert(w, 0, w[0])
        x = np.append(x, x[-1] + x_end)
        w = np.append(w, w[-1])

    x_poly = np.concatenate([x, x[::-1], [x[0]]])
    y_poly = np.concatenate([w, -w[::-1], [w[0]]])

    return np.column_stack((x_poly, y_poly))


profiles = {
      "linear": partial(inverse_polynomial_profile, m=1, n_points=50),
      "parabolic": partial(inverse_polynomial_profile, m=2, n_points=50),
      "sine": partial(sine_profile, n_points=50),
      "exponential": partial(inverse_exponential_profile, m=5, n_points=50)
}

def get_profile(profile):
     linear_profile = partial(inverse_polynomial_profile, m=1, n_points=50)
     if isinstance(profile, str):
          return profiles.get(profile, linear_profile)
     elif profile is None:
          return linear_profile
     else:
          return profile

def build_profile_taper(wi, wf, length, profile=None, extension=None):
    profile_fn = get_profile(profile)
    left, right = extension or (0., 0.)
    vtx = build_taper(profile_fn(wi, wf, length), (-left, right))
    return vtx + np.array([left, 0])


def set_layer_properties(sim, layer, **params):
    layer_param_map = {
        f"t_{layer}": ("thickness", 1e-6),
        f"z0_{layer}": ("start position", 1e-6),
        f"angle_{layer}": ("sidewall angle", 1.0),   # no scaling
        f"dw_{layer}": ("pattern growth delta", 1e-6),
    }

    for param_key, (sim_prop, scale) in layer_param_map.items():
        value = params.get(param_key)
        if value is not None:
            sim.setlayer(layer, sim_prop, value * scale)


def set_geometries_to_layer(sim, geometries, layer, stack="layer group"):
    sim.select(stack)

    lyr_nr = sim.getlayer(layer, "layer number")
    vertices = sim.get("geometry")
    vertices[lyr_nr] = [vtx * 1e-6 for vtx in geometries]
    # vertices = {lyr_nr : [vtx * 1e-6 for vtx in geometries]}
    sim.set("geometry", vertices)


def set_geometries_to_stack(sim, geometries, stack='layer group'):
    sim.select(stack)

    vertices = sim.get("geometry")
    for lyr, polygons in geometries.items():
         lyr_nr = sim.getlayer(lyr, "layer number")
         vertices[lyr_nr] = [vtx * 1e-6 for vtx in polygons]
    sim.set("geometry", vertices)

 

def get_layer_info(sim, layer):
    sim.select("layer group")
    type = sim.getlayer(layer, "process")
    return {"t": np.round(sim.getlayer(layer, "thickness") * 1e6, 6),
            "z0": np.round(sim.getlayer(layer, "start position") * 1e6, 6),
            "angle": sim.getlayer(layer, "sidewall angle"),
            "material": sim.getlayer(layer, "background material" if type == "Background" else "pattern material"),
            }
