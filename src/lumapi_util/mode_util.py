from functools import partial
# from .general_util import *
from .geometry_util import *
from .sweep_utils import sweep_param_result
import re


################################################################################################################
### Basic items ################################################################################################
################################################################################################################

@convert_length_units(["size", "center"])
def define_fde(sim: lumapi.MODE,
			size: list = [None, 15, 15],
			center: list = [0., 0., 0], 
			wavelength: float = 0.97,
			index: float = 1,
			mesh_cells = [200, 200],
            bc = None,
			**kwargs
			):
    
    axis = find_axis(size)
    norm_axes = [ax for ax in "xyz" if ax != axis]

    

    bc = bc or ["Anti-Symmetric", "PML", "PML", "PML"]

    properties = OrderedDict([
        ("solver type", f"2D {axis.upper()} Normal"),
        *rect_properties(size, center, axis),
        ("index", index),
        (f"mesh cells {norm_axes[0]}", mesh_cells[0]),
        (f"mesh cells {norm_axes[1]}", mesh_cells[1]),
        (f"{norm_axes[0]} min bc", bc[0]),
        (f"{norm_axes[0]} max bc", bc[1]),
        (f"{norm_axes[1]} min bc", bc[2]),
        (f"{norm_axes[1]} max bc", bc[3]),
        ("wavelength", wavelength * 1e-6),
    ])
    properties.update(kwargs)

    # sim.addfde(properties=properties)
    ensure_object(sim, "FDE", "FDE", properties)
    



@convert_length_units(["size", "center"])
def define_eme(
        sim: lumapi.MODE,
        size: list = [None, 15, 15],
        center: list = [0., 0., 0], 
        wavelength: float = 0.97,
        cells = None,
        index: float = 1,
        mesh_cells = [200, 200],
        bc = None,
        n_modes = 20,
        **kwargs
):
        wl = wavelength * 1e-6
        norm_axes = "yz"

        if len(size) < 3:
            size = [None, *size]

        bc = bc or ["Anti-Symmetric", "PML", "PML", "PML"]

        properties = OrderedDict([
            ("wavelength", wl),
            ("index", index),
            *rect_properties(size, [None, *center[1:]]),
            ("x min", center[0]),
            (f"mesh cells {norm_axes[0]}", mesh_cells[0]),
            (f"mesh cells {norm_axes[1]}", mesh_cells[1]),
            (f"{norm_axes[0]} min bc", bc[0]),
            (f"{norm_axes[0]} max bc", bc[1]),
            (f"{norm_axes[1]} min bc", bc[2]),
            (f"{norm_axes[1]} max bc", bc[3]),
            ("index", index),
            ("number of modes for all cell groups", n_modes),
            # ("allow custom eigensolver settings", 1),
            *eme_cell_properties(cells),
            ("display cells", 1),
        ])
        properties.update(kwargs)

        ensure_object(sim, "EME", "EME", properties)



def eme_cell_properties(cells):
    if cells is None:
        return []
    else:
        n_cells = np.array([c[0] for c in cells]).astype(int)
        group_spans = np.array([c[1] for c in cells])
        return [
            ("number of cell groups", len(cells)),
            ("group spans", group_spans * 1e-6),
            ("cells", n_cells),
            ("subcell method", n_cells > 1),
        ]
    

################################################################################################################
### Profiles ###################################################################################################
################################################################################################################


@convert_length_units(["size", "center"])
def define_eme_profile(sim, size=[None, 50, 50], center=[0., 0., 0]):
    axis = find_axis(size)

    name = f"profile {axis}-normal"

    properties = OrderedDict([
            ("name", name),
            ("monitor type", f"2D {axis.upper()}-normal"),
            *rect_properties(size, center),    
        ])
    
    ensure_object(sim, "emeprofile", name, properties)



@convert_length_units(["size", "center"])
def define_eme_index(sim, size=[None, 50, 50], center=[0., 0., 0]):
    axis = find_axis(size)

    name = f"profile {axis}-normal"

    properties = OrderedDict([
            ("name", name),
            ("monitor type", f"2D {axis.upper()}-normal"),
            *rect_properties(size, center),    
        ])
    
    ensure_object(sim, "emeindex", name, properties)


################################################################################################################
### Run simulations ############################################################################################
################################################################################################################


def run_eme(sim, filename=None, save_result=True):
    if filename is not None:
        sim.save(filename)
    sim.setactivesolver("EME")
    sim.run()
    if save_result and filename is not None:
        sim.save(filename)


def run_eme_propagation(sim, update=False):

    # turn off all unnecessary EME calculations
    for prop in [
        ("include fast diagnostics", 0),
        ("include slow diagnostics", 0),
        ("update monitors", update),
        ("calculate group delays", 0),
    ]: sim.setemeanalysis(*prop)

    sim.emepropagate()
    return sim.getresult("EME","user s matrix")




@convert_length_units(["size", "center"])
def define_eme_port(sim, port_number, size=None, center=None, location="left", mode_numbers=None, **kwargs):
    name = f"EME::Ports::port_{port_number}"

    properties = [
        ("port location", location),
        ("use full simulation span", False if center is not None else True),
        # ("number of trial modes", n_modes),
    ]
    if center is not None and size is not None:
        properties += rect_properties(size, center)
    
    properties = OrderedDict(properties)
    properties.update(kwargs)

    ensure_object(sim, "emeport", name, properties)

    if mode_numbers is not None:
        set_port_modes(sim, port_number, mode_numbers)

# def define_eme_ports(sim, port_specs):
#     sim.switchtolayout()
#     for spec in port_specs:
#         port = spec["port"]


def set_port_modes(sim, port_number, mode_numbers=1, neff=None):
    
    
    sim.select(f"EME::Ports::port_{port_number}")
    sim.set("mode selection","user select")
    if neff is None:
        sim.seteigensolver("use max index", 1)
    else:
        sim.seteigensolver("use max index", 0)
        sim.seteigensolver("n", neff)

    if isinstance(mode_numbers, int):
        mode_numbers = [mode_numbers]
    sim.updateportmodes(np.array(mode_numbers).astype(int))
    

def set_ports_modes(sim, ports_spec):
    sim.switchtolayout()
    for spec in ports_spec:
        port = spec["port"]
        mode = spec.get("mode", 1)
        neff = spec.get("neff", None)
        set_port_modes(sim, port, mode, neff)

        








def find_neff(sim, n_modes=5, mode_tracking=True):
    """
    Extracts the effective indices (`neff`) of optical modes from a Lumerical MODE simulation.

    Parameters
    ----------
    sim : lumapi.MODE
        An active Lumerical MODE simulation handle.
    
    n_modes : int, optional
        Number of trial modes to request and retrieve (default: 5).
    
    mode_tracking : bool, optional
        If True, use mode tracking to match and update test modes across simulations.
        If False, just read out the raw modes returned by the solver.

    Returns
    -------
    neffs : np.ndarray
        A 1D array of shape (n_modes,) containing the absolute value of the effective indices.
        NaN is returned for any missing modes or errors.
    
    Notes
    -----
    - This function assumes that a mode solver region named "FDE" exists.
    - The simulation must be in layout mode to set solver parameters.
    - For mode tracking, this function attempts to overlap previously stored modes (`test_mode#`)
      with the new simulation results and updates them.
    - Be sure to clear or manage D-cards (`test_mode#`) appropriately outside this function.
    """
    # Ensure simulation is in layout mode
    sim.switchtolayout()

    # Configure mode solver
    sim.setnamed("FDE", "number of trial modes", n_modes)
    sim.findmodes()

    # Retrieve how many modes were found
    n_found = int(sim.nummodes())
    neffs = np.full(n_modes, np.nan)

    if mode_tracking:
        mapping = {}

        # Try to overlap each test mode with found modes
        for n in range(n_found):
            try:
                test_mode = f"test_mode{n + 1}"
                mapping[test_mode] = sim.bestoverlap(test_mode)
            except Exception:
                continue  # Skip if overlap fails

        # Copy overlapping modes and store neff
        for n, (test_mode, matched_mode) in enumerate(mapping.items()):
            neffs[n] = np.abs(np.squeeze(sim.getdata(f"FDE::data::{matched_mode}", "neff")))
            sim.cleardcard(test_mode)
            sim.copydcard(matched_mode, test_mode)

        # Fill in remaining modes not matched via overlap
        used_indices = {int(re.search(r'\d+', v).group()) for v in mapping.values() if re.search(r'\d+', v)}
        rem_modes = [i + 1 for i in range(min(n_modes, n_found)) if (i + 1) not in used_indices]

        for n, mode_nr in enumerate(rem_modes, start=len(mapping)):
            if n >= n_modes:
                break  # Avoid going out of bounds
            mode_name = f"mode{mode_nr}"
            neffs[n] = np.abs(np.squeeze(sim.getdata(f"FDE::data::{mode_name}", "neff")))
            sim.copydcard(mode_name, f"test_mode{n + 1}")

    else:
        # Simply return neff of modes without any tracking
        for n in range(min(n_modes, n_found)):
            neffs[n] = np.abs(np.squeeze(sim.getdata(f"FDE::data::mode{n+1}", "neff")))

    return neffs



################################################################################################################
### Overlap calculation#########################################################################################
################################################################################################################


def calculate_best_overlap(sim, beam, shift=(0, 0, 0), optimize=False):
    sim.findmodes()
    best_mode = sim.bestoverlap(beam)
    if optimize:
        find_nr = lambda s: int(re.search(r'(\d+)$', s).group(1))  # This function is needed to find the trailing number of the string
        sim.setanalysis('shift d-card center', True)
        shift = np.squeeze(sim.optimizeposition(find_nr(best_mode), find_nr(beam)))
    ovp = sim.overlap(best_mode, beam, *shift)
    return ovp[1]



def create_gaussian_beam(sim, beam_diameter, sim_size, pol_ang=0):
    sim.switchtolayout()
    sim.select("FDE")
    sim.cleardcard()
    properties = [
        ("use fully vectorial thin lens beam profile", 0),
        ("define gaussian beam by","waist size and position"),
        ("beam direction","2D X normal"),
        ("waist radius", (beam_diameter * 1e-6) / 2),
        ("distance from waist", 0),
        ("refractive index", 1.45),
        ("theta", 0),
        ("phi", 0),
        ("polarization angle", pol_ang),
        ("sample span", np.array(sim_size)*1e-6),
        ("sample resolution", 300),
    ]
    
    set_properties(sim, "FDE", properties)

    return sim.createbeam()



################################################################################################################
### Modify group properties ####################################################################################
################################################################################################################


def expand_cell_group(sim, cell_group_number):

    eme_props = get_properties(sim, "EME", ["cells", "subcell method", "group spans", "modes"])

    cells = np.squeeze(eme_props["cells"])
    subcell_method = np.squeeze(eme_props["subcell method"])
    group_spans = np.squeeze(eme_props["group spans"])
    modes = np.squeeze(eme_props["modes"])

    index = cell_group_number - 1

    N = int(round(cells[index]))
    ones = np.ones(N)
    
    def repeated(value):
        return value + np.zeros(N)

    # Slice helpers
    before = slice(0, index)
    after = slice(index + 1, None)

    # Expand each property
    new_cells = np.concatenate([cells[before], ones, cells[after]])
    new_subcell = np.concatenate([subcell_method[before], repeated(subcell_method[index]), subcell_method[after]])
    new_spans = np.concatenate([group_spans[before], repeated(group_spans[index] / N), group_spans[after]])
    new_modes = np.concatenate([modes[before], repeated(modes[index]), modes[after]])

    # Apply changes
    set_properties(sim, "EME", {
        "number of cell groups": len(new_cells),
        "cells": new_cells.astype(int),
        "subcell method": new_subcell.astype(int),
        "group spans": new_spans,
        # "allow custom eigensolver settings": 1,
        # "modes": new_modes.astype(int)
    })

    return slice(index, index + int(cells[index]) )


def collapse_cell_group(sim, group_range):
    """
    Collapse a previously expanded cell group defined by a slice or (start, stop) tuple.
    
    Parameters:
        sim: The simulation object.
        group_range: A slice object or a tuple of (start, stop) indicating the expanded group range.
    """

    # Convert tuple to slice if needed
    if isinstance(group_range, tuple):
        group_range = slice(*group_range)

    eme_props = get_properties(sim, "EME", ["cells", "subcell method", "group spans", "modes"])

    cells = np.squeeze(eme_props["cells"])
    subcell_method = np.squeeze(eme_props["subcell method"])
    group_spans = np.squeeze(eme_props["group spans"])
    modes = np.squeeze(eme_props["modes"])

    # Indices
    start = group_range.start
    stop = group_range.stop  # slice is inclusive in your expand function's return

    # Sanity check
    N = stop - start

    # Collapse to single group
    collapsed_cells = np.concatenate([cells[:start], [N], cells[stop:]])
    collapsed_subcell = np.concatenate([subcell_method[:start], [subcell_method[start]], subcell_method[stop:]])
    collapsed_spans = np.concatenate([group_spans[:start], [np.sum(group_spans[start:stop])], group_spans[stop:]])
    collapsed_modes = np.concatenate([modes[:start], [modes[start]], modes[stop:]])

    # Apply updated properties
    set_properties(sim, "EME", {
        "number of cell groups": len(collapsed_cells),
        "cells": collapsed_cells.astype(int),
        "subcell method": collapsed_subcell.astype(int),
        "group spans": collapsed_spans,
        # "allow custom eigensolver settings": 1,
        # "modes": collapsed_modes.astype(int)
    })

    # Return the index of the collapsed group
    return start







def set_expanded_cells_properties(sim, neff, group_cell=None):
    sim.switchtolayout()
    n_cells = sim.getnamed("EME", "cells")

    if sum(n_cells) != len(neff):
        print("neff array dim does not match nr of cells")
        return

    if group_cell is None:
        cell_ranges = [expand_cell_group(sim, sum(n_cells[:i]) + 1) for i in range(len(n_cells))]
    else:
        cell_ranges = [expand_cell_group(sim, group_cell)]
        

    for i, n in enumerate(neff):
        sim.select(f"EME::Cells::cell_{i + 1}")
        sim.seteigensolver("use max index", 0)
        sim.seteigensolver("n", n)

    # sim.setactivesolver("EME")
    # sim.run()

    return cell_ranges





def set_eme_group_lengths(sim, cell_range, lengths):
    if isinstance(cell_range, int):
        cell_range = slice(cell_range, cell_range + 1)
    elif isinstance(cell_range, tuple):
        cell_range = slice(*cell_range)

    if sim.layoutmode():
        print("Operation not possible - Run EME first")
        return  # Possibly we could run EME here
    

    group_spans = sim.getemeanalysis("group spans").squeeze()
    group_spans[cell_range] = np.array(lengths) * 1e-6  # in microns
    sim.setemeanalysis("group spans", group_spans)
    return group_spans


def set_eme_group_by_uniform_length(sim, cell_range, length):
    n = cell_range.stop - cell_range.start if isinstance(cell_range, slice) else 1
    lengths = np.full(n, length) / n
    return set_eme_group_lengths(sim, cell_range, lengths)


def set_eme_group_lengths_by_profile(sim, cell_range, total_length, profile_fn=None):
    n = cell_range.stop - cell_range.start
    if profile_fn is None:
        profile_fn = lambda n: 1 / n
    weights = profile_fn(n)
    lengths = weights * total_length
    return set_eme_group_lengths(sim, cell_range, lengths)



@convert_length_units(["lengths"])
def run_expanded_cells_propagation_sweep(sim, lengths, cell_range):
    S_sweep = sweep_param_result(
        sim,
        sweep_params={"length": lengths},
        geometry_params={"cell_range": cell_range},
        geometry_fn=set_eme_group_lengths,
        result_fn=run_eme_propagation,
    )
    return np.stack(S_sweep)





