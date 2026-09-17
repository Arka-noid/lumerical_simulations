import lumapi
from collections import OrderedDict
from typing import Tuple, List, Optional, Sequence
import numpy as np
from functools import wraps
import pandas as pd
from scipy.interpolate import CubicSpline


# ------------------------------------------------------------
# Global Unit Configuration
# ------------------------------------------------------------

class UnitConfig:
    """
    Global configuration for default units used in simulations.

    Attributes:
        default_length_unit (str): Default unit for length values. Must be one of:
            'nm', 'um', 'cm', 'm'.
    """
    default_length_unit = 'um'

    @classmethod
    def set_unit(cls, unit: str):
        cls.default_length_unit = unit

    @classmethod
    def get_unit(cls) -> str:
        return cls.default_length_unit


def length_convert(x, unit=None):
    factors = {'um': 1e-6, 'nm': 1e-9, 'cm': 1e-2, 'm': 1}
    unit = unit or UnitConfig.get_unit()

    if x is None:
        return None

    has_none = any(xi is None for xi in x)

    converted = [
        round(xi * factors[unit], 9) if xi is not None else None
        for xi in x
    ]

    return converted # if has_none else np.array(converted)




def convert_length_units(keys_to_convert):
    """
    Decorator that automatically converts specified function arguments to meters.

    Args:
        keys_to_convert (list of str): List of parameter names to convert.

    Usage:
        @convert_length_units(['wl0', 'deltawl'])
        def your_function(wl0, deltawl, ...):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, unit=None, **kwargs):
            unit = unit or UnitConfig.get_unit()
            func_params = func.__code__.co_varnames[:func.__code__.co_argcount]
            func_defaults = func.__defaults__ or ()
            default_dict = dict(zip(func_params[-len(func_defaults):], func_defaults))

            for key in keys_to_convert:
                if key in kwargs and kwargs[key] is not None:
                    kwargs[key] = length_convert(kwargs[key], unit)
                elif key in default_dict and default_dict[key] is not None:
                    kwargs[key] = length_convert(default_dict[key], unit)

            kwargs.pop('unit', None)  # Avoid passing 'unit' to the wrapped function
            return func(*args, **kwargs)
        return wrapper
    return decorator


def replace_underscores_dict(d):
    return {k.replace("_", " "): v for k, v in d.items()}

def set_properties( sim: lumapi.Lumerical, 
					item: str, 
					properties: OrderedDict):
    properties = OrderedDict(properties)
    for property, value in properties.items():
        sim.setnamed(item, property.replace("_", " "), value)
    return sim


def get_properties(sim, item: str, properties: list):
    return OrderedDict([(prop, sim.getnamed(item, prop)) for prop in properties])


def lin2db(x):
    return 10 * np.log10(abs(x)) 

def db2lin(x):
     return 10 ** (x / 10)


def add_lumerical_object(sim, 
                         obj_type: str, 
                         properties: Optional[OrderedDict] = None):
    obj_type_alias = {
        "FDTD": "fdtd",
        "FDE": "fde",
        "EME": "eme",
        "rectangle": "rect",
        "eme_port": "emeport",
    }

    obj_type = obj_type_alias.get(obj_type, obj_type)
    properties = properties or OrderedDict()

    add_func = getattr(sim, f"add{obj_type}", None)
    if add_func is None:
        raise ValueError(f"[add_lumerical_object] No such object type: add{obj_type}")

    try:
        return add_func(properties=properties)
    except lumapi.LumApiError as e:
        print(f"[add_lumerical_object] Failed to add '{obj_type}' object with properties:\n{properties}")
        print(f"[add_lumerical_object] Error: {e}")
        raise


        
def ensure_object(sim, obj_type, name, properties=None, defaults=None):
    properties = properties or OrderedDict()
    defaults = defaults or OrderedDict()

    try:
        # Try to update existing object with given properties only
        set_properties(sim, name, properties)
    except lumapi.LumApiError as e:
        not_found_msg = f"in setnamed, no items matching the name '{name}' can be found."

        if not_found_msg in str(e):
            print(f"[ensure_object] '{name}' not found. Creating a new '{obj_type}'...")

            # Ensure name is in defaults if missing
            all_props = OrderedDict([
                *defaults.items(),
                *properties.items()
            ])

            # Insert the name into properties if not already present
            # if "name" not in properties:
            #     properties = OrderedDict([("name", name), *properties.items()])
            
            add_lumerical_object(sim, obj_type, all_props)
        else:
            print(f"[ensure_object] Error setting properties for '{name}': {e}")
            raise


def interp_sim_data(df: pd.DataFrame, columns, parameter=None, interp_function=CubicSpline):
    """
    Interpolates specified columns of a DataFrame using the given interpolation function.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the simulation data.
    columns : list or str
        Columns to interpolate.
    parameter : str or None
        The parameter to interpolate over. If None, uses the index.
    interp_function : callable
        A function/class from scipy.interpolate (e.g., CubicSpline, interp1d, etc.).

    Returns:
    --------
    dict
        A dictionary {column_name: interpolation_function}
    """
    if isinstance(columns, str):
        columns = [columns]

    x = df[parameter].values if parameter else df.index.values

    interpolators = {}
    for col in columns:
        y = df[col].values
        # Remove NaNs to avoid interpolation errors
        mask = ~pd.isna(x) & ~pd.isna(y)
        if mask.sum() < 2:
            raise ValueError(f"Not enough valid points to interpolate column '{col}'.")
        interpolators[col] = interp_function(x[mask], y[mask])

    return interpolators


def unpack_array_column(df, array_col_name, prefix="neff", index=None):
    """
    Unpacks a column of 1D numpy arrays into multiple scalar columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        array_col_name (str): The name of the column containing 1D numpy arrays.
        prefix (str): Prefix for the new column names (default: "neff").

    Returns:
        pd.DataFrame: A new DataFrame with the arrays unpacked into columns.
    """
    # Create a temporary DataFrame with columns for each array element
    array_df = df[array_col_name].apply(
        lambda x: pd.Series(x) if isinstance(x, (list, np.ndarray)) else pd.Series(dtype=float)
    )
    array_df.columns = [f"{prefix}_{i+1}" for i in array_df.columns]

    # Drop the original array column and concatenate the unpacked columns
    df = df.drop(columns=[array_col_name])
    new_df = pd.concat([df, array_df], axis=1)
    if index is not None:
        new_df = new_df.set_index(index)
    return new_df



def add_material_from_file(sim, name, filename):
    mat = sim.addmaterial("Sampled data")

    mat_data = pd.read_csv(filename, sep=r'\s+', header=None)
    c = 299792458

    if mat_data.shape[1] == 2:
        mat_data.columns = ['wl', 'n']
    elif mat_data.shape[1] == 3:
        mat_data = pd.DataFrame({
            'wl': mat_data.iloc[:, 0],
            'n': mat_data.iloc[:, 1] + 1j * mat_data.iloc[:, 2]
        })
    mat_data["f"] = c / mat_data["wl"] * 1e9
    mat_data["eps"] = mat_data["n"] ** 2
    
    sim.setmaterial(mat, "name", name)
    sim.setmaterial(name,"sampled data", mat_data[["f", "eps"]].to_numpy())








        

