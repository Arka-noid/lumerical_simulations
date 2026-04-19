from ast import Call
import numpy as np
import itertools
from collections import OrderedDict
from typing import Callable, Optional, Dict, List, Generator
from functools import partial
import pandas as pd
from tqdm import tqdm
import os

import traceback



## Deprecating this
def sweep_param_result(sim, 
                       sweep_params: dict, 
                       geometry_fn: Callable, 
                       geometry_params: Optional[dict] = None,
                       result_fn: Optional[Callable] = None,
                       result_params: Optional[dict] = None,
                       result_type = object,
                       result_shape: Optional[tuple] = None,
                       pass_geom_output_to_result_fn: bool = False,
                       post_process_fn: Optional[Callable] = None,
                       post_process_params: Optional[dict] = None,
                       save_path: str = "sweep_results.csv",
                       resume: bool = False
                       ):
    """
    Evaluates a result function over an N-dimensional sweep of parameters by modifying a simulation object.

    Parameters
    ----------
    sim : object
        The simulation object to be modified and evaluated. Assumes `geometry_fn` and `result_fn` 
        can operate on it directly.

    sweep_params : dict
        A dictionary (preferably an OrderedDict for deterministic axis order) where each key is a parameter 
        name and each value is an iterable of values to sweep. All combinations of parameter values 
        are evaluated.

    geometry_fn : callable
        A function with signature `geometry_fn(sim, **params)` that modifies the simulation object 
        in-place based on the current set of sweep parameters.

    geometry_params : dict, optional
        Additional keyword arguments passed to `geometry_fn` at each iteration.

    result_fn : callable, optional
        A function with signature `result_fn(sim, **kwargs)` that computes and returns the 
        result of interest after the simulation has been modified. If `geometry_fn` returns a value 
        and `pass_geom_output_to_result_fn` is True, the unpacked result will also be passed as 
        keyword arguments to `result_fn`.

    result_params : dict, optional
        Additional keyword arguments passed to `result_fn` during evaluation.

    result_type : type, optional
        The data type of the result array. Use `float` for scalar metrics, or `object` for general 
        results like arrays or dictionaries. Defaults to `object`.

    pass_geom_output_to_result_fn : bool, optional
        If True, the output of `geometry_fn` (if not None) is unpacked into keyword arguments and 
        passed to `result_fn`. The output of `geometry_fn` must be a dictionary, a scalar, or None.
        Scalars are wrapped as {'output': value}. If False, `result_fn` only receives `result_params`.

    Returns
    -------
    result : np.ndarray
        An N-dimensional array with shape determined by the lengths of the sweep parameter value lists. 
        Each entry contains the result of `result_fn` for the corresponding parameter combination.

    Notes
    -----
    - The order of keys in `sweep_params` determines the axis ordering in the output array.
    - `sim` is modified in-place for each evaluation; ensure it is reusable or resettable between runs.
    - The return value from `geometry_fn`, if any, is passed to `result_fn` as keyword arguments 
      only if `pass_geom_output_to_result_fn=True`. This assumes compatibility of those keys with 
      the signature of `result_fn`.
    """

    if result_params is None:
        result_params = {}
    if geometry_params is None:
        geometry_params = {}
    if post_process_params is None:
        post_process_params = {}

    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    combinations = list(itertools.product(*param_values))
    sweep_shape = [len(values) for values in param_values]
    total = np.prod(sweep_shape)
    param_values = [list(v) for v in param_values]

    # df = pd.DataFrame(columns=param_names + ["result"])
    # Set up results DataFrame with known dtypes
    df = pd.DataFrame({name: [] for name in param_names})
    df["result"] = pd.Series(dtype=result_type)


    if resume and os.path.exists(save_path):
        df = pd.read_csv(save_path)
        done_set = set(tuple(row[param] for param in param_names) for _, row in df.iterrows())
    else:
        done_set = set()

    def unpack_output(output):
        if isinstance(output, dict):
            return output
        elif output is not None:
            return {'output': output}
        else:
            return {}

    # Allocate result array
    full_shape = sweep_shape + list(result_shape) if result_shape else sweep_shape
    result_array = np.empty(full_shape, dtype=result_type)

    try:
        with tqdm(combinations, total=total) as pbar:
            for values in pbar:
                if values in done_set:
                    pbar.set_postfix(skipped=True)
                    continue

                params = {**geometry_params, **OrderedDict(zip(param_names, values))}
                row_dict = OrderedDict(zip(param_names, values))
                # Prepare short summaries for postfix
                param_summary = ', '.join(f" {k}={v}" for k, v in row_dict.items() if k != "result")
                pbar.set_postfix(parameters=param_summary)


                geom_output = geometry_fn(sim, **params)

                if result_fn is not None:
                    kwargs = {**result_params}
                    if pass_geom_output_to_result_fn:
                        kwargs.update(unpack_output(geom_output))
                    output = result_fn(sim, **kwargs)
                else:
                    output = None

                row_dict["result"] = output
                df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

                # Write into result array
                index = tuple(param_values[j].index(values[j]) for j in range(len(param_names)))
                result_array[index] = output

                if post_process_fn is not None:
                    df, result_array = post_process_fn(df, result_array, **post_process_params)

                # Append new row to the dataframe (in-memory)
                df.to_csv(save_path, index=False)


    except Exception as e:
        print("⚠️ An error occurred. Saving partial results...")
        df.to_csv(save_path, index=False)
        traceback.print_exc()
        raise e

    # Save completed results
    df.to_csv(save_path, index=False)
    print(f"✅ Sweep completed. Results saved to: {save_path}")

    return result_array, df

sweep_param_metric = partial(sweep_param_result, result_type=float)
sweep_param_smatrix = partial(sweep_param_result, result_type=float, result_shape=(2, 2))

######################################################################################################################
# New version
######################################################################################################################

from scipy.stats import qmc


def build_product_params(sweep_params: Dict[str, List]) -> List[OrderedDict]:
    """Build a full Cartesian product of parameter sets."""
    param_names = list(sweep_params.keys())
    param_values = [list(v) for v in sweep_params.values()]
    return [
        OrderedDict(zip(param_names, combo))
        for combo in itertools.product(*param_values)
    ]


def build_sequential_params(sweep_params: Dict[str, List]) -> List[OrderedDict]:
    """Sweep one parameter at a time (others fixed at first value)."""
    param_names = list(sweep_params.keys())
    defaults = {k: v[0] for k, v in sweep_params.items()}
    param_sets = []
    for name, values in sweep_params.items():
        for v in values:
            param_sets.append(OrderedDict({**defaults, name: v}))
    return param_sets


def build_lhs_params(
    sweep_params: Dict[str, List],
    n_samples: int,
    seed: Optional[int] = None,
) -> List[OrderedDict]:
    """
    Latin Hypercube Sampling.
    Works when all parameters are numeric ranges [min, max].
    """
    param_names = list(sweep_params.keys())

    # Check ranges
    bounds = []
    for v in sweep_params.values():
        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            bounds.append(v)
        else:
            raise ValueError("SciPy LHS requires numeric ranges [min, max].")

    # Generate normalized LHS samples in [0,1]
    sampler = qmc.LatinHypercube(d=len(param_names), seed=seed)
    sample = sampler.random(n=n_samples)

    # Scale to parameter ranges
    l_bounds = [b[0] for b in bounds]
    u_bounds = [b[1] for b in bounds]
    scaled = qmc.scale(sample, l_bounds, u_bounds)

    # Convert to list of OrderedDicts
    return [
        OrderedDict(zip(param_names, row))
        for row in scaled
    ]



import numpy as np
import pandas as pd
from scipy.stats import qmc, uniform, norm

def lhs_sampling(variables, n_samples, seed=None, optimization='lloyd', output_filename: str = None):
    """
    Latin Hypercube Sampling with named variables.
    
    """
    names = list(variables.keys())
    variables = list(variables.values())
    n_vars = len(variables)
    
    sampler = qmc.LatinHypercube(d=n_vars, seed=seed, optimization=optimization)
    lhs_unit = sampler.random(n=n_samples)  # uniform [0,1] samples
    
    
    samples = np.zeros_like(lhs_unit)
    
    for j, var in enumerate(variables):
        if hasattr(var, "ppf"):  # SciPy distribution
            samples[:, j] = var.ppf(lhs_unit[:, j])
        else:
            low, high = var
            mean = (low + high) / 2
            std = (high - low) / 4
            samples[:, j] = norm(loc=mean, scale=std).ppf(lhs_unit[:, j])
        
    df = pd.DataFrame(samples, columns=names)

    if output_filename is not None:
        df.to_csv(output_filename)


    return df


def build_random_params(
    sweep_params: Dict[str, List],
    n_samples: int,
    seed: Optional[int] = None,
    allow_continuous: bool = True,
) -> List[OrderedDict]:
    """
    Randomly sample parameter values from given lists or ranges.

    """
    rng = np.random.default_rng(seed)
    param_names = list(sweep_params.keys())
    param_values = [list(v) for v in sweep_params.values()]

    def sample_value(values):
        if allow_continuous and len(values) == 2 and all(isinstance(x, (int, float)) for x in values):
            return rng.uniform(values[0], values[1])
        else:
            return rng.choice(values)

    return [
        OrderedDict(zip(param_names, [sample_value(v) for v in param_values]))
        for _ in range(n_samples)
    ]


def build_normal_random_params(
        sweep_params: Dict[str, List],
        n_samples: int,
        seed: Optional[int] = None,
        ):
    rng = np.random.default_rng(seed)
    param_names = list(sweep_params.keys())
    param_sets = []

    for _ in range(n_samples):
        sample = {}
        for k, v in sweep_params.items():
            if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                mean = np.mean(v)
                std = (v[1] - v[0]) / 6  # 99.7% within [min,max]
                sample[k] = np.clip(rng.normal(mean, std), v[0], v[1])
            else:
                sample[k] = rng.choice(v)
        param_sets.append(OrderedDict(sample))
    
    return param_sets


def evaluate_param_sweep(
    sim,
    param_sets: List[OrderedDict],
    setup_fn: Callable,
    setup_params: Optional[dict] = None,
    result_fn: Optional[Callable] = None,
    result_params: Optional[dict] = None,
):
    """
    Iterate over a list of parameter sets and run the simulation.

    Yields:
        row_dict: OrderedDict with parameter values and 'result'
    """
    setup_params = setup_params or {}
    result_params = result_params or {}

    # Convert DataFrame to list of OrderedDict if needed
    if isinstance(param_sets, pd.DataFrame):
        param_sets = [
            OrderedDict(row._asdict()) if hasattr(row, "_asdict") else OrderedDict(row)
            for _, row in param_sets.iterrows()
        ]

    with tqdm(param_sets, total=len(param_sets)) as pbar:
        for params_dict in pbar:
            pbar.set_postfix(params_dict)

            # Merge with fixed setup params
            params = {**setup_params, **params_dict}
            row_dict = OrderedDict(params_dict)

            # Apply setup
            setup_fn(sim, **params)

            # Compute result
            row_dict["result"] = (
                result_fn(sim, **result_params) if result_fn else None
            )

            yield row_dict


def run_param_sets(
    sim,
    param_sets: List[OrderedDict],
    setup_fn: Callable,
    setup_params: Optional[dict] = None,
    result_fn: Optional[Callable] = None,
    result_params: Optional[dict] = None,
    postprocess_fn: Optional[Callable] = None,
    postprocess_params: Optional[dict] = None,
    save_path: str = "sweep_results.csv",
    save_during_run: bool = True,
) -> pd.DataFrame:
    """
    Run a parameter sweep (or set of parameter combinations), collecting results into a DataFrame.

    Args:
        sim: Simulation object (passed to setup and result functions).
        param_sets: List of OrderedDicts, where each dict defines a parameter combination.
        setup_fn: Function to configure the simulation geometry or parameters.
                  Signature: setup_fn(sim, **params).
        setup_params: Extra kwargs passed to setup_fn in addition to the parameter set.
        result_fn: Function to evaluate results from the simulation.
                   Signature: result_fn(sim, **params) -> dict
        result_params: Extra kwargs passed to result_fn.
        postprocess_fn: Optional function to apply to the final DataFrame.
                        Signature: postprocess_fn(df: pd.DataFrame, **params) -> pd.DataFrame
        postprocess_params: Extra kwargs passed to postprocess_fn.
        save_path: File path to save results as CSV.
        save_during_run: If True, append/save results after each parameter set (for robustness).

    Returns:
        pd.DataFrame: Final concatenated DataFrame of results.
    """
    setup_params = setup_params or {}
    result_params = result_params or {}
    postprocess_params = postprocess_params or {}

    df = pd.DataFrame()

    try:
        for row_dict in evaluate_param_sweep(
            sim, param_sets, setup_fn, setup_params, result_fn, result_params
        ):
            df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

            if save_during_run:
                df.to_csv(save_path, index=False)

    except Exception as e:
        print("⚠️ Error occurred, saving partial results.")
        df.to_csv(save_path, index=False)
        traceback.print_exc()
        raise e

    if postprocess_fn is not None:
        df = postprocess_fn(df, **postprocess_params)

    df.to_csv(save_path, index=False)
    print(f"✅ Sweep completed. Results saved to {save_path}")

    return df


# def _build_run(param_builder, **build_args):
#     def run(sim, range_params, *args, **kwargs):
#         return run_param_sets(sim,
#                               param_sets=param_builder(range_params, **build_args),
#                               *args, **kwargs
#                               )
    
# run_sweep = _build_run(build_product_params)
# run_sequential = _build_run(build_sequential_params)
# run_random = _build

def run_sweep(
    sim,
    sweep_params: dict,
    setup_fn: Callable,
    result_fn: Callable,
    postprocess_fn: Optional[Callable] = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Run a full grid search sweep over parameters.

    Args:
        sim: Simulation object.
        sweep_params: Dictionary of parameters to sweep.
                      Keys = parameter names, values = list of values.
        *args, **kwargs: Passed through to run_param_sets.

    Returns:
        pd.DataFrame: Concatenated DataFrame of results.
    """
    return run_param_sets(
        sim,
        param_sets=build_product_params(sweep_params),
        setup_fn=setup_fn,
        result_fn=result_fn,
        postprocess_fn=postprocess_fn,
        *args,
        **kwargs,
    )


def run_random(
    sim,
    sweep_params: dict,
    setup_fn: Callable,
    result_fn: Callable,
    postprocess_fn: Optional[Callable] = None,
    n_samples: int = 50,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Run a random sweep (sampled parameter combinations).

    Args:
        sim: Simulation object.
        sweep_params: Dictionary of parameters with value ranges or distributions.
        *args, **kwargs: Passed through to run_param_sets.
                         May include "n_samples" to control sample size.

    Returns:
        pd.DataFrame: Concatenated DataFrame of results.
    """

    return run_param_sets(
        sim,
        param_sets=build_random_params(sweep_params, n_samples=n_samples),
        setup_fn=setup_fn,
        result_fn=result_fn,
        postprocess_fn=postprocess_fn,
        *args,
        **kwargs,
    )


def run_lhs(
        sim,
        sweep_params: dict,
        setup_fn: Callable,
        result_fn: Callable,
        postprocess_fn: Optional[Callable] = None,
        n_samples: int = 50,
        *args,
        **kwargs,

) -> pd.DataFrame: 
    
    return run_param_sets(
        sim,
        param_sets=build_lhs_params(sweep_params, n_samples=n_samples),
        setup_fn=setup_fn,
        result_fn=result_fn,
        postprocess_fn=postprocess_fn,
        *args,
        **kwargs,
    )