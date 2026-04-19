import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ast
import seaborn as sns
import itertools

from scipy.stats import norm
from pyDOE import lhs
from typing import Union


from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol







def lhs_sampling_from_normal_distributions(input_parameters, output_file: str = None, num_samples=None):
    """
    Generate Latin Hypercube Samples assuming Gaussian distributions
    for each parameter. Ranges in input_parameters are interpreted as ±3σ.
    
    Args:
        input_parameters (dict): {param: [min_val, max_val]} where [min, max] ~ mean ± 3σ
        lhs_filename (str): output CSV filename (without extension)
        num_samples (int): number of samples. Defaults to 10 * num_params.
        
    Returns:
        pd.DataFrame with samples (including ID column).
    """
    
    # Compute mean and std for each parameter
    param_means = {k: (v[0] + v[1]) / 2 for k, v in input_parameters.items()}
    param_stds  = {k: (v[1] - v[0]) / 6 for k, v in input_parameters.items()}  # since ±3σ = range/2

    num_params = len(input_parameters)
    if num_samples is None:
        num_samples = 10 * num_params

    # Step 1: Generate LHS in [0,1]
    lhs_unit = lhs(num_params, samples=num_samples, criterion='maximin')

    # Step 2: Map to Gaussian (mean 0, std 1)
    lhs_gaussian = norm.ppf(lhs_unit)

    # Step 3: Scale and shift to each parameter's mean, std
    samples = np.zeros_like(lhs_gaussian)
    for i, (name, mean) in enumerate(param_means.items()):
        std = param_stds[name]
        samples[:, i] = lhs_gaussian[:, i] * std + mean

    # Step 4: Build DataFrame with IDs
    df = pd.DataFrame(samples, columns=list(input_parameters.keys()))
  

    # Save to CSV
    if output_file is not None:
        df.to_csv(output_file + ".csv", index=False)

    # Preview
    print("✅ Generated LHS samples with Gaussian variability (bounded by ±3σ).")
    for name in input_parameters:
        print(f"{name}: mean≈{df[name].mean():.4f}, std≈{df[name].std():.4f}, range=({df[name].min():.4f}, {df[name].max():.4f})")

    print("\nFirst 5 samples:")
    print(df.head())

    return df



def analyze_montecarlo_results(
    data: Union[str, pd.DataFrame],
    delimiter: str = "\t",
    title: str = None,
    x_target: float = None,
    scalar_outputs: list = None,
    array_outputs: list = None,
    plot_mode: str = "std",          # "std" or "percentile"
    percentile_width: int = 80,      # only for percentile mode
    y_lim: list = None,
    x_lim: list = None,
):
    """
    Perform statistical analysis on Monte Carlo results.

    Args:
        data: Path to file (CSV/TSV) or a pandas DataFrame.
        delimiter: Delimiter if loading from file (default="\t").
        title: Optional title prefix for plots.
        x_target: Vertical marker line for array outputs.
        scalar_outputs: List of scalar output column names.
        array_outputs: List of (x_col, y_col) tuples for array outputs.
        plot_mode: "std" or "percentile" (controls spread visualization).
        percentile_width: Percentile band width for "percentile" mode.
        y_lim, x_lim: Optional axis limits [min, max].

    Returns:
        dict with summary statistics for scalar outputs.
    """
    # ------------------
    # Load Data
    # ------------------
    if isinstance(data, str):
        df = pd.read_csv(data, delimiter=delimiter)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("data must be a file path or a pandas DataFrame")

    # Convert stringified lists into Python lists if present
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
                )
            except Exception:
                pass

    stats = {}

    # ------------------
    # Scalar outputs
    # ------------------
    if scalar_outputs is None:
        scalar_outputs = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]

    lower_p = (100 - percentile_width) / 2
    upper_p = 100 - lower_p
    percentiles = [lower_p, upper_p]

    for col in scalar_outputs:
        data_col = df[col].dropna().astype(float).values

        stats[col] = {
            "mean": np.mean(data_col),
            "std": np.std(data_col),
            "median": np.median(data_col),
            "percentiles": np.percentile(data_col, percentiles),
        }

        # === Plot
        plt.figure(figsize=(8, 4))
        plt.hist(data_col, bins=50, alpha=0.6, color="gray", label="Samples")

        if plot_mode == "std":
            mean, std = stats[col]["mean"], stats[col]["std"]
            plt.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean:.3f}")
            plt.axvline(mean - 3 * std, linestyle=":", label=f"Mean - 3σ = {mean-3*std:.3f}")
            plt.axvline(mean + 3 * std, linestyle=":", label=f"Mean + 3σ = {mean+3*std:.3f}")

        elif plot_mode == "percentile":
            median = stats[col]["median"]
            plt.axvline(median, color="blue", linestyle="-", label=f"Median = {median:.3f}")
            for p, val in zip(percentiles, stats[col]["percentiles"]):
                plt.axvline(val, linestyle=":", label=f"{p:.0f}th = {val:.3f}")

        if y_lim: plt.ylim(y_lim[0], y_lim[1])
        if x_lim: plt.xlim(x_lim[0], x_lim[1])

        plot_title = f"Statistical Analysis of {col}"
        if title:
            plot_title += f" : {title}"
        plt.title(plot_title)
        plt.xlabel(col)
        plt.ylabel("Counts")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------
    # Array outputs
    # ------------------
    if array_outputs:
        for (x_col, y_col) in array_outputs:
            x_vals = np.array(df[x_col].iloc[0]) * 1e6  # µm conversion
            y_arrays = np.vstack(df[y_col].values)

            if plot_mode == "std":
                center = np.mean(y_arrays, axis=0)
                lower = center - np.std(y_arrays, axis=0)
                upper = center + np.std(y_arrays, axis=0)
                label_center, label_band = "Mean", "±1σ"
            else:  # percentile
                center = np.median(y_arrays, axis=0)
                lower = np.percentile(y_arrays, lower_p, axis=0)
                upper = np.percentile(y_arrays, upper_p, axis=0)
                label_center = "Median"
                label_band = f"{lower_p:.0f}–{upper_p:.0f} percentile"

            plt.figure(figsize=(5, 4))
            plt.plot(x_vals, center, label=label_center, color="blue")
            plt.fill_between(x_vals, lower, upper, alpha=0.3, label=label_band, color="blue")

            if x_target:
                plt.axvline(x_target, color="red", linestyle="--", label=f"Target = {x_target} µm")

            if y_lim: plt.ylim(y_lim[0], y_lim[1])
            if x_lim: plt.xlim(x_lim[0], x_lim[1])

            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"Monte Carlo Analysis - {title or ''}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return stats



def build_gpr_model(data, input_parameters=None, target_col='result', kernel=None, return_scaler=True, seed=None):
    """
    Build a Gaussian Process Regression (GPR) model.
    
    Parameters
    ----------
    data : str or pd.DataFrame
        Path to CSV/TSV file or a DataFrame containing the dataset.
    variable_cols : list of str
        Column names for input features.
    target_col : str
        Column name of the target variable (scalar output).
    kernel : sklearn.gaussian_process.kernels.Kernel, optional
        Custom kernel. If None, a default RBF + WhiteKernel is used.
    return_scaler : bool
        If True, return the fitted StandardScaler along with the model.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    gpr : GaussianProcessRegressor
        Trained GPR model.
    scaler : StandardScaler (optional)
        Fitted scaler for input features (returned if return_scaler=True).
    """
    # Load data
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data
    

    
    # Extract features and target
    if isinstance(input_parameters, dict):
        variable_cols = list(input_parameters.keys())
    elif input_parameters is None:
        variable_cols = [k for k in df.columns if k != target_col]
    else:
        variable_cols = input_parameters

    X = df[variable_cols].values
    y = df[target_col].values
    
    # Scale inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define kernel if not provided
    if kernel is None:
        kernel = (C(1.0, (1e-3, 1e3)) *
                  RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                  WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1)))
    
    # Create and fit GPR
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   n_restarts_optimizer=20,
                                   normalize_y=True,
                                   random_state=seed)
    gpr.fit(X_scaled, y)
    
    print("Optimized kernel:", gpr.kernel_)
    
    if return_scaler:
        return gpr, scaler
    return gpr




def gpr_prediction(mc_file,
                   gpr,
                   X_new,
                   input_parameters,
                   separator="\t",
                   ):
    # Load TSV file
    df = pd.read_csv(mc_file, sep=separator)

    # Scale new inputs with the same scaler
    df = pd.read_csv(mc_file, sep=separator)
    X = df[list(input_parameters.keys())].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 
    X_new_scaled = scaler.transform(X_new)

    # Get prediction and standard deviation
    y_pred, y_std = gpr.predict(X_new_scaled, return_std=True)

    print("Predicted IL0:", y_pred[0])
    print("Prediction uncertainty (std):", y_std[0])

    return y_pred[0], y_std[0]




def sobol_analysis(gpr, input_parameters, scaler, N=2**12, top_n=5, show_plots=True):
    """
    Perform Sobol sensitivity analysis using a GPR surrogate.

    Parameters
    ----------
    gpr : trained GaussianProcessRegressor
    input_parameters : dict {name: [min, max]}
    scaler : StandardScaler used for GPR inputs
    N : int, number of base Sobol samples (N*(2D+2) total)
    top_n : int, top interactions to show
    show_plots : bool, whether to plot results

    Returns
    -------
    Si : dict of Sobol indices (S1, ST, S2)
    """
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol
    
    names = list(input_parameters.keys())
    problem = {"num_vars": len(names), "names": names, "bounds": list(input_parameters.values())}

    # Generate Sobol samples
    param_values = sobol_sample.sample(problem, N, calc_second_order=True)
    X_eval = scaler.transform(param_values)
    Y_eval, _ = gpr.predict(X_eval, return_std=True)

    # Sobol analysis
    Si = sobol.analyze(problem, Y_eval, calc_second_order=True, print_to_console=False)

    if show_plots:
        plot_sobol_indices(Si, names, top_n)
    
    return Si


def plot_sobol_indices(Si, names, top_n=5):
    """
    Plot Sobol sensitivity indices (first-order, total-order, second-order).
    
    Parameters
    ----------
    Si : dict
        Output of SALib Sobol analysis containing 'S1', 'ST', 'S2'.
    names : list of str
        Names of input variables.
    top_n : int
        Number of top second-order interactions to show.
    """
    n_vars = len(names)
    indices = np.arange(n_vars)
    width = 0.35

    # --- First-order and Total-order bar chart ---
    plt.figure(figsize=(6,4))
    plt.bar(indices, Si["S1"], width, label="First-order")
    plt.bar(indices + width, Si["ST"], width, label="Total-order")
    plt.xticks(indices + width/2, names, rotation=45)
    plt.ylabel("Sobol index")
    plt.title("First-order and Total-order Sobol indices")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- Heatmap of second-order interactions ---
    S2 = np.nan_to_num(Si["S2"], nan=0.0)
    mask = np.triu(np.ones_like(S2, dtype=bool))
    
    annot = np.empty_like(S2).astype(str)
    for i in range(S2.shape[0]):
        for j in range(S2.shape[1]):
            if mask[i, j]:
                annot[i, j] = ""
            else:
                annot[i, j] = f"{S2[i, j]:.2f}"
    
    plt.figure(figsize=(7,5))
    sns.heatmap(S2, xticklabels=names, yticklabels=names,
                cmap="plasma", annot=annot, fmt="", mask=mask.T,
                cbar_kws={"label": "S2 index"})
    plt.title("Second-order Interaction Effects")
    plt.tight_layout()
    plt.show()
    
    # --- Top N second-order interactions bar chart ---
    pairs, values = [], []
    for i, j in itertools.combinations(range(n_vars), 2):
        pairs.append(f"{names[i]} × {names[j]}")
        values.append(S2[i, j])
    
    pairs = np.array(pairs)
    values = np.array(values)
    sorted_idx = np.argsort(values)[::-1]
    
    plt.figure(figsize=(6,4))
    plt.barh(pairs[sorted_idx][:top_n], values[sorted_idx][:top_n])
    plt.gca().invert_yaxis()  # largest on top
    plt.xlabel("Second-order Sobol index (S2)")
    plt.title(f"Top {top_n} Second-order Interactions")
    plt.tight_layout()
    plt.show()