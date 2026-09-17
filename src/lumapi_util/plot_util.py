import numpy as np
import matplotlib.pyplot as plt


def plot_metric_1d(params: dict, metric: np.ndarray, 
                   x_param: str,
                   fixed_params: dict = None,
                   metric_name="Metric",
                   show=True,
                   **plot_kwargs):
    """
    Plots a 1D slice of an N-dimensional result array: metric vs x_param,
    keeping all other parameters fixed.

    Parameters
    ----------
    params : dict
        Dictionary mapping parameter names to arrays of values.

    metric : np.ndarray
        N-dimensional array of result values.

    x_param : str
        The parameter to plot on the x-axis.

    fixed_params : dict, optional
        Dictionary of parameter names (excluding x_param) to fixed indices.

    metric_name : str, optional
        Label for the y-axis.

    show : bool, optional
        Whether to display the plot immediately. Defaults to True.

    **plot_kwargs : dict
        Extra keyword arguments for `plt.plot()`.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and axes objects for further customization or saving.
    """
    if fixed_params is None:
        fixed_params = {}

    param_names = list(params.keys())
    param_axes = {name: i for i, name in enumerate(param_names)}
    assert x_param in param_names, "x_param must be a key in params"

    # Fix other axes to 0 unless specified
    slicer = [0] * metric.ndim
    x_axis = param_axes[x_param]
    slicer[x_axis] = slice(None)

    for name, idx in fixed_params.items():
        assert name in param_names and name != x_param, f"Invalid fixed param: {name}"
        slicer[param_axes[name]] = idx

    x_values = params[x_param]
    y_values = metric[tuple(slicer)]

    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, **plot_kwargs)
    ax.set_xlabel(x_param)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs {x_param}")
    ax.grid(True)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax




def plot_metric_slice_2d(params: dict, metric: np.ndarray, 
                         x_param: str, varying_param: str,
                         fixed_params: dict = None,
                         metric_name="Metric",
                         **plot_kwargs):
    """
    Plots a 2D slice of an N-dimensional result array: metric vs x_param, for varying values of varying_param.

    Parameters
    ----------
    params : dict
        Dictionary of parameter names to arrays of values. The order and length of each array 
        should match the axes of `metric`.

    metric : np.ndarray
        N-dimensional array of result values. The shape should match the parameter sweep grid.

    x_param : str
        The name of the parameter to plot on the x-axis.

    varying_param : str
        The name of the parameter for which separate curves will be plotted.

    fixed_params : dict, optional
        Dictionary mapping parameter names (not x_param or varying_param) to fixed indices.
        If not provided, remaining parameters are fixed to index 0 by default.

    metric_name : str, optional
        Label for the y-axis. Defaults to "Metric".

    **plot_kwargs : dict
        Additional keyword arguments passed to `plt.plot()` for customizing line style, color, etc.

    Returns
    -------
    None
        Displays a matplotlib plot.
    """
    if fixed_params is None:
        fixed_params = {}

    param_names = list(params.keys())
    param_axes = {name: i for i, name in enumerate(param_names)}

    assert x_param in param_names and varying_param in param_names, \
        "x_param and varying_param must be keys in params"

    # Default remaining fixed parameters to index 0
    for name in param_names:
        if name != x_param and name != varying_param and name not in fixed_params:
            fixed_params[name] = 0

    x_axis = param_axes[x_param]
    varying_axis = param_axes[varying_param]

    # Build slicing tuple
    slicer = [slice(None)] * metric.ndim
    for name, axis in param_axes.items():
        if name == x_param or name == varying_param:
            continue
        slicer[axis] = fixed_params[name]

    sliced_metric = metric[tuple(slicer)]  # Should be 2D

    x_values = params[x_param]
    y_values = params[varying_param]

    plt.figure()
    for i, y_val in enumerate(y_values):
        y_data = sliced_metric[:, i] if x_axis < varying_axis else sliced_metric[i, :]
        plt.plot(x_values, y_data, label=f"{varying_param} = {y_val}", **plot_kwargs)

    plt.xlabel(x_param)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs {x_param} for different {varying_param}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_metric_2d_image(params: dict, metric: np.ndarray,
                         x_param: str, y_param: str,
                         fixed_params: dict = None,
                         metric_name="Metric",
                         cmap="viridis",
                         colorbar=True,
                         **imshow_kwargs):
    """
    Plots a 2D heatmap of the metric with x_param on the horizontal axis
    and y_param on the vertical axis. All other parameters are fixed.

    Parameters
    ----------
    params : dict
        Dictionary mapping parameter names to arrays of values. The order and length 
        should match the axes of `metric`.

    metric : np.ndarray
        N-dimensional array of result values.

    x_param : str
        Name of the parameter for the x-axis.

    y_param : str
        Name of the parameter for the y-axis.

    fixed_params : dict, optional
        Dictionary of parameter names (excluding x_param and y_param) to fixed indices.
        Defaults to index 0 for unspecified parameters.

    metric_name : str, optional
        Label for the colorbar. Defaults to "Metric".

    cmap : str, optional
        Colormap to use for the image. Defaults to "viridis".

    colorbar : bool, optional
        Whether to show the colorbar. Defaults to True.

    **imshow_kwargs : dict
        Additional keyword arguments passed to `plt.imshow()`.

    Returns
    -------
    None
        Displays a matplotlib heatmap.
    """
    if fixed_params is None:
        fixed_params = {}

    param_names = list(params.keys())
    param_axes = {name: i for i, name in enumerate(param_names)}

    assert x_param in params and y_param in params, "x_param and y_param must be keys in params"

    x_axis = param_axes[x_param]
    y_axis = param_axes[y_param]

    # Fix all other parameters
    slicer = [0] * metric.ndim
    slicer[x_axis] = slice(None)
    slicer[y_axis] = slice(None)

    for name, idx in fixed_params.items():
        assert name not in (x_param, y_param), f"{name} is being varied, not fixed"
        assert name in param_names, f"Unknown parameter: {name}"
        slicer[param_axes[name]] = idx

    image_data = metric[tuple(slicer)]

    x_values = params[x_param]
    y_values = params[y_param]

    plt.figure()
    extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]
    plt.imshow(image_data.T, origin='lower', extent=extent, aspect='auto',
               cmap=cmap, **imshow_kwargs)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f"{metric_name} vs {x_param} & {y_param}")
    if colorbar:
        cbar = plt.colorbar()
        cbar.set_label(metric_name)
    plt.tight_layout()
    plt.show()



def mask_array_threshold(data, threshold, above=True):
    masked = data.copy()
    if above:
        masked[masked > threshold] = np.nan
    else:
        masked[masked < threshold] = np.nan
    return masked



########################## New version #####################################


def plot_metric_slice(df,
                      x_param: str,
                      varying_param: str = None,
                      metric_name: str = "result",
                      fixed_params: dict = None,
                      separate_subplots: bool = False,
                      show: bool = True,
                      **plot_kwargs):
    """
    Plot 1D or 2D slice(s) of sweep results stored in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns for all parameters and metric(s).

    x_param : str
        Parameter to plot on the x-axis.

    varying_param : str, optional
        Parameter for which separate curves will be plotted. If None, a single curve is drawn.

    fixed_params : dict, optional
        Dictionary of {param_name: value} to fix other parameters.

    metric_name : str or list of str, optional
        Column name(s) to plot on the y-axis.

    separate_subplots : bool, optional
        If True, plot each metric in a separate subplot when multiple metrics & varying_param.

    show : bool, optional
        Whether to display the plot immediately.

    **plot_kwargs : dict
        Extra keyword arguments passed to plt.plot().

    Returns
    -------
    fig, ax : matplotlib Figure and Axes (or list of Axes if separate_subplots=True)
    """
    fixed_params = fixed_params or {}

    # Filter DataFrame according to fixed_params
    df_slice = df.copy()
    for param, val in fixed_params.items():
        df_slice = df_slice[df_slice[param] == val]

    # Ensure metric_name is a list
    if isinstance(metric_name, str):
        metrics = [metric_name]
    else:
        metrics = metric_name

    ylabel = plot_kwargs.pop('ylabel', " / ".join(metrics) if len(metrics)>1 else metrics[0])
    title = plot_kwargs.pop('title', f"{', '.join(metrics)} vs {x_param}" + (f" for different {varying_param}" if varying_param else ""))



    # Decide on subplot structure
    if separate_subplots and len(metrics) > 1 and varying_param is not None:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 4*len(metrics)), sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            for v_val, sub_df in df_slice.groupby(varying_param):
                sub_df = sub_df.sort_values(x_param)
                ax.plot(sub_df[x_param], sub_df[metric], 
                        label=f"{v_val}", **plot_kwargs)
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)
        axes[-1].set_xlabel(x_param)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    else:
        # Single axis plot
        fig, ax = plt.subplots()
        if varying_param is None:
            # Case: single axis, multiple metrics or single metric
            df_plot = df_slice.sort_values(x_param)
            for metric in metrics:
                ax.plot(df_plot[x_param], df_plot[metric], label=metric, **plot_kwargs)
        else:
            # Case: single metric + varying_param, or multiple metrics + varying_param combined
            for metric in metrics:
                for v_val, sub_df in df_slice.groupby(varying_param):
                    sub_df = sub_df.sort_values(x_param)
                    ax.plot(sub_df[x_param], sub_df[metric], 
                            label=f"{metric}, {v_val}", **plot_kwargs)
        ax.set_xlabel(x_param)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax


def plot_metric_2d_heatmap(df,
                            x_param: str,
                            y_param: str,
                            fixed_params: dict = None,
                            metric_name: str = "result",
                            cmap: str = "viridis",
                            colorbar: bool = True,
                            show: bool = True,
                            **imshow_kwargs):
    """
    Plot a 2D heatmap of a metric vs two parameters using a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns for all parameters and 'result'.

    x_param : str
        Parameter for horizontal axis.

    y_param : str
        Parameter for vertical axis.

    fixed_params : dict, optional
        Dictionary of {param_name: value} to fix all other parameters.

    metric_name : str, optional
        Label for the colorbar.

    cmap : str, optional
        Colormap to use.

    colorbar : bool, optional
        Whether to show a colorbar.

    show : bool, optional
        Whether to display the plot immediately.

    **imshow_kwargs : dict
        Extra keyword arguments passed to plt.imshow().
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fixed_params = fixed_params or {}

    # Filter DataFrame by fixed parameters
    df_slice = df.copy()
    for param, val in fixed_params.items():
        df_slice = df_slice[df_slice[param] == val]

    # Pivot DataFrame to create 2D array
    pivot_table = df_slice.pivot(index=y_param, columns=x_param, values=metric_name)
    x_values = pivot_table.columns.values
    y_values = pivot_table.index.values
    image_data = pivot_table.values

    fig, ax = plt.subplots()
    extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]
    im = ax.imshow(image_data, origin='lower', extent=extent, aspect='auto',
                   cmap=cmap, **imshow_kwargs)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"{metric_name} vs {x_param} & {y_param}")
    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric_name)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax