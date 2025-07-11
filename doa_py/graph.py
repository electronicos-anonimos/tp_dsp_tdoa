import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, os
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

def generate_heatmap(
    measure: str,
    method_graph: str,
    x_variable: str,
    x_label: str,
    x_lim: tuple,
    csv_folder: str,
    boundaries: list = [0, 1.5, 2.5, 4, 5, 7.5, 10, 15, 25, 40],
    colors: list = [
        "#006400", "#00a000", "#92d050", "#ffff66", "#ffc000",
        "#ff9900", "#ff3333", "#cc0000", "#800000"
    ],
    y_label: str = "Error angular [°]",
    y_axis_label: str = "Ángulo de llegada [°]",
    method_column: str = "method",
    graph_mode: str = "interpolated",
    output_dir: str = "img"
):
    """
    Genera un heatmap a partir de múltiples archivos CSV con interpolación o grilla.

    Parámetros obligatorios:
    - measure: columna a graficar en Z (eje de colores)
    - method_graph: método a filtrar (e.g., "classic", "scot")
    - x_variable: nombre de columna a usar como eje X
    - x_label: etiqueta del eje X
    - x_lim: tuple con mínimo y máximo del eje X (sólo para interpolado)
    - csv_folder: carpeta con los CSVs de entrada
    - boundaries: lista de valores límite para la escala de color
    - colors: lista de colores para la paleta

    Parámetros opcionales:
    - y_label: etiqueta de la barra de color
    - y_axis_label: etiqueta del eje Y
    - method_column: columna que indica el método en el CSV
    - graph_mode: "interpolated" o "grid"
    - output_dir: carpeta donde guardar el gráfico
    """

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, len(colors), clip=True)

    pattern_path = os.path.join(csv_folder, "*.csv")
    csv_files = glob.glob(pattern_path)
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos en: {pattern_path}")

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df[method_column] = df[method_column].str.upper()
    df = df[df[method_column] == method_graph.upper()]

    required_cols = [x_variable, "angle", measure]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna: {col}")
    df[x_variable] = pd.to_numeric(df[x_variable], errors="coerce")
    df = df.dropna(subset=required_cols)

    plt.figure(figsize=(8, 6))

    if graph_mode == "interpolated":
        angles = np.linspace(0, 180, 500)
        x_vals = np.linspace(*x_lim, 400)
        X, A = np.meshgrid(x_vals, angles)
        points = df[["angle", x_variable]].values
        values = df[measure].values
        Z = griddata(points, values, (A, X), method="linear")
        Z_masked = np.ma.masked_invalid(Z)
        heatmap = plt.contourf(X, A, Z_masked, levels=boundaries, cmap=cmap, norm=norm, extend="max")
        plt.xlim(x_lim)

    elif graph_mode == "grid":
        df["angle_round"] = df["angle"].round(1)
        df["x_round"] = df[x_variable].round(2)
        pivot = df.pivot_table(index="angle_round", columns="x_round", values=measure).sort_index(ascending=False)
        ny, nx = pivot.shape
        pc = plt.pcolormesh(np.arange(nx + 1), np.arange(ny + 1), pivot.values, cmap=cmap, norm=norm, shading="auto")
        xticks = np.arange(0.5, nx + 0.5)
        yticks = np.arange(0.5, ny + 0.5, max(1, ny // 10))
        plt.xticks(ticks=xticks, labels=[f"{v:.2f}" for v in pivot.columns])
        plt.yticks(ticks=yticks, labels=[f"{v:.0f}" for v in pivot.index[::max(1, ny // 10)]])
        heatmap = pc

    else:
        raise ValueError("Modo de gráfico no reconocido. Usa 'grid' o 'interpolated'.")

    plt.xlabel(x_label)
    plt.ylabel(y_axis_label)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(heatmap, ticks=boundaries)
    cbar.set_label(y_label)

    os.makedirs(output_dir, exist_ok=True)
    x_range = f"{x_lim[0]}to{x_lim[1]}"
    filename = f"{measure}_vs_{x_variable}_{x_range}_{graph_mode}_{method_graph}.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.show()
    



# def graficar_error_vs_variable(
#     csv_folder: str,
#     agrupador_x: str,
#     columna_error: str,
#     method_column: str,
#     metodos_a_graficar: list,
#     x_label: str,
#     y_label: str = "Error promedio [°]",
#     y_axis_label: str = "Método",
#     colores: dict = None,
#     trazos: dict = None,
#     output_dir: str = "img",
#     output_filename: str = None,
#     figsize: tuple = (7, 4.5),
#     x_range: tuple = None,
#     y_range: tuple = None,
#     dispersion_func = np.std
# ):
#     archivos = glob.glob(os.path.join(csv_folder, "*.csv"))
#     if not archivos:
#         raise FileNotFoundError(f"No se encontraron archivos en: {csv_folder}")

#     dfs = []
#     for archivo in archivos:
#         df_tmp = pd.read_csv(archivo)
#         if all(col in df_tmp.columns for col in [agrupador_x, columna_error, method_column]):
#             dfs.append(df_tmp)

#     if not dfs:
#         raise ValueError("Ningún archivo contenía las columnas requeridas.")

#     df_all = pd.concat(dfs, ignore_index=True)
#     df_all[method_column] = df_all[method_column].str.upper()
#     df_plot = df_all[df_all[method_column].isin(metodos_a_graficar)]

#     if x_range:
#         df_plot = df_plot[df_plot[agrupador_x].between(*x_range)]
#     if y_range:
#         df_plot = df_plot[df_plot[columna_error].between(*y_range)]

#     if df_plot.empty:
#         raise ValueError("No hay datos disponibles tras aplicar los filtros.")

#     resumen = df_plot.groupby([agrupador_x, method_column])[columna_error].agg(['mean']).reset_index()
#     resumen.rename(columns={"mean": "error_promedio"}, inplace=True)

#     if dispersion_func is not None:
#         dispersion = df_plot.groupby([agrupador_x, method_column])[columna_error].agg(dispersion_func).reset_index()
#         resumen["error_dispersion"] = dispersion[columna_error]
#         dispersion_tag = dispersion_func.__name__
#     else:
#         resumen["error_dispersion"] = 0
#         dispersion_tag = "nodisp"

#     if colores is None:
#         colores = {met: col for met, col in zip(metodos_a_graficar, plt.rcParams["axes.prop_cycle"].by_key()["color"])}
#     if trazos is None:
#         trazos = {met: "-" for met in metodos_a_graficar}

#     if output_filename is None:
#         met_str = "_".join(m.lower() for m in metodos_a_graficar)
#         output_filename = f"{columna_error}_vs_{agrupador_x}_{dispersion_tag}_{met_str}"

#     plt.figure(figsize=figsize)
#     xticks = sorted(resumen[agrupador_x].unique())

#     for metodo in metodos_a_graficar:
#         df_met = resumen[resumen[method_column] == metodo]
#         if df_met.empty:
#             continue
#         plt.errorbar(
#             df_met[agrupador_x],
#             df_met["error_promedio"],
#             yerr=df_met["error_dispersion"],
#             label=metodo,
#             fmt="o",
#             linestyle=trazos.get(metodo, "-"),
#             color=colores.get(metodo),
#             capsize=4,
#             linewidth=1.5,
#             markersize=5
#         )

#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.xticks(xticks, rotation='vertical')
#     plt.gcf().autofmt_xdate(rotation=45, ha='right')  # 👈 evita solapamiento
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.legend(title=y_axis_label, loc="upper right")
#     plt.tight_layout()

#     os.makedirs(output_dir, exist_ok=True)
#     ruta_salida = os.path.join(output_dir, f"{output_filename}.png")
#     plt.savefig(ruta_salida, dpi=300)
#     plt.show()


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def graficar_error_vs_variable(
    csv_folder: str,
    agrupador_x: str,
    columna_error: str,
    method_column: str,
    metodos_a_graficar: list,
    x_label: str,
    y_label: str = "Error promedio [°]",
    y_axis_label: str = "Método",
    colores: dict = None,
    trazos: dict = None,
    marcadores: dict = None,
    output_dir: str = "img",
    output_filename: str = None,
    figsize: tuple = (7, 4.5),
    x_range: tuple = None,
    y_range: tuple = None,
    dispersion_func = np.std
):
    archivos = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos en: {csv_folder}")

    dfs = []
    for archivo in archivos:
        df_tmp = pd.read_csv(archivo)
        if all(col in df_tmp.columns for col in [agrupador_x, columna_error, method_column]):
            dfs.append(df_tmp)

    if not dfs:
        raise ValueError("Ningún archivo contenía las columnas requeridas.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all[method_column] = df_all[method_column].str.upper()
    df_plot = df_all[df_all[method_column].isin(metodos_a_graficar)]

    if x_range:
        df_plot = df_plot[df_plot[agrupador_x].between(*x_range)]
    if y_range:
        df_plot = df_plot[df_plot[columna_error].between(*y_range)]

    if df_plot.empty:
        raise ValueError("No hay datos disponibles tras aplicar los filtros.")

    resumen = df_plot.groupby([agrupador_x, method_column])[columna_error].agg(['mean']).reset_index()
    resumen.rename(columns={"mean": "error_promedio"}, inplace=True)

    if dispersion_func is not None:
        dispersion = df_plot.groupby([agrupador_x, method_column])[columna_error].agg(dispersion_func).reset_index()
        resumen["error_dispersion"] = dispersion[columna_error]
        dispersion_tag = dispersion_func.__name__
    else:
        resumen["error_dispersion"] = 0
        dispersion_tag = "nodisp"

    default_colors = ["blue", "orange", "green", "red", "purple"]
    default_trazos = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
    default_markers = ["o", "s", "^", "D", "x"]

    if colores is None:
        colores = {met: default_colors[i % len(default_colors)] for i, met in enumerate(metodos_a_graficar)}
    if trazos is None:
        trazos = {met: default_trazos[i % len(default_trazos)] for i, met in enumerate(metodos_a_graficar)}
    if marcadores is None:
        marcadores = {met: default_markers[i % len(default_markers)] for i, met in enumerate(metodos_a_graficar)}

    if output_filename is None:
        met_str = "_".join(m.lower() for m in metodos_a_graficar)
        output_filename = f"{columna_error}_vs_{agrupador_x}_{dispersion_tag}_{met_str}"

    plt.figure(figsize=figsize)
    xticks = sorted(resumen[agrupador_x].unique())

    for metodo in metodos_a_graficar:
        df_met = resumen[resumen[method_column] == metodo]
        if df_met.empty:
            continue

        plt.errorbar(
            df_met[agrupador_x],
            df_met["error_promedio"],
            yerr=df_met["error_dispersion"],
            label=metodo,
            color=colores.get(metodo),
            linestyle=trazos.get(metodo),
            marker=marcadores.get(metodo),
            capsize=4,
            linewidth=1.5,
            markersize=5
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(xticks, rotation='vertical')
    plt.gcf().autofmt_xdate(rotation=45, ha='right')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title=y_axis_label, loc="upper right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    ruta_salida = os.path.join(output_dir, f"{output_filename}.png")
    plt.savefig(ruta_salida, dpi=300)
    plt.show()

# def graficar_error_vs_variable(
#     csv_folder: str,
#     agrupador_x: str,
#     columna_error: str,
#     method_column: str,
#     metodos_a_graficar: list,
#     x_label: str,
#     y_label: str = "Error promedio [°]",
#     y_axis_label: str = "Método",
#     colores: dict = None,
#     trazos: dict = None,
#     output_dir: str = "img",
#     output_filename: str = None,
#     figsize: tuple = (7, 4.5),
#     x_range: tuple = None,
#     y_range: tuple = None,
#     dispersion_func = np.std  # También podés usar scipy.stats.iqr, np.var, etc.
# ):
#     """
#     Genera un gráfico de líneas con barras de dispersión para comparar métodos
#     en función de una variable X, usando columnas internas del CSV.

#     - dispersion_func: función como np.std, np.var o scipy.stats.iqr (o None para no graficar dispersión)
#     """

#     archivos = glob.glob(os.path.join(csv_folder, "*.csv"))
#     if not archivos:
#         raise FileNotFoundError(f"No se encontraron archivos en: {csv_folder}")

#     dfs = []
#     for archivo in archivos:
#         df_tmp = pd.read_csv(archivo)
#         if all(col in df_tmp.columns for col in [agrupador_x, columna_error, method_column]):
#             dfs.append(df_tmp)

#     if not dfs:
#         raise ValueError("Ningún archivo contenía las columnas requeridas.")

#     df_all = pd.concat(dfs, ignore_index=True)
#     df_all[method_column] = df_all[method_column].str.upper()
#     df_plot = df_all[df_all[method_column].isin(metodos_a_graficar)]

#     if x_range:
#         df_plot = df_plot[df_plot[agrupador_x].between(*x_range)]
#     if y_range:
#         df_plot = df_plot[df_plot[columna_error].between(*y_range)]

#     if df_plot.empty:
#         raise ValueError("No hay datos disponibles tras aplicar los filtros.")

#     resumen = df_plot.groupby([agrupador_x, method_column])[columna_error].agg(['mean']).reset_index()
#     resumen.rename(columns={"mean": "error_promedio"}, inplace=True)

#     if dispersion_func is not None:
#         dispersion = df_plot.groupby([agrupador_x, method_column])[columna_error].agg(dispersion_func).reset_index()
#         resumen["error_dispersion"] = dispersion[columna_error]
#         dispersion_tag = dispersion_func.__name__
#     else:
#         resumen["error_dispersion"] = 0
#         dispersion_tag = "nodisp"

#     if colores is None:
#         colores = {met: col for met, col in zip(metodos_a_graficar, plt.rcParams["axes.prop_cycle"].by_key()["color"])}
#     if trazos is None:
#         trazos = {met: "-" for met in metodos_a_graficar}

#     if output_filename is None:
#         met_str = "_".join(m.lower() for m in metodos_a_graficar)
#         output_filename = f"{columna_error}_vs_{agrupador_x}_{dispersion_tag}_{met_str}"

#     plt.figure(figsize=figsize)
#     xticks = sorted(resumen[agrupador_x].unique())

#     for metodo in metodos_a_graficar:
#         df_met = resumen[resumen[method_column] == metodo]
#         if df_met.empty:
#             continue
#         plt.errorbar(
#             df_met[agrupador_x],
#             df_met["error_promedio"],
#             yerr=df_met["error_dispersion"],
#             label=metodo,
#             fmt="o",
#             linestyle=trazos.get(metodo, "-"),
#             color=colores.get(metodo),
#             capsize=4,
#             linewidth=1.5,
#             markersize=5
#         )

#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.xticks(xticks)
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.legend(title=y_axis_label, loc="upper right")
#     plt.tight_layout()

#     os.makedirs(output_dir, exist_ok=True)
#     ruta_salida = os.path.join(output_dir, f"{output_filename}.png")
#     plt.savefig(ruta_salida, dpi=300)
#     plt.show()
