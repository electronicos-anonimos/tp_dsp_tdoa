import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Parámetros
archivo_txt = r'C:\Users\faust\Desktop\Metodología\Plan de Investigación\Simulaciones\Datos Simulados_17.txt'
angulo_objetivo = 17
ruta_salida = r'D:\Gráficos\Boxplot_17.png'

# Cargar datos y transformar
df_raw = pd.read_csv(archivo_txt, sep='\t', encoding='latin1')
df_long = df_raw.melt(id_vars="Fuentes", var_name="Medición", value_name="Error_medido")
df_long["Ángulo_objetivo"] = angulo_objetivo
df_long["Fuentes"] = df_long["Fuentes"].astype(str)

# Estilo
sns.set(style='whitegrid', font_scale=1.2)

print(df_long.groupby("Fuentes")["Error_medido"].describe())

# Crear gráfico
plt.figure(figsize=(9, 6), dpi=300)
ax = sns.boxplot(data=df_long, x='Fuentes', y='Error_medido', 
                 width=0.5, palette="pastel",
                 boxprops=dict(edgecolor='black'),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 flierprops=dict(marker='o', markerfacecolor='gray', markersize=6, linestyle='none'))

# Títulos y etiquetas
plt.title(f"Boxplot – Ángulo de cobertura {angulo_objetivo}°", fontsize=16)
plt.ylabel("Error en cobertura (°)", fontsize=13)
plt.xlabel("Cantidad de fuentes", fontsize=13)
plt.tight_layout()

# Guardar y mostrar
plt.savefig(ruta_salida, format='png', dpi=300)
plt.show()
