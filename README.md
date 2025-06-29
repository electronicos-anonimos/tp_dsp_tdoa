# ESTIMACIÓN DE DIRECCIÓN DE ARRIBO DE FUENTES SONORAS

## Creación entorno virtual windows

Desde la terminal de windows

```bash
pip install virtualenv
```

Ubicando la terminal en el directorio de trabajo: 

```bash
python -m venv .venv
.venv/scripts/activate
```

Luego de activar el entorno virtual intalar todas las librerías. 

```bash
pip install -r requirements.txt
```

Listo, ejecutar el código. 


## SIMULACIONES DESDE GOOGLE SPREADSHETS

Debe incluirse el archivo `simulationsdoa-credenciales.json` en el directorio de trabajo, para poder acceder a la hoja de calculo. Las credenciales se pueden obtener desde la consola de Google Cloud Platform, creando un proyecto y habilitando la API de Google Sheets.

Se pueden hacer las simulaciónes cargando los datos en una hoja de calculo, puede así hacerse un barrido de 0 a 180° de forma muy sencilla. 
La hoja de cálculo debe tener las siguientes columnas:
```plaintext

description	simulation_name	audio	room_x	room_y	room_z	rt60	snr_db	n_mics	mic_d	mic_z	mic_directivity	src_dist	src_z	src_ang_start	src_ang_end	src_ang_step													

```

## EJECUCIÓN DEL CÓDIGO

Debe generarse un sine sweep de al menos 15 segundos de duración. Esto se puede hacer desde la el archivo `sine_sweep.py`.

Luego en la hoja de cálculo se debe cargar el nombre del archivo de audio generado, y los parámetros de la simulación. Luego desde el archivo `main.py` se puede ejecutar el código, que generará los resultados en un archivo CSV y los gráficos correspondientes. Tener en cuenta que primero se generarán todos los audios desde la hoja de cálculo, guardandose en la carpeta `audios/output/nombre_simulacion`. Luego se procesarán todos los audios generados con la correlación cruzada clasica y la correlación cruzada generalizada utilizando los métodos roth, phat y scot para poder estimar los ángulos en el barrido que se haya elegido en la hoja de cálculo. 
Todos los resultados se guardarán en un archivo CSV dentro de la carpeta `csv_results/`.

## GRÁFICOS
Los gráficos se generarán automáticamente al ejecutar el código, y se guardarán en la carpeta `carpeta a definir a futuro por los cracks del codigo`. Por ahora esto está en desarrollo. 


