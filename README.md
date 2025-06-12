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
Se pueden hacer las simulaciónes cargando los datos en una hoja de calculo, puede así hacerse un barrido de 0 a 180° de forma muy sencilla. 

https://docs.google.com/spreadsheets/d/13XTDng98P99pfexK78Dd4Gud1CzZwO7PfVhpyIG1jCM/edit?usp=sharing

## Audios

Hasta el momento se dejan 3 audios para utilizar.
- audio_anecoico_corto.wav: persona hablando
- audio_anecoico_largo.wav: guitarra flamenco
- sine_sweep_24bit.wav: sine sweep de 30 segundos de 20 Hz a 20 kHz

