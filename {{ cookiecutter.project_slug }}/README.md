# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

---

### Instalación rápida de uv

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
### Instalación rápida de cookiecutter

```bash
apt install cookiecutter
```

### Configura tu entorno de proyecto

1. Instala todas las dependencias del proyecto:
   ```bash
   cd <nombre_directorio_creado>
   uv sync
   source .venv/bin/activate
   ```

> Si agregas nuevos paquetes a `pyproject.toml`, recuerda volver a ejecutar `uv sync` para actualizar tu entorno.

---

## Project Organization

        ├── LICENSE
        ├── tasks.py           <- Archivo con tareas que puedes ejecutar con comandos como `notebook`.
        ├── README.md          <- Guía principal para desarrolladores que trabajen con este proyecto.
        ├── install.md         <- Instrucciones paso a paso para instalar y configurar el entorno.
        ├── data
        │   ├── external       <- Datos obtenidos de fuentes externas.
        │   ├── interim        <- Datos intermedios, ya transformados pero no finales.
        │   ├── processed      <- Datos finales listos para ser usados en modelos.
        │   └── raw            <- Datos originales sin modificar.
        │
        ├── models             <- Modelos entrenados, guardados y sus predicciones o reportes.
        │
        ├── notebooks          <- Notebooks de Jupyter. Se nombran con un número (para ordenar),
        │                         iniciales del autor y una breve descripción, por ejemplo:
        │                         `1.0-jqp-exploracion-inicial`.
        │
        ├── references         <- Documentación de apoyo: diccionarios de datos, manuales, etc.
        │
        ├── reports            <- Resultados generados en formatos como HTML, PDF o LaTeX.
        │   └── figures        <- Gráficos e imágenes usados en los reportes.
        │
        ├── pyproject.toml     <- Archivo con las dependencias necesarias para reproducir el entorno.
        │
        ├── .here              <- Archivo que indica el directorio raíz del proyecto.
        │
        └── {{ cookiecutter.project_module_name }}               <- Código fuente principal del proyecto.
            ├── __init__.py    <- Indica que este directorio es un módulo de Python.
            │
            ├── data           <- Scripts para descargar, generar o preparar datos.
            │   └── make_dataset.py
            │
            ├── features       <- Scripts para convertir datos crudos en características útiles.
            │   └── build_features.py
            │
            ├── models         <- Scripts para entrenar modelos y generar predicciones.
            │   ├── predict_model.py
            │   └── train_model.py
            │
            ├── utils          <- Funciones auxiliares para tareas comunes del proyecto.
            │   └── paths.py   <- Funciones para manejar rutas de archivos dentro del proyecto.
            │
            └── visualization  <- Scripts para crear visualizaciones y gráficos de resultados.
                └── visualize.py

---

¿Necesitas instalar algún paquete más tarde? Simplemente agrégalo en `pyproject.toml` y ejecuta de nuevo:

```bash
uv sync
```
