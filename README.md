# Cookiecuter Personal

## Requiremientos

- [git](https://git-scm.com/) >= 2.x
- [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html) >= 1.4.0:

``` bash
apt install cookiecutter
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

## Crear un nuevo proyecto

En el directorio en el que quieras guardar tu proyecto generado:

```bash
cookiecutter https://github.com/cacelass/preparacion-entorno
```
Instalar lo necesario para el proyecto...

```bash
cd <nombre_directorio_creado>
conda env create --file environment.yml  
```


## Estructura de directorios y archivos resultantes

    {{ cookiecutter.project_slug }}
        ├── data
        │   ├── processed      <- Conjuntos de datos finales y limpios, listos para el modelado.
        │   └── raw            <- Datos originales sin modificar, tal como fueron obtenidos.
        │
        ├── notebooks          <- Notebooks de Jupyter. La convención de nombres usa un número
        │                         (para ordenar), las iniciales del autor y una breve descripción
        │                         separada por guiones. Ejemplo:
        │                         `1.0-jvelezmagic-exploracion-inicial-datos`.
        │
        ├── .gitignore         <- Lista de archivos y carpetas que `git` debe ignorar.
        │
        ├── environment.yml    <- Archivo con las dependencias necesarias para reproducir
        │                         el entorno de análisis.
        │
        └── README.md          <- El README principal para desarrolladores que trabajen con este proyecto.
