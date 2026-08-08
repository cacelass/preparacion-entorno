# Linux para agentes que operan una máquina

Referencia práctica para el agente que tiene que ejecutar comandos en una
máquina real: shell, ficheros, procesos, entorno, redes y automatización.
Cada sección: el principio, la práctica y el fallo típico.

## Fundamentos de shell

**Pipes y redirección.**

```bash
cmd1 | cmd2          # stdout de cmd1 → stdin de cmd2
cmd > file           # stdout → file (sobrescribe)
cmd >> file          # stdout → file (añade)
cmd 2> err.log       # stderr → err.log
cmd 1> out.log 2> err.log
```

**Sustitución de comandos.**

```bash
git_sha=$(git rev-parse HEAD)   # $() anida mejor que ``
ls "$(dirname "$file")"
```

**Exit codes.** 0 = éxito, != 0 = fallo. Los pipelines devuelven el código
del último comando. `set -euo pipefail` convierte los errores silenciosos en
fallos del script:

```bash
set -euo pipefail
# -e:    cualquier comando que falle aborta el script
# -u:    variables sin definir son error (no se expanden a "" en silencio)
# pipefail: un fallo en medio del pipe falla el pipe entero
```

**Trampas de comillas.** Las variables se expanden entre comillas dobles y no
entre simples. Un path con espacios sin comillas se parte en dos argumentos:

```bash
f="mi archivo.csv"
cat "$f"        # bien
cat $f          # mal: dos argumentos
```

**Cómo falla en un script DS.** Un script sin `set -euo pipefail` que
continúa después de que `make train` fallara y "manda" el deploy de un modelo
roto; o un path con espacios en `for f in $(ls)` que parte cada nombre.

## Sistema de ficheros

**Permisos.** `rwx` por usuario/grupo/otros. Simbólico vs numérico:

```bash
chmod u+x script.sh    # simbólico: añade ejecutable al dueño
chmod 755 script.sh    # numérico: rwxr-xr-x (755 = 7 5 5)
chown user:group file
```

En ficheros de datos no hace falta `+x`; en scripts, sí. `chmod 777` es casi
siempre un error.

**Buscar ficheros.**

```bash
find data/ -name "*.parquet" -mtime -7          # por nombre y edad
rg "TODO" src/                                   # contenido, con regex (ripgrep)
fd -e py                                          # por extensión (alternativa a find)
```

`find` busca por atributos (nombre, fecha, tamaño); `rg`/`fd` buscan por
contenido/nombre con mejor UX. Para un agente, `rg` y `fd` casi siempre
ganan; `find` para las consultas de atributos que no cubren.

**Uso de disco.**

```bash
du -sh data/ models/           # tamaño de cada directorio
df -h                           # espacio libre por mount
df -h .                         # ¿el disco actual?
```

**Symlinks.** `ln -s target link` crea un enlace que apunta a `target`. Útil
para "esta carpeta es esta otra" (p.ej. `data/ -> /mnt/disco/data`). Cuidado:
un symlink roto (`ls` con fondo rojo) falla silenciosamente al usarse.

**Cómo falla.** `df -h` muestra el disco lleno al 98% y nadie sabe qué es:
`du -sh */* | sort -h | tail` lo localiza en un paso. Un symlink a una ruta
que se desmontó hace que "el directorio no existe" con datos intactos al otro
lado.

## Procesamiento de texto

Cuándo usar cada herramienta (y cuándo Python es mejor):

| Herramienta | Para qué | Ejemplo |
|-------------|----------|---------|
| `rg`/`grep` | Filtrar líneas por regex | `rg "auc=" reports/` |
| `sed` | Sustitución en streaming | `sed 's/foo/bar/g' file` |
| `awk` | Columnas y sumas | `awk -F, '{print $1, $3}'` |
| `cut` | Cortar columnas por delimitador | `cut -d, -f1 file` |
| `sort`/`uniq` | Ordenar y contar duplicados | `sort file | uniq -c | sort -rn` |
| `head`/`tail` | Principio/fin de stream | `tail -f log/train.log` |
| `wc` | Contar líneas/palabras | `wc -l features.csv` |
| `xargs` | Pasar líneas como argumentos | `find . -name "*.tmp" | xargs rm` |

```bash
# patrones que se usan solos
tail -f nohup.out                        # seguir un log de entrenamiento
sort -t, -k2 -n results.csv | head -10   # top-10 por columna numérica
rg -c "error" logs/ | sort -t: -k2 -nr   # ¿qué log tiene más errores?
```

**Cuándo usar Python.** Más de un paso de parsing, tipos, uniones o
cualquier cosa que vaya a repetirse: `python` (o el venv del proyecto) es más
legible y testeable que un pipeline de `sed|awk|grep` encadenado. La regla:
shell para operación de una vez; Python para lógica que merece un script.

**Cómo falla.** `awk`/`sed` con un CSV que tiene comas dentro de comillas (el
parser rompe columnas), o un `grep -v exclude` que filtra más de lo que
parece por expresión mal anclada. Cuando el "parseo rápido" tiene cinco
pasos, ya no es rápido.

## Procesos

**Ver y matar.**

```bash
ps aux | rg "train.py"     # ¿qué procesos hay?
top / htop                 # consumidores en vivo
kill -TERM <pid>           # pedida de parada amistosa (por defecto)
kill -KILL <pid>           # matar a la fuerza (no ejecuta cleanup)
```

`TERM` permite al proceso guardar estado (checkpoint); `KILL` es el último
recurso. En un entrenamiento, matar con `TERM` y que el código capture
`SIGTERM` para checkpointear es la diferencia entre "reanudo" y "empiezo de
cero".

**Jobs y largas ejecuciones.**

```bash
command &               # job en background del shell actual
jobs; fg; bg; Ctrl-Z    # control de jobs
nohup make train > train.log 2>&1 &
echo $! > train.pid     # el pid, para matar/consultar después
```

Para entrenamientos largos, mejor `tmux` (o `screen`): despega el proceso del
shell y permite re-conectarse.

```bash
tmux new -s train       # nueva sesión
# ... lanza el entrenamiento ...
Ctrl-b d                 # despegar (el proceso sigue)
tmux attach -t train    # volver
```

**Cómo falla.** Lanzar `make train` a secas y cerrar el terminal: un SIGHUP
mata el entrenamiento en la época 30. La disciplina es `nohup ... &` + log +
pidfile, o tmux; y el `kill -9` de un entrenamiento con checkpoints a medio
escribir puede corromper los artefactos.

## Entorno

**Variables y PATH.**

```bash
export VAR=value        # solo para este shell y sus hijos
PATH=$PATH:/opt/mi/bin   # añadir a la búsqueda de comandos
echo $VAR                # leer
```

**Shell de login vs no-login.** `~/.bashrc` se lee en shells interactivos
no-login (la terminal normal); `~/.profile`/`~/.bashrc` de login en el login
SSH. Regla práctica: config en `.bashrc` y `source ~/.bashrc` tras editarlo;
los crons no leen ninguno de los dos (ver Automatización).

**Secretos NO viven en el entorno del repo.** Las variables de entorno son
visibles por cualquier proceso del usuario (`/proc/<pid>/environ`) y las
claves exportadas quedan en la historia del shell. Los secretos van en un
gestor (o en `.env` fuera de git, leído al arranque) y las claves de
servicios en el secret manager del entorno. El agente no imprime variables
con claves ni las pasa por la línea de comandos (visible en `ps aux`).

**Cómo falla.** `export AWS_KEY=...` copiado de un script y luego el script
commiteado: la clave queda en el historial y en `/proc`. O el cron que no
encuentra el comando porque el PATH de login no está en el entorno de cron.

## Inspección del sistema

| Qué | Comando | Señal de alarma |
|-----|---------|-----------------|
| Memoria | `free -h` | `available` bajo; swap en uso |
| CPU | `lscpu` / `top` | load > núcleos; un proceso al 100% |
| Disco | `df -h` | uso > 85% (deja margen a los checkpoints) |
| GPU | `nvidia-smi` | VRAM al límite; ECC errors; proceso zombi de otro run |
| Quién consume | `top`, `lsof` | proceso desconocido con CPU/RAM |

**GPU — el relevante para DS.** `nvidia-smi` muestra memoria y procesos por
GPU. Dos checkpoints reales:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
ps aux | rg python           # ¿es MI entrenamiento o un proceso huérfano?
```

`lsof` lista qué procesos tienen abierto un fichero o puerto:

```bash
lsof -i :8000                # ¿quién sirve en el puerto 8000?
lsof data/dataset.parquet    # ¿quién tiene el fichero abierto?
```

**Cómo falla.** Un entrenamiento "no avanza" porque hay otro proceso ocupando
la GPU a la vez; o `nvidia-smi` muestra VRAM llena sin proceso visible
(proceso de otro usuario, o un zombie): hay que ver con `ps aux` quién es
realmente.
{% if use_docker %}
**Dentro de un contenedor.** `nvidia-smi` solo ve la GPU si el runtime del
contenedor pasa los dispositivos (`--gpus all` o `nvidia-container-toolkit`).
Sin eso, "no hay GPU" dentro del contenedor aunque el host la tenga. Y los
límites de `docker stats` (CPU/RAM) son los que valen dentro: un `free -h`
del contenedor muestra solo la parte que le asignaron.
{% endif %}

## Redes

**Puertos.**

```bash
ss -tlnp            # puertos en escucha y qué proceso
lsof -i :8080       # ¿quién usa el 8080?
```

**curl — la herramienta que más usa un agente.**

```bash
curl -sS http://localhost:8000/health        # ¿responde el servicio?
curl -sS -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": {...}}'
curl -sS -o file URL                         # descargar a fichero
curl -sS -f URL                              # -f: falla (exit != 0) si el HTTP da error
```

`-sS` silencia el progreso pero muestra los errores; `-f` hace que un 404/500
falle el comando (crucial dentro de `set -e`).

**localhost vs contenedores.** `localhost` es el loopback del host. Dentro de
un contenedor, `localhost` es el propio contenedor: para llegar a un servicio
del host hay que usar la IP del host (o `host.docker.internal` según el
driver). Dos servicios en contenedores se hablan por la red de Docker, no por
localhost.

**Cómo falla.** Un agente que hace `curl localhost:8000` desde dentro de un
contenedor y no encuentra el servicio del host: está golpeando la puerta de
su propio contenedor.

## Automatización

**cron / systemd timers.**

```bash
# cron: cada día a las 03:00
0 3 * * * cd /ruta/proyecto && ./retrain.sh >> /var/log/retrain.log 2>&1
```

- cron no carga el entorno de login: las rutas y variables deben declararse
  dentro del script (PATH, PYTHONPATH).
- systemd timers (`systemctl --user`) dan logs, unidades y arranque al
  reiniciar; para "reentrenar cada noche", un timer systemd es más robusto
  que cron.

**Rotación de logs.** Sin rotación, `train.log` crece sin límite y llena el
disco que el entrenamiento necesita. `logrotate` por tamaño/día, o un mínimo
en el propio script (guardar `train.log.1`, `train.log.2`, ...).

**Limpieza de temporales.** Los `nohup.out`, `/tmp/*` y los checkpoints
viejos se acumulan. Un cron semanal de limpieza con retención explícita
("borrar > 7 días") vale más que el debate sobre el tamaño.

**Cómo falla.** Un cron de retrain que falla en silencio porque el PATH del
venv no estaba en el entorno de cron, y nadie se entera porque la salida se
perdió en `/dev/null`; o el disco lleno por logs sin rotar a mitad de un
entrenamiento.

## Higiene de seguridad

- **No ser root.** Ejecutar como usuario normal; `sudo` solo para el comando
  concreto que lo necesita. `sudo` no es para "ir más rápido".
- **`sudo` en pipelines**: `sudo` solo envuelve un comando; en
  `sudo cat f > /root/f` la redirección la hace tu shell como no-root y
  falla. Usar `tee` (`echo x | sudo tee /root/f`) o `sudo sh -c '...'`.
- **Claves SSH**: `~/.ssh/` con permisos estrictos (`chmod 700` el dir,
  `600` la clave); la clave privada nunca se comparte ni se copia a máquinas
  no propias.
- **Sin secretos en la historia ni en scripts.** Nada de
  `curl -H "Authorization: Bearer $TOKEN"` donde el TOKEN venga en el propio
  comando visible en `ps` y en `.bash_history`. Los scripts con claves
  hardcodeadas acaban en un repo tarde o temprano.
- **Comandos destructivos**: `rm -rf` de un directorio, `chmod -R`, `mv` sobre
  otro fichero: confirmar antes. El agente revisa el path dos veces.

**Cómo falla.** Un script con `sudo make install` y claves en variables de
entorno commiteado; o un `rm -rf data/` con un `.` de más que borra el
proyecto entero. La disciplina es mínima: comando específico, path explícito,
sin secretos en el texto.

## Checklist: primeros 5 minutos en una máquina nueva

```text
□ ¿Quién soy?        whoami; sudo -n true 2>/dev/null && echo "sudo OK" || echo "sin sudo"
□ ¿Tengo el repo?    git status; ./init.sh (¿la puerta del proyecto está verde?)
□ ¿Disco y RAM?      df -h .; free -h; nvidia-smi (¿GPU libre? ¿otro proceso?)
□ ¿Puerto del servicio? ss -tlnp | rg 8000   (¿ya corre algo?)
□ ¿Venv y lock?      ls .venv/bin/python && uv sync   (¿el entorno es el del lock?)
□ ¿Hay un job mío corriendo?  ps aux | rg "train.py|python"  (¿huérfano de antes?)
```

## Checklist: rescatar un job atascado

```text
□ ¿Está vivo o colgado?    ps aux | rg <pid>; top -p <pid>   (¿CPU 0% pero no muere?)
□ ¿Qué dice el log?        tail -50 train.log                (¿error real o solo lento?)
□ ¿Está la GPU ocupada?    nvidia-smi                        (¿OOM por otro proceso?)
□ ¿Ha hecho checkpoints?   ls -lat models/checkpoints/       (¿hasta dónde reanudar?)
□ Si hay que matar:        kill -TERM <pid> → esperar → verificar checkpoint → kill -KILL
□ ¿El disco está lleno?    df -h .                           (mataría cualquier escritura)
□ Reanudar con evidencia:  nohup make train > train.log 2>&1 & echo $! > train.pid
```

## Fuentes

- Documentación GNU coreutils (find, sort, sed, awk, grep, xargs, du, df).
- GNU Bash manual: redirecciones, sustitución de comandos, `set -euo pipefail`.
- Documentación de ripgrep (rg) y fd.
- Documentación de tmux y screen (sesiones despegadas).
- Documentación de NVIDIA (`nvidia-smi`) y del nvidia-container-toolkit.
- Documentación de systemd (timers) y cron.
- Las reglas del proyecto: `AGENTS.md` (la puerta de `init.sh`, evidencia
  sobre afirmaciones, política de permisos) e `ingenieria/reglas-codigo.md`.
