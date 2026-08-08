# Git para proyectos dirigidos por agentes

Referencia para el lider y el reviewer sobre cómo trabajar con git cuando el
que opera es un agente: modelo de objetos, flujos, higiene de commits,
recuperación y las reglas del arnés de este proyecto.

## El modelo de objetos

**Principio.** Git es un grafo acíclico dirigido de objetos inmutables y
content-addressed: blobs (contenido de fichero), trees (directorios), commits
(snapshot con metadata y padres) y refs (punteros con nombre). Entender eso
explica casi todos los comandos.

**Práctica.**

```text
blob   = contenido de un fichero, direccionado por hash (SHA-1)
tree   = lista de {modo, nombre, hash de blob|tree}  → directorio
commit = {tree, parent(s), autor, mensaje}           → snapshot
ref    = nombre → hash de commit (main, v1.2.0, HEAD)
index  = staging area: lo que será el próximo commit
```

- `git add` escribe un blob y registra el path en el index; `git commit`
  envuelve el árbol del index en un commit cuyo parent es HEAD y mueve la ref.
- `git checkout main~2` mueve HEAD; las refs no se borran al hacer checkout:
  por eso "recuperar" es casi siempre "volver a apuntar una ref".
- `.gitignore` no es git: es una instrucción para `git add`, que decide qué
  entra en el index.

**Cómo falla en un proyecto DS.** Tratar git como "copias de carpetas con
fechas": `git add .` de todo (incluidos datasets y `.env`), `git reset --hard`
para "deshacer", y commits que mezclan veinte cambios porque "ya que estoy
guardando".

## Flujos de ramas

| Flujo | Cómo es | Cuándo |
|-------|---------|--------|
| GitHub Flow | `main` siempre desplegable + ramas cortas con PR | Default en equipos pequeños con CI fuerte |
| GitFlow | `develop` + `feature/` + `release/` + `hotfix/` | Releases versionados con ritmo propio |
| Trunk-based | Todos commitean a `main` en ramas de horas | Equipos con CI y tests rápidos |

**Para un proyecto dirigido por agentes: GitHub Flow / trunk-based.** El
agente produce unidades pequeñas y verificables (una feature del backlog →
una rama → un commit). Ramas de vida corta, commits pequeños y `main` siempre
verde son las tres propiedades que el arnés puede verificar por máquina.

**Cómo falla.** GitFlow en un proyecto de un agente: el arnés vive en `main`
(backlog, harness) y fusionar feature branches a través de `develop` duplica
la maquinaria sin aportar estabilidad. El flujo complejo es otro tipo de
deuda.

## Higiene de commits

**Conventional commits.** El prefijo declara la intención y permite
`git log --oneline` legible, changelogs automáticos y semver:

```text
feat:  nueva funcionalidad (sube minor en semver)
fix:   corrección de bug (sube patch)
docs:  documentación
chore: mantenimiento, refactor sin cambio de comportamiento
```

**Commits atómicos.** Un commit = un cambio lógico, completo en sí mismo.
Debe compilar y pasar tests sin sus vecinos. Si necesitas "y además..." o "y
por cierto..." en el mensaje, son dos commits.

**El cuerpo explica el "por qué".** El título es qué; el cuerpo, por qué se
hizo así y qué se descartó. El "cómo" ya lo dice el diff:

```text
fix: semilla no se propagaba a sklearn

Los estimators se creaban antes de set_seed, así que los splits
variaban entre runs. Mover la semilla al arranque de train hace el
pipeline reproducible; el test test_reproducible pincha este caso.
```

**Fuera de los commits:** secretos (`.env`, claves, tokens), artefactos
generados (`models/*.joblib`, `data/*.parquet`, `.rag-index/`, `__pycache__`),
ficheros de editor. Se excluyen en `.gitignore`; si algo generado entra por
error, se saca con `git rm --cached` (no con `git rm`: eso borra el fichero
del working tree).

**Cómo falla.** Un commit "arregla cosas" con 40 ficheros tocados: el `git
bisect` no puede aislar nada, la review es imposible y el revert revierte
media feature. Un `.env` commiteado una vez queda en el historial aunque se
borre después — la clave se rota, no se confía en el borrado.

## Merge vs rebase

- **Merge**: crea un commit de unión. Preserva el orden real y el contexto de
  "desde dónde" se trabajó. No reescribe nada.
- **Rebase**: re-aplica los commits de la rama sobre otra base. Linealiza la
  historia, pero reescribe hashes.

**Cuándo cada uno.**

- **Rebase para alinear con upstream**: `git rebase main` sobre una feature
  branch antes del PR, para que la diff contra `main` sea solo la feature.
- **Merge para integrar**: el PR se cierra con merge (o squash) cuando la
  rama va a quedar compartida.
- **Regla de oro**: nunca rebasear historia compartida (lo que ya está en
  `main` o en una rama que otros clonaron). El rebase reescribe hashes y
  rompe los clonos de todos. Esa es la razón técnica de por qué el arnés
  prohíbe el force-push.

**`git cherry-pick <hash>`**: aplica un commit suelto sobre HEAD. Útil para
llevar un fix de una rama a otra sin traer todo el árbol.

**`git commit --amend`**: reemplaza el último commit (título, mensaje,
contenido) y crea un hash nuevo. Seguro solo mientras ese commit no se haya
compartido. Nunca tras un push, salvo que seas el único que tiene la rama.

## Rebase interactivo

**`git rebase -i <base>`** abre una lista de commits y permite reordenar,
`squash` (fusiona con el anterior), `fixup` (fusiona sin mensaje), `reword`
(cambia mensaje), `drop`. Combinado con el autosquash:

```bash
git commit --fixup=<hash>    # marca un commit como "arreglo de <hash>"
git rebase -i --autosquash   # lo coloca automáticamente tras su objetivo
```

**Cómo falla.** Squashear todo en un único commit "feature completa": el PR
pierde la historia de decisiones y el bisect a un cambio concreto se vuelve
imposible. El squash unitario está bien; el squash total convierte una
historia de trabajo en una única caja negra. La regla: squashear la basura
(typos, "fix CI", debug) y conservar los pasos con sentido.

## Bisect

**Objetivo.** Encontrar el commit que rompió algo en O(log n) ejecuciones.

**Flujo de agente.**

```text
1. Reproducir:   construir un test que falle en HEAD (y pase en un commit bueno)
2. git bisect start HEAD <primer-commit-bueno>
3. git bisect run pytest tests/test_caso_roto.py   # 0 bueno, !=0 malo
4. git bisect reset
```

`git bisect run` ejecuta un comando por commit: el arnés lo aprovecha porque
"reproducir" ya es un test. Sin un test que reproduzca, el bisect manual
depende de inspeccionar cada commit.

**Cómo falla.** Bisect sobre commits gigantes y mezclados: el commit culpable
toca 30 ficheros y el fix no se puede aislar. El bisect presupone higiene de
commits; es otro argumento para commits atómicos.

## Reflog: la red de seguridad

**`git reflog`** registra los movimientos de HEAD y de las refs (checkouts,
resets, rebases, commits) durante ~90 días. Es el "undo" de git.

```bash
git reflog                          # ¿dónde estaba HEAD ayer?
git reset --hard HEAD@{2}           # volver a un estado que creías perdido
git reflog main                     # una rama, no solo HEAD
```

Recupera un `git reset --hard` equivocado, un rebase interrumpido y un commit
borrado "sin querer". La condición: la operación debe estar en el reflog, es
decir, los objetos no deben haberse recogido por el garbage collector.

**Cómo falla.** `git reset --hard` + cierre de terminal, o un clon nuevo que
no tiene tu reflog local: si el trabajo no se pusheó, no hay red. El push
regular de ramas de trabajo (aunque sea un WIP a un remoto propio) es la
única copia de seguridad real.

## Conflictos

Un conflicto ocurre cuando dos ramas cambian las mismas líneas de forma
incompatible. Git deja ambos lados en el working tree con marcadores
`<<<<<<<` / `=======` / `>>>>>>>`.

**Resolución.**

```bash
git merge main            # o rebase
git status                # ficheros en conflicto (UU)
# editar cada fichero: decidir cuál versión (o combinar)
git add <fichero>         # marcarlo resuelto
git commit                # merge: cierra; rebase: rebase --continue
```

**rerere** (`git config rerere.enabled true`) recuerda cómo resolviste un
conflicto y lo reaplica automáticamente si reaparece. Útil cuando el mismo
merge se rehace (rebase sobre main que avanza).

**Evitarlos.** Commits pequeños y frecuentes, `git pull`/`git rebase` a
menudo, no reescribir los mismos ficheros en dos ramas a la vez. En un
proyecto de agentes: una feature por rama y el pull antes de empezar a
implementar. Los conflictos de artefactos binarios (modelos, parquets) son
irresolubles por merge: otra razón para no versionarlos en git.

## Tags y releases

- **Semver** `MAJOR.MINOR.PATCH`: MAJOR rompe compatibilidad, MINOR añade
  funcionalidad compatible, PATCH corrige bugs. Un `feat:` sube MINOR; un
  `fix:` sube PATCH (el arnés lo hace en `commit_feature`).
- **Annotated tags** para releases: `git tag -a v0.1.1 -m "..."` — llevan
  mensaje, autor y fecha (a diferencia de las lightweight). Se empujan con
  `git push --tags` (o `--follow-tags`).
- **Releases** = tag + artefactos + changelog en el remoto. El CHANGELOG se
  genera desde los mensajes convencionales.

**Cómo falla.** Tags lightweight sin mensaje ("¿qué había en v0.1.0?"), o
etiquetar sobre commits no probados: un release debe apuntar a un commit que
pasó la puerta (`./init.sh` verde), no al último commit "que pintaba bien".

## Submódulos, LFS y datos grandes

- **Submódulos**: un repo dentro de un repo, anclado a un commit. Suelen ser
  una trampa en proyectos DS: cada checkout requiere `git submodule update
  --init`, el versionado compartido se vuelve frágil y los conflictos de
  "qué commit apunta el submódulo" confunden al equipo. Solo tienen sentido
  para pinchar librerías de terceros estables; no para datos ni modelos.
- **git-lfs**: almacena punteros en git y el contenido en un servidor LFS.
  Sirve para ficheros grandes que SÍ son código (assets), pero añade un
  servidor y cuotas.
- **Datos y modelos NO van en git.** Un dataset de 2 GB infla el clon para
  siempre y corrompe la historia (cada cambio = 2 GB de objetos). El
  versionado de datos es otro sistema (DVC, hashes, almacenamiento externo):
  el repo guarda la versión/referencia, no el contenido.

**Cómo falla.** `models/` con los `.joblib` commiteados "para no perderlos":
el clon pesa gigas, los merges sobre binarios son irresolubles y el historial
es irrecuperable. La regla del proyecto: artefactos fuera de git, referencia
a su versión dentro.

{% if use_sdd %}
## Las reglas del arnés con spec-driven

Con `use_sdd` el historial tiene una propiedad extra que el reviewer debe
verificar: **el contrato va antes que el código y el backlog no se toca a
mano**.

```text
1. Los únicos que editan features/ y harness/ es el agente `harness`
   (un recurso, un dueño). Un commit que toque features/*.feature o
   harness/* por fuera es un REJECT automático.
2. La secuencia de commits de una feature con spec-driven:
   harness write_feature → [aprobación humana] → implementer (TDD)
   → reviewer → mutation (¿los tests muerden?) → finish.
3. Un commit de "código" sin su features/<ID>.feature aprobado en el
   historial indica un flujo que se saltó la puerta humana.
4. Los escenarios Gherkin son parte del contrato: si el código cambia el
   comportamiento, el .feature cambia antes (o a la vez) — nunca después.
```

El `git log --oneline` de una feature cerrada debe contar la historia en ese
orden: spec, tests, código, evidencia. Un commit que mezcle "spec + código +
tests" en uno solo es una bandera roja igual que en cualquier otro proyecto.
{% endif %}

## Las reglas del arnés (este proyecto)

Reglas operativas que el lider y el reviewer aplican a cada ciclo:

| Regla | Detalle |
|-------|---------|
| `git commit_feature` | Sube **patch** en `pyproject.toml`, actualiza CHANGELOG y propone el commit |
| `--dry-run` primero | Propone versión, mensaje y ficheros; pide OK antes del commit real |
| No force-push | Prohibido. El push es siempre decisión explícita del humano |
| Evidencia antes de commit | Un commit de cierre exige `./init.sh` en verde (la puerta de `harness finish`) |
| Commits convencionales | `feat:`/`fix:`/`docs:`/`chore:` en `git log` |

```bash
uv run python -m agents --json run git commit_feature --id DATA-001 --title "..." --dry-run true
# revisar propuesta, confirmar
uv run python -m agents --json run git commit_feature --id DATA-001 --title "..."
```

**Cómo falla.** Un agente que commitea sin evidencia "porque los tests pasan
en mi cabeza", o que fuerza el push para "arreglar" la rama compartida: el
historial compartido se reescribe y el resto del equipo (y el arnés) pierde
referencias. La puerta de permisos (`agents/contracts.py` marca los commits
como `destructive`) existe para que el commit/push no ocurra sin el humano.

## Checklist mínimo para el reviewer

```text
□ ¿Es atómico? Un cambio lógico, compila y pasa tests solo.      (si no → split)
□ ¿El mensaje explica el porqué (no solo el qué)?                (cuerpo, no solo título)
□ ¿Prefijo convencional coherente? (feat/fix/docs/chore)         (y semver coherente)
□ ¿Hay secretos?  (grep api_key|token|password; .env no listado)
□ ¿Hay artefactos generados o binarios?  (git ls-files | grep parquet|joblib|rag-index)
□ ¿Toca ficheros fuera del alcance de la feature?                (diff mínimo)
□ ¿La rama está alineada con main?  (rebase previo; sin conflictos)
□ ¿La evidencia (init.sh / pytest) está pegada en el PR/commit?  (regla del arnés)
□ ¿Cumple las reglas de spec-driven si aplica?  (features/ y harness/ intactos)
```

## Fuentes

- Scott Chacon y Ben Straub, "Pro Git" (2ª ed., Apress) — el modelo de
  objetos, rebase, bisect, reflog, submódulos y LFS.
- Documentación de Conventional Commits y de semver.org.
- Documentación de GitHub Flow (github.com) y GitFlow (nvie).
- Documentación de git: `git help <comando>`, `git help workflows`.
- Documentación de DVC (versionado de datos) y git-lfs.
- Las reglas del proyecto: `AGENTS.md` (protocolo del arnés, `commit_feature`,
  la puerta de permisos) e `ingenieria/reglas-codigo.md` (R17–R21).
