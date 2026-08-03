"""
agents.tools.rag_tool — RAG local con ChromaDB: búsqueda híbrida sin API key.

Indexa documentación del proyecto (código, prompts, docs, vault, README, y la
memoria del arnés: harness/progress/ y harness/featureslist.json) y permite buscar en lenguaje
natural. Sin GPU, sin API key, funciona offline.

La búsqueda es **híbrida**: funde el ranking vectorial de ChromaDB con un BM25
léxico en stdlib. No es un adorno — el embedder por defecto está entrenado en
inglés y este proyecto se documenta en español, así que la mitad de la señal
útil es literal ("train_model", "drift", "GradientBoosting") y el vector solo
no la ve. Para embeddings multilingües de verdad: DSKIT_RAG_EMBEDDER, abajo.

Dependencias: chromadb. Opcional: sentence-transformers (embedder multilingüe).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from agents.tools.registry import register_tool

try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


COLLECTION_NAME = "dskit-rag"

#: El embedder por defecto de chroma (all-MiniLM-L6-v2 ONNX) trunca la entrada
#: a 256 tokens wordpiece — unos 1.000 caracteres. Lo que pase de ahí se guarda
#: en la base pero NO entra en el vector: queda indexado e irrecuperable, sin
#: aviso. Por eso el techo no es una preferencia de estilo, es el límite físico
#: del modelo, y se aplica a todos los chunks vengan del troceador que vengan.
_MAX_CHUNK_CHARS = 1000

#: Solape entre trozos de un mismo bloque: una frase partida por el techo tiene
#: que aparecer entera en alguno de los dos lados o deja de ser buscable.
_CHUNK_OVERLAP_CHARS = 120

#: Suelo. Solo aplica a los cortes `fallback`, que parten a ciegas por tamaño:
#: ahí un fragmento diminuto no significa nada y solo mete ruido. Un corte
#: estructural (encabezado, función, clase) se indexa aunque sea corto.
_MIN_CHUNK_CHARS = 100

#: Embedders disponibles. `onnx` es el de chroma: cero dependencias extra, pero
#: entrenado en inglés. `multilingual` entiende español de verdad a cambio de
#: arrastrar sentence-transformers (y torch). Se elige con la variable de
#: entorno DSKIT_RAG_EMBEDDER y queda grabado en los metadatos de la colección:
#: ambos dan vectores de 384 dimensiones, así que mezclarlos NO daría error de
#: chroma — daría resultados sin sentido. Por eso se detecta y se exige rebuild.
_EMBEDDERS = {
    "onnx": "all-MiniLM-L6-v2 (ONNX, inglés, sin dependencias extra)",
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2 (español + 50 idiomas)",
}

#: Corpus léxico cacheado por ruta de índice. Lo invalida `index_project`.
_CORPUS_CACHE: dict[str, dict[str, Any]] = {}

_PALABRA = re.compile(r"[a-z0-9_]+")


def _file_type(source: str) -> str:
    if source.startswith("url:"):
        return "url"
    if source.endswith(".py"):
        return "code"
    if "/prompts/" in source:
        return "prompt"
    if "/vault/" in source or source.startswith("vault/"):
        return "vault"
    if source.startswith("harness/"):
        return "harness"
    return "doc"


#: Frases que en un documento recuperado no son información sino un intento de
#: dar órdenes. La lista es corta y honesta: NO es una defensa —una inyección
#: reformulada la esquiva sin esfuerzo—, es una señal para poder marcar el
#: fragmento y para que un test pueda detectar que alguien envenenó el índice.
#: La defensa de verdad es que un dato no pueda provocar nada irreversible:
#: eso lo da la puerta de `agents/permissions.py`, no esta lista.
_INYECCION = re.compile(
    r"(?i)("
    r"ignor(a|e|ar|ing)\s+(todas\s+)?(las\s+)?(instrucciones|instructions)"
    r"|disregard\s+(all\s+)?(previous|prior)"
    r"|olvida\s+(todo\s+)?lo\s+anterior"
    r"|(new|nuevas?)\s+(system\s+)?(prompt|instrucciones)"
    r"|system\s*prompt\s*[:=]"
    r"|eres\s+un\s+asistente\s+que"
    r"|you\s+are\s+now\s+"
    r"|<\s*/?\s*(system|assistant)\s*>"
    r")"
)

#: De dónde salió el texto. `repo` es lo que vive en tu repositorio y pasa por
#: tus revisiones; `externo` es lo que se descargó de una URL y no lo ha
#: revisado nadie. La distinción existe para que la respuesta pueda
#: presentarlos separados en vez de mezclados y con la misma pinta.
CONFIANZA_REPO = "repo"
CONFIANZA_EXTERNA = "externo"


def _confianza(source: str) -> str:
    return CONFIANZA_EXTERNA if source.startswith("url:") else CONFIANZA_REPO


def _parece_inyeccion(texto: str) -> bool:
    return bool(_INYECCION.search(texto))


def _embedder_id() -> str:
    """Qué embedder toca. Un id desconocido cae al de siempre, no rompe."""
    elegido = os.environ.get("DSKIT_RAG_EMBEDDER", "onnx").strip().lower()
    return elegido if elegido in _EMBEDDERS else "onnx"


class RagConfigError(RuntimeError):
    """El RAG está mal configurado (embedder pedido que no se puede cargar)."""


def _embedding_function(embedder: str) -> Any:
    """`None` = el ONNX por defecto de chroma. No se pasa el kwarg y ya."""
    if embedder != "multilingual":
        return None
    try:
        from chromadb.utils import embedding_functions

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
    except Exception as exc:  # noqa: BLE001 — chroma lanza ValueError, no ImportError
        raise RagConfigError(
            "DSKIT_RAG_EMBEDDER=multilingual pero sentence-transformers no está "
            "instalado. Ejecuta 'uv sync --extra rag_multilingual' o vuelve a "
            f"DSKIT_RAG_EMBEDDER=onnx. Detalle: {str(exc)[:120]}"
        ) from exc


def _tokenizar(texto: str) -> list[str]:
    """
    Tokens para el BM25. Sin acentos a propósito: "cómo" y "como" son la misma
    palabra para quien busca, y media documentación de este repo va acentuada.
    """
    plano = unicodedata.normalize("NFKD", texto.lower())
    plano = "".join(c for c in plano if not unicodedata.combining(c))
    return _PALABRA.findall(plano)


class _Bm25:
    """
    BM25 Okapi en stdlib, sobre índice invertido.

    El índice es invertido por dos razones concretas. Una: puntuar recorría
    antes los `tf` de TODOS los chunks por cada término de la consulta, así
    que una búsqueda costaba O(términos × chunks) aunque el término
    apareciera en dos ficheros. Dos: las postings caben en un JSON, y eso es
    lo que permite persistir el índice (`a_estado`/`desde_estado`) en vez de
    releer y retokenizar el proyecto entero en cada invocación de la CLI,
    que es de un solo disparo y no aprovecha ninguna caché en memoria.
    """

    def __init__(self, documentos: list[list[str]] | None = None) -> None:
        documentos = documentos or []
        self.n = len(documentos)
        self.largos = [len(d) for d in documentos]
        self.medio = (sum(self.largos) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(documentos):
            for termino, veces in Counter(doc).items():
                self.postings.setdefault(termino, []).append((i, veces))
        self.idf = _idf_desde_postings(self.postings, self.n)

    def puntua(self, consulta: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
        if not self.n or not self.medio:
            return [0.0] * self.n
        marcadores = [0.0] * self.n
        for termino in consulta:
            idf = self.idf.get(termino)
            if idf is None:
                continue
            for i, f in self.postings[termino]:
                norma = 1 - b + b * (self.largos[i] / self.medio)
                marcadores[i] += idf * (f * (k1 + 1)) / (f + k1 * norma)
        return marcadores

    def a_estado(self) -> dict[str, Any]:
        """Estado serializable. El `idf` no se guarda: se deriva de las postings."""
        return {"largos": self.largos, "postings": self.postings}

    @classmethod
    def desde_estado(cls, estado: dict[str, Any]) -> _Bm25:
        obj = cls()
        obj.largos = estado["largos"]
        obj.n = len(obj.largos)
        obj.medio = (sum(obj.largos) / obj.n) if obj.n else 0.0
        obj.postings = estado["postings"]
        obj.idf = _idf_desde_postings(obj.postings, obj.n)
        return obj


def _idf_desde_postings(postings: dict[str, list], n: int) -> dict[str, float]:
    return {
        t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()
    }


class _SoloTexto(HTMLParser):
    """Extrae texto de HTML. Indexar `<div class=...>` no ayuda a nadie."""

    _MUDOS = {"script", "style", "noscript", "head", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.partes: list[str] = []
        self._silencio = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._MUDOS:
            self._silencio += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._MUDOS and self._silencio:
            self._silencio -= 1
        elif tag in ("p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr"):
            self.partes.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._silencio and data.strip():
            self.partes.append(data.strip())

    def texto(self) -> str:
        crudo = " ".join(self.partes)
        return re.sub(r"\n\s*\n\s*\n+", "\n\n", crudo).strip()


def _html_a_texto(contenido: str) -> str:
    if "<html" not in contenido[:2000].lower() and "<body" not in contenido[:2000].lower():
        return contenido  # ya venía en texto plano o markdown
    parser = _SoloTexto()
    try:
        parser.feed(contenido)
    except Exception:  # noqa: BLE001 — HTML roto no debe tumbar el indexado
        return contenido
    return parser.texto() or contenido


@register_tool("rag")
class RagTool:
    @staticmethod
    def available() -> bool:
        return HAS_CHROMA

    @staticmethod
    def _client(root: Path) -> chromadb.ClientAPI | None:
        if not HAS_CHROMA:
            return None
        db_dir = root / ".rag-index"
        db_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

    @staticmethod
    def _collection(client: chromadb.ClientAPI, embedder: str | None = None) -> Any:
        """
        `get_or_create` y no `get` + `except`: la excepción de «no existe» ha
        cambiado de tipo entre versiones de chroma (hoy `NotFoundError`, que no
        hereda de `ValueError`), y capturar la equivocada dejaba la colección
        sin crear y el RAG entero muerto en el primer `index`.
        """
        embedder = embedder or _embedder_id()
        kwargs: dict[str, Any] = {
            "metadata": {"hnsw:space": "cosine", "embedder": embedder},
        }
        funcion = _embedding_function(embedder)
        if funcion is not None:
            kwargs["embedding_function"] = funcion
        return client.get_or_create_collection(COLLECTION_NAME, **kwargs)

    @staticmethod
    def _abrir(root: Path) -> tuple[Any | None, str | None]:
        """
        (colección, error). Un solo sitio donde abrir el índice, porque las
        tres formas de que falle —sin chromadb, sin el embedder pedido, o con
        un índice de otro embedder— tienen que llegar al usuario como un
        mensaje accionable y no como una traza de chroma.
        """
        client = RagTool._client(root)
        if client is None:
            return None, "chromadb no disponible. Instala: uv sync --extra rag"
        try:
            collection = RagTool._collection(client)
        except RagConfigError as exc:
            return None, str(exc)
        return collection, RagTool._desajuste_embedder(collection)

    @staticmethod
    def _desajuste_embedder(collection: Any) -> str | None:
        """El índice de otro embedder no da error: da basura. Mejor pararlo."""
        grabado = (collection.metadata or {}).get("embedder", "onnx")
        actual = _embedder_id()
        if grabado == actual or collection.count() == 0:
            return None
        return (
            f"El índice se construyó con el embedder '{grabado}' y ahora está "
            f"activo '{actual}'. Los vectores no son comparables: reconstruye "
            f"con 'rag index --rebuild' o vuelve a DSKIT_RAG_EMBEDDER={grabado}."
        )

    # -- ID helpers ----------------------------------------------------------

    @staticmethod
    def _chunk_id(source: str, line: int, text: str, section_type: str) -> str:
        """
        El hash cubre el texto ENTERO. Con un prefijo (antes: 80 caracteres) dos
        chunks que empiezan igual comparten id, y `collection.add` se queda en
        silencio con el primero: la edición se perdía sin error.
        """
        crudo = f"{source}:{line}:{section_type}:{text}".encode()
        return hashlib.md5(crudo).hexdigest()[:16]

    @staticmethod
    def _add_chunk(chunks: list, text: str, source: str, line: int,
                   section_type: str = "fallback", file_hash: str = "") -> None:
        """
        Apila el chunk, partiéndolo si no cabe en la ventana del embedder.

        **La estructura gana al tamaño, pero el techo gana a la estructura.** Un
        corte estructural se indexa aunque sea corto; el suelo solo aplica a los
        `fallback`. Lo que no se negocia es el máximo: un chunk de 18.000
        caracteres no es un chunk grande, es un chunk del que solo se indexan
        los primeros 1.000 y el resto es humo.
        """
        text = text.strip()
        if not text:
            return
        if section_type == "fallback" and len(text) < _MIN_CHUNK_CHARS:
            return
        for trozo, salto in RagTool._partir(text):
            chunk = RagTool._make_chunk(trozo, source, line + salto, section_type, file_hash)
            if chunk is not None:
                chunks.append(chunk)

    @staticmethod
    def _solape(lineas: list[str]) -> list[str]:
        """Cola del trozo anterior que se repite al principio del siguiente."""
        cola: list[str] = []
        largo = 0
        for linea in reversed(lineas):
            if largo >= _CHUNK_OVERLAP_CHARS:
                break
            cola.insert(0, linea)
            largo += len(linea) + 1
        return cola

    @staticmethod
    def _partir(texto: str) -> list[tuple[str, int]]:
        """
        Trocea por líneas hasta el techo, con solape. Devuelve (trozo, líneas
        de desplazamiento) para que el metadato `line` siga apuntando a algo.
        """
        if len(texto) <= _MAX_CHUNK_CHARS:
            return [(texto, 0)]

        lineas = texto.split("\n")
        piezas: list[tuple[str, int]] = []
        actual: list[str] = []
        largo = 0
        inicio = 0
        paso = _MAX_CHUNK_CHARS - _CHUNK_OVERLAP_CHARS
        i = 0

        while i < len(lineas):
            linea = lineas[i]
            if actual and largo + len(linea) + 1 > _MAX_CHUNK_CHARS:
                piezas.append(("\n".join(actual), inicio))
                cola = RagTool._solape(actual)
                largo_cola = sum(len(x) + 1 for x in cola)
                if largo_cola + len(linea) + 1 > _MAX_CHUNK_CHARS:
                    cola, largo_cola = [], 0  # el solape no cabe: se sacrifica
                actual, largo, inicio = list(cola), largo_cola, i - len(cola)

            if not actual and len(linea) > _MAX_CHUNK_CHARS:
                # una sola línea más larga que el techo (JSON minificado, tablas)
                for j in range(0, len(linea), paso):
                    piezas.append((linea[j : j + _MAX_CHUNK_CHARS], i))
                i += 1
                inicio = i
                continue

            actual.append(linea)
            largo += len(linea) + 1
            i += 1

        if actual:
            piezas.append(("\n".join(actual), inicio))
        return [(p.strip(), off) for p, off in piezas if p.strip()]

    @staticmethod
    def _ensure_not_empty(chunks: list, content: str, source: str,
                          section_type: str = "fallback", file_hash: str = "") -> list:
        """
        Un documento nunca desaparece del índice por ser corto.

        Si ningún fragmento llegó al suelo, se emite uno con el texto entero:
        una nota de 90 caracteres sigue siendo buscable, y perderla sin aviso
        es peor que indexar un chunk pequeño.
        """
        if chunks or not content.strip():
            return chunks
        sueltos: list[dict[str, Any]] = []
        RagTool._add_chunk(sueltos, content, source, 0, section_type, file_hash)
        if sueltos:
            return sueltos
        chunk = RagTool._make_chunk(content, source, 0, section_type, file_hash)
        return [chunk] if chunk is not None else []

    @staticmethod
    def _make_chunk(text: str, source: str, line: int, section_type: str = "fallback",
                    file_hash: str = "") -> dict[str, Any] | None:
        """Construye el chunk. Sin política: solo rechaza el texto vacío."""
        text = text.strip()
        if not text:
            return None
        return {
            "id": RagTool._chunk_id(source, line, text, section_type),
            "text": text,
            "metadata": {
                "source": source,
                "line": line,
                "char_len": len(text),
                "file_type": _file_type(source),
                "section_type": section_type,
                "file_hash": file_hash,
                "trust": _confianza(source),
                "injection_flag": _parece_inyeccion(text),
            },
        }

    # -- Recursive chunking --------------------------------------------------

    @staticmethod
    def _con_contexto(texto: str, source: str, clase: str | None) -> str:
        """
        Una cabecera con la ruta del fichero (y la clase, si la hay) delante de
        cada chunk de código. Sin ella, `def fit(self, X, y)` es un fragmento
        anónimo: no hay nada en el vector que lo relacione con
        `models/train_model.py`, que es justo por lo que pregunta la gente.
        """
        if not texto.strip():
            return ""
        cabecera = f"# {source}" + (f" · class {clase}" if clase else "")
        return f"{cabecera}\n{texto}"

    @staticmethod
    def chunk_py(content: str, source: str, file_hash: str = "") -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        lines = content.split("\n")
        actual: list[str] = []
        inicio = 0
        tipo = "module"
        clase: str | None = None

        def _cerrar(clase_del_bloque: str | None) -> None:
            RagTool._add_chunk(
                chunks,
                RagTool._con_contexto("\n".join(actual), source, clase_del_bloque),
                source, inicio, tipo, file_hash,
            )

        for i, line in enumerate(lines):
            match_def = re.match(r"^(\s*)(async\s+)?def\s+(\w+)", line)
            match_class = re.match(r"^(\s*)class\s+(\w+)", line)

            if not (match_def or match_class):
                actual.append(line)
                continue

            _cerrar(clase)
            actual, inicio = [line], i
            if match_class:
                # solo las clases de primer nivel dan contexto a sus métodos
                clase = match_class.group(2) if not match_class.group(1) else clase
                tipo = "class"
            else:
                tipo = "function"
                if not match_def.group(1):
                    clase = None  # función suelta: se sale de la clase anterior

        _cerrar(clase)

        if not chunks:
            chunks = RagTool._chunk_by_size(content, source, file_hash)
        return RagTool._ensure_not_empty(chunks, content, source, "module", file_hash)

    @staticmethod
    def chunk_md(content: str, source: str, file_hash: str = "") -> list[dict[str, Any]]:
        """
        Trocea por encabezados y le cuelga a cada sección la ruta de sus
        ancestros: un `## Uso` suelto no dice de qué va, `README > Docker > Uso`
        sí. El troceado por tamaño lo resuelve `_add_chunk`, que parte con
        solape lo que no quepa en la ventana del embedder.
        """
        chunks: list[dict[str, Any]] = []
        lines = content.split("\n")
        seccion: list[str] = []
        inicio = 0
        ruta: list[tuple[int, str]] = []  # (nivel, título) de la sección abierta

        def _cerrar() -> None:
            if not seccion:
                return
            texto = "\n".join(seccion)
            migas = " > ".join(t for _, t in ruta[:-1])
            if migas:
                texto = f"[{source} — {migas}]\n{texto}"
            RagTool._add_chunk(
                chunks, texto, source, inicio,
                "heading" if ruta else "paragraph", file_hash,
            )

        for i, line in enumerate(lines):
            m = re.match(r"^(#+)\s+(.*)", line)
            if not m:
                seccion.append(line)
                continue
            _cerrar()
            nivel, titulo = len(m.group(1)), m.group(2).strip()
            while ruta and ruta[-1][0] >= nivel:
                ruta.pop()
            ruta.append((nivel, titulo))
            seccion, inicio = [line], i

        _cerrar()

        if not chunks:
            chunks = RagTool._chunk_by_size(content, source, file_hash)
        return RagTool._ensure_not_empty(
            chunks, content, source, "heading" if ruta else "paragraph", file_hash
        )

    @staticmethod
    def _chunk_by_size(content: str, source: str, file_hash: str = "") -> list[dict[str, Any]]:
        """
        Último recurso: agrupa frases hasta el techo. El `line` que emite es el
        número de línea real del fichero, no el índice de la frase — un
        resultado que apunta a `t.md:249` en un fichero de 12 líneas no sirve
        para abrir nada.
        """
        chunks: list[dict[str, Any]] = []
        frases = re.split(r"(?<=[.!?])\s+", content)
        if not frases:
            return chunks

        acumulado: list[str] = []
        largo = 0
        consumido = 0
        linea_inicio = 0

        def _volcar() -> None:
            if acumulado:
                RagTool._add_chunk(
                    chunks, " ".join(acumulado), source, linea_inicio, "fallback", file_hash
                )

        for frase in frases:
            if acumulado and largo + len(frase) > _MAX_CHUNK_CHARS:
                _volcar()
                linea_inicio = content.count("\n", 0, consumido)
                acumulado, largo = [], 0
            acumulado.append(frase)
            largo += len(frase) + 1
            consumido += len(frase) + 1

        _volcar()  # la cola también se indexa: antes se tiraba en silencio
        return chunks

    # -- Indexing ------------------------------------------------------------

    #: De donde sale el indice. Eran cinco bloques copiados con el mismo
    #: esqueleto (existe? -> glob -> leer -> trocear): anadir una fuente
    #: significaba copiar el sexto. Ahora es una fila mas en la tabla.
    #: (subdirectorio, patron, recursivo, sufijos admitidos o None, troceador)
    FUENTES = (
        ("{{ project_slug }}", "*.py", True, None, "py"),
        # El codigo que se despliega tambien es documentacion: sin estas cuatro
        # filas, preguntar por la API o por el drift solo devolvia los prompts
        # que los describen, nunca la implementacion que los hace. Los
        # subdirectorios que no existan (extras desactivados) se saltan solos.
        ("api", "*.py", True, None, "py"),
        ("chat", "*.py", True, None, "py"),
        ("monitoring", "*.py", True, None, "py"),
        ("tuning", "*.py", True, None, "py"),
        ("agents", "*.py", True, None, "py"),
        ("agents/prompts", "*.md", False, None, "md"),
        ("docs", "*.*", True, (".md", ".rst"), "md"),
        ("vault", "*.md", True, None, "md"),
        # Memoria del arnes: el historico de features cerradas y sus decisiones
        # es lo que un agente nuevo necesita buscar en lenguaje natural sin
        # releer todo harness/progress/.
        ("harness/progress", "*.md", False, None, "md"),
    )

    #: Ruido que nunca aporta a una busqueda semantica.
    EXCLUIDOS = ("__pycache__", "/tests/", "/.venv/", "/node_modules/", "/build/")

    #: Ficheros sueltos de la raiz que tambien entran al indice.
    FICHEROS_RAIZ = ("README.md", "AGENTS.md", "CHANGELOG.md", "CONTRIBUTING.md")

    @staticmethod
    def _documentos(root: Path) -> list[tuple[str, str, str]]:
        """(origen, texto, troceador) de cada fichero indexable."""
        docs: list[tuple[str, str, str]] = []
        vistos: set[str] = set()

        def _leer(fichero: Path) -> str | None:
            try:
                return fichero.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                return None  # un fichero ilegible no tumba el indexado entero

        for sub, patron, recursivo, sufijos, tipo in RagTool.FUENTES:
            base = root / sub
            if not base.exists():
                continue
            for fichero in sorted(base.rglob(patron) if recursivo else base.glob(patron)):
                if sufijos and fichero.suffix not in sufijos:
                    continue
                origen = str(fichero.relative_to(root))
                if origen in vistos:
                    continue  # dos fuentes pueden solaparse (agents, agents/prompts)
                if any(malo in f"/{origen}" for malo in RagTool.EXCLUIDOS):
                    continue
                texto = _leer(fichero)
                if texto is not None:
                    vistos.add(origen)
                    docs.append((origen, texto, tipo))

        for nombre in RagTool.FICHEROS_RAIZ:
            texto = _leer(root / nombre) if (root / nombre).exists() else None
            if texto is not None and nombre not in vistos:
                vistos.add(nombre)
                docs.append((nombre, texto, "md"))

        backlog = RagTool._backlog_a_markdown(root)
        if backlog:
            docs.append(("harness/featureslist.json", backlog, "md"))
        return docs

    @staticmethod
    def _trocear(origen: str, texto: str, tipo: str, huella: str) -> list[dict[str, Any]]:
        return (
            RagTool.chunk_py(texto, origen, huella) if tipo == "py"
            else RagTool.chunk_md(texto, origen, huella)
        )

    @staticmethod
    def _recolectar(root: Path) -> list[dict[str, Any]]:
        """Recorre las fuentes declaradas y devuelve todos los chunks."""
        trozos: list[dict[str, Any]] = []
        for origen, texto, tipo in RagTool._documentos(root):
            trozos.extend(RagTool._trocear(origen, texto, tipo, RagTool._huella(texto)))
        return trozos

    @staticmethod
    def _huella(texto: str) -> str:
        return hashlib.md5(texto.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _backlog_a_markdown(root: Path) -> str:
        """
        El backlog se aplana a markdown antes de indexarlo: buscar
        "criterios de aceptacion" contra JSON crudo no devuelve nada util.
        """
        backlog = root / "harness/featureslist.json"
        if not backlog.exists():
            return ""
        try:
            doc = json.loads(backlog.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        bloques = []
        for feat in doc.get("features", []):
            criterios = "\n".join("- " + c for c in feat.get("acceptance_criteria", []))
            bloques.append(
                "## {} — {} [{}]\n\n{}\n\nCriterios de aceptación:\n{}\n".format(
                    feat.get("id"), feat.get("title"), feat.get("status"),
                    feat.get("description", ""), criterios,
                )
            )
        return "\n".join(bloques)

    @staticmethod
    def _chunks_backlog(root: Path) -> list[dict[str, Any]]:
        texto = RagTool._backlog_a_markdown(root)
        return RagTool.chunk_md(texto, "harness/featureslist.json", RagTool._huella(texto)) if texto else []

    @staticmethod
    def index_project(root: Path, rebuild: bool = False) -> dict[str, Any]:
        """
        Indexa por fichero y con huella de contenido.

        El add-only de antes no borraba nada: editar un fichero dejaba el chunk
        viejo dentro para siempre y borrarlo no lo sacaba del índice. Con
        `harness/progress/` y `harness/featureslist.json` dentro —que cambian en cada feature—
        eso contamina la memoria del arnés al ritmo al que se trabaja. Ahora
        cada fichero lleva su huella: si no ha cambiado no se re-embebe, y si
        cambió o desapareció, sus chunks se borran antes de reescribirlos.
        """
        client = RagTool._client(root)
        if client is None:
            return {"error": "chromadb no disponible. Instala: uv sync --extra rag"}

        if rebuild:
            with contextlib.suppress(Exception):  # no existía: es el caso feliz
                client.delete_collection(COLLECTION_NAME)
        collection, aviso = RagTool._abrir(root)
        if collection is None:
            return {"error": aviso}
        if aviso and not rebuild:
            return {"error": aviso}

        docs = RagTool._documentos(root)
        vivos = {origen: RagTool._huella(texto) for origen, texto, _ in docs}

        ids_por_origen: dict[str, list[str]] = {}
        huella_por_origen: dict[str, str] = {}
        guardados = collection.get(include=["metadatas"])
        for cid, meta in zip(guardados["ids"], guardados["metadatas"] or [], strict=False):
            origen = (meta or {}).get("source", "?")
            ids_por_origen.setdefault(origen, []).append(cid)
            huella_por_origen[origen] = (meta or {}).get("file_hash", "")

        # 1. ficheros que ya no existen (las URLs no se tocan: no vienen de disco)
        huerfanos = [
            cid
            for origen, cids in ids_por_origen.items()
            if origen not in vivos and not origen.startswith("url:")
            for cid in cids
        ]
        if huerfanos:
            collection.delete(ids=huerfanos)

        # 2. ficheros nuevos o modificados
        nuevos: list[dict[str, Any]] = []
        actualizados = 0
        sin_cambios = 0
        for origen, texto, tipo in docs:
            huella = vivos[origen]
            if huella_por_origen.get(origen) == huella and ids_por_origen.get(origen):
                sin_cambios += 1
                continue
            if ids_por_origen.get(origen):
                collection.delete(ids=ids_por_origen[origen])
                actualizados += 1
            nuevos.extend(RagTool._trocear(origen, texto, tipo, huella))

        if nuevos:
            unicos: dict[str, dict[str, Any]] = {c["id"]: c for c in nuevos}
            lote = list(unicos.values())
            for i in range(0, len(lote), 500):
                bloque = lote[i : i + 500]
                collection.add(
                    ids=[c["id"] for c in bloque],
                    documents=[c["text"] for c in bloque],
                    metadatas=[c["metadata"] for c in bloque],
                )

        _CORPUS_CACHE.pop(str(root / ".rag-index"), None)
        RagTool._guardar_indice_lexico(root, collection)
        return {
            "total_chunks": collection.count(),
            "new_chunks": len(nuevos),
            "updated_files": actualizados,
            "unchanged_files": sin_cambios,
            "deleted_chunks": len(huerfanos),
            "sources": len(vivos),
            "embedder": _embedder_id(),
        }

    @staticmethod
    def index_url(root: Path, url: str, text: str) -> dict[str, Any]:
        collection, aviso = RagTool._abrir(root)
        if collection is None or aviso:
            return {"error": aviso}

        origen = f"url:{url}"
        previos = collection.get(where={"source": origen}, include=[])["ids"]
        if previos:
            collection.delete(ids=previos)  # reindexar una URL la reemplaza

        limpio = _html_a_texto(text)
        chunks = RagTool.chunk_md(limpio, origen, RagTool._huella(limpio))
        unicos = list({c["id"]: c for c in chunks}.values())
        if unicos:
            collection.add(
                ids=[c["id"] for c in unicos],
                documents=[c["text"] for c in unicos],
                metadatas=[{**c["metadata"], "type": "url"} for c in unicos],
            )
        _CORPUS_CACHE.pop(str(root / ".rag-index"), None)
        RagTool._guardar_indice_lexico(root, collection)
        return {"chunks_indexed": len(unicos), "url": url, "replaced": len(previos)}

    # -- Search --------------------------------------------------------------

    @staticmethod
    def _ruta_bm25(root: Path) -> Path:
        return root / ".rag-index" / "bm25.json"

    @staticmethod
    def _guardar_indice_lexico(root: Path, collection: Any) -> None:
        """
        Vuelca el índice léxico a disco al terminar de indexar.

        Lo escriben `index_project` e `index_url`, que son los dos únicos
        sitios donde la colección cambia. Si falla, no pasa nada: la búsqueda
        lo reconstruye en memoria. Por eso se traga la excepción en vez de
        tumbar un indexado que ya ha ido bien.
        """
        try:
            datos = collection.get(include=["documents"])
            documentos = datos["documents"] or []
            bm25 = _Bm25([_tokenizar(d) for d in documentos])
            payload = {"count": len(documentos), "ids": datos["ids"], "estado": bm25.a_estado()}
            ruta = RagTool._ruta_bm25(root)
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:  # noqa: BLE001 — es una caché, no la fuente de verdad
            with contextlib.suppress(OSError):
                RagTool._ruta_bm25(root).unlink(missing_ok=True)

    @staticmethod
    def _indice_lexico(root: Path, collection: Any) -> dict[str, Any]:
        """
        Índice BM25: de la caché en memoria, del disco, o reconstruido.

        El `count` es la validación: la colección solo la tocan `index_project`
        e `index_url`, y ambos reescriben el fichero, así que un count que
        cuadra significa que el volcado corresponde a lo que hay indexado.
        Si no cuadra (o no hay fichero) se reconstruye y se vuelve a guardar.
        """
        clave = str(root / ".rag-index")
        total = collection.count()

        guardado = _CORPUS_CACHE.get(clave)
        if guardado is not None and guardado["n"] == total:
            return guardado

        ruta = RagTool._ruta_bm25(root)
        if ruta.exists():
            try:
                payload = json.loads(ruta.read_text(encoding="utf-8"))
                if payload.get("count") == total:
                    indice = {
                        "n": total,
                        "ids": payload["ids"],
                        "bm25": _Bm25.desde_estado(payload["estado"]),
                    }
                    _CORPUS_CACHE[clave] = indice
                    return indice
            except (OSError, ValueError, KeyError):
                pass  # volcado corrupto o de otra versión: se rehace

        datos = collection.get(include=["documents"])
        documentos = datos["documents"] or []
        indice = {
            "n": total,
            "ids": datos["ids"],
            "bm25": _Bm25([_tokenizar(d) for d in documentos]),
        }
        _CORPUS_CACHE[clave] = indice
        RagTool._guardar_indice_lexico(root, collection)
        return indice

    @staticmethod
    def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
        """
        Reciprocal Rank Fusion: funde rankings sin necesidad de que sus scores
        sean comparables entre sí (uno es coseno, el otro BM25 sin normalizar).
        """
        fusion: dict[str, float] = {}
        for ranking in rankings:
            for puesto, cid in enumerate(ranking):
                fusion[cid] = fusion.get(cid, 0.0) + 1.0 / (k + puesto + 1)
        return fusion

    @staticmethod
    def _coseno(a: list[float], b: list[float]) -> float:
        num = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return (num / (na * nb)) if na and nb else 0.0

    @staticmethod
    def _completar_similitudes(collection: Any, query: str, ids: list[str],
                               similitudes: dict[str, float]) -> None:
        """
        Calcula el coseno de los candidatos que entraron solo por BM25.

        Sin esto, un acierto léxico salía con `score` 0.0 —el valor por
        defecto de un id que no estaba en la respuesta vectorial— y cualquier
        `min_score` lo borraba. Es decir: el filtro de calidad se comía
        exactamente los resultados que el híbrido existe para rescatar, y el
        número que se imprimía al lado no ordenaba nada. Se resuelve pidiendo
        a chroma los vectores de esos ids concretos (`get` por id, no un
        barrido) y comparándolos con el de la consulta.
        """
        faltan = [cid for cid in ids if cid not in similitudes]
        if not faltan:
            return
        try:
            funcion = _embedding_function(_embedder_id())
            if funcion is None:
                from chromadb.utils import embedding_functions

                funcion = embedding_functions.DefaultEmbeddingFunction()
            vector_consulta = list(funcion([query])[0])
            datos = collection.get(ids=faltan, include=["embeddings"])
            for cid, vector in zip(datos["ids"], datos["embeddings"] or [], strict=False):
                similitudes[cid] = round(
                    RagTool._coseno(vector_consulta, list(vector)), 4
                )
        except Exception:  # noqa: BLE001 — sin coseno se ordena igual por RRF
            return

    @staticmethod
    def _expandir(collection: Any, resultados: list[dict[str, Any]], vecinos: int) -> None:
        """
        Añade a cada resultado el texto de los chunks contiguos de su fichero.

        El techo de `_MAX_CHUNK_CHARS` es el límite del embedder: manda sobre
        lo que se vectoriza, no sobre lo que se devuelve. Recuperar por trozo
        pequeño y responder con su vecindario da el contexto que le falta a un
        fragmento suelto, sin tocar el índice.
        """
        por_fuente: dict[str, list[dict]] = {}
        for r in resultados:
            por_fuente.setdefault(r["source"], [])
        for fuente in por_fuente:
            try:
                datos = collection.get(where={"source": fuente}, include=["documents", "metadatas"])
            except Exception:  # noqa: BLE001
                continue
            trozos = [
                {"id": cid, "text": doc, "line": (meta or {}).get("line", 0)}
                for cid, doc, meta in zip(
                    datos["ids"], datos["documents"] or [], datos["metadatas"] or [], strict=False
                )
            ]
            por_fuente[fuente] = sorted(trozos, key=lambda t: t["line"])

        for r in resultados:
            trozos = por_fuente.get(r["source"]) or []
            posicion = next((i for i, t in enumerate(trozos) if t["id"] == r["id"]), None)
            if posicion is None:
                r["context"] = r["text"]
                continue
            desde = max(0, posicion - vecinos)
            hasta = min(len(trozos), posicion + vecinos + 1)
            r["context"] = "\n\n".join(t["text"] for t in trozos[desde:hasta])
            r["context_lines"] = [trozos[desde]["line"], trozos[hasta - 1]["line"]]

    @staticmethod
    def search(root: Path, query: str, top_k: int = 10, hybrid: bool = True,
               min_score: float = 0.0, *, file_type: str | None = None,
               source: str | None = None, max_per_source: int = 0,
               expand: int = 0) -> list[dict[str, Any]]:
        """
        Búsqueda híbrida (vectorial + BM25 léxico fundidos con RRF).

        Parámetros
        ----------
        hybrid : bool
            `False` deja solo el vector.
        min_score : float
            Umbral de **similitud coseno**. Con el embedder por defecto, por
            debajo de ~0.35 el resultado casi nunca tiene que ver con la
            pregunta. Se aplica antes de cortar por `top_k`, no después: si
            no, filtrar devolvía menos resultados de los pedidos habiendo
            candidatos válidos esperando.
        file_type : str | None
            `code | doc | prompt | vault | harness | url`. Se filtra en chroma
            (`where`), así que no consume presupuesto de resultados.
        source : str | None
            Prefijo de ruta (`harness/progress/`, `agents/tools/`). Chroma no tiene
            operador de prefijo, así que este se aplica después de recuperar
            —por eso se pide de más antes de filtrar.
        max_per_source : int
            Tope de fragmentos por fichero. 0 = sin tope. Evita que un `top_k`
            se lo coma entero un solo módulo largo.
        expand : int
            Nº de chunks vecinos a cada lado que se devuelven en `context`.

        En cada resultado, `score` es lo que **ordena** (RRF si es híbrido,
        coseno si no) y `similarity` es el coseno, o `None` si no se pudo
        calcular.
        """
        collection, aviso = RagTool._abrir(root)
        if collection is None or aviso:
            return [{"error": aviso}]
        if collection.count() == 0:
            return []

        # Se pide de más porque los filtros de `source` y `max_per_source` se
        # aplican después de recuperar: pedir justo `top_k` haría que un filtro
        # devolviera media lista.
        pedidos = min(collection.count(), max(top_k * 3, 10))
        consulta_kwargs: dict[str, Any] = {"query_texts": [query], "n_results": pedidos}
        if file_type:
            consulta_kwargs["where"] = {"file_type": file_type}
        vectorial = collection.query(**consulta_kwargs)

        similitudes: dict[str, float] = {}
        textos: dict[str, str] = {}
        metadatos: dict[str, dict] = {}
        metas_vec = vectorial.get("metadatas") or [[]]
        for i, cid in enumerate(vectorial["ids"][0]):
            similitudes[cid] = round(1 - vectorial["distances"][0][i], 4)
            textos[cid] = vectorial["documents"][0][i]
            metadatos[cid] = (metas_vec[0][i] if metas_vec[0] else {}) or {}

        def _pasa(cid: str) -> bool:
            meta = metadatos.get(cid) or {}
            if file_type and meta.get("file_type") != file_type:
                return False
            return not (source and not str(meta.get("source", "")).startswith(source))

        orden_vectorial = [cid for cid in vectorial["ids"][0] if _pasa(cid)]
        en_lexico: set[str] = set()

        if not hybrid:
            candidatos = orden_vectorial
            fusion: dict[str, float] = {}
        else:
            indice = RagTool._indice_lexico(root, collection)
            marcadores = indice["bm25"].puntua(_tokenizar(query))
            mejores = sorted(
                (i for i, s in enumerate(marcadores) if s > 0),
                key=lambda i: marcadores[i],
                reverse=True,
            )[:pedidos]
            ids_lexicos = [indice["ids"][i] for i in mejores]

            desconocidos = [cid for cid in ids_lexicos if cid not in metadatos]
            if desconocidos:
                datos = collection.get(ids=desconocidos, include=["documents", "metadatas"])
                for cid, doc, meta in zip(
                    datos["ids"], datos["documents"] or [], datos["metadatas"] or [], strict=False
                ):
                    textos.setdefault(cid, doc)
                    metadatos.setdefault(cid, meta or {})

            orden_lexico = [cid for cid in ids_lexicos if _pasa(cid)]
            en_lexico = set(orden_lexico)
            fusion = RagTool._rrf([orden_vectorial, orden_lexico])
            candidatos = sorted(fusion, key=lambda c: fusion[c], reverse=True)

        candidatos = candidatos[: max(top_k * 3, top_k)]
        RagTool._completar_similitudes(collection, query, candidatos, similitudes)

        salida: list[dict[str, Any]] = []
        por_fuente: Counter = Counter()
        en_vector = set(orden_vectorial)
        for cid in candidatos:
            similitud = similitudes.get(cid)
            if min_score and (similitud is None or similitud < min_score):
                continue
            meta = metadatos.get(cid, {})
            fuente = meta.get("source", "?")
            if max_per_source and por_fuente[fuente] >= max_per_source:
                continue
            por_fuente[fuente] += 1
            salida.append({
                "id": cid,
                "text": textos.get(cid, "")[:500],
                "source": fuente,
                "line": meta.get("line", 0),
                "file_type": meta.get("file_type", "?"),
                "section_type": meta.get("section_type", "?"),
                "score": round(fusion[cid], 5) if fusion else (similitud or 0.0),
                "similarity": similitud,
                # `_confianza(fuente)` como respaldo: un índice construido
                # antes de que existiera este campo no lo trae en sus
                # metadatos, y ahí adivinarlo por el origen es exacto.
                "trust": meta.get("trust") or _confianza(fuente),
                "injection_flag": bool(meta.get("injection_flag", False)),
                "match": (
                    "ambos" if cid in en_vector and cid in en_lexico
                    else "lexico" if cid in en_lexico else "vector"
                ),
            })
            if len(salida) >= top_k:
                break

        if expand > 0 and salida:
            RagTool._expandir(collection, salida, expand)
        return salida

    @staticmethod
    def status(root: Path) -> dict[str, Any]:
        """
        Estado del índice, incluido si está al día.

        Buscar sobre un índice viejo no da error: da la respuesta de ayer, en
        silencio. Y como `make index-rag` es manual, ese silencio dura hasta
        que alguien se acuerda. Comparando las huellas de los ficheros en
        disco con las que quedaron grabadas en los metadatos, el desfase se
        puede contar y avisar.
        """
        collection, aviso = RagTool._abrir(root)
        if collection is None:
            return {"available": False, "mismatch": aviso}
        grabado = (collection.metadata or {}).get("embedder", "onnx")

        huella_por_origen: dict[str, str] = {}
        if collection.count():
            for meta in collection.get(include=["metadatas"])["metadatas"] or []:
                origen = (meta or {}).get("source", "?")
                huella_por_origen[origen] = (meta or {}).get("file_hash", "")

        vivos = {origen: RagTool._huella(texto) for origen, texto, _ in RagTool._documentos(root)}
        desfasados = sorted(
            o for o, h in vivos.items() if o in huella_por_origen and huella_por_origen[o] != h
        )
        nuevos = sorted(o for o in vivos if o not in huella_por_origen)
        # Las URLs no vienen de disco: no estar en `vivos` es su estado normal.
        borrados = sorted(
            o for o in huella_por_origen if o not in vivos and not o.startswith("url:")
        )

        return {
            "available": True,
            "total_chunks": collection.count(),
            "collection": COLLECTION_NAME,
            "sources": len(huella_por_origen),
            "embedder": grabado,
            "embedder_desc": _EMBEDDERS.get(grabado, grabado),
            "mismatch": aviso,
            "up_to_date": not (desfasados or nuevos or borrados),
            "stale_files": desfasados,
            "new_files": nuevos,
            "deleted_files": borrados,
        }
