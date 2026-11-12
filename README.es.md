# detector-ia

[English](README.md) · **Español**

Herramienta local que da una **estimación orientativa** de si un texto lo ha escrito un LLM. Funciona en tu propio equipo con modelos abiertos de Hugging Face, así que ningún texto sale de tu ordenador.

Pasa el texto por pares de modelos de lenguaje pequeños (un modelo base y su versión instruct) y combina tres señales estadísticas publicadas en una «sospecha» del 0 al 100 %. Te da la sospecha del documento entero y la de cada fragmento, y además un informe HTML.

> ⚠️ **No es una prueba de autoría.** Es una pista estadística. Un texto humano muy formal o pulido puede puntuar alto, y un texto de IA editado puede puntuar bajo. Mientras no lo [calibres](#calibración-muy-recomendable), los umbrales son genéricos.

Tienes un informe de ejemplo en [`ejemplos/informe_ia.html`](ejemplos/informe_ia.html). Descárgalo y ábrelo en el navegador.

## Cómo funciona

Los dos modelos de cada **par** comparten tokenizador: el base hace de «observer» y el instruct de «performer». Con cada par, el script calcula:

| Señal | Artículo | Qué mide | Más «IA» cuanto… |
|---|---|---|---|
| **Binoculars** | Hans et al., 2024 | La perplejidad del modelo instruct dividida entre la entropía cruzada de los dos modelos | más bajo |
| **Fast-DetectGPT** (analítico) | Bao et al., 2024 | Cuánto más probables son las palabras elegidas que las que saldrían muestreando del modelo. Normalizado a unos 100 tokens. | más alto |
| **GLTR top-10** | Gehrmann et al., 2019 | El porcentaje de tokens que estaban entre las 10 predicciones más probables del modelo base | más alto |

Cada combinación de modelo y métrica es una **señal**. Cada señal se convierte en una sospecha entre 0 y 1, y el resultado final es su media ponderada. Sin calibrar, todas las señales pesan lo mismo. Con calibración, cada señal pesa según lo bien que separó *tus* textos (su AUC), y las que tienen un AUC por debajo de 0,55 quedan fuera.

El texto se divide en fragmentos de unas 50 palabras. Los cortes respetan las frases y los párrafos, y cada fragmento tiene su propia puntuación.

## Instalación

Necesitas Python 3.9 o superior. Se recomienda mucho una GPU NVIDIA con CUDA. Sin GPU también funciona, pero va lento.

```bash
git clone https://github.com/amt911/detector-ia.git
cd detector-ia
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`bitsandbytes` solo hace falta para cargar los modelos en 4 u 8 bits en la GPU. En la CPU los modelos se cargan siempre sin cuantizar. `pypdf` solo hace falta para leer PDF cuando no está instalado `pdftotext` (Poppler); `pdftotext` separa mejor las palabras, así que es preferible. `weasyprint` solo hace falta para `--pdf`; además necesita Pango instalado en el sistema, que la mayoría de escritorios Linux ya traen.

## Archivos admitidos

| Formato | Qué se analiza |
|---|---|
| `.txt` | El archivo entero, leído como UTF-8 |
| `.md` | El archivo entero, sin la sintaxis de markdown (ver abajo) |
| `.pdf` | La capa de texto de cada página, leída con `pdftotext` (Poppler) si está instalado, o con `pypdf` si no. Las letras acentuadas que los PDF de LaTeX dejan partidas (`Adem´ as`, `a˜ nadir`) se recomponen, y las palabras cortadas con guion al final de línea se vuelven a unir. Los PDF escaneados no tienen capa de texto y se rechazan, porque no hay OCR. |
| `.ipynb` | Solo las celdas markdown, sin la sintaxis de markdown. El código y las salidas se ignoran porque no son prosa. |
| `.docx` | El texto de cada párrafo. Cada párrafo de Word cuenta como un párrafo al dividir en fragmentos. |

En los `.md` y en los notebooks, antes de analizar se quita la sintaxis de markdown: las marcas de título, la negrita y la cursiva, las viñetas, los enlaces (se queda su texto), las imágenes, las etiquetas HTML, las tablas (se queda el texto de las celdas), los bloques de código y las fórmulas LaTeX. Esas marcas son muy predecibles para los modelos e inflan las métricas, sobre todo GLTR top-10. El texto que sale en el informe es el ya limpio.

Cualquier otra extensión se lee como texto plano. En los PDF, comprueba que los encabezados, los pies y los números de página no ocupen gran parte del texto, porque pueden falsear el resultado.

## Uso

```bash
python detector_ia.py modelos                         # lista los modelos disponibles
python detector_ia.py analizar mi_texto.txt           # crea informes/informe_mi_texto.html
python detector_ia.py analizar a.txt b.txt --json     # varios archivos y, además, los datos en JSON
python detector_ia.py analizar mi_texto.txt --pdf      # crea además informes/informe_mi_texto.pdf
python detector_ia.py analizar trabajo.pdf apuntes.ipynb memoria.docx
```

Por defecto usa tres pares de familias distintas. Los carga uno detrás de otro para no pasarse de VRAM:

- `qwen2.5-1.5b`: Qwen2.5-1.5B y Qwen2.5-1.5B-Instruct.
- `qwen3-1.7b`: Qwen3-1.7B-Base y Qwen3-1.7B.
- `salamandra-2b`: Salamandra 2B, del BSC-LT (Barcelona Supercomputing Center). Se entrenó con mucho español.

Para usar otros pares, pásalos con `--modelos`, por ejemplo `--modelos qwen2.5-1.5b,salamandra-2b,qwen2.5-7b`. También puedes indicar un par propio con `"org/base+org/instruct"`; los dos modelos tienen que usar el mismo tokenizador. Los modelos de 3B o más se cargan automáticamente en 4 bits.

La primera vez se descargan de Hugging Face todos los modelos que uses. Con los de por defecto son unos 20 GB.

`llama3.2-3b` y `gemma2-2b` son modelos con acceso restringido. Antes de usarlos, acepta su licencia en Hugging Face y ejecuta `hf auth login`.

### Opciones

| Opción | Comandos | Por defecto | Descripción |
|---|---|---|---|
| `--modelos` | analizar, calibrar | `qwen2.5-1.5b,qwen3-1.7b,salamandra-2b` | Lista de modelos o pares propios, separados por comas |
| `--quant` | analizar, calibrar | `auto` | `auto`, `none`, `8bit` o `4bit`. Con `auto`, los modelos de 3B o más se cargan en 4 bits. |
| `--min-palabras` | analizar, calibrar | `50` | Tamaño mínimo de cada fragmento, en palabras |
| `--ventana` | analizar, calibrar | `384` | Tokens por pasada. Bájalo a 256 si te quedas sin VRAM. |
| `--max-ram` | analizar, calibrar | 40 % de la RAM disponible | GiB de RAM para los pesos que no quepan en la GPU. Lo que sobre va a disco. |
| `--calibracion` | analizar | — | El JSON que genera `calibrar` |
| `--salida` | analizar / calibrar | `informes/` / `calibracion.json` | Carpeta de los informes o archivo de la calibración |
| `--top` | analizar | `6` | Número de fragmentos sospechosos que se destacan |
| `--json` | analizar | desactivado | Guarda también las puntuaciones en un JSON junto al informe |
| `--pdf` | analizar | desactivado | Guarda también el informe en PDF (A4), junto al HTML. Necesita `weasyprint`. |

## Uso de GPU y RAM

Los dos modelos de un par tienen que estar cargados a la vez, porque cada métrica compara sus predicciones token a token. Los pares se cargan uno detrás de otro, y cada par se libera antes de cargar el siguiente.

Los modelos sin cuantizar (lo normal en los de menos de 3B) se colocan solos:

1. En la GPU va todo lo que quepa. Se deja libre el espacio que necesita el propio cálculo. El modelo base puede quedarse como mucho la mitad de la VRAM libre, para que al instruct también le toque una parte. Un modelo sin ninguna parte en la GPU se calcularía en la CPU, que es muy lento; si pasa, la consola te avisa.
2. Lo que no quepa va a la RAM en bf16, hasta `--max-ram` GiB. Por defecto es el 40 % de la RAM disponible al empezar.
3. Lo que pase de ahí va a una carpeta temporal en disco, que se borra al terminar.

El cálculo se sigue haciendo en la GPU: cada capa se copia a ella cuando le toca. Va más lento, pero el resultado es el mismo, así que una calibración sigue valiendo. En la consola verás cuánto de cada modelo ha ido a RAM o a disco.

Los modelos cuantizados (`--quant 4bit/8bit`, o los de 3B o más con `auto`) van siempre enteros en la GPU. Si alguno no cabe, usa un modelo más pequeño o libera VRAM.

## Qué trae el informe

- Un **porcentaje global de sospecha**, con cuántas de las señales apuntan a IA.
- Una **tabla por modelo** con las tres métricas y sus umbrales.
- Los **fragmentos más sospechosos**, cada uno citado y con el motivo en lenguaje normal. Por ejemplo: «el 88 % de los tokens estaban entre los 10 más probables».
- El **texto completo coloreado por fragmentos**, de verde (pocas señales de IA) a rojo (muchas).
- El **texto completo coloreado palabra a palabra, al estilo GLTR**, con un selector para elegir el modelo.

En el PDF, la vista palabra a palabra sale para todos los modelos, uno detrás de otro, porque el selector solo funciona en el HTML. Con textos largos, el PDF puede tener decenas de páginas y tardar un rato en generarse.

## Calibración (muy recomendable)

Sin calibrar, los umbrales son genéricos y el informe lo avisa en rojo. Calibrar cambia mucho el resultado. Para calibrar:

1. Mete en `humanos/` 20 o más textos escritos por **ti**, mejor de antes de 2023, en cualquiera de los [formatos admitidos](#archivos-admitidos). El mínimo son 5.
2. Mete en `ia/` otros 20 o más generados con ChatGPT, Claude, Gemini o similares, sobre temas parecidos.
3. Ejecuta:

```bash
python detector_ia.py calibrar --humanos humanos/ --ia ia/
python detector_ia.py analizar mi_texto.txt --calibracion calibracion.json
```

La calibración te enseña el **AUC de cada modelo y métrica con tus propios textos**, así sabrás qué señales funcionan de verdad en tu caso. Ajusta un umbral para cada señal, tanto para el documento como para los fragmentos. Las señales con un AUC por debajo de 0,55 dejan de contar, y las que separan mejor tus textos pesan más.

Analiza con los mismos `--modelos` y `--quant` con los que calibraste. Si no coinciden, el informe te avisa.

## Limitaciones

- Estas técnicas se han evaluado sobre todo en inglés. En español rinden peor y varían mucho según el tipo de texto.
- Una paráfrasis o una edición ligera las engaña con facilidad.
- Penalizan la escritura muy formal o técnica, los textos hechos con fórmulas fijas y los textos muy conocidos que los modelos han memorizado, como citas o leyes.
- Con menos de unas 200 palabras, la fiabilidad cae mucho.
- Probado de principio a fin con los modelos por defecto en una GPU de 12 GB (RTX 3080 Ti) compartida con otros programas, con un notebook en español de unos 17.000 tokens y parte de salamandra-2b descargada a RAM. También se ha ejecutado con el par pequeño `qwen2.5-0.5b` en CPU. La carga en 4/8 bits y `calibrar` todavía no se han probado con datos reales.

## Licencia

[MIT](LICENSE).
