# detector-ia

**English** · [Español](README.es.md)

A local tool that gives a **rough estimate** of whether a text was written by an LLM. It runs on your own machine with open models from Hugging Face. No text leaves your computer.

It runs pairs of small language models (a base model plus its instruct version) over the text and combines three published statistical signals into one "suspicion" score from 0 to 100 %. You get the score for the whole document and for each passage, plus an HTML report.

> ⚠️ **This is not proof of authorship.** It is a statistical hint. Very formal or polished human writing can score high, and edited AI text can score low. Until you [calibrate](#calibration-strongly-recommended) it, the thresholds are generic.

An example report is in [`ejemplos/informe_ia.html`](ejemplos/informe_ia.html). Download it and open it in a browser.

## How it works

Each **model pair** shares one tokenizer: the base model is the "observer" and the instruct model is the "performer". For every pair, the script computes:

| Signal | Paper | Meaning | More "AI" when… |
|---|---|---|---|
| **Binoculars** | Hans et al., 2024 | The instruct model's perplexity divided by the cross-entropy between the two models | lower |
| **Fast-DetectGPT** (analytic) | Bao et al., 2024 | How much more probable the chosen words are than words sampled from the model would be. Normalised to ~100 tokens. | higher |
| **GLTR top-10** | Gehrmann et al., 2019 | Share of tokens that were among the base model's 10 most likely predictions | higher |

Each combination of model and metric is one **signal**. Each signal becomes a suspicion value between 0 and 1. The final score is their weighted average. Without calibration, every signal has the same weight. With calibration, each signal is weighted by how well it separated *your* texts (its AUC), and signals with an AUC below 0.55 are left out.

The text is split into passages of about 50 words. Passage boundaries follow sentences and paragraphs. Each passage gets its own score.

## Installation

You need Python 3.9 or newer. An NVIDIA GPU with CUDA is strongly recommended. The script also runs on the CPU, but slowly.

```bash
git clone https://github.com/amt911/detector-ia.git
cd detector-ia
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`bitsandbytes` is only needed to load models in 4 or 8 bits on a GPU. On the CPU, models are always loaded without quantization. `pypdf` is only needed to read PDF files. `weasyprint` is only needed for `--pdf`; it also needs Pango installed on the system, which most Linux desktops already have.

## Supported files

| Format | What is analysed |
|---|---|
| `.txt`, `.md` | The whole file, read as UTF-8 |
| `.pdf` | The text layer of every page. Words split with a hyphen at the end of a line are joined back together. Scanned PDFs have no text layer and are rejected, because there is no OCR. |
| `.ipynb` | Only the markdown cells. Code and outputs are skipped because they are not prose. |
| `.docx` | The text of every paragraph. Each Word paragraph becomes one paragraph for splitting. |

Any other extension is read as plain text. For PDFs, check that headers, footers and page numbers do not take up a large part of the text, because they can skew the result.

## Usage

The commands and options are in Spanish: `analizar` means analyze, `calibrar` means calibrate, and `modelos` means models.

```bash
python detector_ia.py modelos                       # list the available model presets
python detector_ia.py analizar my_text.txt          # writes informes/informe_my_text.html
python detector_ia.py analizar a.txt b.txt --json   # several files, plus the raw data as JSON
python detector_ia.py analizar my_text.txt --pdf     # also writes informes/informe_my_text.pdf
python detector_ia.py analizar essay.pdf notes.ipynb report.docx
```

By default the script uses three pairs from different model families. It loads them one at a time, so they fit in limited VRAM:

- `qwen2.5-1.5b`: Qwen2.5-1.5B and Qwen2.5-1.5B-Instruct
- `qwen3-1.7b`: Qwen3-1.7B-Base and Qwen3-1.7B
- `salamandra-2b`: BSC-LT Salamandra 2B, from the Barcelona Supercomputing Center. It was trained on a lot of Spanish.

To choose other pairs, pass `--modelos`, for example `--modelos qwen2.5-1.5b,salamandra-2b,qwen2.5-7b`. You can also pass your own pair as `"org/base+org/instruct"`; both models must use the same tokenizer. Models of 3B or more are loaded in 4 bits automatically.

The first run downloads every model it uses from Hugging Face. With the default pairs, that is roughly 20 GB.

The `llama3.2-3b` and `gemma2-2b` presets are gated. To use them, accept their license on Hugging Face and run `hf auth login` first.

### Options

| Option | Commands | Default | Description |
|---|---|---|---|
| `--modelos` | analizar, calibrar | `qwen2.5-1.5b,qwen3-1.7b,salamandra-2b` | Comma-separated list of model presets or custom pairs |
| `--quant` | analizar, calibrar | `auto` | `auto`, `none`, `8bit` or `4bit`. `auto` loads models of 3B or more in 4 bits. |
| `--min-palabras` | analizar, calibrar | `50` | Minimum size of each passage, in words |
| `--ventana` | analizar, calibrar | `384` | Tokens per forward pass. Lower it to 256 if you run out of VRAM. |
| `--calibracion` | analizar | — | Calibration JSON produced by `calibrar` |
| `--salida` | analizar / calibrar | `informes/` / `calibracion.json` | Output folder for reports, or output file for the calibration |
| `--top` | analizar | `6` | Number of suspicious passages to highlight |
| `--json` | analizar | off | Also save the scores as JSON next to the report |
| `--pdf` | analizar | off | Also save the report as PDF (A4), next to the HTML. Needs `weasyprint`. |

## What the report shows

- An **overall suspicion score**, and how many of the signals point to AI.
- A **table per model** with the three metrics and their thresholds.
- The **most suspicious passages**. Each one is quoted, with the reason in plain language, for example "88 % of tokens were among the 10 most likely".
- The **full text, coloured by passage**, from green (few AI signals) to red (many).
- The **full text coloured word by word, GLTR-style**, with a selector to pick the model.

The report itself is written in Spanish. In the PDF, the word-by-word view is shown for every model, one after another, because the model selector only works in the HTML. With long texts, the PDF can run to dozens of pages and take a while to generate.

## Calibration (strongly recommended)

Without calibration, the thresholds are generic, and the report shows a red warning. Calibration changes the result a lot. To calibrate:

1. Put 20 or more texts that **you** wrote, ideally before 2023, in `humanos/`. Any [supported format](#supported-files) works. The minimum is 5.
2. Put 20 or more texts generated by ChatGPT, Claude, Gemini and similar tools, on similar topics, in `ia/`.
3. Run:

```bash
python detector_ia.py calibrar --humanos humanos/ --ia ia/
python detector_ia.py analizar my_text.txt --calibracion calibracion.json
```

Calibration prints the **AUC of every model and metric on your own texts**, so you can see which signals work in your case. It fits a threshold per signal, at both document and passage level. Signals with an AUC below 0.55 no longer count, and the signals that separate your texts better get more weight.

Use the same `--modelos` and `--quant` for analysis that you used for calibration. The report warns you when they differ.

## Limitations

- These techniques have been evaluated mostly on English. They work worse on Spanish and vary a lot with the type of text.
- Paraphrasing or light editing fools them easily.
- They penalise very formal or technical writing, text built from set phrases, and well-known texts that models have memorised, such as quotes or laws.
- Below about 200 words, reliability drops sharply.
- The script has only been tested end to end with the small `qwen2.5-0.5b` pair, on a CPU, with two sample texts. It has not been tested with the default models or on a GPU yet.
