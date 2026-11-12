#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detector_ia.py — Detector local y ORIENTATIVO de texto generado por IA.

Para cada par de modelos (base + instruct, mismo tokenizador) calcula tres señales:

  · Binoculars (Hans et al., 2024): perplejidad del modelo instruct dividida por la
    entropía cruzada entre ambos modelos. Más bajo = más «IA».
  · Fast-DetectGPT analítico (Bao et al., 2024): cuánto más probables son las palabras
    elegidas que lo que saldría muestreando del modelo. Normalizado a ~100 tokens.
    Más alto = más «IA».
  · GLTR (Gehrmann et al., 2019): % de tokens que estaban en el top-10 del modelo base.
    Más alto = más «IA».

Combina todas las señales de todos los modelos en una «sospecha» 0-100 % para el
documento y para cada fragmento, y genera un informe HTML.

IMPORTANTE: sin calibrar, los umbrales son genéricos y el resultado es muy orientativo.
Calibra con textos tuyos (humanos) y textos de IA:  ver `calibrar` abajo.

Uso:
  python detector_ia.py modelos
  python detector_ia.py analizar texto.txt [otro.txt ...] [--modelos a,b,c] [--calibracion calib.json]
  python detector_ia.py calibrar --humanos carpeta_humanos/ --ia carpeta_ia/ [--modelos a,b,c]

Requisitos: pip install torch transformers accelerate   (+ bitsandbytes para 4/8 bits, + pypdf para PDF)
Formatos: .txt, .md, .pdf, .ipynb (solo celdas markdown) y .docx
"""

import argparse
import bisect
import datetime as dt
import gc
import html
import json
import math
import re
import statistics
import sys
from pathlib import Path

# ----------------------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------------------
# clave: (modelo base = "observer", modelo instruct = "performer", tamaño en miles de millones, nota)
PRESETS = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", 0.5, "Muy rápido, para pruebas"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", 1.5, "Rápido, buen español"),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", 3.1, "4 bits por defecto"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", 7.6, "4 bits obligatorio con 12 GB"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B", 1.7, "Familia más reciente de Qwen"),
    "qwen3-4b": ("Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B", 4.0, "4 bits por defecto"),
    "salamandra-2b": ("BSC-LT/salamandra-2b", "BSC-LT/salamandra-2b-instruct", 2.3,
                      "BSC (Barcelona); entrenado con mucho español"),
    "salamandra-7b": ("BSC-LT/salamandra-7b", "BSC-LT/salamandra-7b-instruct", 7.8, "4 bits por defecto"),
    "llama3.2-3b": ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", 3.2,
                    "Requiere aceptar la licencia en Hugging Face y `hf auth login`"),
    "gemma2-2b": ("google/gemma-2-2b", "google/gemma-2-2b-it", 2.6,
                  "Requiere aceptar la licencia en Hugging Face y `hf auth login`"),
}
DEFAULT_MODELOS = "qwen2.5-1.5b,qwen3-1.7b,salamandra-2b"

MIN_TOKENS_FRAGMENTO = 20
TOKENS_DOC_CORTO = 150

# ----------------------------------------------------------------------------------------
# Métricas y umbrales por defecto (genéricos, NO calibrados)
# ----------------------------------------------------------------------------------------
# dir = +1: valores más altos = más IA ; dir = -1: valores más bajos = más IA
METRICAS = {
    "binoculars": {"nombre": "Binoculars", "t": 0.90, "s": 0.05, "dir": -1},
    "fastdetect": {"nombre": "Fast-DetectGPT", "t": 1.0, "s": 1.0, "dir": 1},
    "top10": {"nombre": "GLTR top-10", "t": 0.78, "s": 0.06, "dir": 1},
}


def fmt_valor(met, v):
    if met == "top10":
        return f"{v:.0%}"
    if met == "binoculars":
        return f"{v:.3f}"
    return f"{v:.2f}"


# ----------------------------------------------------------------------------------------
# Segmentación en fragmentos
# ----------------------------------------------------------------------------------------
_PARRAFO = re.compile(r"(?:(?!\n[ \t]*\n).)+", re.S)
_FRASE = re.compile(r"\S.*?(?:[.!?…]+[»\"”’)\]]*(?=\s)|\Z)", re.S)


def _palabras(s):
    return len(re.findall(r"\w+", s))


def segmentar(texto, min_palabras=50):
    """Devuelve una lista de (inicio, fin) en caracteres. Agrupa frases hasta ~min_palabras."""
    frases = []
    for m in _PARRAFO.finditer(texto):
        g = m.group()
        if not g.strip():
            continue
        base = m.start()
        for f in _FRASE.finditer(g):
            frases.append((base + f.start(), base + f.end(), _palabras(f.group())))
        frases.append(None)  # marca de fin de párrafo

    trozos, ini, fin, n = [], None, None, 0
    for fr in frases:
        if fr is None:  # fin de párrafo: cierra si el trozo pendiente tiene entidad
            if ini is not None and n >= min_palabras // 2:
                trozos.append((ini, fin))
                ini, n = None, 0
            continue
        s, e, w = fr
        if ini is None:
            ini = s
        fin, n = e, n + w
        if n >= min_palabras:
            trozos.append((ini, fin))
            ini, n = None, 0
    if ini is not None:
        if trozos and n < min_palabras // 2:
            trozos[-1] = (trozos[-1][0], fin)
        else:
            trozos.append((ini, fin))
    return trozos


# ----------------------------------------------------------------------------------------
# Carga de modelos y cálculo por token
# ----------------------------------------------------------------------------------------
def resolver_modelo(clave):
    if clave in PRESETS:
        return PRESETS[clave]
    if "+" in clave:  # par personalizado: "org/base+org/instruct"
        b, i = clave.split("+", 1)
        return (b.strip(), i.strip(), None, "personalizado")
    sys.exit(f"Modelo desconocido: «{clave}». Lista: python detector_ia.py modelos  "
             f"(o usa un par personalizado «org/base+org/instruct»)")


def cargar_par(clave, quant, device):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base, inst, tam, _ = resolver_modelo(clave)
    q = quant
    if q == "auto":
        q = "4bit" if (tam or 0) >= 3 else "none"
    if device == "cpu" and q != "none":
        print(f"  ! Sin GPU no se puede cuantizar con bitsandbytes; cargo {clave} sin cuantizar.")
        q = "none"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # transformers ≥ 4.56 usa `dtype`; versiones anteriores, `torch_dtype`
    ver = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
    kw = {("dtype" if ver >= (4, 56) else "torch_dtype"): dtype, "low_cpu_mem_usage": True}
    if q in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
            if q == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
    if device == "cuda":
        kw["device_map"] = {"": 0}

    print(f"→ Cargando {clave}: {base} + {inst} (cuantización: {q})")
    tok = AutoTokenizer.from_pretrained(base)
    if not tok.is_fast:
        sys.exit(f"El tokenizador de {base} no es «fast» (necesito offsets de caracteres).")
    obs = AutoModelForCausalLM.from_pretrained(base, **kw).eval()
    perf = AutoModelForCausalLM.from_pretrained(inst, **kw).eval()
    return obs, perf, tok, {"clave": clave, "base": base, "instruct": inst, "quant": q}


def stats_por_token(obs, perf, ids, device, ventana=384, contexto=128):
    """Para cada token j ≥ 1 calcula, con el contexto anterior:
       ll (log-prob en el modelo base), rank (posición en el base), nll_perf (−log-prob en instruct),
       xent (entropía cruzada base→instruct), mu/var (media/varianza analíticas de Fast-DetectGPT)."""
    import torch

    n = len(ids)
    claves = ("ll", "rank", "nll_perf", "xent", "mu", "var")
    st = {k: torch.zeros(n, dtype=torch.float32) for k in claves}
    ids_t = torch.tensor(ids, dtype=torch.long)
    pos = 1
    while pos < n:
        fin = min(n, pos + ventana)
        ini = max(0, pos - contexto)
        inp = ids_t[ini:fin].unsqueeze(0).to(device)
        with torch.inference_mode():
            lo_all = obs(input_ids=inp).logits[0]
            lp_all = perf(input_ids=inp).logits[0]
            V = min(lo_all.shape[-1], lp_all.shape[-1])
            a, b = pos - ini - 1, fin - ini - 1  # logits en a..b-1 predicen tokens pos..fin-1
            lo = torch.log_softmax(lo_all[a:b, :V].float(), -1)
            lp = torch.log_softmax(lp_all[a:b, :V].float(), -1)
            del lo_all, lp_all
            tgt = inp[0, a + 1:b + 1].unsqueeze(-1)
            ll = lo.gather(-1, tgt).squeeze(-1)
            rank = (lo > ll.unsqueeze(-1)).sum(-1) + 1
            nll_perf = -lp.gather(-1, tgt).squeeze(-1)
            xent = -(lo.exp() * lp).sum(-1)                 # Binoculars: H(p_base, q_instruct)
            pp = lp.exp()                                    # Fast-DetectGPT: muestreo = instruct,
            mu = (pp * lo).sum(-1)                           #                 puntuación = base
            var = ((pp * lo.square()).sum(-1) - mu.square()).clamp_min(0)
            for k, v in zip(claves, (ll, rank.float(), nll_perf, xent, mu, var)):
                st[k][pos:fin] = v.float().cpu()
            del lo, lp, pp
        pos = fin
    return st


def agregar(st, idx):
    import torch

    if len(idx) < MIN_TOKENS_FRAGMENTO:
        return None
    it = torch.tensor(idx, dtype=torch.long)
    n = len(idx)
    ll, mu, var = st["ll"][it], st["mu"][it], st["var"][it]
    bino = (st["nll_perf"][it].mean() / st["xent"][it].mean().clamp_min(1e-6)).item()
    fdg = ((ll.sum() - mu.sum()) / var.sum().clamp_min(1e-6).sqrt()).item()
    fdg_n = fdg / math.sqrt(n / 100)  # normalizado a ~100 tokens para comparar longitudes
    top10 = (st["rank"][it] <= 10).float().mean().item()
    return {"binoculars": bino, "fastdetect": fdg_n, "top10": top10, "n_tokens": n}


def analizar_textos(textos, claves, args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("! No se detecta GPU CUDA: funcionará, pero lento.")
    res = [{"segmentos": segmentar(t, args.min_palabras), "modelos": {}} for t in textos]

    for clave in claves:
        obs, perf, tok, info = cargar_par(clave, args.quant, device)
        for i, texto in enumerate(textos):
            enc = tok(texto, return_offsets_mapping=True, add_special_tokens=True)
            ids = list(enc["input_ids"])
            offs = [tuple(o) for o in enc["offset_mapping"]]
            if tok.bos_token_id is not None and (not ids or ids[0] != tok.bos_token_id):
                ids, offs = [tok.bos_token_id] + ids, [(0, 0)] + offs
            st = stats_por_token(obs, perf, ids, device, args.ventana)

            validos = [j for j in range(1, len(ids)) if offs[j][1] > offs[j][0]]
            segs = res[i]["segmentos"]
            inicios = [s for s, _ in segs]
            idx_seg = [[] for _ in segs]
            for j in validos:
                k = bisect.bisect_right(inicios, offs[j][0]) - 1
                if k >= 0:
                    idx_seg[k].append(j)
            res[i]["modelos"][clave] = {
                "info": info,
                "doc": agregar(st, validos),
                "seg": [agregar(st, ix) for ix in idx_seg],
                "tokens": [(offs[j][0], offs[j][1], int(st["rank"][j])) for j in validos],
            }
            print(f"  {clave}: texto {i + 1}/{len(textos)} ({len(validos)} tokens)")
        del obs, perf
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------------------------------
# Puntuación combinada
# ----------------------------------------------------------------------------------------
def cfg_metrica(calib, clave, nivel, met):
    c = (calib or {}).get("modelos", {}).get(clave, {}).get(nivel, {}).get(met)
    if c:
        return dict(c, calibrado=True)
    d = METRICAS[met]
    return {"t": d["t"], "s": d["s"], "dir": d["dir"], "peso": 1.0, "calibrado": False}


def sospecha(valor, cfg):
    d = cfg["dir"] * (valor - cfg["t"]) / max(cfg["s"], 1e-6)
    d = max(-20.0, min(20.0, d))
    return 1.0 / (1.0 + math.exp(-1.5 * d))


def _combinar(senales):
    if not senales:
        return None
    tot = sum(s["peso"] for s in senales)
    if tot <= 0:  # ninguna métrica calibrada separa bien: pesos iguales
        return sum(s["susp"] for s in senales) / len(senales)
    return sum(s["susp"] * s["peso"] for s in senales) / tot


def evaluar(r, calib):
    def senales(nivel, getter):
        out = []
        for clave, m in r["modelos"].items():
            v = getter(m)
            if not v:
                continue
            for met in METRICAS:
                cfg = cfg_metrica(calib, clave, nivel, met)
                out.append({"clave": clave, "met": met, "valor": v[met], "cfg": cfg,
                            "susp": sospecha(v[met], cfg), "peso": cfg.get("peso", 1.0)})
        return out

    doc = senales("doc", lambda m: m["doc"])
    ev = {"doc": {"susp": _combinar(doc), "senales": doc,
                  "n_ia": sum(s["susp"] >= 0.5 for s in doc), "n": len(doc)},
          "segs": []}
    for i in range(len(r["segmentos"])):
        ss = senales("seg", lambda m, i=i: m["seg"][i])
        ev["segs"].append({"susp": _combinar(ss), "senales": ss,
                           "n_ia": sum(s["susp"] >= 0.5 for s in ss), "n": len(ss)})
    return ev


def etiqueta(p):
    if p is None:
        return "sin datos"
    return "alta" if p >= 0.65 else "media" if p >= 0.4 else "baja"


def explicar(s):
    v, t, met, c = s["valor"], s["cfg"]["t"], s["met"], s["clave"]
    if met == "top10":
        return (f"<b>{c}</b>: el {v:.0%} de los tokens estaban entre los 10 más probables "
                f"para el modelo (umbral {t:.0%})")
    if met == "binoculars":
        lado = "por debajo" if v < t else "por encima"
        return (f"<b>{c}</b>: Binoculars {v:.3f}, {lado} del umbral {t:.3f} "
                f"(cuanto más bajo, menos «sorprende» el texto a los modelos)")
    lado = "por encima" if v > t else "por debajo"
    return (f"<b>{c}</b>: Fast-DetectGPT {v:.2f}, {lado} del umbral {t:.2f} "
            f"(palabras más probables de lo esperable al escribir con naturalidad)")


# ----------------------------------------------------------------------------------------
# Informe HTML
# ----------------------------------------------------------------------------------------
def _color(p, luz=92):
    if p is None:
        return "hsl(0,0%,94%)", "hsl(0,0%,70%)"
    h = 120 * (1 - p)
    return f"hsl({h:.0f},70%,{luz}%)", f"hsl({h:.0f},60%,40%)"


def _esc(s):
    return html.escape(s, quote=False)


def html_gltr(texto, tokens):
    out, cur = [], 0
    for s, e, r in sorted(tokens):
        if e <= cur:
            continue
        s = max(s, cur)
        if s > cur:
            out.append(_esc(texto[cur:s]))
        cls = "g1" if r <= 10 else "g2" if r <= 100 else "g3" if r <= 1000 else "g4"
        out.append(f'<span class="{cls}" title="posición {r}">{_esc(texto[s:e])}</span>')
        cur = e
    out.append(_esc(texto[cur:]))
    return "".join(out)


CSS = """
:root{--txt:#1d1d1f;--sub:#5f6368;--borde:#e3e3e3}
*{box-sizing:border-box}
body{font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--txt);
     max-width:980px;margin:0 auto;padding:32px 20px 80px;background:#fafafa}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:36px 0 12px;border-bottom:1px solid var(--borde);padding-bottom:6px}
.sub{color:var(--sub);font-size:13px}
.aviso{background:#fff8e1;border:1px solid #f0d58a;border-radius:8px;padding:10px 14px;margin:16px 0;font-size:14px}
.aviso.rojo{background:#fdecea;border-color:#f3b5ad}
.veredicto{display:flex;gap:24px;align-items:center;background:#fff;border:1px solid var(--borde);
           border-radius:12px;padding:20px 24px;margin-top:16px}
.pct{font-size:44px;font-weight:700;line-height:1}
.pct small{display:block;font-size:14px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;margin-top:6px}
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid var(--borde);border-radius:8px;overflow:hidden;font-size:14px}
th,td{padding:8px 10px;border-bottom:1px solid var(--borde);text-align:left;vertical-align:top}
th{background:#f3f3f3;font-weight:600}
td.m{font-variant-numeric:tabular-nums}
.card{background:#fff;border:1px solid var(--borde);border-left:6px solid;border-radius:8px;padding:12px 16px;margin:12px 0}
.card .cab{font-weight:600;margin-bottom:6px}
.card blockquote{margin:6px 0 8px;padding:6px 12px;background:#f7f7f7;border-radius:6px;font-style:italic;white-space:pre-wrap}
.card ul{margin:4px 0 0;padding-left:20px;font-size:14px}
.seg{position:relative;white-space:pre-wrap;border-left:5px solid;border-radius:6px;padding:8px 12px 8px 12px;margin:6px 0}
.badge{float:right;font-size:12px;font-weight:700;padding:1px 8px;border-radius:10px;background:#fff;margin-left:10px}
.gltr{white-space:pre-wrap;background:#fff;border:1px solid var(--borde);border-radius:8px;padding:14px 16px;line-height:1.9}
.g1{background:#c8f0c8}.g2{background:#fff3a6}.g3{background:#ffc9c2}.g4{background:#e3cdf5}
.ley span{display:inline-block;padding:0 8px;border-radius:4px;margin-right:6px;font-size:13px}
select{font:inherit;padding:4px 8px;margin:4px 0 10px}
details{background:#fff;border:1px solid var(--borde);border-radius:8px;padding:10px 14px;margin-top:10px}
.solo-print{display:none}
@page{size:A4;margin:14mm 12mm}
@media print{body{background:#fff;max-width:none;padding:0;font-size:12px}.gltr[hidden]{display:block!important}
  select{display:none}.solo-print{display:block;font-size:14px;margin:14px 0 6px}
  .card,.veredicto,tr{break-inside:avoid}h2{break-after:avoid}}
"""


def informe_html(nombre, texto, r, ev, calib, args):
    doc = ev["doc"]
    p = doc["susp"]
    bg, fg = _color(p, 90)
    calibrado = any(s["cfg"]["calibrado"] for s in doc["senales"])
    n_tok = max((m["doc"] or {}).get("n_tokens", 0) for m in r["modelos"].values())
    claves = list(r["modelos"].keys())

    avisos = []
    if not calibrado:
        avisos.append('<div class="aviso rojo"><b>Sin calibrar.</b> Los umbrales son genéricos y no están '
                      'ajustados a tu forma de escribir ni al español. Toma el porcentaje como una pista, no como '
                      'un veredicto. Para calibrar: <code>python detector_ia.py calibrar --humanos ... --ia ...</code></div>')
    if n_tok < TOKENS_DOC_CORTO:
        avisos.append(f'<div class="aviso">Texto corto ({n_tok} tokens). Con menos de ~{TOKENS_DOC_CORTO} '
                      'tokens las señales son poco fiables.</div>')
    if calib:
        for c in claves:
            qc = calib.get("modelos", {}).get(c, {}).get("quant")
            qa = r["modelos"][c]["info"]["quant"]
            if qc and qc != qa:
                avisos.append(f'<div class="aviso">{c}: calibrado con cuantización «{qc}» pero analizado con '
                              f'«{qa}». Los umbrales pueden no corresponder.</div>')
            if c not in calib.get("modelos", {}):
                avisos.append(f'<div class="aviso">{c} no está en el archivo de calibración: usa umbrales genéricos.</div>')

    if p is None:
        frase = "No hay suficiente texto para analizar."
    elif p >= 0.65:
        frase = "Bastantes señales estadísticas típicas de texto generado por LLM."
    elif p >= 0.4:
        frase = "Señales mixtas: algunos rasgos típicos de LLM, pero no concluyentes."
    else:
        frase = "Pocas señales estadísticas típicas de texto generado por LLM."

    # Tabla por modelo
    filas = []
    for c in claves:
        m = r["modelos"][c]
        celdas = [f"<td><b>{c}</b><br><span class='sub'>{m['info']['quant']}</span></td>"]
        for met in METRICAS:
            s = next((x for x in doc["senales"] if x["clave"] == c and x["met"] == met), None)
            if not s:
                celdas.append("<td>—</td>")
                continue
            b, _ = _color(s["susp"], 88)
            extra = f" · AUC {s['cfg']['auc']:.2f}" if s["cfg"].get("auc") is not None else ""
            celdas.append(f"<td class='m' style='background:{b}'>{fmt_valor(met, s['valor'])}"
                          f"<br><span class='sub'>umbral {fmt_valor(met, s['cfg']['t'])}{extra}</span></td>")
        filas.append("<tr>" + "".join(celdas) + "</tr>")
    cab = "".join(f"<th>{METRICAS[m]['nombre']}</th>" for m in METRICAS)
    tabla = f"<table><tr><th>Modelo</th>{cab}</tr>{''.join(filas)}</table>"

    # Fragmentos más sospechosos
    orden = sorted([(e["susp"], i) for i, e in enumerate(ev["segs"]) if e["susp"] is not None], reverse=True)
    tarjetas = []
    for susp, i in orden[:args.top]:
        if susp < 0.5:
            break
        s0, s1 = r["segmentos"][i]
        frag = texto[s0:s1].strip()
        if len(frag) > 420:
            frag = frag[:400].rsplit(" ", 1)[0] + " […]"
        sen = sorted([x for x in ev["segs"][i]["senales"] if x["susp"] >= 0.6], key=lambda x: -x["susp"])
        _, f = _color(susp)
        motivos = "".join(f"<li>{explicar(x)}</li>" for x in sen[:6]) or "<li>Señales moderadas en varias métricas.</li>"
        tarjetas.append(f"<div class='card' style='border-left-color:{f}'><div class='cab'>Fragmento {i + 1} · "
                        f"sospecha {susp:.0%} ({etiqueta(susp)}) · {ev['segs'][i]['n_ia']}/{ev['segs'][i]['n']} señales</div>"
                        f"<blockquote>{_esc(frag)}</blockquote><ul>{motivos}</ul></div>")
    if not tarjetas:
        tarjetas.append("<p>Ningún fragmento supera el 50 % de sospecha.</p>")

    # Texto por fragmentos
    bloques = []
    for i, (s0, s1) in enumerate(r["segmentos"]):
        susp = ev["segs"][i]["susp"]
        b, f = _color(susp)
        txt = "muy corto" if susp is None else f"{susp:.0%}"
        bloques.append(f"<div class='seg' style='background:{b};border-left-color:{f}' id='f{i + 1}'>"
                       f"<span class='badge' style='color:{f}'>#{i + 1} · {txt}</span>{_esc(texto[s0:s1])}</div>")

    # Vista GLTR
    opciones = "".join(f"<option value='{i}'>{c}</option>" for i, c in enumerate(claves))
    vistas = "".join(f"<h3 class='solo-print'>Modelo: {_esc(c)}</h3>"
                     f"<div class='gltr' id='gltr-{i}' {'' if i == 0 else 'hidden'}>"
                     f"{html_gltr(texto, r['modelos'][c]['tokens'])}</div>" for i, c in enumerate(claves))

    modelos_txt = ", ".join(f"{c} ({r['modelos'][c]['info']['base']} + {r['modelos'][c]['info']['instruct']})"
                            for c in claves)
    pct_txt = "—" if p is None else f"{p:.0%}"

    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Informe IA · {_esc(nombre)}</title><style>{CSS}</style></head><body>
<h1>Informe de detección de IA</h1>
<div class="sub">{_esc(nombre)} · {dt.datetime.now():%d/%m/%Y %H:%M} · {n_tok} tokens · {len(r['segmentos'])} fragmentos<br>
Modelos: {_esc(modelos_txt)}<br>Calibración: {'sí — ' + _esc(args.calibracion) if calibrado else 'no (umbrales genéricos)'}</div>
{''.join(avisos)}

<div class="veredicto" style="background:{bg};border-color:{fg}">
  <div class="pct" style="color:{fg}">{pct_txt}<small>sospecha {etiqueta(p)}</small></div>
  <div><b>{frase}</b><br>{doc['n_ia']} de {doc['n']} señales (modelos × métricas) apuntan a IA en el documento completo.
  <br><span class="sub">Es una estimación estadística, no una prueba de autoría. Un texto humano muy formal o
  pulido puede puntuar alto, y un texto de IA editado puede puntuar bajo.</span></div>
</div>

<h2>Señales por modelo (documento completo)</h2>
{tabla}
<p class="sub">Binoculars: más bajo = más «IA». Fast-DetectGPT y GLTR top-10: más alto = más «IA».
El color de cada celda indica hacia qué lado cae respecto a su umbral.</p>

<h2>Fragmentos más sospechosos</h2>
{''.join(tarjetas)}

<h2>Texto completo por fragmentos</h2>
<p class="sub">Verde = pocas señales de IA · rojo = muchas. Gris = fragmento demasiado corto para puntuar.</p>
{''.join(bloques)}

<h2>Texto completo por palabras (estilo GLTR)</h2>
<p class="ley"><span class="g1">top 10</span><span class="g2">top 100</span><span class="g3">top 1000</span><span class="g4">resto</span>
<span class="sub">Posición de cada token en la predicción del modelo base. Zonas casi enteras en verde = muy predecibles.
Pasa el ratón por encima para ver la posición exacta.</span></p>
<select onchange="document.querySelectorAll('.gltr').forEach(d=>d.hidden=d.id!=='gltr-'+this.value)">{opciones}</select>
{vistas}

<h2>Cómo leer este informe</h2>
<details open><summary><b>Qué mide cada señal</b></summary>
<p><b>GLTR top-10</b>: porcentaje de palabras (tokens) que el modelo habría puesto entre sus 10 primeras opciones.
Los LLM eligen casi siempre palabras muy probables; las personas usan más palabras «inesperadas».</p>
<p><b>Binoculars</b>: compara lo sorprendente que es el texto para un modelo instruct con lo sorprendente que
«debería» ser según el modelo base. Valores bajos = texto típico de un modelo.</p>
<p><b>Fast-DetectGPT</b>: mide si las palabras elegidas son sistemáticamente más probables que las que saldrían
muestreando del modelo. Valores altos = texto típico de un modelo.</p>
<p><b>Sospecha</b>: media ponderada de todas las señales de todos los modelos{'; con calibración, cada señal pesa según lo bien que separó tus textos (AUC)' if calibrado else ''}.</p>
</details>
<details><summary><b>Limitaciones</b></summary>
<p>Estas técnicas están evaluadas sobre todo en inglés; en español rinden peor y varían mucho según el tipo de
texto. Son fáciles de engañar con paráfrasis o edición. Penalizan la escritura muy formal, técnica o con
fórmulas fijas, y los textos muy conocidos (citas, leyes) que los modelos han memorizado. Con textos de
menos de ~200 palabras la fiabilidad cae mucho.</p>
</details>
</body></html>"""


# ----------------------------------------------------------------------------------------
# Calibración
# ----------------------------------------------------------------------------------------
def auc(h, a):
    """P(valor_ia > valor_humano), con empates a medias."""
    tot = 0.0
    for x in a:
        for y in h:
            tot += 1.0 if x > y else 0.5 if x == y else 0.0
    return tot / (len(a) * len(h))


def ajustar_umbral(h, a):
    au = auc(h, a)
    d = 1 if au >= 0.5 else -1
    vals = sorted(set(h + a))
    cands = [(u + v) / 2 for u, v in zip(vals, vals[1:])] or vals
    mejor = (-2.0, cands[0], 0.0, 0.0)
    for t in cands:
        tpr = sum(d * (x - t) > 0 for x in a) / len(a)
        fpr = sum(d * (y - t) > 0 for y in h) / len(h)
        if tpr - fpr > mejor[0]:
            mejor = (tpr - fpr, t, tpr, fpr)
    sd = lambda xs: statistics.pstdev(xs) if len(xs) > 1 else 0.0
    s = math.sqrt((sd(h) ** 2 + sd(a) ** 2) / 2) or 1e-3
    sep = max(au, 1 - au)
    return {"t": mejor[1], "s": s, "dir": d, "auc": round(sep, 4),
            "peso": round(2 * (sep - 0.5), 4) if sep >= 0.55 else 0.0,
            "tpr": round(mejor[2], 3), "fpr": round(mejor[3], 3), "n_humano": len(h), "n_ia": len(a)}


EXTENSIONES = (".txt", ".md", ".pdf", ".ipynb", ".docx")


def _leer_pdf(p):
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("Para leer PDF necesito pypdf:  pip install pypdf")
    paginas = [(pag.extract_text() or "").strip() for pag in PdfReader(str(p)).pages]
    texto = "\n\n".join(t for t in paginas if t)
    return re.sub(r"(\w)-\n(\w)", r"\1\2", texto)  # une palabras cortadas con guion al final de línea


def _leer_ipynb(p):
    """Solo las celdas markdown: el código y las salidas no son prosa."""
    nb = json.loads(Path(p).read_text(encoding="utf-8"))
    celdas = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "markdown":
            src = c.get("source", "")
            celdas.append(("".join(src) if isinstance(src, list) else src).strip())
    return "\n\n".join(t for t in celdas if t)


def _leer_docx(p):
    """Texto de los párrafos de word/document.xml, sin dependencias externas."""
    import zipfile
    import xml.etree.ElementTree as ET

    w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(p) as z:
        raiz = ET.fromstring(z.read("word/document.xml"))
    parrafos = []
    for par in raiz.iter(f"{w}p"):
        trozos = []
        for el in par.iter():
            if el.tag == f"{w}t":
                trozos.append(el.text or "")
            elif el.tag == f"{w}tab":
                trozos.append("\t")
            elif el.tag in (f"{w}br", f"{w}cr"):
                trozos.append("\n")
        parrafos.append("".join(trozos))
    # un párrafo de Word = un párrafo para la segmentación
    return "\n\n".join(t.strip() for t in parrafos if t.strip())


def _leer(p):
    p = Path(p)
    ext = p.suffix.lower()
    try:
        if ext == ".pdf":
            texto = _leer_pdf(p)
        elif ext == ".ipynb":
            texto = _leer_ipynb(p)
        elif ext == ".docx":
            texto = _leer_docx(p)
        else:
            texto = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # PDF dañado, JSON inválido, zip roto…
        sys.exit(f"No he podido leer {p}: {e}")
    if not texto.strip():
        pista = " ¿Es un PDF escaneado? Necesitaría OCR." if ext == ".pdf" else ""
        sys.exit(f"{p} no contiene texto que analizar.{pista}")
    return texto


def cmd_calibrar(args):
    def listar(carpeta):
        return sorted(p for p in Path(carpeta).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES)

    hum = listar(args.humanos)
    ia = listar(args.ia)
    if len(hum) < 5 or len(ia) < 5:
        sys.exit(f"Necesito al menos 5 textos en cada carpeta (hay {len(hum)} humanos y {len(ia)} de IA). "
                 f"Lo recomendable son 20+ de cada.")
    claves = [c.strip() for c in args.modelos.split(",") if c.strip()]
    textos = [_leer(p) for p in hum + ia]
    y = [0] * len(hum) + [1] * len(ia)
    res = analizar_textos(textos, claves, args)

    calib = {"version": 1, "creado": dt.datetime.now().isoformat(timespec="seconds"),
             "n_humanos": len(hum), "n_ia": len(ia), "min_palabras": args.min_palabras, "modelos": {}}
    print("\nResultados de calibración (AUC: 0,5 = azar, 1 = separación perfecta)")
    print(f"{'modelo':<16}{'nivel':<6}{'métrica':<16}{'AUC':>6}{'umbral':>10}{'TPR':>7}{'FPR':>7}")
    for c in claves:
        calib["modelos"][c] = {"quant": res[0]["modelos"][c]["info"]["quant"], "doc": {}, "seg": {}}
        for nivel in ("doc", "seg"):
            for met in METRICAS:
                h, a = [], []
                for r, lab in zip(res, y):
                    m = r["modelos"][c]
                    for v in ([m["doc"]] if nivel == "doc" else m["seg"]):
                        if v:
                            (a if lab else h).append(v[met])
                if len(h) < 3 or len(a) < 3:
                    continue
                cfg = ajustar_umbral(h, a)
                calib["modelos"][c][nivel][met] = cfg
                print(f"{c:<16}{nivel:<6}{METRICAS[met]['nombre']:<16}{cfg['auc']:>6.2f}"
                      f"{fmt_valor(met, cfg['t']):>10}{cfg['tpr']:>7.0%}{cfg['fpr']:>7.0%}")
    Path(args.salida).write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {args.salida}. Las métricas con AUC < 0,55 no cuentan en la sospecha.")


# ----------------------------------------------------------------------------------------
# Análisis
# ----------------------------------------------------------------------------------------
def cmd_analizar(args):
    calib = None
    if args.calibracion:
        calib = json.loads(Path(args.calibracion).read_text(encoding="utf-8"))
        if calib.get("min_palabras") and calib["min_palabras"] != args.min_palabras:
            print(f"! La calibración usó --min-palabras {calib['min_palabras']}; uso ese valor.")
            args.min_palabras = calib["min_palabras"]
    if args.pdf:
        try:
            import weasyprint  # noqa: F401
        except (ImportError, OSError) as e:
            sys.exit(f"Para --pdf necesito WeasyPrint (pip install weasyprint) y sus librerías del sistema: {e}")
    claves = [c.strip() for c in args.modelos.split(",") if c.strip()]
    rutas = [Path(p) for p in args.archivos]
    textos = [_leer(p) for p in rutas]
    res = analizar_textos(textos, claves, args)

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    for ruta, texto, r in zip(rutas, textos, res):
        ev = evaluar(r, calib)
        out = salida / f"informe_{ruta.stem}.html"
        doc_html = informe_html(ruta.name, texto, r, ev, calib, args)
        out.write_text(doc_html, encoding="utf-8")
        p = ev["doc"]["susp"]
        print(f"\n{ruta.name}: sospecha {'—' if p is None else f'{p:.0%}'} ({etiqueta(p)}) → {out}")
        if args.pdf:
            import weasyprint
            out_pdf = out.with_suffix(".pdf")
            print("   generando PDF (con textos largos tarda un rato)…")
            weasyprint.HTML(string=doc_html).write_pdf(out_pdf)
            print(f"   PDF → {out_pdf}")
        top = sorted([(e["susp"], i) for i, e in enumerate(ev["segs"]) if e["susp"] is not None], reverse=True)[:3]
        for s, i in top:
            frag = " ".join(texto[slice(*r["segmentos"][i])].split())[:90]
            print(f"   #{i + 1:<3} {s:4.0%}  {frag}…")
        if args.json:
            datos = {"archivo": ruta.name, "sospecha": p, "segmentos": [
                {"n": i + 1, "inicio": s0, "fin": s1, "sospecha": ev["segs"][i]["susp"],
                 "metricas": {c: r["modelos"][c]["seg"][i] for c in claves}}
                for i, (s0, s1) in enumerate(r["segmentos"])],
                "documento": {c: r["modelos"][c]["doc"] for c in claves}}
            (salida / f"informe_{ruta.stem}.json").write_text(
                json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_modelos(_):
    print(f"Por defecto: {DEFAULT_MODELOS}\n")
    for k, (b, i, t, nota) in PRESETS.items():
        print(f"  {k:<15} {t:>4.1f}B  {b} + {i}\n  {'':<15}        {nota}")
    print("\nTambién puedes pasar un par propio: --modelos \"org/base+org/instruct\" (mismo tokenizador).")


def main():
    ap = argparse.ArgumentParser(description="Detector local y orientativo de texto generado por IA.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def comunes(p):
        p.add_argument("--modelos", default=DEFAULT_MODELOS, help=f"lista separada por comas (def.: {DEFAULT_MODELOS})")
        p.add_argument("--quant", default="auto", choices=["auto", "none", "8bit", "4bit"],
                       help="auto = 4 bits para modelos ≥3B, sin cuantizar los pequeños")
        p.add_argument("--min-palabras", type=int, default=50, help="tamaño mínimo de cada fragmento (def.: 50)")
        p.add_argument("--ventana", type=int, default=384, help="tokens por pasada; bájalo si te quedas sin VRAM")

    pa = sub.add_parser("analizar", help="analiza uno o varios archivos (.txt, .md, .pdf, .ipynb, .docx) y genera informes HTML")
    pa.add_argument("archivos", nargs="+")
    pa.add_argument("--calibracion", help="archivo JSON generado con `calibrar`")
    pa.add_argument("--salida", default="informes", help="carpeta de salida (def.: informes/)")
    pa.add_argument("--top", type=int, default=6, help="nº de fragmentos sospechosos a destacar")
    pa.add_argument("--json", action="store_true", help="guarda también los datos en JSON")
    pa.add_argument("--pdf", action="store_true", help="guarda también el informe en PDF (necesita weasyprint)")
    comunes(pa)
    pa.set_defaults(func=cmd_analizar)

    pc = sub.add_parser("calibrar", help="ajusta umbrales con textos humanos y de IA conocidos")
    pc.add_argument("--humanos", required=True, help="carpeta con textos escritos por ti (antes de 2023 mejor)")
    pc.add_argument("--ia", required=True, help="carpeta con textos generados por LLMs")
    pc.add_argument("--salida", default="calibracion.json")
    comunes(pc)
    pc.set_defaults(func=cmd_calibrar)

    pm = sub.add_parser("modelos", help="lista los modelos disponibles")
    pm.set_defaults(func=cmd_modelos)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
