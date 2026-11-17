"""Tests sin modelos: limpieza de markdown, explicación de señales y avisos de dirección invertida.
Ejecutar con:  python -m pytest tests   (o  python tests/test_detector.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import detector_ia as d  # noqa: E402


def test_codigo_en_linea_con_etiqueta_html_se_conserva():
    assert d.limpiar_markdown("restos del tag HTML `<br />`, que sale como `/>` y pasar todo") == \
        "restos del tag HTML <br />, que sale como /> y pasar todo"


def test_codigo_en_linea_con_angulos_sueltos_se_conserva():
    assert d.limpiar_markdown("quitar todo lo que haya entre `<` y `>` borraría") == \
        "quitar todo lo que haya entre < y > borraría"


def test_angulos_en_prosa_no_son_etiquetas():
    assert d.limpiar_markdown("quitar todo lo que haya entre < y > borraría; si x < 3 y y > 2") == \
        "quitar todo lo que haya entre < y > borraría; si x < 3 y y > 2"


def test_etiquetas_html_fuera_de_codigo_se_quitan():
    assert d.limpiar_markdown("hola<br/>mundo <b>negrita</b>") == "holamundo negrita"


def test_bloque_de_codigo_se_quita():
    assert d.limpiar_markdown("antes\n\n```python\nx = `<a>`\n```\n\ndespués") == "antes\n\ndespués"


def _senal(met, valor, t, dir_):
    return {"clave": "m", "met": met, "valor": valor,
            "cfg": {"t": t, "s": 1.0, "dir": dir_, "calibrado": True}}


def test_explicar_usa_direccion_calibrada():
    # Binoculars calibrado con dir=+1: por encima del umbral apunta a IA y debe decirse.
    txt = d.explicar(_senal("binoculars", 1.238, 1.039, +1))
    assert "por encima" in txt and "más alto = más «IA»" in txt and "invertida" in txt


def test_explicar_direccion_teorica_sin_aviso():
    txt = d.explicar(_senal("fastdetect", 2.0, 1.0, +1))
    assert "más alto = más «IA»" in txt and "invertida" not in txt


def test_explicar_top10_dice_el_lado():
    txt = d.explicar(_senal("top10", 0.71, 0.89, -1))
    assert "por debajo" in txt and "más bajo = más «IA»" in txt and "invertida" in txt


def test_senales_invertidas_ignora_metricas_sin_peso():
    calib = {"modelos": {"m": {"quant": "none", "doc": {
        "binoculars": {"dir": 1, "peso": 0.8}, "fastdetect": {"dir": 1, "peso": 0.9},
        "top10": {"dir": -1, "peso": 0.0}}, "seg": {}}}}
    assert d.senales_invertidas(calib) == [("m", "doc", "binoculars")]


if __name__ == "__main__":
    fallos = 0
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            try:
                f()
                print("ok  ", nombre)
            except AssertionError as e:
                fallos += 1
                print("FALLO", nombre, e)
    sys.exit(1 if fallos else 0)
