import pdfplumber
from pathlib import Path

PDF_PATH = Path("data/raw/Confesiones-de-San-Agustin-Tomo-I.pdf")

def inspeccionar_fuentes():
    with pdfplumber.open(PDF_PATH) as pdf:
        pagina = pdf.pages[9]
        chars = pagina.chars

        # Mostrar combinaciones únicas de tamaño y fuente
        combinaciones = sorted(set(
            (round(c['size'], 1), c['fontname'])
            for c in chars
        ), reverse=True)

        print("Tamaño | Fuente")
        print("-" * 60)
        for tamanio, fuente in combinaciones:
            print(f"{tamanio:6.1f} | {fuente}")

if __name__ == "__main__":
    inspeccionar_fuentes()