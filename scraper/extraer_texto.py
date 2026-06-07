import pdfplumber
from pathlib import Path

# Rutas
PDF_PATH = Path("data/raw/Confesiones-de-San-Agustin-Tomo-I.pdf")
OUTPUT_PATH = Path("data/corpus/confesiones_limpio.txt")

# Tamaño de fuente del texto principal de Agustín
TAMANIO_PRINCIPAL = 12.0


def extraer_texto_principal(pdf_path, output_path):
    """
    Extrae solo el texto con tamaño de fuente 12.0 y fuente normal
    (no itálica) — el cuerpo principal del texto de Agustín,
    descartando encabezados, notas del traductor y títulos de capítulo.
    """
    paginas_texto = []

    with pdfplumber.open(pdf_path) as pdf:
        total_paginas = len(pdf.pages)
        print(f"Total de páginas: {total_paginas}")

        for i, pagina in enumerate(pdf.pages):

            # Filtrar solo texto principal: tamaño 12.0 y fuente normal
            chars_principales = [
                c for c in pagina.chars
                if abs(c['size'] - TAMANIO_PRINCIPAL) < 0.1
                and 'Oblique' not in c['fontname']
                and 'Italic' not in c['fontname']
            ]

            # Reconstruir texto desde los caracteres filtrados
            if chars_principales:
                texto = ''.join(c['text'] for c in chars_principales)
                texto = texto.strip()
                if texto:
                    paginas_texto.append(texto)

            if (i + 1) % 10 == 0:
                print(f"  Procesadas {i + 1}/{total_paginas} páginas...")

    # Unir páginas con salto de línea
    texto_completo = '\n'.join(paginas_texto)

    # Guardar
    output_path.write_text(texto_completo, encoding="utf-8")

    print(f"\nListo.")
    print(f"Caracteres extraídos: {len(texto_completo)}")
    print(f"Guardado en: {output_path}")


if __name__ == "__main__":
    extraer_texto_principal(PDF_PATH, OUTPUT_PATH)