import re
from pathlib import Path

INPUT_PATH = Path("data/corpus/confesiones_limpio.txt")
OUTPUT_PATH = Path("data/corpus/confesiones_final.txt")


def limpiar_corpus(input_path, output_path):
    texto = input_path.read_text(encoding="utf-8")

    # 1. Eliminar todo antes de "1. Grande sois" — el texto real de Agustín
    marcador = "1. Grande sois"
    pos = texto.find(marcador)
    if pos != -1:
        texto = texto[pos:]
        print("✓ Prólogo del traductor eliminado")

    # 2. Eliminar números de página pegados al inicio de línea
    # Patrón: número solo en una línea, o número seguido de espacio al inicio
    texto = re.sub(r'^\d+\s*\n', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'^\d+\s+', '', texto, flags=re.MULTILINE)
    print("✓ Números de página eliminados")

    # 3. Normalizar espacios múltiples
    texto = re.sub(r'  +', ' ', texto)
    print("✓ Espacios múltiples normalizados")

    # 4. Normalizar saltos de línea múltiples
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    print("✓ Saltos de línea normalizados")

    output_path.write_text(texto.strip(), encoding="utf-8")

    print(f"\nCaracteres antes: {len(input_path.read_text(encoding='utf-8'))}")
    print(f"Caracteres después: {len(texto)}")
    print(f"Guardado en: {output_path}")


if __name__ == "__main__":
    limpiar_corpus(INPUT_PATH, OUTPUT_PATH)