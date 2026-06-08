import json
from pathlib import Path


class TokenizadorCharLevel:
    """
    Tokenizador a nivel de carácter.

    Atributos:
        char_a_idx: diccionario que mapea carácter → entero
        idx_a_char: diccionario que mapea entero → carácter
        vocab_size: cantidad total de caracteres únicos
    """

    def __init__(self):
        self.char_a_idx = {}
        self.idx_a_char = {}
        self.vocab_size = 0

    def construir_vocabulario(self, texto):
        """
        Recorre el texto completo, encuentra todos los
        caracteres únicos, y les asigna un índice entero.

        """
        # Encontrar todos los caracteres únicos
        caracteres_unicos = sorted(set(texto))

        # Construir los dos diccionarios
        self.char_a_idx = {
            char: idx
            for idx, char in enumerate(caracteres_unicos)
        }
        self.idx_a_char = {
            idx: char
            for idx, char in enumerate(caracteres_unicos)
        }

        self.vocab_size = len(caracteres_unicos)

    def codificar(self, texto):
        """
        Convierte una cadena de texto en una lista de enteros.

        Ejemplo:
            "Vos" → [54, 71, 75]
        """
        return [self.char_a_idx[c] for c in texto]

    def decodificar(self, indices):
        """
        Convierte una lista de enteros en una cadena de texto.

        Ejemplo:
            [54, 71, 75] → "Vos"
        """
        return ''.join(self.idx_a_char[i] for i in indices)

    def guardar(self, path):
        """
        Guarda el vocabulario en un archivo JSON para
        no tener que reconstruirlo cada vez.
        """
        datos = {
            'char_a_idx': self.char_a_idx,
            'vocab_size': self.vocab_size
        }
        Path(path).write_text(
            json.dumps(datos, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    def cargar(self, path):
        """
        Carga un vocabulario previamente guardado.
        Reconstruye idx_a_char desde char_a_idx.
        """
        datos = json.loads(Path(path).read_text(encoding='utf-8'))
        self.char_a_idx = datos['char_a_idx']
        self.idx_a_char = {
            int(idx): char
            for char, idx in self.char_a_idx.items()
        }
        self.vocab_size = datos['vocab_size']


if __name__ == "__main__":
    # Probar el tokenizador con el corpus
    CORPUS_PATH = Path("data/corpus/confesiones_final.txt")
    VOCAB_PATH = Path("data/corpus/vocabulario.json")

    # Leer el corpus
    texto = CORPUS_PATH.read_text(encoding="utf-8")
    print(f"Caracteres en el corpus: {len(texto)}")

    # Construir vocabulario
    tok = TokenizadorCharLevel()
    tok.construir_vocabulario(texto)
    print(f"Tamaño del vocabulario: {tok.vocab_size} caracteres únicos")

    # Mostrar el vocabulario completo
    print(f"\nCaracteres en el vocabulario:")
    print(repr(''.join(tok.char_a_idx.keys())))

    # Probar codificación y decodificación
    frase = "está inquieto nuestro corazón"
    codificada = tok.codificar(frase)
    decodificada = tok.decodificar(codificada)

    print(f"\nPrueba de codificación:")
    print(f"  Original:    '{frase}'")
    print(f"  Codificada:  {codificada}")
    print(f"  Decodificada: '{decodificada}'")
    print(f"  Correcto: {frase == decodificada}")

    # Guardar vocabulario
    tok.guardar(VOCAB_PATH)
    print(f"\nVocabulario guardado en: {VOCAB_PATH}")
