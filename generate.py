import numpy as np
from pathlib import Path
from tokenizer import TokenizadorCharLevel
from model import TransformerLM
from train import cargar_modelo

VOCAB_PATH      = Path("data/corpus/vocabulario.json")
CHECKPOINT_PATH = Path("checkpoints/modelo_20000pasos.npz")

D_MODEL  = 256
N_HEADS  = 8
N_LAYERS = 4
D_FF     = 1024
CHECKPOINT_PATH = Path("checkpoints/modelo_50000pasos.npz")


def softmax(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


def generar(modelo, tok, prompt, max_tokens=200, temperatura=0.8):
    indices = np.array(tok.codificar(prompt), dtype=np.int32)
    texto_generado = prompt

    for _ in range(max_tokens):
        contexto = indices[-512:]
        logits = modelo.forward(contexto)
        logits_ultimo = logits[-1, :] / temperatura
        probs = softmax(logits_ultimo)
        siguiente = np.random.choice(len(probs), p=probs)
        indices = np.append(indices, siguiente)
        texto_generado += tok.decodificar([siguiente])

    return texto_generado


if __name__ == "__main__":
    tok = TokenizadorCharLevel()
    tok.cargar(VOCAB_PATH)

    modelo = TransformerLM(
        vocab_size = tok.vocab_size,
        d_model    = D_MODEL,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        d_ff       = D_FF
    )

    cargar_modelo(modelo, CHECKPOINT_PATH)

    print("\n=== Generación con modelo entrenado ===\n")

    prompt = "Señor, mi corazón"
    resultado = generar(modelo, tok, prompt, max_tokens=150, temperatura=1.2)
    print(resultado)