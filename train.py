import numpy as np
from pathlib import Path
from tokenizer import TokenizadorCharLevel
from model import TransformerLM
from engine import cross_entropy_loss, gradiente_cross_entropy
from optim import Adam


# ── Hiperparámetros ──────────────────────────────────────────────
CORPUS_PATH  = Path("data/corpus/confesiones_final.txt")
VOCAB_PATH   = Path("data/corpus/vocabulario.json")

CONTEXT_LEN  = 128
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 2
D_FF         = 512
LR           = 3e-4
PASOS        = 5000
LOG_CADA     = 100
SEED         = 42
# ─────────────────────────────────────────────────────────────────


def obtener_batch(datos, context_len):
    pos = np.random.randint(0, len(datos) - context_len - 1)
    x = datos[pos     : pos + context_len]
    y = datos[pos + 1 : pos + context_len + 1]
    return x, y


def guardar_modelo(modelo, path):
    pesos = {}
    pesos['embedding']      = modelo.embedding.pesos
    pesos['cabeza']         = modelo.cabeza
    pesos['ln_final_gamma'] = modelo.ln_final.gamma
    pesos['ln_final_beta']  = modelo.ln_final.beta

    for i, bloque in enumerate(modelo.bloques):
        p = f'bloque_{i}'
        pesos[f'{p}_Wq']       = bloque.attn.Wq
        pesos[f'{p}_Wk']       = bloque.attn.Wk
        pesos[f'{p}_Wv']       = bloque.attn.Wv
        pesos[f'{p}_Wo']       = bloque.attn.Wo
        pesos[f'{p}_W1']       = bloque.ff.W1
        pesos[f'{p}_b1']       = bloque.ff.b1
        pesos[f'{p}_W2']       = bloque.ff.W2
        pesos[f'{p}_b2']       = bloque.ff.b2
        pesos[f'{p}_ln1_gamma'] = bloque.ln1.gamma
        pesos[f'{p}_ln1_beta']  = bloque.ln1.beta
        pesos[f'{p}_ln2_gamma'] = bloque.ln2.gamma
        pesos[f'{p}_ln2_beta']  = bloque.ln2.beta

    np.savez(path, **pesos)
    print(f"Modelo guardado en: {path}")


def cargar_modelo(modelo, path):
    pesos = np.load(path)

    modelo.embedding.pesos  = pesos['embedding']
    modelo.cabeza           = pesos['cabeza']
    modelo.ln_final.gamma   = pesos['ln_final_gamma']
    modelo.ln_final.beta    = pesos['ln_final_beta']

    for i, bloque in enumerate(modelo.bloques):
        p = f'bloque_{i}'
        bloque.attn.Wq   = pesos[f'{p}_Wq']
        bloque.attn.Wk   = pesos[f'{p}_Wk']
        bloque.attn.Wv   = pesos[f'{p}_Wv']
        bloque.attn.Wo   = pesos[f'{p}_Wo']
        bloque.ff.W1     = pesos[f'{p}_W1']
        bloque.ff.b1     = pesos[f'{p}_b1']
        bloque.ff.W2     = pesos[f'{p}_W2']
        bloque.ff.b2     = pesos[f'{p}_b2']
        bloque.ln1.gamma = pesos[f'{p}_ln1_gamma']
        bloque.ln1.beta  = pesos[f'{p}_ln1_beta']
        bloque.ln2.gamma = pesos[f'{p}_ln2_gamma']
        bloque.ln2.beta  = pesos[f'{p}_ln2_beta']

    print(f"Modelo cargado desde: {path}")
    return modelo


def entrenar():
    np.random.seed(SEED)

    print("Cargando corpus...")
    texto = CORPUS_PATH.read_text(encoding="utf-8")

    tok = TokenizadorCharLevel()
    tok.cargar(VOCAB_PATH)

    datos = np.array(tok.codificar(texto), dtype=np.int32)
    np.random.seed(SEED)
    print(f"Corpus: {len(datos):,} tokens")
    print(f"Vocabulario: {tok.vocab_size} caracteres\n")

    modelo = TransformerLM(
        vocab_size = tok.vocab_size,
        d_model    = D_MODEL,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        d_ff       = D_FF
    )

    optim = Adam(lr=LR)

    print(f"Iniciando entrenamiento — {PASOS} pasos")
    print("-" * 40)

    perdidas = []

    for paso in range(1, PASOS + 1):
        x, y = obtener_batch(datos, CONTEXT_LEN)
        logits = modelo.forward(x)
        loss, probs = cross_entropy_loss(logits, y)
        perdidas.append(loss)
        grad_logits = gradiente_cross_entropy(probs, y)
        modelo.backward(grad_logits)
        optim.paso(modelo)

        if paso % LOG_CADA == 0:
            perdida_promedio = np.mean(perdidas[-LOG_CADA:])
            print(f"Paso {paso:>5} | pérdida: {perdida_promedio:.4f}")

    print("-" * 40)
    print(f"Entrenamiento completado")
    print(f"Pérdida final: {np.mean(perdidas[-100:]):.4f}")

    # Guardar modelo entrenado
    guardar_modelo(modelo, Path("checkpoints/modelo_5000pasos.npz"))


if __name__ == "__main__":
    entrenar()