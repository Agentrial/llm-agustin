import time
import numpy as np
from pathlib import Path
from tokenizer import TokenizadorCharLevel
from model import TransformerLM
from engine import cross_entropy_loss, gradiente_cross_entropy
from optim import Adam


def contar_parametros(modelo):
    total = 0
    total += modelo.embedding.pesos.size
    total += modelo.cabeza.size
    total += modelo.ln_final.gamma.size
    total += modelo.ln_final.beta.size
    for bloque in modelo.bloques:
        total += bloque.attn.Wq.size + bloque.attn.Wk.size
        total += bloque.attn.Wv.size + bloque.attn.Wo.size
        total += bloque.ff.W1.size + bloque.ff.b1.size
        total += bloque.ff.W2.size + bloque.ff.b2.size
        total += bloque.ln1.gamma.size + bloque.ln1.beta.size
        total += bloque.ln2.gamma.size + bloque.ln2.beta.size
    return total


def medir_velocidad(modelo, datos, n_pasos=10):
    optim = Adam(lr=3e-4)
    for _ in range(3):
        pos = np.random.randint(0, len(datos) - 129)
        x, y = datos[pos:pos+128], datos[pos+1:pos+129]
        logits = modelo.forward(x)
        loss, probs = cross_entropy_loss(logits, y)
        grad = gradiente_cross_entropy(probs, y)
        modelo.backward(grad)
        optim.paso(modelo)
    inicio = time.time()
    for _ in range(n_pasos):
        pos = np.random.randint(0, len(datos) - 129)
        x, y = datos[pos:pos+128], datos[pos+1:pos+129]
        logits = modelo.forward(x)
        loss, probs = cross_entropy_loss(logits, y)
        grad = gradiente_cross_entropy(probs, y)
        modelo.backward(grad)
        optim.paso(modelo)
    return n_pasos / (time.time() - inicio)


def entrenar_rapido(modelo, datos, n_pasos=500):
    """Entrena el modelo por n_pasos y devuelve la pérdida final."""
    optim = Adam(lr=3e-4)
    perdidas = []
    for paso in range(n_pasos):
        pos = np.random.randint(0, len(datos) - 129)
        x, y = datos[pos:pos+128], datos[pos+1:pos+129]
        logits = modelo.forward(x)
        loss, probs = cross_entropy_loss(logits, y)
        perdidas.append(loss)
        grad = gradiente_cross_entropy(probs, y)
        modelo.backward(grad)
        optim.paso(modelo)
        if (paso + 1) % 100 == 0:
            print(f"    paso {paso+1:>4} | pérdida: {np.mean(perdidas[-100:]):.4f}")
    return np.mean(perdidas[-100:])


def softmax(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


def generar_muestra(modelo, tok, prompt="Grande sois, Señor", max_tokens=150, temperatura=1.0):
    """Genera texto de muestra con el modelo actual."""
    indices = np.array(tok.codificar(prompt), dtype=np.int32)
    texto = prompt
    for _ in range(max_tokens):
        contexto = indices[-512:]
        logits = modelo.forward(contexto)
        logits_ultimo = logits[-1, :] / temperatura
        probs = softmax(logits_ultimo)
        siguiente = np.random.choice(len(probs), p=probs)
        indices = np.append(indices, siguiente)
        texto += tok.decodificar([siguiente])
    return texto


def benchmark_configuracion(nombre, vocab_size, d_model, n_heads, n_layers, d_ff, datos, tok, n_pasos=500):
    print(f"\n{'='*60}")
    print(f"CONFIG: {nombre} | d_model={d_model} n_heads={n_heads} n_layers={n_layers} d_ff={d_ff}")
    print(f"{'='*60}")

    np.random.seed(42)
    modelo = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff
    )

    params = contar_parametros(modelo)
    velocidad = medir_velocidad(modelo, datos)
    memoria = sum(
        arr.nbytes for arr in [
            modelo.embedding.pesos, modelo.cabeza,
            modelo.ln_final.gamma, modelo.ln_final.beta
        ] + [
            arr for bloque in modelo.bloques
            for arr in [
                bloque.attn.Wq, bloque.attn.Wk,
                bloque.attn.Wv, bloque.attn.Wo,
                bloque.ff.W1, bloque.ff.b1,
                bloque.ff.W2, bloque.ff.b2,
                bloque.ln1.gamma, bloque.ln1.beta,
                bloque.ln2.gamma, bloque.ln2.beta,
            ]
        ]
    ) / (1024 * 1024)

    print(f"  Parámetros: {params:,} | Memoria: {memoria:.1f} MB | Velocidad: {velocidad:.2f} pasos/s")
    print(f"  Tiempo estimado 20k pasos: {20000/velocidad/60:.1f} min")
    print(f"\n  Entrenando {n_pasos} pasos...")

    loss_final = entrenar_rapido(modelo, datos, n_pasos)

    print(f"\n  Pérdida final ({n_pasos} pasos): {loss_final:.4f}")
    print(f"\n  Generación con temperatura 0.8:")
    print(f"  {'-'*50}")
    texto_08 = generar_muestra(modelo, tok, temperatura=0.8)
    print(f"  {texto_08}")
    print(f"\n  Generación con temperatura 1.2:")
    print(f"  {'-'*50}")
    texto_12 = generar_muestra(modelo, tok, temperatura=1.2)
    print(f"  {texto_12}")

    return {
        'nombre': nombre,
        'params': params,
        'memoria_mb': memoria,
        'pasos_por_segundo': velocidad,
        'loss_500_pasos': loss_final,
        'tiempo_20k_min': 20000/velocidad/60,
    }


if __name__ == "__main__":
    np.random.seed(42)

    tok = TokenizadorCharLevel()
    tok.cargar(Path('data/corpus/vocabulario.json'))
    texto = Path('data/corpus/confesiones_final.txt').read_text(encoding='utf-8')
    datos = np.array(tok.codificar(texto), dtype=np.int32)
    vocab_size = tok.vocab_size

    print(f"BENCHMARK COMPLETO — TransformerLM desde cero")
    print(f"Corpus: {len(datos):,} tokens | Vocabulario: {vocab_size} chars")
    print(f"Entrenamiento por configuración: 500 pasos")

    configuraciones = [
        ("tiny",         128,  4, 2,  512),
        ("small",        256,  8, 4, 1024),
        ("medium-deep",  256,  8, 6, 1024),
    ]

    resultados = []
    for nombre, d_model, n_heads, n_layers, d_ff in configuraciones:
        r = benchmark_configuracion(
            nombre, vocab_size, d_model, n_heads, n_layers, d_ff,
            datos, tok, n_pasos=500
        )
        resultados.append(r)

    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"{'Nombre':<15} {'Params':>10} {'MB':>6} {'Steps/s':>8} {'Loss 500':>9} {'20k min':>8}")
    print(f"{'-'*60}")
    for r in resultados:
        print(f"{r['nombre']:<15} {r['params']:>10,} {r['memoria_mb']:>6.1f} "
              f"{r['pasos_por_segundo']:>8.2f} {r['loss_500_pasos']:>9.4f} "
              f"{r['tiempo_20k_min']:>8.1f}")