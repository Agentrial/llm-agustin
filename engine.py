import numpy as np


def softmax(x):
    """
    Softmax numéricamente estable.

    Args:
        x: array de forma (seq_len, vocab_size)

    Returns:
        array de la misma forma con valores entre 0 y 1
        que suman 1 a lo largo del último eje
    """
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(logits, targets):
    """
    Función de pérdida cross-entropy para modelado de lenguaje.

    Para cada posición en la secuencia, mide qué tan equivocado
    estuvo el modelo al predecir el siguiente token.

    La pérdida total es el promedio sobre todas las posiciones.

    Args:
        logits:  array de forma (seq_len, vocab_size)
                 valores sin normalizar del modelo
        targets: array de forma (seq_len,)
                 índices de los tokens correctos

    Returns:
        loss: escalar — pérdida promedio sobre la secuencia
        probs: array de forma (seq_len, vocab_size)
               probabilidades después del softmax
    """
    seq_len = logits.shape[0]

    # Convertir logits a probabilidades
    probs = softmax(logits)

    # Extraer la probabilidad asignada a cada token correcto
    # probs[i, targets[i]] es la probabilidad del token correcto
    # en la posición i
    probs_correctas = probs[np.arange(seq_len), targets]

    # Cross-entropy: -log de la probabilidad correcta
    # Promediada sobre toda la secuencia
    loss = -np.mean(np.log(probs_correctas + 1e-10))

    return loss, probs


def gradiente_cross_entropy(probs, targets):
    """
    Gradiente de la pérdida cross-entropy respecto a los logits.


    Args:
        probs:   array de forma (seq_len, vocab_size)
                 probabilidades después del softmax
        targets: array de forma (seq_len,)
                 índices de los tokens correctos

    Returns:
        grad: array de forma (seq_len, vocab_size)
              gradiente de la pérdida respecto a los logits
    """
    seq_len = probs.shape[0]

    # Empezamos con una copia de las probabilidades
    grad = probs.copy()

    # Restar 1 en la posición del token correcto
    # Esto implementa: probs - one_hot(targets)
    grad[np.arange(seq_len), targets] -= 1

    # Dividir por seq_len porque la pérdida es un promedio
    grad = grad / seq_len

    return grad