import numpy as np


class Adam:
    """
    Optimizador Adam (Adaptive Moment Estimation).
    Kingma & Ba, 2014.

    Mantiene promedios móviles de gradientes (m) y gradientes
    al cuadrado (v) para adaptar el learning rate por parámetro.

    Args:
        lr:    learning rate (típicamente 3e-4)
        beta1: decaimiento del primer momento (típicamente 0.9)
        beta2: decaimiento del segundo momento (típicamente 0.999)
        eps:   estabilidad numérica (típicamente 1e-8)
    """

    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0          # contador de pasos
        self.m = {}         # momentos de primer orden
        self.v = {}         # momentos de segundo orden

    def actualizar(self, nombre, peso, grad):
        """
        Actualiza un peso usando Adam.

        Returns:
            peso actualizado
        """
        # Inicializar momentos en cero la primera vez
        if nombre not in self.m:
            self.m[nombre] = np.zeros_like(peso)
            self.v[nombre] = np.zeros_like(peso)

        # Actualizar momentos
        self.m[nombre] = self.beta1 * self.m[nombre] + (1 - self.beta1) * grad
        self.v[nombre] = self.beta2 * self.v[nombre] + (1 - self.beta2) * grad ** 2

        # Bias correction
        m_hat = self.m[nombre] / (1 - self.beta1 ** self.t)
        v_hat = self.v[nombre] / (1 - self.beta2 ** self.t)

        # Actualizar peso
        peso -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return peso

    def paso(self, modelo):
        """
        Actualiza todos los pesos del modelo.

        Args:
            modelo: instancia de TransformerLM
        """
        self.t += 1

        # Cabeza de lenguaje
        modelo.cabeza = self.actualizar(
            'cabeza', modelo.cabeza, modelo.grad_cabeza
        )

        # LayerNorm final
        self._actualizar_ln(modelo.ln_final, 'ln_final')

        # Bloques transformer
        for i, bloque in enumerate(modelo.bloques):
            prefijo = f'bloque_{i}'

            # Atención
            attn = bloque.attn
            attn.Wq = self.actualizar(f'{prefijo}_Wq', attn.Wq, attn.grad_Wq)
            attn.Wk = self.actualizar(f'{prefijo}_Wk', attn.Wk, attn.grad_Wk)
            attn.Wv = self.actualizar(f'{prefijo}_Wv', attn.Wv, attn.grad_Wv)
            attn.Wo = self.actualizar(f'{prefijo}_Wo', attn.Wo, attn.grad_Wo)

            # FeedForward
            ff = bloque.ff
            ff.W1 = self.actualizar(f'{prefijo}_W1', ff.W1, ff.grad_W1)
            ff.b1 = self.actualizar(f'{prefijo}_b1', ff.b1, ff.grad_b1)
            ff.W2 = self.actualizar(f'{prefijo}_W2', ff.W2, ff.grad_W2)
            ff.b2 = self.actualizar(f'{prefijo}_b2', ff.b2, ff.grad_b2)

            # LayerNorms del bloque
            self._actualizar_ln(bloque.ln1, f'{prefijo}_ln1')
            self._actualizar_ln(bloque.ln2, f'{prefijo}_ln2')

        # Embedding
        modelo.embedding.pesos = self.actualizar(
            'embedding', modelo.embedding.pesos, modelo.embedding.grad_pesos
        )

    def _actualizar_ln(self, ln, nombre):
        """Actualiza los parámetros de una LayerNorm."""
        ln.gamma = self.actualizar(f'{nombre}_gamma', ln.gamma, ln.grad_gamma)
        ln.beta = self.actualizar(f'{nombre}_beta', ln.beta, ln.grad_beta)