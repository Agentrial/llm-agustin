import numpy as np


class Embedding:
    """
    Capa de embedding — convierte índices de tokens en vectores densos.

    Args:
        vocab_size: cantidad de tokens en el vocabulario
        d_model: dimensión de cada vector de embedding
    """

    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pesos = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)

    def forward(self, indices):
        self.indices = indices
        return self.pesos[indices]

    def backward(self, grad_output):
        self.grad_pesos = np.zeros_like(self.pesos)
        np.add.at(self.grad_pesos, self.indices, grad_output)


class PositionalEncoding:
    """
    Codificación posicional sinusoidal.
    'Attention Is All You Need' (Vaswani et al., 2017).

    Args:
        d_model: dimensión del modelo
        max_len: longitud máxima de secuencia soportada
    """

    def __init__(self, d_model, max_len=512):
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        posiciones = np.arange(max_len).reshape(-1, 1)
        denominadores = np.power(
            10000.0,
            np.arange(0, d_model, 2) / d_model
        )
        pe[:, 0::2] = np.sin(posiciones / denominadores)
        pe[:, 1::2] = np.cos(posiciones / denominadores)
        self.pe = pe

    def forward(self, x):
        if x.ndim == 2:
            seq_len = x.shape[0]
        else:
            seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]

    def backward(self, grad_output):
        # El positional encoding es una suma de constantes
        # El gradiente pasa directo sin modificación
        return grad_output


class SelfAttention:
    """
    Mecanismo de self-attention con máscara causal.

    Args:
        d_model: dimensión del modelo
        n_heads: número de cabezas de atención
    """

    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, \
            "d_model debe ser divisible por n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        escala = np.sqrt(self.d_k)
        self.Wq = np.random.randn(d_model, d_model) / escala
        self.Wk = np.random.randn(d_model, d_model) / escala
        self.Wv = np.random.randn(d_model, d_model) / escala
        self.Wo = np.random.randn(d_model, d_model) / escala

    def _softmax(self, x):
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _mascara_causal(self, seq_len):
        mascara = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mascara * -1e9

    def forward(self, x):
        self.x = x
        seq_len, d_model = x.shape

        # Proyectar a Q, K, V
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # Dividir en cabezas
        Q = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Guardar para backward
        self.Q = Q
        self.K = K
        self.V = V

        # Scores de atención
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)
        scores = scores + self._mascara_causal(seq_len)

        # Softmax
        self.pesos_attn = self._softmax(scores)

        # Combinar con Values
        salida = self.pesos_attn @ V

        # Reunir cabezas
        salida = salida.transpose(1, 0, 2).reshape(seq_len, d_model)
        self.salida_pre_wo = salida

        return salida @ self.Wo


class FeedForward:
    """
    Red feed-forward aplicada token por token.

    Args:
        d_model: dimensión de entrada y salida
        d_ff: dimensión interna (típicamente 4 × d_model)
    """

    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.x = x
        self.h = self._relu(x @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2

    def backward(self, grad_output):
        # Gradientes segunda capa
        self.grad_W2 = self.h.T @ grad_output
        self.grad_b2 = np.sum(grad_output, axis=0)
        grad_h = grad_output @ self.W2.T

        # Gradiente a través de ReLU
        # ReLU'(x) = 1 si x > 0, 0 si x <= 0
        grad_relu = grad_h * (self.h > 0)

        # Gradientes primera capa
        self.grad_W1 = self.x.T @ grad_relu
        self.grad_b1 = np.sum(grad_relu, axis=0)
        grad_x = grad_relu @ self.W1.T

        return grad_x


class LayerNorm:
    """
    Normalización de capa.

    Args:
        d_model: dimensión sobre la que se normaliza
        eps: valor pequeño para estabilidad numérica
    """

    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        self.x = x
        self.media = np.mean(x, axis=-1, keepdims=True)
        self.std = np.std(x, axis=-1, keepdims=True)
        self.x_norm = (x - self.media) / (self.std + self.eps)
        return self.gamma * self.x_norm + self.beta

    def backward(self, grad_output):
        # Gradientes de gamma y beta
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)

        # Gradiente respecto a x_norm
        grad_x_norm = grad_output * self.gamma

        # Gradiente respecto a x — propagación a través de la normalización
        N = self.d_model
        grad_x = (1 / (N * (self.std + self.eps))) * (
            N * grad_x_norm
            - np.sum(grad_x_norm, axis=-1, keepdims=True)
            - self.x_norm * np.sum(grad_x_norm * self.x_norm, axis=-1, keepdims=True)
        )
        return grad_x


class BloqueTransformer:
    """
    Un bloque completo del Transformer decoder.

    Estructura:
        x = x + SelfAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    Args:
        d_model: dimensión del modelo
        n_heads: número de cabezas de atención
        d_ff: dimensión interna del feed-forward
    """

    def __init__(self, d_model, n_heads, d_ff):
        self.attn = SelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x):
        self.x_entrada = x

        # Sub-bloque 1: atención con conexión residual
        self.x_ln1 = self.ln1.forward(x)
        self.x_attn = self.attn.forward(self.x_ln1)
        x = x + self.x_attn

        # Sub-bloque 2: feed-forward con conexión residual
        self.x_ln2 = self.ln2.forward(x)
        self.x_ff = self.ff.forward(self.x_ln2)
        x = x + self.x_ff

        return x

    def backward(self, grad_output):
        # Backward sub-bloque 2
        # Conexión residual: grad pasa directo + grad del ff
        grad_ff = self.ff.backward(grad_output)
        grad_ln2 = self.ln2.backward(grad_ff)
        grad_output = grad_output + grad_ln2

        # Backward sub-bloque 1
        grad_attn = self.attn_backward(grad_output)
        grad_ln1 = self.ln1.backward(grad_attn)
        grad_output = grad_output + grad_ln1

        return grad_output

    def attn_backward(self, grad_output):
        # Gradiente a través de Wo
        attn = self.attn
        grad_salida_pre_wo = grad_output @ attn.Wo.T
        attn.grad_Wo = attn.salida_pre_wo.T @ grad_output

        # Separar en cabezas
        seq_len = grad_salida_pre_wo.shape[0]
        grad_salida = grad_salida_pre_wo.reshape(
            seq_len, attn.n_heads, attn.d_k
        ).transpose(1, 0, 2)

        # Gradiente respecto a V
        attn.grad_V = attn.pesos_attn.transpose(0, 2, 1) @ grad_salida

        # Gradiente respecto a pesos_attn
        grad_pesos = grad_salida @ attn.V.transpose(0, 2, 1)

        # Gradiente a través del softmax
        p = attn.pesos_attn
        grad_scores = p * (grad_pesos - np.sum(grad_pesos * p, axis=-1, keepdims=True))
        grad_scores = grad_scores / np.sqrt(attn.d_k)

        # Gradientes respecto a Q y K
        attn.grad_Q = grad_scores @ attn.K
        attn.grad_K = grad_scores.transpose(0, 2, 1) @ attn.Q

        # Reunir cabezas y calcular gradientes de Wq, Wk, Wv
        def reunir(g):
            return g.transpose(1, 0, 2).reshape(seq_len, attn.d_model)

        grad_Q = reunir(attn.grad_Q)
        grad_K = reunir(attn.grad_K)
        grad_V = reunir(attn.grad_V)

        attn.grad_Wq = attn.x.T @ grad_Q
        attn.grad_Wk = attn.x.T @ grad_K
        attn.grad_Wv = attn.x.T @ grad_V

        return grad_Q @ attn.Wq.T + grad_K @ attn.Wk.T + grad_V @ attn.Wv.T


class TransformerLM:
    """
    Modelo de lenguaje basado en Transformer decoder.

    Args:
        vocab_size: tamaño del vocabulario
        d_model: dimensión del modelo
        n_heads: número de cabezas de atención
        n_layers: número de bloques transformer
        d_ff: dimensión interna del feed-forward
        max_len: longitud máxima de secuencia
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=512):
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.bloques = [
            BloqueTransformer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
        self.ln_final = LayerNorm(d_model)
        self.cabeza = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)

    def forward(self, indices):
        x = self.embedding.forward(indices)
        x = self.pos_encoding.forward(x)

        for bloque in self.bloques:
            x = bloque.forward(x)

        x = self.ln_final.forward(x)
        self.x_antes_cabeza = x
        logits = x @ self.cabeza

        return logits

    def backward(self, grad_logits):
        # Gradiente de la cabeza
        self.grad_cabeza = self.x_antes_cabeza.T @ grad_logits
        grad_x = grad_logits @ self.cabeza.T

        # Backward LayerNorm final
        grad_x = self.ln_final.backward(grad_x)

        # Backward bloques en orden inverso
        for bloque in reversed(self.bloques):
            grad_x = bloque.backward(grad_x)

        # Backward positional encoding
        grad_x = self.pos_encoding.backward(grad_x)

        # Backward embedding
        self.embedding.backward(grad_x)