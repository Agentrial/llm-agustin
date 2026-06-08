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

        # Inicialización con distribución normal escalada
        # Dividir por sqrt(d_model) mantiene la varianza estable
        # independientemente de la dimensión
        self.pesos = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)

    def forward(self, indices):
        """
        Dado un array de índices, devuelve sus vectores de embedding.

        Args:
            indices: array de forma (batch_size, seq_len) o (seq_len,)

        Returns:
            array de forma (batch_size, seq_len, d_model) o (seq_len, d_model)
        """
        return self.pesos[indices]
class PositionalEncoding:
    """
    Codificación posicional sinusoidal del paper original
    'Attention Is All You Need' (Vaswani et al., 2017).


    Args:
        d_model: dimensión del modelo
        max_len: longitud máxima de secuencia soportada
    """

    def __init__(self, d_model, max_len=512):
        self.d_model = d_model

        # Construir la matriz de encodings de forma (max_len, d_model)
        pe = np.zeros((max_len, d_model))

        # Vector de posiciones: [0, 1, 2, ..., max_len-1]
        # shape: (max_len, 1)
        posiciones = np.arange(max_len).reshape(-1, 1)

        # Denominadores para cada par de dimensiones
        # shape: (d_model/2,)
        denominadores = np.power(
            10000.0,
            np.arange(0, d_model, 2) / d_model
        )

        # Dimensiones pares — seno
        pe[:, 0::2] = np.sin(posiciones / denominadores)

        # Dimensiones impares — coseno
        pe[:, 1::2] = np.cos(posiciones / denominadores)

        self.pe = pe

    def forward(self, x):
        """
        Suma el positional encoding a los embeddings.

        Args:
            x: array de forma (seq_len, d_model) o
               (batch_size, seq_len, d_model)

        Returns:
            array de la misma forma que x
        """
        if x.ndim == 2:
            seq_len = x.shape[0]
            return x + self.pe[:seq_len, :]
        else:
            seq_len = x.shape[1]
            return x + self.pe[:seq_len, :]

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
        self.d_k = d_model // n_heads  # dimensión por cabeza

        # Matrices de proyección para Q, K, V
        # Cada una transforma (d_model) → (d_model)
        escala = np.sqrt(self.d_k)
        self.Wq = np.random.randn(d_model, d_model) / escala
        self.Wk = np.random.randn(d_model, d_model) / escala
        self.Wv = np.random.randn(d_model, d_model) / escala

        # Proyección final que combina las cabezas
        self.Wo = np.random.randn(d_model, d_model) / escala

    def _softmax(self, x):
        """
        Softmax numéricamente estable.

        """
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _mascara_causal(self, seq_len):
        """
        Crea una máscara triangular inferior.

        Posiciones futuras reciben -inf para que después
        del softmax tengan peso 0.

        Ejemplo para seq_len=4:
            [[0,   -inf, -inf, -inf],
             [0,   0,    -inf, -inf],
             [0,   0,    0,    -inf],
             [0,   0,    0,    0   ]]
        """
        mascara = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mascara * -1e9

    def forward(self, x):
        """
        Forward pass del self-attention.

        Args:
            x: array de forma (seq_len, d_model)

        Returns:
            array de forma (seq_len, d_model)
        """
        seq_len, d_model = x.shape

        # Paso 1 — Proyectar a Q, K, V
        Q = x @ self.Wq  # (seq_len, d_model)
        K = x @ self.Wk  # (seq_len, d_model)
        V = x @ self.Wv  # (seq_len, d_model)

        # Paso 2 — Dividir en cabezas
        # Reshape: (seq_len, d_model) → (seq_len, n_heads, d_k)
        # Transpose: (seq_len, n_heads, d_k) → (n_heads, seq_len, d_k)
        Q = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Paso 3 — Calcular scores de atención
        # (n_heads, seq_len, d_k) @ (n_heads, d_k, seq_len)
        # → (n_heads, seq_len, seq_len)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)

        # Paso 4 — Aplicar máscara causal
        scores = scores + self._mascara_causal(seq_len)

        # Paso 5 — Softmax
        pesos = self._softmax(scores)  # (n_heads, seq_len, seq_len)

        # Paso 6 — Combinar con Values
        # (n_heads, seq_len, seq_len) @ (n_heads, seq_len, d_k)
        # → (n_heads, seq_len, d_k)
        salida = pesos @ V

        # Paso 7 — Reunir cabezas
        # Transpose: (n_heads, seq_len, d_k) → (seq_len, n_heads, d_k)
        # Reshape: (seq_len, n_heads, d_k) → (seq_len, d_model)
        salida = salida.transpose(1, 0, 2).reshape(seq_len, d_model)

        # Paso 8 — Proyección final
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

        # Primera capa: d_model → d_ff
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)

        # Segunda capa: d_ff → d_model
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)

    def _relu(self, x):
        """
        Rectified Linear Unit.
        Reemplaza valores negativos con cero.
        ReLU(x) = max(0, x)
        """
        return np.maximum(0, x)

    def forward(self, x):
        """
        Args:
            x: array de forma (seq_len, d_model)

        Returns:
            array de forma (seq_len, d_model)
        """
        # Primera capa lineal + ReLU
        h = self._relu(x @ self.W1 + self.b1)  # (seq_len, d_ff)

        # Segunda capa lineal
        return h @ self.W2 + self.b2            # (seq_len, d_model)

class LayerNorm:
    """
    Normalización de capa aplicada sobre la dimensión d_model.

    Args:
        d_model: dimensión sobre la que se normaliza
        eps: valor pequeño para estabilidad numérica
    """

    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps

        # Parámetros aprendibles
        # gamma inicializado en 1 — sin escala al inicio
        # beta inicializado en 0 — sin desplazamiento al inicio
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        """
        Args:
            x: array de forma (seq_len, d_model)

        Returns:
            array de forma (seq_len, d_model)
        """
        # Calcular media y std sobre la última dimensión
        # keepdims=True mantiene la forma para broadcasting
        media = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)

        # Normalizar
        x_norm = (x - media) / (std + self.eps)

        # Escalar y desplazar con parámetros aprendibles
        return self.gamma * x_norm + self.beta

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
        """
        Args:
            x: array de forma (seq_len, d_model)

        Returns:
            array de forma (seq_len, d_model)
        """
        # Sub-bloque 1: atención con conexión residual
        x = x + self.attn.forward(self.ln1.forward(x))

        # Sub-bloque 2: feed-forward con conexión residual
        x = x + self.ff.forward(self.ln2.forward(x))

        return x

class TransformerLM:
    """
    Modelo de lenguaje basado en Transformer decoder.

    Arquitectura completa:
        1. Embedding — índices → vectores densos
        2. Positional Encoding — agrega información de posición
        3. N bloques Transformer — self-attention + feed-forward
        4. LayerNorm final — normalización antes de la proyección
        5. Cabeza de lenguaje — vectores → logits sobre vocabulario

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

        # Capas
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.bloques = [
            BloqueTransformer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
        self.ln_final = LayerNorm(d_model)

        # Cabeza de lenguaje — proyecta d_model → vocab_size
        # Sin bias — práctica común en LLMs modernos
        self.cabeza = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)

    def forward(self, indices):
        """
        Forward pass completo.

        Args:
            indices: array de forma (seq_len,) con índices de tokens

        Returns:
            logits: array de forma (seq_len, vocab_size)
                    valores sin normalizar — uno por token del vocabulario
        """
        # 1. Embedding + positional encoding
        x = self.embedding.forward(indices)
        x = self.pos_encoding.forward(x)

        # 2. Pasar por cada bloque transformer
        for bloque in self.bloques:
            x = bloque.forward(x)

        # 3. Normalización final
        x = self.ln_final.forward(x)

        # 4. Proyección a logits sobre el vocabulario
        logits = x @ self.cabeza

        return logits