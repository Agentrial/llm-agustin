# LLM desde cero — Confesiones de San Agustín

Implementación de un Large Language Model desde cero, usando solo Python y NumPy.
Sin frameworks de ML. Cada componente construido e implementado a mano.

## Dominio
Texto especializado en las Confesiones de San Agustín (siglo IV, traducción al español del siglo XVIII).

## Stack
- Python 3.14
- NumPy (operaciones matriciales únicamente)
- - pdfplumber (extracción de texto del PDF — solo para el pipeline de datos)

## Fases
1. Dataset — extracción y limpieza del PDF
2. Tokenizador — char-level y BPE desde cero
3. Arquitectura Transformer decoder en NumPy
4. Backpropagation manual
5. Loop de entrenamiento
6. Inferencia y generación de texto