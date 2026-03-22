# Laboratório 05 — Treinamento Fim-a-Fim do Transformer

 Instituto de Ensino Superior ICEV.

## Estrutura dos Arquivos

```
transformer_lab05/
├── attention.py       # MultiHeadAttention + scaled_dot_product_attention
├── add_norm.py        # Camada Add & Norm (residual + LayerNorm)
├── ffn.py             # Feed-Forward Network
├── utils.py           # Máscaras causal/padding + PositionalEncoding
├── encoder.py         # EncoderBlock + Encoder (N camadas)
├── decoder.py         # DecoderBlock + Decoder (N camadas)
├── transformer.py     # Classe principal Transformer
├── dataset.py         # Tarefa 1 e 2: Dataset + Tokenização
├── train.py           # Tarefa 3: Training Loop
├── inference.py       # Tarefa 4: Loop Autoregressivo + Overfitting Test
├── requirements.txt
└── README.md
```

## Como Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Treinar e testar (roda tudo)
```bash
python inference.py
```

### 3. Apenas treinar
```bash
python train.py
```

## Tarefas Implementadas

### Tarefa 1 — Dataset Real (Hugging Face)
- Dataset: `Helsinki-NLP/opus_books` (par `en-pt`)
- Subconjunto de 1.000 frases conforme o enunciado
- Arquivo: `dataset.py`

### Tarefa 2 — Tokenização Básica
- Tokenizador: `bert-base-multilingual-cased` (multilingual, cobre en e pt)
- Conversão de pares de frases em listas de inteiros
- Tokens especiais: `[CLS]` = `<START>`, `[SEP]` = `<EOS>`
- Padding com zeros para equalizar comprimentos no batch
- Arquivo: `dataset.py`

### Tarefa 3 — Training Loop
- Modelo: `d_model=128`, `h=4`, `N=2` (viável para CPU/Colab)
- Loss: `CrossEntropyLoss` com `ignore_index=pad_idx`
- Otimizador: `Adam` com `betas=(0.9, 0.98)` (paper original)
- 15 epochs com impressão do Loss a cada época
- Arquivo: `train.py`

### Tarefa 4 — Prova de Fogo (Overfitting Test)
- Loop autoregressivo token a token
- Encoder roda 1 única vez; Decoder roda incrementalmente
- Para ao gerar `[SEP]` = `<EOS>`
- Arquivo: `inference.py`

## Ferramentas de IA Generativa Utilizadas
Conforme exigido pelo enunciado (identificação no README):

- **Claude (Anthropic)** — auxiliou na estruturação das Tarefas 1 e 2
  (manipulação do dataset HuggingFace e tokenização)
- O fluxo de **Forward/Backward (Tarefa 3)** interage estritamente com
  as classes `EncoderBlock`, `DecoderBlock` e `Transformer` construídas
  nos laboratórios anteriores (migradas para PyTorch para habilitar
  backpropagation automático).
