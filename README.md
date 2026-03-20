# BitNetSharp

A C# implementation of Microsoft BitNet – the 1-bit/ternary-weight transformer architecture for large language models.

## Overview

BitNet replaces all dense (linear) layers in a standard Transformer with **BitLinear** layers whose weights are quantized to ternary values `{-1, 0, +1}` during training (absmean quantization), and activations are quantized to 8-bit integers (absmax quantization).  The result is a model that requires roughly **16× less memory** than FP16 while maintaining competitive performance.

Reference: [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

## Architecture

```
token_ids
    │
    ▼
Token Embedding  (vocabSize × embedDim)
    │
    ▼
┌─────────────────────────────┐
│   BitNetBlock (× NumLayers) │
│  ─────────────────────────  │
│  RMSNorm                    │
│  └─ BitNetAttention         │
│      ├─ BitLinear (Q)       │
│      ├─ BitLinear (K)       │
│      ├─ BitLinear (V)       │
│      ├─ RoPE (Q, K)         │
│      ├─ Scaled dot-product  │
│      │  attention + causal  │
│      │  mask                │
│      └─ BitLinear (Out)     │
│  + Residual                 │
│  RMSNorm                    │
│  └─ BitNetFFN               │
│      ├─ BitLinear (expand)  │
│      ├─ Squared ReLU        │
│      └─ BitLinear (project) │
│  + Residual                 │
└─────────────────────────────┘
    │
    ▼
RMSNorm
    │
    ▼
LM Head  (embedDim → vocabSize)
    │
    ▼
Logits  (seqLen × vocabSize)
```

### Key Components

| Component | File | Description |
|---|---|---|
| `Tensor` | `Core/Tensor.cs` | Lightweight flat-array tensor with 1-D/2-D ops |
| `Quantization` | `Core/Quantization.cs` | Absmean (weights) and absmax (activations) quantization |
| `RMSNorm` | `Core/RMSNorm.cs` | Root Mean Square Layer Normalization |
| `RotaryEmbedding` | `Core/RotaryEmbedding.cs` | Rotary Position Embeddings (RoPE) |
| `BitLinear` | `Layers/BitLinear.cs` | Ternary-weight linear layer with SubLN |
| `BitNetAttention` | `Layers/BitNetAttention.cs` | Multi-head self-attention with BitLinear + RoPE |
| `BitNetFFN` | `Layers/BitNetFFN.cs` | Feed-forward network with BitLinear + Squared ReLU |
| `BitNetBlock` | `Layers/BitNetBlock.cs` | Single transformer block (Attention + FFN + residuals) |
| `BitNetConfig` | `Models/BitNetConfig.cs` | Model hyperparameter configuration |
| `BitNetModel` | `Models/BitNetModel.cs` | Full model: embedding → blocks → LM head |

## Quick Start

```csharp
using BitNetSharp.Models;
using BitNetSharp.Core;

// Create a small model
var config = BitNetConfig.Small();   // vocabSize=1000, embedDim=64, 2 layers, 4 heads
var model  = new BitNetModel(config);

// Run a forward pass
int[] tokenIds = { 1, 42, 7 };
Tensor logits = model.Forward(tokenIds);   // shape: (3, 1000)

// Greedy next-token prediction
int nextToken = model.PredictNextToken(tokenIds);
Console.WriteLine($"Predicted next token: {nextToken}");
```

### Predefined Configurations

```csharp
var small  = BitNetConfig.Small();   // 64-dim, 2 layers,  4 heads, ffn=256
var medium = BitNetConfig.Medium();  // 512-dim, 6 layers,  8 heads, ffn=2048
var large  = BitNetConfig.Large();   // 1024-dim,24 layers,16 heads, ffn=4096
```

## Solution Structure

```
BitNetSharp.slnx
├── src/BitNetSharp/          # Library
│   ├── Core/
│   │   ├── Tensor.cs
│   │   ├── Quantization.cs
│   │   ├── RMSNorm.cs
│   │   └── RotaryEmbedding.cs
│   ├── Layers/
│   │   ├── BitLinear.cs
│   │   ├── BitNetAttention.cs
│   │   ├── BitNetFFN.cs
│   │   └── BitNetBlock.cs
│   └── Models/
│       ├── BitNetConfig.cs
│       └── BitNetModel.cs
└── tests/BitNetSharp.Tests/  # xUnit tests
    ├── TensorTests.cs
    ├── QuantizationTests.cs
    ├── RMSNormTests.cs
    ├── RotaryEmbeddingTests.cs
    ├── BitLinearTests.cs
    └── BitNetModelTests.cs
```

## Building and Testing

```bash
dotnet build BitNetSharp.slnx
dotnet test  BitNetSharp.slnx
```

## License

See [LICENSE](LICENSE).
