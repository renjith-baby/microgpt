# Quick Reference: GPT Components

## 🎯 Key Concepts (One-Liners)

| Concept | What It Does | Where in Code |
|---------|--------------|---------------|
| **Autograd** | Automatically computes gradients (derivatives) | `Value` class (lines 109-197) |
| **Tokenization** | Converts text → numbers | Lines 343-347 |
| **Embedding** | Converts token IDs → dense vectors | Lines 276-278 |
| **Attention** | Lets positions "look at" other positions | Lines 286-316 |
| **Softmax** | Converts scores → probabilities | Lines 217-222 |
| **Loss** | Measures prediction error | Line 398 |
| **Backprop** | Propagates error backwards | `Value.backward()` (line 175) |
| **Adam** | Updates weights with adaptive learning rate | Lines 408-414 |

## 🔍 Debugging Checklist

### Forward Pass Flow
```
Input Token → Embedding → Normalize → Attention → MLP → Logits → Softmax → Probabilities
```

### Key Variables to Watch
- `x`: Hidden state (changes through layers)
- `logits`: Raw prediction scores (before softmax)
- `probs`: Probabilities (after softmax, sum to 1)
- `loss.data`: Current loss value (should decrease)
- `p.grad`: Gradient for parameter `p` (how much to adjust)

### Important Breakpoints
1. **Line 276**: Start of forward pass (see embeddings)
2. **Line 286-288**: Q, K, V computation (attention setup)
3. **Line 306**: Attention weights (which positions attend to which)
4. **Line 330**: Final logits (raw predictions)
5. **Line 398**: Loss calculation (how wrong is the prediction?)
6. **Line 405**: Backward pass (gradient computation)
7. **Line 414**: Weight update (learning happens here)

## 📊 Data Flow Example

**Training Step:**
```
Input: "emma"
  ↓ Tokenize
[BOS, 'e', 'm', 'm', 'a', BOS]
  ↓ For each position
Position 0: BOS → predict 'e' → loss₁
Position 1: 'e' → predict 'm' → loss₂
Position 2: 'm' → predict 'm' → loss₃
Position 3: 'm' → predict 'a' → loss₄
  ↓ Average
loss = (loss₁ + loss₂ + loss₃ + loss₄) / 4
  ↓ Backward
Compute gradients for all weights
  ↓ Update
Adjust weights using Adam optimizer
```

**Inference Step:**
```
Start: BOS
  ↓ Forward
Logits: [2.1, 0.5, -0.3, ...] (27 scores)
  ↓ Softmax
Probs: [0.15, 0.05, 0.02, ...] (27 probabilities)
  ↓ Sample
Pick token based on probabilities (e.g., 'm')
  ↓ Add to sequence
[BOS, 'm']
  ↓ Repeat
Forward again with context → sample next token → ...
```

## 🧮 Math Formulas (Simplified)

**Attention:**
```
scores = (Q · K^T) / √d
weights = softmax(scores)
output = weights · V
```

**Softmax:**
```
exp_i = exp(logit_i - max(logits))
prob_i = exp_i / sum(exp)
```

**Loss (Cross-Entropy):**
```
loss = -log(prob[target])
```

**Adam Update:**
```
m = β₁·m + (1-β₁)·gradient
v = β₂·v + (1-β₂)·gradient²
weight = weight - lr·m̂ / (√v̂ + ε)
```

## 🎓 Learning Path

1. **Start here:** Set breakpoint at line 337 (`main()`)
2. **Step through:** One training iteration (lines 374-421)
3. **Deep dive:** `gptForward` function (lines 265-331)
4. **Understand autograd:** `Value.backward()` (lines 175-196)
5. **See learning:** Weight updates (lines 408-414)

## 💡 Common Questions

**Q: Why does attention need Q, K, V?**
A: Q = "What am I looking for?", K = "What do I represent?", V = "What info do I have?"
   Attention computes: "How much should I attend to each position?"

**Q: Why residual connections?**
A: Allows gradients to flow directly through, prevents vanishing gradients.

**Q: Why normalize (RMSNorm)?**
A: Keeps activations in reasonable range, stabilizes training.

**Q: Why softmax?**
A: Converts arbitrary scores into valid probabilities (sum to 1).

**Q: What does Adam do differently?**
A: Adapts learning rate per parameter based on gradient history.

## 🔗 Code → Guide Mapping

| Code Section | Guide Section |
|--------------|---------------|
| `Value` class | Part 2.1: Autograd |
| Tokenization | Part 2.2: Tokenization |
| Embeddings | Part 2.3: Embeddings |
| `gptForward` | Part 2.4: Attention, Part 3.1: Forward Pass |
| Training loop | Part 3.2: Training Loop |
| Adam optimizer | Part 2.9: Adam Optimizer |

---

**Tip:** Keep this file open while debugging, and refer to `GUIDE.md` for detailed explanations!

