// Web Worker for GPT model training and inference
// Ported from microgpt.ts - no Node.js dependencies

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------

const FALLBACK_NAMES_DATASET = [
    "emma",
    "olivia",
    "ava",
    "isabella",
    "sophia",
    "charlotte",
    "mia",
    "amelia",
    "harper",
    "evelyn"
];

async function loadDocs() {
    const inputUrl = new URL("input.txt", self.location.href).toString();
    try {
        const response = await fetch(inputUrl, { cache: "no-store" });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const text = await response.text();
        const loadedDocs = text
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter((line) => line.length > 0);

        if (loadedDocs.length === 0) {
            throw new Error("input.txt is empty");
        }
        return loadedDocs;
    } catch (error) {
        postMessage({
            type: "warning",
            message: `Falling back to built-in dataset because input.txt could not be loaded (${error.message}).`
        });
        return [...FALLBACK_NAMES_DATASET];
    }
}

// ---------------------------------------------------------------------------
// Seeded PRNG (replaces Python's random.seed(42))
// ---------------------------------------------------------------------------

class SeededRandom {
    constructor(seed) {
        this.state = seed >>> 0;
    }

    // Mulberry32 — fast, good quality 32-bit PRNG
    next() {
        this.state = (this.state + 0x6d2b79f5) >>> 0;
        let z = this.state;
        z = Math.imul(z ^ (z >>> 15), z | 1);
        z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
        return ((z ^ (z >>> 14)) >>> 0) / 0x100000000;
    }

    /** Normally-distributed sample via Box-Muller */
    gauss(mean = 0, std = 1) {
        const u1 = this.next();
        const u2 = this.next();
        return mean + std * Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    }

    /** Shuffle an array in place (Fisher-Yates) */
    shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }

    /** Weighted random choice — returns index */
    choices(weights) {
        const total = weights.reduce((a, b) => a + b, 0);
        let r = this.next() * total;
        for (let i = 0; i < weights.length; i++) {
            r -= weights[i];
            if (r <= 0) return i;
        }
        return weights.length - 1;
    }
}

const rng = new SeededRandom(42);

// ---------------------------------------------------------------------------
// Autograd: scalar Value node
// ---------------------------------------------------------------------------

class Value {
    constructor(data, children = [], localGrads = []) {
        this.data = data;
        this.grad = 0;
        this._children = children;
        this._localGrads = localGrads;
    }

    add(other) {
        const o = other instanceof Value ? other : new Value(other);
        return new Value(this.data + o.data, [this, o], [1, 1]);
    }

    mul(other) {
        const o = other instanceof Value ? other : new Value(other);
        return new Value(this.data * o.data, [this, o], [o.data, this.data]);
    }

    pow(exp) {
        return new Value(
            Math.pow(this.data, exp),
            [this],
            [exp * Math.pow(this.data, exp - 1)]
        );
    }

    log() {
        return new Value(Math.log(this.data), [this], [1 / this.data]);
    }

    exp() {
        const e = Math.exp(this.data);
        return new Value(e, [this], [e]);
    }

    relu() {
        return new Value(
            Math.max(0, this.data),
            [this],
            [this.data > 0 ? 1 : 0]
        );
    }

    neg() {
        return this.mul(-1);
    }

    sub(other) {
        const o = other instanceof Value ? other : new Value(other);
        return this.add(o.neg());
    }

    div(other) {
        const o = other instanceof Value ? other : new Value(other);
        return this.mul(o.pow(-1));
    }

    backward() {
        const topo = [];
        const visited = new Set();

        function buildTopo(v) {
            if (!visited.has(v)) {
                visited.add(v);
                for (const child of v._children) buildTopo(child);
                topo.push(v);
            }
        }

        buildTopo(this);
        this.grad = 1;

        for (let i = topo.length - 1; i >= 0; i--) {
            const v = topo[i];
            for (let j = 0; j < v._children.length; j++) {
                v._children[j].grad += v._localGrads[j] * v.grad;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Model architecture helpers
// ---------------------------------------------------------------------------

function makeMatrix(nout, nin, std = 0.08) {
    return Array.from({ length: nout }, () =>
        Array.from({ length: nin }, () => new Value(rng.gauss(0, std)))
    );
}

function linear(x, w) {
    return w.map((row) =>
        row.reduce((acc, wi, i) => acc.add(wi.mul(x[i])), new Value(0))
    );
}

function softmax(logits) {
    const maxVal = Math.max(...logits.map((v) => v.data));
    const exps = logits.map((v) => v.sub(maxVal).exp());
    const total = exps.reduce((a, b) => a.add(b), new Value(0));
    return exps.map((e) => e.div(total));
}

function rmsnorm(x) {
    const ms = x
        .reduce((acc, xi) => acc.add(xi.mul(xi)), new Value(0))
        .div(x.length);
    const scale = ms.add(1e-5).pow(-0.5);
    return x.map((xi) => xi.mul(scale));
}

// ---------------------------------------------------------------------------
// Model state dict & GPT forward pass
// ---------------------------------------------------------------------------

function buildStateDict(nLayer, nEmbd, blockSize, vocabSize) {
    const sd = {
        wte: makeMatrix(vocabSize, nEmbd),
        wpe: makeMatrix(blockSize, nEmbd),
        lm_head: makeMatrix(vocabSize, nEmbd),
    };
    for (let i = 0; i < nLayer; i++) {
        sd[`layer${i}.attn_wq`] = makeMatrix(nEmbd, nEmbd);
        sd[`layer${i}.attn_wk`] = makeMatrix(nEmbd, nEmbd);
        sd[`layer${i}.attn_wv`] = makeMatrix(nEmbd, nEmbd);
        sd[`layer${i}.attn_wo`] = makeMatrix(nEmbd, nEmbd);
        sd[`layer${i}.mlp_fc1`] = makeMatrix(4 * nEmbd, nEmbd);
        sd[`layer${i}.mlp_fc2`] = makeMatrix(nEmbd, 4 * nEmbd);
    }
    return sd;
}

function gptForward(tokenId, posId, keys, values, sd, nLayer, nHead, headDim, nEmbd) {
    const tokEmb = sd.wte[tokenId];
    const posEmb = sd.wpe[posId];
    let x = tokEmb.map((t, i) => t.add(posEmb[i]));
    x = rmsnorm(x);

    for (let li = 0; li < nLayer; li++) {
        // --- Multi-head Attention ---
        const xResidual = x;
        x = rmsnorm(x);

        const q = linear(x, sd[`layer${li}.attn_wq`]);
        const k = linear(x, sd[`layer${li}.attn_wk`]);
        const v = linear(x, sd[`layer${li}.attn_wv`]);

        keys[li].push(k);
        values[li].push(v);

        const xAttn = [];
        for (let h = 0; h < nHead; h++) {
            const hs = h * headDim;
            const qH = q.slice(hs, hs + headDim);
            const kH = keys[li].map((ki) => ki.slice(hs, hs + headDim));
            const vH = values[li].map((vi) => vi.slice(hs, hs + headDim));
            const scale = 1 / Math.sqrt(headDim);

            const attnLogits = kH.map((kt) =>
                qH
                    .reduce((acc, qj, j) => acc.add(qj.mul(kt[j])), new Value(0))
                    .mul(scale)
            );
            const attnWeights = softmax(attnLogits);

            for (let j = 0; j < headDim; j++) {
                xAttn.push(
                    attnWeights.reduce(
                        (acc, w, t) => acc.add(w.mul(vH[t][j])),
                        new Value(0)
                    )
                );
            }
        }

        x = linear(xAttn, sd[`layer${li}.attn_wo`]);
        x = x.map((xi, i) => xi.add(xResidual[i]));

        // --- MLP ---
        const xRes2 = x;
        x = rmsnorm(x);
        x = linear(x, sd[`layer${li}.mlp_fc1`]);
        x = x.map((xi) => xi.relu());
        x = linear(x, sd[`layer${li}.mlp_fc2`]);
        x = x.map((xi, i) => xi.add(xRes2[i]));
    }

    return linear(x, sd.lm_head);
}

// ---------------------------------------------------------------------------
// State serialization
// ---------------------------------------------------------------------------

function serializeState(sd) {
    const serialized = {};
    for (const [key, matrix] of Object.entries(sd)) {
        serialized[key] = matrix.map(row => row.map(v => v.data));
    }
    return serialized;
}

function deserializeState(serialized, nLayer, nEmbd, blockSize, vocabSize) {
    const sd = {};
    for (const [key, matrix] of Object.entries(serialized)) {
        sd[key] = matrix.map(row => row.map(data => new Value(data)));
    }
    return sd;
}

// ---------------------------------------------------------------------------
// Worker state
// ---------------------------------------------------------------------------

let modelState = null;
let docs = null;
let allChars = null;
let BOS = null;
let vocabSize = null;
let nLayer = 1;
let nEmbd = 16;
let blockSize = 16;
let nHead = 4;
let headDim = nEmbd / nHead;
let runId = 0;

// ---------------------------------------------------------------------------
// Initialize model
// ---------------------------------------------------------------------------

async function initModel() {
    // Load dataset
    docs = await loadDocs();
    rng.shuffle(docs);

    // Tokenizer
    allChars = [...new Set(docs.join(""))].sort();
    BOS = allChars.length;
    vocabSize = allChars.length + 1;

    // Build model
    modelState = buildStateDict(nLayer, nEmbd, blockSize, vocabSize);
    runId = 0;

    postMessage({ type: 'ready', vocabSize, numDocs: docs.length });
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

function train(learningRate, numSteps, temperature) {
    if (!modelState) {
        postMessage({ type: 'error', message: 'Model not initialized' });
        return;
    }

    runId++;
    const params = Object.values(modelState).flatMap((mat) =>
        mat.flatMap((row) => row)
    );

    // Adam state
    const beta1 = 0.85;
    const beta2 = 0.99;
    const epsAdam = 1e-8;
    const mBuf = new Float64Array(params.length);
    const vBuf = new Float64Array(params.length);

    // Initialize finalLoss to track loss across steps
    let finalLoss = 0;
    for (let step = 0; step < numSteps; step++) {
        const doc = docs[step % docs.length];
        const tokens = [BOS, ...doc.split("").map((ch) => allChars.indexOf(ch)), BOS];
        const n = Math.min(blockSize, tokens.length - 1);

        if (n === 0) {
            // Skip this step but still report progress
            postMessage({
                type: 'progress',
                step: step + 1,
                total: numSteps,
                loss: finalLoss || 0,
                runId
            });
            continue;
        }

        const keys = Array.from({ length: nLayer }, () => []);
        const values = Array.from({ length: nLayer }, () => []);
        const losses = [];

        for (let posId = 0; posId < n; posId++) {
            const tokenId = tokens[posId];
            const targetId = tokens[posId + 1];
            const logits = gptForward(
                tokenId,
                posId,
                keys,
                values,
                modelState,
                nLayer,
                nHead,
                headDim,
                nEmbd
            );
            const probs = softmax(logits);
            losses.push(probs[targetId].log().neg());
        }

        const loss = losses
            .reduce((acc, l) => acc.add(l), new Value(0))
            .div(n);

        finalLoss = loss.data;
        
        try {
            loss.backward();
        } catch (err) {
            postMessage({ 
                type: 'error', 
                message: `Backward pass error: ${err.message}`,
                stack: err.stack 
            });
            return;
        }

        const lrT = learningRate * (1 - step / numSteps);
        for (let i = 0; i < params.length; i++) {
            const p = params[i];
            mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * p.grad;
            vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * p.grad ** 2;
            const mHat = mBuf[i] / (1 - beta1 ** (step + 1));
            const vHat = vBuf[i] / (1 - beta2 ** (step + 1));
            p.data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
            p.grad = 0;
        }

        // Post progress - use finalLoss which is always defined
        postMessage({
            type: 'progress',
            step: step + 1,
            total: numSteps,
            loss: finalLoss,
            runId
        });
    }

    postMessage({
        type: 'complete',
        finalLoss: finalLoss,
        runId
    });
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

function infer(temperature, count) {
    if (!modelState) {
        postMessage({ type: 'error', message: 'Model not initialized' });
        return;
    }

    const samples = [];
    for (let sampleIdx = 0; sampleIdx < count; sampleIdx++) {
        const keys = Array.from({ length: nLayer }, () => []);
        const values = Array.from({ length: nLayer }, () => []);
        let tokenId = BOS;
        const sample = [];
        const tokenProbs = [];

        for (let posId = 0; posId < blockSize; posId++) {
            const logits = gptForward(
                tokenId,
                posId,
                keys,
                values,
                modelState,
                nLayer,
                nHead,
                headDim,
                nEmbd
            );
            const scaledLogits = logits.map((l) => new Value(l.data / temperature));
            const probs = softmax(scaledLogits);
            
            // Extract probabilities for visualization
            const probData = probs.map(p => p.data);
            tokenId = rng.choices(probData);
            
            // Store probabilities before checking for BOS
            tokenProbs.push({
                token: tokenId === BOS ? '<BOS>' : allChars[tokenId],
                probabilities: probData.map((p, idx) => ({
                    char: idx === BOS ? '<BOS>' : allChars[idx],
                    prob: p
                })).sort((a, b) => b.prob - a.prob).slice(0, 10) // Top 10
            });
            
            if (tokenId === BOS) break;
            
            sample.push(allChars[tokenId]);
        }

        samples.push({
            text: sample.join(""),
            tokenProbs
        });
    }

    postMessage({
        type: 'samples',
        samples
    });
}

// ---------------------------------------------------------------------------
// Reset model
// ---------------------------------------------------------------------------

function resetModel() {
    modelState = buildStateDict(nLayer, nEmbd, blockSize, vocabSize);
    runId = 0;
    postMessage({ type: 'reset' });
}

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

self.onmessage = async function(e) {
    const { type, ...data } = e.data;

    try {
        switch (type) {
            case 'init':
                await initModel();
                break;
            case 'train':
                train(data.learningRate, data.steps, data.temperature);
                break;
            case 'infer':
                infer(data.temperature, data.count || 10);
                break;
            case 'reset':
                resetModel();
                break;
            default:
                postMessage({ type: 'error', message: `Unknown message type: ${type}` });
        }
    } catch (error) {
        postMessage({ type: 'error', message: error.message, stack: error.stack });
    }
};
