/**
 * The most atomic way to train and run inference for a GPT in pure, dependency-free TypeScript.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * TypeScript port of @karpathy's Python original.
 *
 * Usage:
 *   npx ts-node nanogpt.ts
 *   # or compile and run:
 *   npx tsc nanogpt.ts --target es2020 --module commonjs && node nanogpt.js
 *
 * Expects `input.txt` in the same directory (one document per line, e.g. names).
 * Will download names.txt from karpathy/makemore if input.txt is absent and you have net access.
 */

import * as fs from "fs";
import * as https from "https";
import * as path from "path";

// ---------------------------------------------------------------------------
// Seeded PRNG (replaces Python's random.seed(42))
// ---------------------------------------------------------------------------

class SeededRandom {
    private state: number;

    constructor(seed: number) {
        this.state = seed >>> 0;
    }

    // Mulberry32 — fast, good quality 32-bit PRNG
    private next(): number {
        this.state = (this.state + 0x6d2b79f5) >>> 0;
        let z = this.state;
        z = Math.imul(z ^ (z >>> 15), z | 1);
        z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
        return ((z ^ (z >>> 14)) >>> 0) / 0x100000000;
    }

    /** Normally-distributed sample via Box-Muller */
    gauss(mean = 0, std = 1): number {
        const u1 = this.next();
        const u2 = this.next();
        return mean + std * Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    }

    /** Shuffle an array in place (Fisher-Yates) */
    shuffle<T>(arr: T[]): void {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }

    /** Weighted random choice — returns index */
    choices(weights: number[]): number {
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
// Dataset
// ---------------------------------------------------------------------------

const inputPath = path.join(__dirname, "input.txt");

function downloadFile(url: string, dest: string): Promise<void> {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(dest);
        https
            .get(url, (res) => {
                res.pipe(file);
                file.on("finish", () => {
                    file.close();
                    resolve();
                });
            })
            .on("error", (err) => {
                fs.unlink(dest, () => { });
                reject(err);
            });
    });
}

async function loadDocs(): Promise<string[]> {
    if (!fs.existsSync(inputPath)) {
        const url =
            "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt";
        console.log(`Downloading names from ${url} ...`);
        await downloadFile(url, inputPath);
    }
    const lines = fs.readFileSync(inputPath, "utf-8").split("\n");
    return lines.map((l) => l.trim()).filter((l) => l.length > 0);
}

// ---------------------------------------------------------------------------
// Autograd: scalar Value node
// ---------------------------------------------------------------------------

class Value {
    data: number;
    grad: number;
    private _children: Value[];
    private _localGrads: number[];

    constructor(
        data: number,
        children: Value[] = [],
        localGrads: number[] = []
    ) {
        this.data = data;
        this.grad = 0;
        this._children = children;
        this._localGrads = localGrads;
    }

    add(other: Value | number): Value {
        const o = other instanceof Value ? other : new Value(other);
        return new Value(this.data + o.data, [this, o], [1, 1]);
    }

    mul(other: Value | number): Value {
        const o = other instanceof Value ? other : new Value(other);
        return new Value(this.data * o.data, [this, o], [o.data, this.data]);
    }

    pow(exp: number): Value {
        return new Value(
            Math.pow(this.data, exp),
            [this],
            [exp * Math.pow(this.data, exp - 1)]
        );
    }

    log(): Value {
        return new Value(Math.log(this.data), [this], [1 / this.data]);
    }

    exp(): Value {
        const e = Math.exp(this.data);
        return new Value(e, [this], [e]);
    }

    relu(): Value {
        return new Value(
            Math.max(0, this.data),
            [this],
            [this.data > 0 ? 1 : 0]
        );
    }

    neg(): Value {
        return this.mul(-1);
    }

    sub(other: Value | number): Value {
        const o = other instanceof Value ? other : new Value(other);
        return this.add(o.neg());
    }

    div(other: Value | number): Value {
        const o = other instanceof Value ? other : new Value(other);
        return this.mul(o.pow(-1));
    }

    backward(): void {
        const topo: Value[] = [];
        const visited = new Set<Value>();

        function buildTopo(v: Value): void {
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

type Matrix = Value[][];

function makeMatrix(nout: number, nin: number, std = 0.08): Matrix {
    return Array.from({ length: nout }, () =>
        Array.from({ length: nin }, () => new Value(rng.gauss(0, std)))
    );
}

function linear(x: Value[], w: Matrix): Value[] {
    return w.map((row) =>
        row.reduce((acc, wi, i) => acc.add(wi.mul(x[i])), new Value(0))
    );
}

function softmax(logits: Value[]): Value[] {
    const maxVal = Math.max(...logits.map((v) => v.data));
    const exps = logits.map((v) => v.sub(maxVal).exp());
    const total = exps.reduce((a, b) => a.add(b), new Value(0));
    return exps.map((e) => e.div(total));
}

function rmsnorm(x: Value[]): Value[] {
    const ms = x
        .reduce((acc, xi) => acc.add(xi.mul(xi)), new Value(0))
        .div(x.length);
    const scale = ms.add(1e-5).pow(-0.5);
    return x.map((xi) => xi.mul(scale));
}

// ---------------------------------------------------------------------------
// Model state dict & GPT forward pass
// ---------------------------------------------------------------------------

interface StateDict {
    wte: Matrix;
    wpe: Matrix;
    lm_head: Matrix;
    [key: string]: Matrix;
}

function buildStateDict(
    nLayer: number,
    nEmbd: number,
    blockSize: number,
    vocabSize: number
): StateDict {
    const sd: StateDict = {
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

function gptForward(
    tokenId: number,
    posId: number,
    keys: Value[][][],
    values: Value[][][],
    sd: StateDict,
    nLayer: number,
    nHead: number,
    headDim: number,
    nEmbd: number
): Value[] {
    const tokEmb = sd.wte[tokenId];
    const posEmb = sd.wpe[posId];
    let x: Value[] = tokEmb.map((t, i) => t.add(posEmb[i]));
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

        const xAttn: Value[] = [];
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
// Main: train + infer
// ---------------------------------------------------------------------------

async function main() {
    // Load data
    const docs = await loadDocs();
    rng.shuffle(docs);
    console.log(`num docs: ${docs.length}`);

    // Tokenizer
    const allChars = Array.from(new Set(docs.join(""))).sort() as string[];
    const BOS = allChars.length;
    const vocabSize = allChars.length + 1;
    console.log(`vocab size: ${vocabSize}`);

    // Hyperparameters
    const nLayer = 1;
    const nEmbd = 16;
    const blockSize = 16;
    const nHead = 4;
    const headDim = nEmbd / nHead;

    // Model
    const sd = buildStateDict(nLayer, nEmbd, blockSize, vocabSize);
    const params: Value[] = Object.values(sd).flatMap((mat) =>
        mat.flatMap((row) => row)
    );
    console.log(`num params: ${params.length}`);

    // Adam state
    const learningRate = 0.01;
    const beta1 = 0.85;
    const beta2 = 0.99;
    const epsAdam = 1e-8;
    const mBuf = new Float64Array(params.length);
    const vBuf = new Float64Array(params.length);

    // Training loop
    const numSteps = 1000;

    for (let step = 0; step < numSteps; step++) {
        const doc = docs[step % docs.length];
        const tokens = [BOS, ...doc.split("").map((ch) => allChars.indexOf(ch)), BOS];
        const n = Math.min(blockSize, tokens.length - 1);

        const keys: Value[][][] = Array.from({ length: nLayer }, () => []);
        const values: Value[][][] = Array.from({ length: nLayer }, () => []);
        const losses: Value[] = [];

        for (let posId = 0; posId < n; posId++) {
            const tokenId = tokens[posId];
            const targetId = tokens[posId + 1];
            const logits = gptForward(
                tokenId,
                posId,
                keys,
                values,
                sd,
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

        loss.backward();

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

        process.stdout.write(
            `\rstep ${(step + 1).toString().padStart(4)} / ${numSteps} | loss ${loss.data.toFixed(4)}`
        );
    }

    // Inference
    const temperature = 0.5;
    console.log("\n--- inference (new, hallucinated names) ---");
    for (let sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
        const keys: Value[][][] = Array.from({ length: nLayer }, () => []);
        const values: Value[][][] = Array.from({ length: nLayer }, () => []);
        let tokenId = BOS;
        const sample: string[] = [];

        for (let posId = 0; posId < blockSize; posId++) {
            const logits = gptForward(
                tokenId,
                posId,
                keys,
                values,
                sd,
                nLayer,
                nHead,
                headDim,
                nEmbd
            );
            const scaledLogits = logits.map((l) => new Value(l.data / temperature));
            const probs = softmax(scaledLogits);
            tokenId = rng.choices(probs.map((p) => p.data));
            if (tokenId === BOS) break;
            sample.push(allChars[tokenId]);
        }

        console.log(`sample ${(sampleIdx + 1).toString().padStart(2)}: ${sample.join("")}`);
    }
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});