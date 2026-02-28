import { useEffect, useMemo, useRef, useState } from 'react';

const PRESETS = {
  safe: { lr: 0.005, steps: 700, temp: 0.4, label: 'Safe: stable, slower' },
  balanced: { lr: 0.01, steps: 1000, temp: 0.7, label: 'Balanced: recommended default' },
  experimental: { lr: 0.03, steps: 1400, temp: 1.2, label: 'Experimental: high variance' }
};

const STAGES = ['embed', 'norm', 'attn', 'mlp', 'logits', 'sample'];

const STAGE_LABELS = {
  embed: 'Embedding',
  norm: 'Normalize',
  attn: 'Attention',
  mlp: 'MLP',
  logits: 'Logits',
  sample: 'Sampling'
};

const PIPELINE = ['Input', 'Embed', 'Norm', 'Attention', 'MLP', 'Logits', 'Output'];

export default function App() {
  const workerRef = useRef(null);
  const lossCanvasRef = useRef(null);
  const flowCanvasRef = useRef(null);
  const cinematicCanvasRef = useRef(null);

  const [status, setStatus] = useState('Booting worker...');
  const [isReady, setIsReady] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

  const [learningRate, setLearningRate] = useState(0.01);
  const [numSteps, setNumSteps] = useState(500);
  const [temperature, setTemperature] = useState(0.5);

  const [runs, setRuns] = useState([]);
  const [currentRunId, setCurrentRunId] = useState(null);
  const [samples, setSamples] = useState([]);
  const [vocab, setVocab] = useState([]);
  const [latestViz, setLatestViz] = useState(null);

  const [sampleIndex, setSampleIndex] = useState(0);
  const [tokenIndex, setTokenIndex] = useState(0);
  const [whatIfTemp, setWhatIfTemp] = useState(0.5);

  const [cinematicEnabled, setCinematicEnabled] = useState(false);
  const [cinemaStageIndex, setCinemaStageIndex] = useState(0);
  const [cinemaPlaying, setCinemaPlaying] = useState(false);
  const cinemaTimerRef = useRef(null);

  const currentRun = useMemo(() => runs.find((r) => r.id === currentRunId) || null, [runs, currentRunId]);
  const currentTrace = useMemo(() => (samples[sampleIndex]?.trace || []), [samples, sampleIndex]);
  const currentStep = useMemo(() => currentTrace[tokenIndex] || null, [currentTrace, tokenIndex]);

  const canGenerate = isReady && !isTraining && runs.length > 0 && currentRun?.finalLoss != null;
  const hasTrace = currentTrace.length > 0;

  useEffect(() => {
    const worker = new Worker(`${import.meta.env.BASE_URL}worker.js`);
    workerRef.current = worker;

    worker.onmessage = (event) => {
      const { type, ...data } = event.data;

      if (type === 'ready') {
        setIsReady(true);
        setStatus(`Model ready (vocab ${data.vocabSize}, docs ${data.numDocs})`);
        return;
      }

      if (type === 'progress') {
        setStatus(`Step ${data.step}/${data.total} | Loss ${Number(data.loss).toFixed(4)}`);
        setRuns((prev) =>
          prev.map((r) =>
            r.id === data.runId
              ? {
                  ...r,
                  points: [...r.points, { x: data.step, y: data.loss }]
                }
              : r
          )
        );
        return;
      }

      if (type === 'complete') {
        setIsTraining(false);
        setRuns((prev) =>
          prev.map((r) =>
            r.id === data.runId
              ? {
                  ...r,
                  finalLoss: data.finalLoss
                }
              : r
          )
        );
        setStatus(`Training complete | Final loss ${Number(data.finalLoss).toFixed(4)}`);
        return;
      }

      if (type === 'samples') {
        setSamples((data.samples || []).map(withFallbackTrace));
        setVocab(data.vocab || []);
        setSampleIndex(0);
        setTokenIndex(0);
        setCinemaStageIndex(0);
        stopCinematic();
        setStatus('Inference complete. Explore replay and cinematic walkthrough.');
        return;
      }

      if (type === 'network_viz') {
        setLatestViz(data);
        return;
      }

      if (type === 'warning') {
        setStatus(`Warning: ${data.message}`);
        return;
      }

      if (type === 'error') {
        setIsTraining(false);
        stopCinematic();
        setStatus(`Error: ${data.message}`);
      }
    };

    worker.postMessage({ type: 'init' });

    return () => {
      stopCinematic();
      worker.terminate();
      workerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    drawLoss(lossCanvasRef.current, runs);
  }, [runs]);

  useEffect(() => {
    drawLiveFlow(flowCanvasRef.current, latestViz);
  }, [latestViz]);

  useEffect(() => {
    if (!cinematicEnabled) return;
    drawCinematic(cinematicCanvasRef.current, currentStep, cinemaStageIndex);
  }, [cinematicEnabled, currentStep, cinemaStageIndex]);

  useEffect(() => {
    if (!cinematicEnabled) {
      stopCinematic();
      return;
    }
    setCinemaStageIndex(0);
    drawCinematic(cinematicCanvasRef.current, currentStep, 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokenIndex, sampleIndex, cinematicEnabled]);

  useEffect(() => {
    setWhatIfTemp(temperature);
  }, [temperature]);

  function applyPreset(key) {
    const p = PRESETS[key];
    if (!p) return;
    setLearningRate(p.lr);
    setNumSteps(p.steps);
    setTemperature(p.temp);
    setStatus(`Preset applied: ${p.label}`);
  }

  function startTraining() {
    if (!workerRef.current || isTraining || !isReady) return;

    const runId = runs.length + 1;
    setIsTraining(true);
    stopCinematic();
    setSamples([]);
    setCurrentRunId(runId);
    setRuns((prev) => [
      ...prev,
      {
        id: runId,
        lr: learningRate,
        steps: numSteps,
        temp: temperature,
        points: [],
        finalLoss: null
      }
    ]);

    workerRef.current.postMessage({
      type: 'train',
      learningRate,
      steps: numSteps,
      temperature
    });
  }

  function generate() {
    if (!workerRef.current || isTraining || !canGenerate) return;
    stopCinematic();
    setStatus('Generating samples and traces...');
    workerRef.current.postMessage({ type: 'infer', temperature, count: 8 });
  }

  function goToken(index) {
    if (!hasTrace) return;
    const clamped = Math.max(0, Math.min(index, currentTrace.length - 1));
    setTokenIndex(clamped);
  }

  function softmaxFromLogits(rawLogits, temp) {
    if (!Array.isArray(rawLogits) || rawLogits.length === 0) return [];
    const t = Math.max(0.05, temp);
    const scaled = rawLogits.map((v) => v / t);
    const max = Math.max(...scaled);
    const exps = scaled.map((x) => Math.exp(x - max));
    const denom = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((x) => x / denom);
  }

  const topK = useMemo(() => {
    if (!currentStep) return [];

    if (Array.isArray(currentStep.rawLogits) && currentStep.rawLogits.length > 0) {
      const probs = softmaxFromLogits(currentStep.rawLogits, whatIfTemp);
      return probs
        .map((prob, idx) => ({
          token: vocab[idx] || '?',
          prob
        }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 10);
    }

    if (Array.isArray(currentStep.topK)) {
      return currentStep.topK.slice(0, 10).map((x) => ({ token: x.token || '?', prob: x.prob || 0 }));
    }

    return [];
  }, [currentStep, vocab, whatIfTemp]);

  const stageRows = useMemo(() => {
    if (!currentStep) return [];
    const embed = currentStep.stages?.embed || 0;
    const norm = currentStep.stages?.normIn || 0;
    const attn = avg(currentStep.stages?.attn || []);
    const mlp = avg(currentStep.stages?.mlp || []);
    const logits = currentStep.stages?.logits || 0;
    const out = currentStep.confidence || 0;

    return [
      { label: 'Embed', value: embed, detail: `Input token -> dense vector ${embed.toFixed(3)}` },
      { label: 'Norm', value: norm, detail: `Scale adjusted by ${(norm - embed).toFixed(3)}` },
      { label: 'Attention', value: attn, detail: 'Context routing across prior positions' },
      { label: 'MLP', value: mlp, detail: `Feature remix delta ${(mlp - attn).toFixed(3)}` },
      { label: 'Logits', value: logits, detail: 'Vocabulary preference field' },
      { label: 'Output', value: out, detail: `Chosen token confidence ${(out * 100).toFixed(1)}%` }
    ];
  }, [currentStep]);

  function toggleCinematic() {
    if (!cinematicEnabled) {
      setCinematicEnabled(true);
      setCinemaStageIndex(0);
      return;
    }
    setCinematicEnabled(false);
    stopCinematic();
  }

  function stopCinematic() {
    setCinemaPlaying(false);
    if (cinemaTimerRef.current) {
      clearInterval(cinemaTimerRef.current);
      cinemaTimerRef.current = null;
    }
  }

  function stepCinematicNext() {
    if (!currentStep) return;
    if (cinemaStageIndex < STAGES.length - 1) {
      setCinemaStageIndex((s) => s + 1);
      return;
    }
    if (tokenIndex < currentTrace.length - 1) {
      setTokenIndex((t) => t + 1);
      setCinemaStageIndex(0);
      return;
    }
    stopCinematic();
  }

  function cinematicPrev() {
    stopCinematic();
    if (!currentStep) return;
    if (cinemaStageIndex > 0) {
      setCinemaStageIndex((s) => s - 1);
      return;
    }
    if (tokenIndex > 0) {
      setTokenIndex((t) => t - 1);
      setCinemaStageIndex(STAGES.length - 1);
    }
  }

  function cinematicNext() {
    stopCinematic();
    stepCinematicNext();
  }

  function cinematicPlayPause() {
    if (!currentStep) return;
    if (cinemaPlaying) {
      stopCinematic();
      return;
    }
    setCinemaPlaying(true);
    cinemaTimerRef.current = setInterval(stepCinematicNext, 650);
  }

  const runSummary = useMemo(() => {
    if (!currentRun || currentRun.finalLoss == null) return null;
    const prevRun = runs.find((r) => r.id === currentRun.id - 1);
    const prevLoss = prevRun?.finalLoss;

    const trend = Number.isFinite(prevLoss)
      ? (() => {
          const delta = prevLoss - currentRun.finalLoss;
          const pct = Math.abs((delta / prevLoss) * 100);
          return delta >= 0
            ? `Loss improved by ${pct.toFixed(1)}% from previous run.`
            : `Loss worsened by ${pct.toFixed(1)}% from previous run.`;
        })()
      : `Initial run final loss is ${currentRun.finalLoss.toFixed(4)}.`;

    const lrHint =
      currentRun.lr >= 0.025
        ? 'LR is aggressive; reduce if curve spikes.'
        : currentRun.lr <= 0.006
          ? 'LR is conservative; increase slightly if learning is too slow.'
          : 'LR is in a stable middle range.';

    const tempHint =
      currentRun.temp >= 1.0
        ? 'High temperature favors diversity over certainty.'
        : currentRun.temp <= 0.5
          ? 'Low temperature favors consistency.'
          : 'Moderate temperature balances diversity and coherence.';

    return `${trend} ${lrHint} ${tempHint}`;
  }, [currentRun, runs]);

  return (
    <div className="studio-shell min-h-screen">
      <div className="studio-noise" />
      <div className="mx-auto max-w-7xl p-4 md:p-6">
        <header className="hero-card mb-4 rounded-2xl p-4 md:p-5">
          <h1 className="text-2xl font-semibold tracking-tight text-slate-100">MicroGPT Developer Studio</h1>
          <p className="mt-1 text-sm text-slate-300">
            Fast by default. Rich replay + cinematic walkthrough available on demand.
          </p>
        </header>

        <div className="studio-card mb-4 rounded-lg px-3 py-2 text-sm text-sky-200">
          {status}
        </div>

        <div className="grid gap-4 xl:grid-cols-[370px_1fr]">
          <aside className="space-y-4">
            <section className="studio-card rounded-xl p-4">
              <h2 className="mb-2 text-lg font-medium text-slate-100">Training Controls</h2>
              <p className="mb-3 text-xs text-slate-400">
                1) Choose a preset or tune knobs. 2) Train. 3) Generate and inspect token-level flow.
              </p>

              <div className="mb-3 flex flex-wrap gap-2">
                {Object.entries(PRESETS).map(([key, p]) => (
                  <button
                    key={key}
                    className="rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-xs text-slate-200 hover:bg-slate-700"
                    onClick={() => applyPreset(key)}
                  >
                    {p.label}
                  </button>
                ))}
              </div>

              <SliderKnob
                label="Learning rate"
                value={learningRate}
                min={0.001}
                max={0.1}
                step={0.001}
                hint="Higher learns faster but may destabilize."
                onChange={setLearningRate}
              />
              <SliderKnob
                label="Training steps"
                value={numSteps}
                min={100}
                max={2000}
                step={100}
                integer
                hint="More steps improve fit but increase runtime."
                onChange={setNumSteps}
              />
              <SliderKnob
                label="Inference temperature"
                value={temperature}
                min={0.1}
                max={2}
                step={0.1}
                hint="Lower is deterministic; higher is diverse/noisy."
                onChange={setTemperature}
              />

              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  className="rounded-md bg-sky-500 px-4 py-2 text-sm font-semibold text-slate-950 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
                  onClick={startTraining}
                  disabled={!isReady || isTraining}
                >
                  {isTraining ? 'Training...' : 'Train'}
                </button>
                <button
                  className="rounded-md border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-semibold text-slate-200 disabled:cursor-not-allowed disabled:bg-slate-900 disabled:text-slate-500"
                  onClick={generate}
                  disabled={!canGenerate}
                >
                  Generate
                </button>
              </div>
            </section>

            <section className="studio-card rounded-xl p-4">
              <h3 className="mb-2 text-sm font-semibold text-slate-200">Explain This Run</h3>
              <p className="text-xs leading-relaxed text-slate-300">{runSummary || 'Complete a training run to get automated analysis and next-step advice.'}</p>
            </section>
          </aside>

          <main className="space-y-4">
            <section className="studio-card rounded-xl p-4">
              <h2 className="mb-3 text-lg font-medium text-slate-100">Loss Curve</h2>
              <canvas ref={lossCanvasRef} width={1100} height={280} className="w-full rounded-lg border border-slate-700 bg-slate-950" />
              <p className="mt-2 text-xs text-slate-400">Lower curve usually means better next-token prediction quality.</p>
            </section>

            <section className="studio-card rounded-xl p-4">
              <h2 className="mb-3 text-lg font-medium text-slate-100">Live Network Flow</h2>
              <canvas ref={flowCanvasRef} width={1100} height={220} className="w-full rounded-lg border border-slate-700 bg-slate-950" />
              <div className="mt-2 text-xs text-slate-300">
                {latestViz
                  ? `${latestViz.mode === 'train' ? `Train ${latestViz.step}/${latestViz.total}` : `Infer sample ${latestViz.sample} token ${latestViz.tokenIndex}`} | ${latestViz.token ? `token "${latestViz.token}"` : ''}`
                  : 'Run training or inference to visualize live activation flow.'}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {PIPELINE.map((node) => (
                  <span key={node} className="rounded-full border border-slate-600 bg-slate-800/70 px-2 py-1 text-xs text-slate-300">
                    {node}
                  </span>
                ))}
              </div>
            </section>

            <section className="studio-card rounded-xl p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-lg font-medium text-slate-100">Inference Replay</h2>
                <div className="flex items-center gap-2 text-xs text-slate-300">
                  <label htmlFor="sampleSelect">Sample</label>
                  <select
                    id="sampleSelect"
                    className="rounded border border-slate-600 bg-slate-900 px-2 py-1"
                    value={sampleIndex}
                    onChange={(e) => {
                      stopCinematic();
                      setSampleIndex(Number(e.target.value));
                      setTokenIndex(0);
                      setCinemaStageIndex(0);
                    }}
                    disabled={samples.length === 0}
                  >
                    {samples.map((_, i) => (
                      <option key={i} value={i}>
                        Sample {i + 1}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="mb-3 flex gap-2 overflow-x-auto pb-1">
                {currentTrace.map((step, idx) => (
                  <button
                    key={idx}
                    className={`min-w-[44px] rounded-full border px-2 py-1 font-mono text-xs ${
                      idx === tokenIndex
                        ? 'border-sky-400 bg-sky-700/60 text-sky-50'
                        : 'border-slate-600 bg-slate-900 text-slate-300 hover:bg-slate-800'
                    }`}
                    onClick={() => {
                      stopCinematic();
                      goToken(idx);
                      setCinemaStageIndex(0);
                    }}
                  >
                    {step.outputToken}
                  </button>
                ))}
              </div>

              <div className="mb-3 rounded-lg border border-slate-700 bg-slate-950/70 px-3 py-2 text-xs text-sky-200">
                {currentStep
                  ? `Token step ${currentStep.tokenIndex}: "${currentStep.inputToken}" -> "${currentStep.outputToken}" | confidence ${(currentStep.confidence * 100).toFixed(1)}%`
                  : 'Generate samples to inspect token-level traces.'}
              </div>

              <div className="grid gap-3 xl:grid-cols-[1.2fr_1fr]">
                <div className="rounded-lg border border-slate-700 bg-slate-950/60 p-3">
                  <div className="mb-2 text-sm font-semibold text-slate-200">Stage Flow Values</div>
                  <div className="space-y-2">
                    {stageRows.map((row) => (
                      <FlowRow key={row.label} label={row.label} value={row.value} max={Math.max(1e-6, ...stageRows.map((r) => r.value))} detail={row.detail} />
                    ))}
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="rounded-lg border border-slate-700 bg-slate-950/60 p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <div className="text-sm font-semibold text-slate-200">Top-k Token Decisions</div>
                      <div className="flex items-center gap-2 text-xs text-slate-300">
                        <span>What-if T</span>
                        <input
                          className="range w-24"
                          type="range"
                          min={0.1}
                          max={2}
                          step={0.1}
                          value={whatIfTemp}
                          onChange={(e) => setWhatIfTemp(parseFloat(e.target.value))}
                        />
                        <span className="mono w-8 text-right text-slate-200">{whatIfTemp.toFixed(1)}</span>
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      {topK.slice(0, 8).map((item, i) => (
                        <TopKBar key={`${item.token}-${i}`} token={item.token} prob={item.prob} highlight={item.token === currentStep?.outputToken} />
                      ))}
                    </div>
                  </div>

                  <div className="rounded-lg border border-slate-700 bg-slate-950/60 p-3">
                    <div className="mb-2 text-sm font-semibold text-slate-200">Attention Heads (last layer)</div>
                    {renderAttentionHeads(currentStep)}
                  </div>
                </div>
              </div>

              <div className="mt-4 rounded-xl border border-blue-900/60 bg-gradient-to-r from-slate-900 to-blue-950 p-3">
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <div className="text-sm font-semibold text-slate-100">Cinematic Walkthrough (opt-in)</div>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      className={`rounded border px-2 py-1 text-xs ${cinematicEnabled ? 'border-sky-400 bg-sky-700/50 text-sky-50' : 'border-slate-600 bg-slate-800 text-slate-200'}`}
                      onClick={toggleCinematic}
                      disabled={!hasTrace}
                    >
                      {cinematicEnabled ? 'Disable Cinematic' : 'Enable Cinematic'}
                    </button>
                    <button className="rounded border border-slate-600 bg-slate-800 px-2 py-1 text-xs text-slate-200" onClick={cinematicPrev} disabled={!cinematicEnabled || !hasTrace}>
                      Step Back
                    </button>
                    <button className="rounded border border-slate-600 bg-slate-800 px-2 py-1 text-xs text-slate-200" onClick={cinematicPlayPause} disabled={!cinematicEnabled || !hasTrace}>
                      {cinemaPlaying ? 'Pause' : 'Autoplay'}
                    </button>
                    <button className="rounded border border-slate-600 bg-slate-800 px-2 py-1 text-xs text-slate-200" onClick={cinematicNext} disabled={!cinematicEnabled || !hasTrace}>
                      Step Forward
                    </button>
                  </div>
                </div>

                {cinematicEnabled ? (
                  <>
                    <div className="mb-2 flex flex-wrap gap-2">
                      {STAGES.map((stage, idx) => (
                        <span
                          key={stage}
                          className={`rounded-full border px-2 py-1 text-xs ${idx === cinemaStageIndex ? 'border-sky-300 bg-sky-700/50 text-sky-50' : 'border-slate-600 bg-slate-900 text-slate-300'}`}
                        >
                          {STAGE_LABELS[stage]}
                        </span>
                      ))}
                    </div>
                    <div className="mb-2 text-sm leading-relaxed text-sky-100">{cinematicCaption(currentStep, cinemaStageIndex)}</div>
                    <canvas ref={cinematicCanvasRef} width={1100} height={220} className="w-full rounded-lg border border-slate-700 bg-slate-950" />
                    <div className="mt-2 text-xs text-slate-300">{cinematicDetail(currentStep, topK, cinemaStageIndex)}</div>
                  </>
                ) : (
                  <div className="text-xs text-slate-300">Cinematic mode is disabled for speed. Enable it when you want guided stage-by-stage animation.</div>
                )}
              </div>

              <div className="mt-4">
                <details className="rounded-lg border border-slate-700 bg-slate-900/60 p-3">
                  <summary className="cursor-pointer text-sm font-semibold text-slate-200">Generated Samples</summary>
                  <div className="mt-3 grid gap-3 md:grid-cols-2">
                    {samples.map((s, i) => (
                      <article key={i} className="rounded-lg border border-slate-700 bg-slate-950/60 p-3">
                        <div className="mb-1 text-xs text-slate-400">Sample {i + 1}</div>
                        <div className="mono text-sm text-slate-100">{s.text || '(empty)'}</div>
                      </article>
                    ))}
                  </div>
                </details>
              </div>
            </section>
          </main>
        </div>
      </div>
    </div>
  );
}

function SliderKnob({ label, value, min, max, step, hint, onChange, integer = false }) {
  return (
    <div className="mb-3">
      <label className="mb-1 block text-sm text-slate-300">{label}</label>
      <div className="flex items-center gap-3">
        <input
          className="range"
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(integer ? parseInt(e.target.value, 10) : parseFloat(e.target.value))}
        />
        <span className="mono w-14 text-right text-sm text-slate-200">
          {integer ? value : Number(value).toFixed(3).replace(/0+$/, '').replace(/\.$/, '')}
        </span>
      </div>
      <p className="mt-1 text-xs text-slate-500">{hint}</p>
    </div>
  );
}

function FlowRow({ label, value, max, detail }) {
  const pct = Math.max(2, (value / Math.max(max, 1e-6)) * 100);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs text-slate-300">
        <span>{label}</span>
        <span className="mono">{value.toFixed(3)}</span>
      </div>
      <div className="h-2 rounded bg-slate-700">
        <div className="h-2 rounded bg-gradient-to-r from-cyan-400 to-blue-500" style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-1 text-[11px] text-slate-500">{detail}</p>
    </div>
  );
}

function TopKBar({ token, prob, highlight }) {
  return (
    <div className="grid grid-cols-[56px_1fr_44px] items-center gap-2 text-xs">
      <span className={`mono ${highlight ? 'text-emerald-300' : 'text-slate-300'}`}>{token}</span>
      <div className="h-2 rounded bg-slate-700">
        <div
          className={`h-2 rounded ${highlight ? 'bg-gradient-to-r from-emerald-400 to-green-500' : 'bg-gradient-to-r from-cyan-400 to-blue-500'}`}
          style={{ width: `${Math.max(1, prob * 100)}%` }}
        />
      </div>
      <span className="mono text-right text-slate-300">{(prob * 100).toFixed(1)}%</span>
    </div>
  );
}

function renderAttentionHeads(step) {
  const layers = step?.attention || [];
  if (layers.length === 0) return <div className="text-xs text-slate-500">No attention data for this step.</div>;

  const last = layers[layers.length - 1] || [];
  return (
    <div className="space-y-2">
      {last.map((weights, i) => (
        <div key={i}>
          <div className="mb-1 text-xs text-slate-400">Head {i + 1}</div>
          <div className="flex gap-1 overflow-x-auto pb-1">
            {(weights || []).map((w, j) => (
              <div
                key={j}
                className="h-4 w-5 shrink-0 rounded border border-slate-700"
                style={{ backgroundColor: `rgba(56, 189, 248, ${Math.max(0.1, Math.min(1, w))})` }}
                title={`${(w * 100).toFixed(1)}%`}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function cinematicCaption(step, stageIndex) {
  if (!step) return 'Generate samples to begin cinematic walkthrough.';
  const stage = STAGES[Math.max(0, Math.min(stageIndex, STAGES.length - 1))];

  const embed = step.stages?.embed || 0;
  const norm = step.stages?.normIn || 0;
  const attn = avg(step.stages?.attn || []);
  const mlp = avg(step.stages?.mlp || []);
  const logits = step.stages?.logits || 0;

  if (stage === 'embed') return `Embedding maps token "${step.inputToken}" into latent space (${embed.toFixed(3)}).`;
  if (stage === 'norm') return `Normalization stabilizes scale ${embed.toFixed(3)} -> ${norm.toFixed(3)}.`;
  if (stage === 'attn') return `Attention integrates context from prior positions (${attn.toFixed(3)}).`;
  if (stage === 'mlp') return `MLP transforms contextual features (${attn.toFixed(3)} -> ${mlp.toFixed(3)}).`;
  if (stage === 'logits') return `Logits project into vocabulary preference scores (${logits.toFixed(3)}).`;
  return `Sampling chooses "${step.outputToken}" with ${(step.confidence * 100).toFixed(1)}% confidence.`;
}

function cinematicDetail(step, topK, stageIndex) {
  if (!step) return 'No token selected.';
  const stage = STAGES[Math.max(0, Math.min(stageIndex, STAGES.length - 1))];
  const top = topK.slice(0, 3).map((x) => `${x.token} ${(x.prob * 100).toFixed(1)}%`).join(' | ');
  if (stage === 'sample') return `Top alternatives: ${top || 'N/A'}`;
  return `Current token transition: "${step.inputToken}" -> "${step.outputToken}"`;
}

function withFallbackTrace(sample) {
  if (sample && Array.isArray(sample.trace) && sample.trace.length > 0) return sample;

  const tokenProbs = Array.isArray(sample?.tokenProbs) ? sample.tokenProbs : [];
  const trace = tokenProbs.map((tp, idx) => {
    const topK = Array.isArray(tp.probabilities)
      ? tp.probabilities.map((p) => ({ token: p.char, prob: p.prob }))
      : [];
    const best = topK[0] || { token: tp.token || '?', prob: 0 };
    return {
      tokenIndex: idx + 1,
      inputToken: idx === 0 ? '<BOS>' : (sample?.text?.[idx - 1] || '?'),
      outputToken: tp.token || best.token,
      confidence: Number.isFinite(best.prob) ? best.prob : 0,
      rawLogits: [],
      topK,
      nodes: [
        { label: 'Input', value: 1 },
        { label: 'Embed', value: 0.5 },
        { label: 'Norm', value: 0.5 },
        { label: 'Attention', value: 0.5 },
        { label: 'MLP', value: 0.5 },
        { label: 'Logits', value: 0.5 },
        { label: 'Output', value: Number.isFinite(best.prob) ? best.prob : 0 }
      ],
      stages: {
        embed: 0.5,
        normIn: 0.5,
        attn: [0.5],
        mlp: [0.5],
        logits: 0.5
      },
      attention: []
    };
  });

  return { ...(sample || {}), trace };
}

function drawLoss(canvas, runs) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const pad = { top: 20, right: 20, bottom: 28, left: 46 };
  const cw = width - pad.left - pad.right;
  const ch = height - pad.top - pad.bottom;

  ctx.fillStyle = '#020617';
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (ch * i) / 5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }

  if (!runs.length) return;
  const points = runs.flatMap((r) => r.points);
  if (!points.length) return;

  const maxX = Math.max(...points.map((p) => p.x), 1);
  const minY = Math.min(...points.map((p) => p.y));
  const maxY = Math.max(...points.map((p) => p.y));
  const yr = Math.max(1e-6, maxY - minY);

  const colors = ['#38bdf8', '#22c55e', '#f59e0b', '#ef4444', '#a78bfa', '#f97316'];

  runs.forEach((run, idx) => {
    if (!run.points.length) return;
    ctx.strokeStyle = colors[idx % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();

    run.points.forEach((p, i) => {
      const x = pad.left + (p.x / maxX) * cw;
      const y = pad.top + (1 - (p.y - minY) / yr) * ch;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.stroke();
  });
}

function drawLiveFlow(canvas, viz) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  ctx.fillStyle = '#020617';
  ctx.fillRect(0, 0, width, height);

  if (!viz?.nodes?.length) {
    ctx.fillStyle = '#64748b';
    ctx.font = '14px sans-serif';
    ctx.fillText('Waiting for network activity...', 20, 30);
    return;
  }

  const nodes = viz.nodes;
  const values = nodes.map((n) => (Number.isFinite(n.value) ? n.value : 0));
  const maxV = Math.max(1e-6, ...values);
  const left = 42;
  const right = width - 42;
  const y = Math.floor(height / 2);
  const gap = nodes.length > 1 ? (right - left) / (nodes.length - 1) : 0;

  for (let i = 0; i < nodes.length - 1; i++) {
    const a = nodes[i];
    const b = nodes[i + 1];
    const edge = Math.max(a.value || 0, b.value || 0) / maxV;
    ctx.strokeStyle = `rgba(56, 189, 248, ${0.2 + edge * 0.8})`;
    ctx.lineWidth = 1 + edge * 5;
    ctx.beginPath();
    ctx.moveTo(left + i * gap + 16, y);
    ctx.lineTo(left + (i + 1) * gap - 16, y);
    ctx.stroke();
  }

  nodes.forEach((node, i) => {
    const x = left + i * gap;
    const v = Number.isFinite(node.value) ? node.value : 0;
    const t = Math.min(1, v / maxV);

    ctx.fillStyle = `rgba(${30 + Math.round(t * 50)}, ${90 + Math.round(t * 130)}, ${180 + Math.round(t * 70)}, 1)`;
    ctx.beginPath();
    ctx.arc(x, y, 16, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.textAlign = 'center';
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '11px sans-serif';
    ctx.fillText(node.label, x, y - 25);
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(v.toFixed(3), x, y + 32);
  });
}

function drawCinematic(canvas, step, stageIndex) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.fillStyle = '#020617';
  ctx.fillRect(0, 0, width, height);

  if (!step) {
    ctx.fillStyle = '#94a3b8';
    ctx.font = '14px sans-serif';
    ctx.fillText('Enable cinematic and choose a token step.', 20, 30);
    return;
  }

  const nodes = [
    { key: 'embed', label: 'Embed', value: step.stages?.embed || 0 },
    { key: 'norm', label: 'Norm', value: step.stages?.normIn || 0 },
    { key: 'attn', label: 'Attn', value: avg(step.stages?.attn || []) },
    { key: 'mlp', label: 'MLP', value: avg(step.stages?.mlp || []) },
    { key: 'logits', label: 'Logits', value: step.stages?.logits || 0 },
    { key: 'sample', label: 'Sample', value: step.confidence || 0 }
  ];

  const maxV = Math.max(1e-6, ...nodes.map((n) => n.value));
  const left = 42;
  const right = width - 42;
  const y = Math.floor(height / 2);
  const gap = (right - left) / (nodes.length - 1);

  for (let i = 0; i < nodes.length - 1; i++) {
    const x1 = left + i * gap;
    const x2 = left + (i + 1) * gap;
    ctx.strokeStyle = 'rgba(71,85,105,0.65)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x1 + 17, y);
    ctx.lineTo(x2 - 17, y);
    ctx.stroke();
  }

  const activeStage = STAGES[Math.max(0, Math.min(STAGES.length - 1, stageIndex))];
  const activeIdx = STAGES.indexOf(activeStage);

  nodes.forEach((node, i) => {
    const x = left + i * gap;
    const t = Math.min(1, node.value / maxV);
    const active = i === activeIdx;

    ctx.beginPath();
    ctx.arc(x, y, active ? 20 : 16, 0, Math.PI * 2);
    ctx.fillStyle = active ? 'rgba(56,189,248,0.95)' : `rgba(59,130,246,${0.22 + t * 0.6})`;
    ctx.fill();
    ctx.strokeStyle = active ? '#e0f2fe' : '#1f2937';
    ctx.lineWidth = active ? 3 : 2;
    ctx.stroke();

    ctx.textAlign = 'center';
    ctx.fillStyle = '#dbeafe';
    ctx.font = '11px sans-serif';
    ctx.fillText(node.label, x, y - 30);
    ctx.fillStyle = '#93c5fd';
    ctx.fillText(node.value.toFixed(3), x, y + 34);
  });

  const progress = Math.min(1, (stageIndex + 1) / STAGES.length);
  const travel = progress * (right - left);
  const now = Date.now() / 1000;

  for (let i = 0; i < 16; i++) {
    const phase = ((i / 16) + now * 0.35) % 1;
    const px = left + phase * travel;
    const py = y + Math.sin(phase * 8 + now * 2.6) * 8;
    ctx.beginPath();
    ctx.arc(px, py, 2.4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(147,197,253,0.85)';
    ctx.fill();
  }
}

function avg(values) {
  if (!values || values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}
