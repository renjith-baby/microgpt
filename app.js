// Main application logic for GPT Training Visualizer

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let worker = null;
let isTraining = false;
let runs = [];
let currentRun = null;
let chartData = [];
let colorPalette = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4'];
let previousLoss = null;
let latestNetworkViz = null;
let replayState = {
    samples: [],
    vocab: [],
    sampleIndex: 0,
    stepIndex: 0,
    playing: false,
    timer: null
};
const cinematicStages = ['embed', 'norm', 'attn', 'mlp', 'logits', 'sample'];
let cinemaState = {
    stageIndex: 0,
    playing: false,
    timer: null
};
let cinemaEnabled = false;

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------

let lossChart, ctx, learningRateSlider, learningRateValue, numStepsSlider, numStepsValue;
let temperatureSlider, temperatureValue, trainButton, generateButton, statusDiv;
let summaryDiv, historyBody, samplesContainer, currentTemperatureSpan;
let presetSafeButton, presetBalancedButton, presetExperimentalButton, runExplainDiv;
let networkCanvas, networkCtx, networkLegend;
let sampleSelect, replayPrevButton, replayPlayButton, replayNextButton, replayTimeline;
let replayMeta, stageFlowRows, topKList, attentionHeads;
let whatIfTempSlider, whatIfTempValue;
let cinemaCanvas, cinemaCtx, cinemaCaption, cinemaMeta;
let cinemaPrevButton, cinemaPlayButton, cinemaNextButton, pipelineMap;
let cinemaEnableToggle, cinemaBody, cinemaProcess;

// ---------------------------------------------------------------------------
// Chart Configuration
// ---------------------------------------------------------------------------

const chartPadding = { top: 20, right: 40, bottom: 40, left: 60 };
let chartWidth, chartHeight;

function updateChartDimensions() {
    if (!lossChart) return;
    chartWidth = lossChart.width - chartPadding.left - chartPadding.right;
    chartHeight = lossChart.height - chartPadding.top - chartPadding.bottom;
}

function resizeCanvasToDisplaySize(canvas) {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const nextWidth = Math.floor(rect.width || canvas.width);
    const nextHeight = Math.floor(rect.height || canvas.height);
    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
        canvas.width = nextWidth;
        canvas.height = nextHeight;
    }
}

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------

function init() {
    // Get DOM elements
    lossChart = document.getElementById('lossChart');
    if (!lossChart) {
        console.error('lossChart element not found');
        return;
    }
    ctx = lossChart.getContext('2d');
    
    // Ensure canvas has proper dimensions
    resizeCanvasToDisplaySize(lossChart);
    
    // Update chart dimensions
    updateChartDimensions();
    
    learningRateSlider = document.getElementById('learningRate');
    learningRateValue = document.getElementById('learningRateValue');
    numStepsSlider = document.getElementById('numSteps');
    numStepsValue = document.getElementById('numStepsValue');
    temperatureSlider = document.getElementById('temperature');
    temperatureValue = document.getElementById('temperatureValue');
    trainButton = document.getElementById('trainButton');
    generateButton = document.getElementById('generateButton');
    presetSafeButton = document.getElementById('presetSafe');
    presetBalancedButton = document.getElementById('presetBalanced');
    presetExperimentalButton = document.getElementById('presetExperimental');
    statusDiv = document.getElementById('status');
    summaryDiv = document.getElementById('summary');
    runExplainDiv = document.getElementById('runExplain');
    historyBody = document.getElementById('historyBody');
    samplesContainer = document.getElementById('samplesContainer');
    currentTemperatureSpan = document.getElementById('currentTemperature');
    networkCanvas = document.getElementById('networkCanvas');
    networkLegend = document.getElementById('networkLegend');
    sampleSelect = document.getElementById('sampleSelect');
    replayPrevButton = document.getElementById('replayPrev');
    replayPlayButton = document.getElementById('replayPlay');
    replayNextButton = document.getElementById('replayNext');
    replayTimeline = document.getElementById('replayTimeline');
    replayMeta = document.getElementById('replayMeta');
    stageFlowRows = document.getElementById('stageFlowRows');
    topKList = document.getElementById('topKList');
    attentionHeads = document.getElementById('attentionHeads');
    whatIfTempSlider = document.getElementById('whatIfTemp');
    whatIfTempValue = document.getElementById('whatIfTempValue');
    cinemaCanvas = document.getElementById('cinemaCanvas');
    cinemaCaption = document.getElementById('cinemaCaption');
    cinemaMeta = document.getElementById('cinemaMeta');
    cinemaPrevButton = document.getElementById('cinemaPrev');
    cinemaPlayButton = document.getElementById('cinemaPlay');
    cinemaNextButton = document.getElementById('cinemaNext');
    pipelineMap = document.getElementById('pipelineMap');
    cinemaEnableToggle = document.getElementById('cinemaEnable');
    cinemaBody = document.getElementById('cinemaBody');
    cinemaProcess = document.getElementById('cinemaProcess');

    if (networkCanvas) {
        networkCtx = networkCanvas.getContext('2d');
        resizeCanvasToDisplaySize(networkCanvas);
    }
    if (cinemaCanvas) {
        cinemaCtx = cinemaCanvas.getContext('2d');
        resizeCanvasToDisplaySize(cinemaCanvas);
    }

    try {
        // Initialize worker
        worker = new Worker('worker.js');
        worker.onmessage = handleWorkerMessage;
        worker.onerror = (error) => {
            console.error('Worker error:', error);
            statusDiv.textContent = `Error: ${error.message || 'Worker failed to load. Make sure worker.js is in the same directory.'}`;
            isTraining = false;
            updateUI();
        };

        // Initialize worker
        worker.postMessage({ type: 'init' });
    } catch (error) {
        console.error('Failed to create worker:', error);
        statusDiv.textContent = `Error: Failed to create worker. ${error.message}`;
    }

    // Setup event listeners
    if (learningRateSlider && learningRateValue) {
        learningRateSlider.addEventListener('input', (e) => {
            learningRateValue.textContent = parseFloat(e.target.value).toFixed(3);
        });
    }

    if (numStepsSlider && numStepsValue) {
        numStepsSlider.addEventListener('input', (e) => {
            numStepsValue.textContent = e.target.value;
        });
    }

    if (temperatureSlider && temperatureValue && currentTemperatureSpan) {
        temperatureSlider.addEventListener('input', (e) => {
            const temp = parseFloat(e.target.value).toFixed(1);
            temperatureValue.textContent = temp;
            currentTemperatureSpan.textContent = temp;
        });
    }

    if (trainButton) {
        trainButton.addEventListener('click', startTraining);
    }
    if (presetSafeButton) {
        presetSafeButton.addEventListener('click', () => applyPreset('safe'));
    }
    if (presetBalancedButton) {
        presetBalancedButton.addEventListener('click', () => applyPreset('balanced'));
    }
    if (presetExperimentalButton) {
        presetExperimentalButton.addEventListener('click', () => applyPreset('experimental'));
    }
    if (generateButton) {
        generateButton.addEventListener('click', generateSamples);
    }
    if (sampleSelect) {
        sampleSelect.addEventListener('change', () => {
            stopReplayPlayback();
            stopCinematicPlayback();
            replayState.sampleIndex = parseInt(sampleSelect.value, 10) || 0;
            replayState.stepIndex = 0;
            cinemaState.stageIndex = 0;
            renderReplayStudio();
        });
    }
    if (replayPrevButton) replayPrevButton.addEventListener('click', replayPrevStep);
    if (replayPlayButton) replayPlayButton.addEventListener('click', toggleReplayPlayback);
    if (replayNextButton) replayNextButton.addEventListener('click', replayNextStep);
    if (cinemaPrevButton) cinemaPrevButton.addEventListener('click', cinematicPrev);
    if (cinemaPlayButton) cinemaPlayButton.addEventListener('click', toggleCinematicPlayback);
    if (cinemaNextButton) cinemaNextButton.addEventListener('click', cinematicNext);
    if (cinemaEnableToggle) {
        cinemaEnableToggle.addEventListener('change', () => {
            setCinematicEnabled(!!cinemaEnableToggle.checked);
        });
    }
    if (whatIfTempSlider) {
        whatIfTempSlider.addEventListener('input', () => {
            if (whatIfTempValue) whatIfTempValue.textContent = parseFloat(whatIfTempSlider.value).toFixed(1);
            renderReplayDetail();
        });
    }

    // Initial chart render
    drawChart();
    drawNetworkGraph();
    renderReplayStudio();
    setCinematicEnabled(false);

    window.addEventListener('resize', () => {
        resizeCanvasToDisplaySize(lossChart);
        resizeCanvasToDisplaySize(networkCanvas);
        resizeCanvasToDisplaySize(cinemaCanvas);
        updateChartDimensions();
        drawChart();
        drawNetworkGraph();
        if (cinemaEnabled) renderCinematicScene();
    });
}

// ---------------------------------------------------------------------------
// Worker Message Handler
// ---------------------------------------------------------------------------

function handleWorkerMessage(e) {
    const { type, ...data } = e.data;

    switch (type) {
        case 'ready':
            if (statusDiv) {
                statusDiv.textContent = `Model ready (vocab size: ${data.vocabSize}, docs: ${data.numDocs})`;
            }
            break;

        case 'progress':
            updateTrainingProgress(data.step, data.total, data.loss, data.runId);
            break;

        case 'complete':
            completeTraining(data.finalLoss, data.runId);
            break;

        case 'samples':
            displaySamples(data.samples);
            hydrateReplayState(data.samples || [], data.vocab || []);
            break;

        case 'network_viz':
            updateNetworkViz(data);
            break;

        case 'error':
            if (statusDiv) {
                statusDiv.textContent = `Error: ${data.message}`;
            }
            stopReplayPlayback();
            stopCinematicPlayback();
            isTraining = false;
            updateUI();
            break;

        case 'warning':
            if (statusDiv) {
                statusDiv.textContent = `Warning: ${data.message}`;
            }
            break;

        case 'reset':
            statusDiv.textContent = 'Model reset';
            stopReplayPlayback();
            stopCinematicPlayback();
            break;
    }
}

function applyPreset(preset) {
    if (!learningRateSlider || !numStepsSlider || !temperatureSlider) return;
    if (isTraining) return;

    const configs = {
        safe: { lr: 0.005, steps: 800, temp: 0.4 },
        balanced: { lr: 0.01, steps: 1000, temp: 0.7 },
        experimental: { lr: 0.03, steps: 1400, temp: 1.2 }
    };

    const selected = configs[preset];
    if (!selected) return;

    learningRateSlider.value = String(selected.lr);
    numStepsSlider.value = String(selected.steps);
    temperatureSlider.value = String(selected.temp);

    if (learningRateValue) learningRateValue.textContent = selected.lr.toFixed(3);
    if (numStepsValue) numStepsValue.textContent = String(selected.steps);
    if (temperatureValue) temperatureValue.textContent = selected.temp.toFixed(1);
    if (currentTemperatureSpan) currentTemperatureSpan.textContent = selected.temp.toFixed(1);

    if (statusDiv) {
        statusDiv.textContent = `Preset applied: ${preset}`;
    }
}

function explainRun(run, finalLoss, previous) {
    if (!run) return '';
    const lr = run.learningRate;
    const steps = run.steps;
    const temp = run.temperature;

    let trendText = `Final loss is ${finalLoss.toFixed(4)}.`;
    if (Number.isFinite(previous)) {
        const improvement = previous - finalLoss;
        const pct = Math.abs((improvement / previous) * 100);
        trendText = improvement >= 0
            ? `Loss improved by ${pct.toFixed(1)}% compared to previous run.`
            : `Loss worsened by ${pct.toFixed(1)}% compared to previous run.`;
    }

    const stabilityText = lr >= 0.025
        ? 'Learning rate is aggressive; if loss is noisy, reduce LR to 0.008-0.015.'
        : lr <= 0.006
            ? 'Learning rate is conservative; increase LR slightly if training is too slow.'
            : 'Learning rate is in a stable middle range for this toy model.';

    const stepText = steps >= 1300
        ? 'Step count is high, useful for convergence but slower to iterate.'
        : steps <= 500
            ? 'Step count is low; increase steps if loss plateaus too early.'
            : 'Step count is balanced for fast iteration and visible improvement.';

    const tempText = temp >= 1.0
        ? 'Temperature is high, so inference will be diverse with lower confidence tokens.'
        : temp <= 0.5
            ? 'Temperature is low, so inference should be more deterministic and repetitive.'
            : 'Temperature is moderate, balancing novelty and coherence.';

    return `${trendText} ${stabilityText} ${stepText} ${tempText}`;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

function startTraining() {
    if (isTraining || !worker) return;

    const learningRate = parseFloat(learningRateSlider.value);
    const steps = parseInt(numStepsSlider.value);
    const temperature = parseFloat(temperatureSlider.value);

    isTraining = true;
    currentRun = {
        id: runs.length + 1,
        color: colorPalette[(runs.length) % colorPalette.length],
        learningRate,
        steps,
        temperature,
        data: []
    };
    runs.push(currentRun);
    chartData.push([]);

    previousLoss = runs.length > 1 ? runs[runs.length - 2].finalLoss : null;

    updateUI();
    summaryDiv.style.display = 'none';
    if (runExplainDiv) {
        runExplainDiv.style.display = 'none';
        runExplainDiv.textContent = '';
    }

    worker.postMessage({
        type: 'train',
        learningRate,
        steps,
        temperature
    });
}

function updateTrainingProgress(step, total, loss, runId) {
    if (currentRun && currentRun.id === runId) {
        currentRun.data.push({ step, loss });
        const runIndex = runs.length - 1;
        chartData[runIndex] = currentRun.data;
        
        if (statusDiv) {
            statusDiv.textContent = `Step ${step} / ${total} | Loss: ${loss.toFixed(4)}`;
        }
        drawChart();
    }
}

function completeTraining(finalLoss, runId) {
    if (currentRun && currentRun.id === runId) {
        currentRun.finalLoss = finalLoss;
        isTraining = false;
        
        // Update summary
        if (summaryDiv) {
            if (previousLoss !== null) {
                const improvement = previousLoss - finalLoss;
                const percent = ((improvement / previousLoss) * 100).toFixed(1);
                if (improvement > 0) {
                    summaryDiv.textContent = `Loss improved from ${previousLoss.toFixed(4)} → ${finalLoss.toFixed(4)} (${percent}% better)`;
                    summaryDiv.style.color = '#10b981';
                } else {
                    summaryDiv.textContent = `Loss changed from ${previousLoss.toFixed(4)} → ${finalLoss.toFixed(4)} (${Math.abs(percent)}% worse)`;
                    summaryDiv.style.color = '#ef4444';
                }
            } else {
                summaryDiv.textContent = `Training complete. Final loss: ${finalLoss.toFixed(4)}`;
                summaryDiv.style.color = '#10b981';
            }
            summaryDiv.style.display = 'block';
        }

        // Update history
        updateHistory();
        if (runExplainDiv) {
            runExplainDiv.textContent = explainRun(currentRun, finalLoss, previousLoss);
            runExplainDiv.style.display = 'block';
        }
        updateUI();
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

function generateSamples() {
    if (isTraining || !worker) return;

    const temperature = parseFloat(temperatureSlider.value);
    stopReplayPlayback();
    stopCinematicPlayback();
    samplesContainer.innerHTML = '<p style="color: #666;">Generating samples...</p>';
    if (replayMeta) replayMeta.textContent = 'Generating trace...';
    if (cinemaEnabled && cinemaCaption) cinemaCaption.textContent = 'Generating trace and cinematic scenes...';

    worker.postMessage({
        type: 'infer',
        temperature,
        count: 10
    });
}

function displaySamples(samples) {
    samplesContainer.innerHTML = '';

    samples.forEach((sample, idx) => {
        const sampleDiv = document.createElement('div');
        sampleDiv.className = 'sample';

        const textDiv = document.createElement('div');
        textDiv.className = 'sample-text';
        textDiv.textContent = sample.text && sample.text.length > 0 ? sample.text : '(empty)';
        sampleDiv.appendChild(textDiv);

        if (sample.tokenProbs && sample.tokenProbs.length > 0) {
            const probsDiv = document.createElement('div');
            probsDiv.className = 'token-probs';

            sample.tokenProbs.forEach((tokenProb, tokenIdx) => {
                const tokenItem = document.createElement('div');
                tokenItem.className = 'token-prob-item';

                const label = document.createElement('span');
                label.className = 'token-label';
                label.textContent = `"${tokenProb.token}"`;
                tokenItem.appendChild(label);

                const barsContainer = document.createElement('div');
                barsContainer.style.display = 'flex';
                barsContainer.style.gap = '4px';
                barsContainer.style.alignItems = 'center';
                barsContainer.style.flexWrap = 'wrap';

                tokenProb.probabilities.forEach(prob => {
                    const barWrapper = document.createElement('div');
                    barWrapper.style.display = 'flex';
                    barWrapper.style.alignItems = 'center';
                    barWrapper.style.gap = '4px';
                    barWrapper.style.marginRight = '8px';

                    const isSelectedChar = prob.char === tokenProb.token || 
                                          (prob.char === '<BOS>' && tokenProb.token === '<BOS>');

                    const charLabel = document.createElement('span');
                    charLabel.style.fontSize = '10px';
                    charLabel.style.color = isSelectedChar ? '#10b981' : '#b0b0b0';
                    charLabel.style.fontFamily = 'monospace';
                    charLabel.style.minWidth = '20px';
                    charLabel.textContent = prob.char;
                    barWrapper.appendChild(charLabel);

                    const barContainer = document.createElement('div');
                    barContainer.className = 'prob-bar-container';
                    barContainer.style.width = '80px';

                    const bar = document.createElement('div');
                    bar.className = 'prob-bar';
                    if (isSelectedChar) {
                        bar.classList.add('selected');
                    }
                    bar.style.width = `${prob.prob * 100}%`;
                    barContainer.appendChild(bar);
                    barWrapper.appendChild(barContainer);

                    const value = document.createElement('span');
                    value.style.fontSize = '10px';
                    value.style.color = '#b0b0b0';
                    value.style.fontVariantNumeric = 'tabular-nums';
                    value.textContent = `${(prob.prob * 100).toFixed(1)}%`;
                    barWrapper.appendChild(value);

                    barsContainer.appendChild(barWrapper);
                });

                tokenItem.appendChild(barsContainer);
                probsDiv.appendChild(tokenItem);
            });

            sampleDiv.appendChild(probsDiv);
        }

        samplesContainer.appendChild(sampleDiv);
    });
}

// ---------------------------------------------------------------------------
// Inference Replay Studio
// ---------------------------------------------------------------------------

function hydrateReplayState(samples, vocab) {
    replayState.samples = (samples || []).map((sample) => {
        if (sample && Array.isArray(sample.trace) && sample.trace.length > 0) {
            return sample;
        }

        const fallbackTrace = (sample && Array.isArray(sample.tokenProbs) ? sample.tokenProbs : []).map((tp, idx) => {
            const topK = Array.isArray(tp.probabilities) ? tp.probabilities.map((p) => ({
                token: p.char,
                prob: p.prob
            })) : [];
            const best = topK.length > 0 ? topK[0] : { token: tp.token || '?', prob: 0 };
            return {
                tokenIndex: idx + 1,
                inputToken: idx === 0 ? '<BOS>' : (sample.text && sample.text[idx - 1]) || '?',
                outputToken: tp.token || best.token || '?',
                confidence: Number.isFinite(best.prob) ? best.prob : 0,
                rawLogits: [],
                topK,
                nodes: [
                    { id: 'input', label: 'Input', value: 1 },
                    { id: 'embed', label: 'Embed', value: 0.5 },
                    { id: 'norm', label: 'Norm', value: 0.5 },
                    { id: 'attn', label: 'Attention', value: 0.5 },
                    { id: 'mlp', label: 'MLP', value: 0.5 },
                    { id: 'logits', label: 'Logits', value: 0.5 },
                    { id: 'output', label: 'Output', value: Number.isFinite(best.prob) ? best.prob : 0 }
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

        return {
            ...(sample || {}),
            trace: fallbackTrace
        };
    });
    replayState.vocab = vocab || [];
    replayState.sampleIndex = 0;
    replayState.stepIndex = 0;
    cinemaState.stageIndex = 0;
    stopReplayPlayback();
    stopCinematicPlayback();
    renderReplayStudio();
    renderCinematicScene();
    updateUI();
}

function stopReplayPlayback() {
    replayState.playing = false;
    if (replayState.timer) {
        clearInterval(replayState.timer);
        replayState.timer = null;
    }
    if (replayPlayButton) replayPlayButton.textContent = 'Play';
}

function toggleReplayPlayback() {
    const trace = getCurrentTrace();
    if (!trace || trace.length === 0) return;
    if (replayState.playing) {
        stopReplayPlayback();
        return;
    }
    stopCinematicPlayback();
    replayState.playing = true;
    if (replayPlayButton) replayPlayButton.textContent = 'Pause';
    replayState.timer = setInterval(() => {
        const current = getCurrentTrace();
        if (!current || current.length === 0) {
            stopReplayPlayback();
            return;
        }
        if (replayState.stepIndex >= current.length - 1) {
            stopReplayPlayback();
            return;
        }
        replayState.stepIndex += 1;
        cinemaState.stageIndex = 0;
        renderReplayStudio();
    }, 300);
}

function replayPrevStep() {
    stopReplayPlayback();
    stopCinematicPlayback();
    replayState.stepIndex = Math.max(0, replayState.stepIndex - 1);
    cinemaState.stageIndex = 0;
    renderReplayStudio();
}

function replayNextStep() {
    stopReplayPlayback();
    stopCinematicPlayback();
    const trace = getCurrentTrace();
    if (!trace || trace.length === 0) return;
    replayState.stepIndex = Math.min(trace.length - 1, replayState.stepIndex + 1);
    cinemaState.stageIndex = 0;
    renderReplayStudio();
}

function getCurrentTrace() {
    const sample = replayState.samples[replayState.sampleIndex];
    if (!sample || !sample.trace) return [];
    return sample.trace;
}

function renderReplayStudio() {
    const traces = replayState.samples.map((s) => s.trace || []);
    const hasSamples = replayState.samples.length > 0;
    const hasData = traces.some((t) => t.length > 0);

    if (sampleSelect) {
        sampleSelect.innerHTML = '';
        replayState.samples.forEach((sample, idx) => {
            const option = document.createElement('option');
            option.value = String(idx);
            option.textContent = `Sample ${idx + 1}`;
            sampleSelect.appendChild(option);
        });
        sampleSelect.disabled = !hasSamples;
        if (sampleSelect.options.length > 0) {
            sampleSelect.value = String(Math.min(replayState.sampleIndex, sampleSelect.options.length - 1));
        }
    }

    if (!hasSamples) {
        if (replayTimeline) replayTimeline.innerHTML = '';
        if (replayMeta) replayMeta.textContent = 'Generate to inspect token-by-token data flow.';
        if (stageFlowRows) stageFlowRows.innerHTML = '';
        if (topKList) topKList.innerHTML = '';
        if (attentionHeads) attentionHeads.innerHTML = '';
        if (cinemaEnabled) renderCinematicScene();
        return;
    }

    if (!hasData) {
        if (replayTimeline) replayTimeline.innerHTML = '';
        if (replayMeta) replayMeta.textContent = 'Replay trace unavailable for current worker payload. Hard refresh and regenerate.';
        if (stageFlowRows) stageFlowRows.innerHTML = '';
        if (topKList) topKList.innerHTML = '';
        if (attentionHeads) attentionHeads.innerHTML = '';
        if (cinemaEnabled) renderCinematicScene();
        return;
    }

    replayState.sampleIndex = Math.min(replayState.sampleIndex, replayState.samples.length - 1);
    const trace = getCurrentTrace();
    replayState.stepIndex = Math.min(replayState.stepIndex, Math.max(0, trace.length - 1));

    renderReplayTimeline(trace);
    renderReplayDetail();
}

function renderReplayTimeline(trace) {
    if (!replayTimeline) return;
    replayTimeline.innerHTML = '';
    trace.forEach((step, idx) => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = `token-chip${idx === replayState.stepIndex ? ' active' : ''}`;
        chip.textContent = step.outputToken;
        chip.title = `Step ${step.tokenIndex}: ${step.inputToken} -> ${step.outputToken}`;
        chip.addEventListener('click', () => {
            stopReplayPlayback();
            stopCinematicPlayback();
            replayState.stepIndex = idx;
            cinemaState.stageIndex = 0;
            renderReplayStudio();
        });
        replayTimeline.appendChild(chip);
    });
}

function softmaxFromLogits(rawLogits, temperature) {
    if (!rawLogits || rawLogits.length === 0) return [];
    const safeTemp = Math.max(0.05, temperature);
    const scaled = rawLogits.map((v) => v / safeTemp);
    const maxVal = Math.max(...scaled);
    const exps = scaled.map((v) => Math.exp(v - maxVal));
    const denom = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((v) => v / denom);
}

function renderReplayDetail() {
    const trace = getCurrentTrace();
    if (!trace || trace.length === 0) return;
    const step = trace[replayState.stepIndex];
    if (!step) return;

    const whatIfTemp = whatIfTempSlider ? parseFloat(whatIfTempSlider.value) : 0.5;
    if (whatIfTempValue) whatIfTempValue.textContent = whatIfTemp.toFixed(1);

    if (replayMeta) {
        replayMeta.textContent = `Step ${step.tokenIndex} | input "${step.inputToken}" -> output "${step.outputToken}" | confidence ${(step.confidence * 100).toFixed(1)}%`;
    }

    renderStageFlow(step);
    renderTopK(step, whatIfTemp);
    renderAttention(step);
    if (cinemaEnabled) renderCinematicScene();
}

function renderStageFlow(step) {
    if (!stageFlowRows) return;
    stageFlowRows.innerHTML = '';
    const layers = [
        { label: 'Embed', value: step.stages.embed || 0 },
        { label: 'Norm', value: step.stages.normIn || 0 },
        { label: 'Attention', value: avg(step.stages.attn || []) },
        { label: 'MLP', value: avg(step.stages.mlp || []) },
        { label: 'Logits', value: step.stages.logits || 0 },
        { label: 'Output P', value: step.confidence || 0 }
    ];
    const maxVal = Math.max(1e-6, ...layers.map((x) => x.value));

    layers.forEach((layer) => {
        const row = document.createElement('div');
        row.className = 'flow-row';
        const label = document.createElement('span');
        label.textContent = layer.label;
        const bar = document.createElement('div');
        bar.className = 'flow-bar';
        const fill = document.createElement('div');
        fill.className = 'flow-fill';
        fill.style.width = `${Math.min(100, (layer.value / maxVal) * 100)}%`;
        bar.appendChild(fill);
        const value = document.createElement('span');
        value.textContent = layer.value.toFixed(3);
        row.appendChild(label);
        row.appendChild(bar);
        row.appendChild(value);
        stageFlowRows.appendChild(row);
    });
}

function renderTopK(step, temp) {
    if (!topKList) return;
    topKList.innerHTML = '';
    let ranked = [];

    if (Array.isArray(step.rawLogits) && step.rawLogits.length > 0) {
        const probs = softmaxFromLogits(step.rawLogits || [], temp);
        ranked = probs
            .map((p, idx) => ({
                idx,
                token: replayState.vocab[idx] || '?',
                prob: p
            }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 10);
    } else if (Array.isArray(step.topK) && step.topK.length > 0) {
        ranked = step.topK
            .map((item, idx) => ({
                idx,
                token: item.token || '?',
                prob: Number.isFinite(item.prob) ? item.prob : 0
            }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 10);
    }

    if (ranked.length === 0) {
        topKList.textContent = 'No top-k data available for this step.';
        return;
    }

    const maxP = Math.max(1e-6, ...ranked.map((r) => r.prob));
    ranked.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'topk-item';
        const label = document.createElement('span');
        label.textContent = `"${item.token}"`;
        const bar = document.createElement('div');
        bar.className = 'flow-bar';
        const fill = document.createElement('div');
        fill.className = 'flow-fill';
        fill.style.width = `${(item.prob / maxP) * 100}%`;
        if (item.token === step.outputToken) {
            fill.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
        }
        bar.appendChild(fill);
        const val = document.createElement('span');
        val.textContent = `${(item.prob * 100).toFixed(1)}%`;
        row.appendChild(label);
        row.appendChild(bar);
        row.appendChild(val);
        topKList.appendChild(row);
    });
}

function renderAttention(step) {
    if (!attentionHeads) return;
    attentionHeads.innerHTML = '';
    const layers = step.attention || [];
    if (layers.length === 0) {
        attentionHeads.textContent = 'No attention data for this step.';
        return;
    }
    const lastLayer = layers[layers.length - 1] || [];
    lastLayer.forEach((headWeights, headIdx) => {
        const wrapper = document.createElement('div');
        const title = document.createElement('div');
        title.className = 'stage-title';
        title.style.marginBottom = '4px';
        title.textContent = `Head ${headIdx + 1}`;
        const row = document.createElement('div');
        row.className = 'attn-row';
        (headWeights || []).forEach((w) => {
            const cell = document.createElement('div');
            cell.className = 'attn-cell';
            const alpha = Math.max(0.1, Math.min(1, w));
            cell.style.backgroundColor = `rgba(56, 189, 248, ${alpha})`;
            cell.title = `${(w * 100).toFixed(1)}%`;
            row.appendChild(cell);
        });
        wrapper.appendChild(title);
        wrapper.appendChild(row);
        attentionHeads.appendChild(wrapper);
    });
}

function getCurrentReplayStep() {
    const trace = getCurrentTrace();
    if (!trace || trace.length === 0) return null;
    return trace[replayState.stepIndex] || null;
}

function setCinematicEnabled(enabled) {
    cinemaEnabled = !!enabled;
    if (cinemaBody) {
        cinemaBody.style.display = cinemaEnabled ? 'block' : 'none';
    }
    if (!cinemaEnabled) {
        stopCinematicPlayback();
        updateUI();
        return;
    }
    renderCinematicScene();
    updateUI();
}

function stopCinematicPlayback() {
    cinemaState.playing = false;
    if (cinemaState.timer) {
        clearInterval(cinemaState.timer);
        cinemaState.timer = null;
    }
    if (cinemaPlayButton) {
        cinemaPlayButton.textContent = 'Autoplay';
    }
}

function toggleCinematicPlayback() {
    const step = getCurrentReplayStep();
    if (!step) return;
    if (cinemaState.playing) {
        stopCinematicPlayback();
        return;
    }
    cinemaState.playing = true;
    if (cinemaPlayButton) cinemaPlayButton.textContent = 'Pause';
    cinemaState.timer = setInterval(() => {
        const maxStage = cinematicStages.length - 1;
        const trace = getCurrentTrace();
        if (!trace || trace.length === 0) {
            stopCinematicPlayback();
            return;
        }

        if (cinemaState.stageIndex < maxStage) {
            cinemaState.stageIndex += 1;
        } else if (replayState.stepIndex < trace.length - 1) {
            replayState.stepIndex += 1;
            cinemaState.stageIndex = 0;
            renderReplayTimeline(trace);
            renderReplayDetail();
            return;
        } else {
            stopCinematicPlayback();
            return;
        }
        renderCinematicScene();
    }, 700);
}

function cinematicPrev() {
    stopCinematicPlayback();
    if (cinemaState.stageIndex > 0) {
        cinemaState.stageIndex -= 1;
    } else {
        const trace = getCurrentTrace();
        replayState.stepIndex = Math.max(0, replayState.stepIndex - 1);
        cinemaState.stageIndex = cinematicStages.length - 1;
        if (trace) renderReplayTimeline(trace);
        renderReplayDetail();
        return;
    }
    renderCinematicScene();
}

function cinematicNext() {
    stopCinematicPlayback();
    const trace = getCurrentTrace();
    if (!trace || trace.length === 0) return;
    if (cinemaState.stageIndex < cinematicStages.length - 1) {
        cinemaState.stageIndex += 1;
        renderCinematicScene();
        return;
    }
    if (replayState.stepIndex < trace.length - 1) {
        replayState.stepIndex += 1;
        cinemaState.stageIndex = 0;
        renderReplayTimeline(trace);
        renderReplayDetail();
    }
}

function stageCaption(stage, step) {
    const token = step.outputToken;
    const confidence = `${(step.confidence * 100).toFixed(1)}%`;
    const embed = (step.stages.embed || 0).toFixed(3);
    const norm = (step.stages.normIn || 0).toFixed(3);
    const attn = avg(step.stages.attn || []).toFixed(3);
    const mlp = avg(step.stages.mlp || []).toFixed(3);
    const logits = (step.stages.logits || 0).toFixed(3);
    switch (stage) {
        case 'embed':
            return `Embedding: "${step.inputToken}" becomes a dense vector. Magnitude now ${embed}.`;
        case 'norm':
            return `Normalization: vector scale is stabilized ${embed} -> ${norm} before mixing context.`;
        case 'attn':
            return `Attention: context routing updates representation to ${attn} using weighted past positions.`;
        case 'mlp':
            return `MLP: nonlinear transformation refines features ${attn} -> ${mlp}.`;
        case 'logits':
            return `Logits: projection to vocabulary preferences with average magnitude ${logits}.`;
        case 'sample':
            return `Sampling: selected token "${token}" with confidence ${confidence}.`;
        default:
            return 'Follow the token as it moves across each transformer stage.';
    }
}

function currentTopAlternatives(step, maxItems = 3) {
    if (!step) return [];
    if (Array.isArray(step.rawLogits) && step.rawLogits.length > 0) {
        const temp = whatIfTempSlider ? parseFloat(whatIfTempSlider.value) : 0.5;
        const probs = softmaxFromLogits(step.rawLogits, temp);
        return probs
            .map((prob, idx) => ({
                token: replayState.vocab[idx] || '?',
                prob
            }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, maxItems);
    }
    if (Array.isArray(step.topK)) {
        return step.topK.slice(0, maxItems).map((x) => ({
            token: x.token || '?',
            prob: Number.isFinite(x.prob) ? x.prob : 0
        }));
    }
    return [];
}

function renderCinematicProcess(stage, step) {
    if (!cinemaProcess) return;
    cinemaProcess.innerHTML = '';
    if (!step) return;

    const embed = step.stages.embed || 0;
    const norm = step.stages.normIn || 0;
    const attnAvg = avg(step.stages.attn || []);
    const mlpAvg = avg(step.stages.mlp || []);
    const logits = step.stages.logits || 0;
    const top3 = currentTopAlternatives(step, 3);
    const topText = top3.map((t) => `"${t.token}" ${(t.prob * 100).toFixed(1)}%`).join(' | ') || 'N/A';
    const deltaNorm = (norm - embed).toFixed(3);
    const deltaMlp = (mlpAvg - attnAvg).toFixed(3);

    const rows = [
        {
            key: 'embed',
            title: 'Input -> Embedding',
            text: `Token "${step.inputToken}" mapped to vector magnitude ${embed.toFixed(3)}.`
        },
        {
            key: 'norm',
            title: 'Embedding -> Normalize',
            text: `Scale adjusted by ${deltaNorm} to ${norm.toFixed(3)} for stable downstream computation.`
        },
        {
            key: 'attn',
            title: 'Normalize -> Attention',
            text: `Context mix magnitude ${attnAvg.toFixed(3)}. Last-layer head strengths are shown below.`
        },
        {
            key: 'mlp',
            title: 'Attention -> MLP',
            text: `Feature remix delta ${deltaMlp} ending at ${mlpAvg.toFixed(3)}.`
        },
        {
            key: 'logits',
            title: 'MLP -> Logits',
            text: `Vocabulary preference field created (avg magnitude ${logits.toFixed(3)}).`
        },
        {
            key: 'sample',
            title: 'Logits -> Sampled Token',
            text: `Picked "${step.outputToken}" (${(step.confidence * 100).toFixed(1)}%). Top options: ${topText}.`
        }
    ];

    rows.forEach((row) => {
        const el = document.createElement('div');
        el.className = 'process-row';
        const isActive = row.key === stage;
        el.style.borderColor = isActive ? '#60a5fa' : '#334155';
        el.style.background = isActive ? 'rgba(30, 64, 175, 0.25)' : 'rgba(15, 23, 42, 0.8)';
        el.innerHTML = `<span class="process-title">${row.title}</span>${row.text}`;
        cinemaProcess.appendChild(el);
    });
}

function renderCinematicScene() {
    if (!cinemaCtx || !cinemaCanvas) return;
    const step = getCurrentReplayStep();

    if (!step) {
        cinemaCtx.clearRect(0, 0, cinemaCanvas.width, cinemaCanvas.height);
        cinemaCtx.fillStyle = '#94a3b8';
        cinemaCtx.font = '14px sans-serif';
        cinemaCtx.fillText('Generate samples to begin cinematic replay.', 16, 28);
        if (cinemaCaption) cinemaCaption.textContent = 'Generate samples to begin guided replay.';
        if (cinemaMeta) cinemaMeta.textContent = 'No token selected.';
        if (pipelineMap) {
            pipelineMap.querySelectorAll('.stage-pill').forEach((pill) => pill.classList.remove('active'));
        }
        if (cinemaProcess) cinemaProcess.innerHTML = '';
        return;
    }

    const stage = cinematicStages[Math.min(cinemaState.stageIndex, cinematicStages.length - 1)];
    if (pipelineMap) {
        pipelineMap.querySelectorAll('.stage-pill').forEach((pill) => {
            pill.classList.toggle('active', pill.dataset.stage === stage);
        });
    }
    if (cinemaCaption) cinemaCaption.textContent = stageCaption(stage, step);
    if (cinemaMeta) {
        cinemaMeta.textContent = `Token step ${step.tokenIndex} | Stage ${cinemaState.stageIndex + 1}/${cinematicStages.length} | "${step.inputToken}" -> "${step.outputToken}"`;
    }
    renderCinematicProcess(stage, step);

    const w = cinemaCanvas.width;
    const h = cinemaCanvas.height;
    cinemaCtx.fillStyle = '#020617';
    cinemaCtx.fillRect(0, 0, w, h);

    const nodes = [
        { key: 'embed', label: 'Embed', value: step.stages.embed || 0 },
        { key: 'norm', label: 'Norm', value: step.stages.normIn || 0 },
        { key: 'attn', label: 'Attn', value: avg(step.stages.attn || []) },
        { key: 'mlp', label: 'MLP', value: avg(step.stages.mlp || []) },
        { key: 'logits', label: 'Logits', value: step.stages.logits || 0 },
        { key: 'sample', label: 'Sample', value: step.confidence || 0 }
    ];
    const maxVal = Math.max(1e-6, ...nodes.map((n) => n.value));
    const left = 40;
    const right = w - 40;
    const y = Math.floor(h / 2) + 10;
    const gap = (right - left) / (nodes.length - 1);

    for (let i = 0; i < nodes.length - 1; i++) {
        const fromX = left + i * gap;
        const toX = left + (i + 1) * gap;
        cinemaCtx.strokeStyle = 'rgba(71, 85, 105, 0.55)';
        cinemaCtx.lineWidth = 2;
        cinemaCtx.beginPath();
        cinemaCtx.moveTo(fromX + 18, y);
        cinemaCtx.lineTo(toX - 18, y);
        cinemaCtx.stroke();
    }

    const activeIndex = cinematicStages.indexOf(stage);
    nodes.forEach((node, idx) => {
        const x = left + idx * gap;
        const normalized = Math.min(1, node.value / maxVal);
        const base = `rgba(59,130,246,${0.28 + normalized * 0.6})`;
        const active = idx === activeIndex;

        cinemaCtx.beginPath();
        cinemaCtx.arc(x, y, active ? 20 : 16, 0, Math.PI * 2);
        cinemaCtx.fillStyle = active ? 'rgba(56,189,248,0.95)' : base;
        cinemaCtx.fill();
        cinemaCtx.strokeStyle = active ? '#e0f2fe' : '#1f2937';
        cinemaCtx.lineWidth = active ? 3 : 2;
        cinemaCtx.stroke();

        cinemaCtx.fillStyle = '#dbeafe';
        cinemaCtx.font = '11px sans-serif';
        cinemaCtx.textAlign = 'center';
        cinemaCtx.fillText(node.label, x, y - 30);
        cinemaCtx.fillStyle = '#93c5fd';
        cinemaCtx.fillText(node.value.toFixed(3), x, y + 34);
    });

    // Animated data particles from start to current stage
    const particleCount = 18;
    const progress = Math.min(1, (cinemaState.stageIndex + 1) / cinematicStages.length);
    const travel = progress * (right - left);
    const now = Date.now() / 1000;
    for (let i = 0; i < particleCount; i++) {
        const phase = ((i / particleCount) + (now * 0.35)) % 1;
        const px = left + phase * travel;
        const py = y + Math.sin((phase * 8) + now * 3) * 8;
        cinemaCtx.beginPath();
        cinemaCtx.arc(px, py, 2.5, 0, Math.PI * 2);
        cinemaCtx.fillStyle = 'rgba(147,197,253,0.85)';
        cinemaCtx.fill();
    }
}

function avg(values) {
    if (!values || values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
}

// ---------------------------------------------------------------------------
// Neural Network Visualization
// ---------------------------------------------------------------------------

function updateNetworkViz(vizData) {
    latestNetworkViz = vizData;
    drawNetworkGraph();
}

function drawNetworkGraph() {
    if (!networkCtx || !networkCanvas) return;

    const width = networkCanvas.width;
    const height = networkCanvas.height;
    networkCtx.fillStyle = '#0f0f0f';
    networkCtx.fillRect(0, 0, width, height);

    if (!latestNetworkViz || !latestNetworkViz.nodes || latestNetworkViz.nodes.length === 0) {
        networkCtx.fillStyle = '#6b7280';
        networkCtx.font = '13px sans-serif';
        networkCtx.fillText('Run training or inference to visualize activations.', 16, 30);
        return;
    }

    const nodes = latestNetworkViz.nodes;
    const values = nodes.map((n) => Number.isFinite(n.value) ? n.value : 0);
    const maxValue = Math.max(1e-6, ...values);
    const minX = 36;
    const maxX = width - 36;
    const y = Math.floor(height / 2);
    const radius = 18;
    const gap = nodes.length > 1 ? (maxX - minX) / (nodes.length - 1) : 0;

    for (let i = 0; i < nodes.length - 1; i++) {
        const from = nodes[i];
        const to = nodes[i + 1];
        const fromValue = Number.isFinite(from.value) ? from.value : 0;
        const toValue = Number.isFinite(to.value) ? to.value : 0;
        const edgeIntensity = Math.max(fromValue, toValue) / maxValue;
        networkCtx.strokeStyle = `rgba(59, 130, 246, ${0.2 + edgeIntensity * 0.8})`;
        networkCtx.lineWidth = 1 + edgeIntensity * 4;
        networkCtx.beginPath();
        networkCtx.moveTo(minX + gap * i + radius, y);
        networkCtx.lineTo(minX + gap * (i + 1) - radius, y);
        networkCtx.stroke();
    }

    nodes.forEach((node, idx) => {
        const x = minX + gap * idx;
        const value = Number.isFinite(node.value) ? node.value : 0;
        const intensity = Math.min(1, value / maxValue);
        const red = Math.round(30 + intensity * 40);
        const green = Math.round(90 + intensity * 120);
        const blue = Math.round(180 + intensity * 60);

        networkCtx.fillStyle = `rgb(${red},${green},${blue})`;
        networkCtx.beginPath();
        networkCtx.arc(x, y, radius, 0, Math.PI * 2);
        networkCtx.fill();

        networkCtx.strokeStyle = '#1f2937';
        networkCtx.lineWidth = 2;
        networkCtx.stroke();

        networkCtx.fillStyle = '#e5e7eb';
        networkCtx.font = '11px sans-serif';
        networkCtx.textAlign = 'center';
        networkCtx.fillText(node.label, x, y - 28);
        networkCtx.fillStyle = '#9ca3af';
        networkCtx.fillText(value.toFixed(3), x, y + 36);
    });

    if (networkLegend) {
        if (latestNetworkViz.mode === 'train') {
            const lossText = Number.isFinite(latestNetworkViz.loss) ? latestNetworkViz.loss.toFixed(4) : '-';
            networkLegend.textContent =
                `Training step ${latestNetworkViz.step}/${latestNetworkViz.total} | Loss ${lossText} | Node color/intensity = activation magnitude`;
        } else if (latestNetworkViz.mode === 'infer') {
            const confidencePct = Number.isFinite(latestNetworkViz.confidence) ? (latestNetworkViz.confidence * 100).toFixed(1) : '-';
            networkLegend.textContent =
                `Inference sample ${latestNetworkViz.sample}, token ${latestNetworkViz.tokenIndex}: "${latestNetworkViz.token}" | Confidence ${confidencePct}%`;
        }
    }
}

// ---------------------------------------------------------------------------
// Chart Rendering
// ---------------------------------------------------------------------------

function drawChart() {
    if (!ctx || !lossChart) return;
    
    // Update dimensions in case canvas was resized
    updateChartDimensions();
    
    // Clear canvas
    ctx.fillStyle = '#0f0f0f';
    ctx.fillRect(0, 0, lossChart.width, lossChart.height);

    if (chartData.length === 0) {
        drawEmptyChart();
        return;
    }

    // Find data bounds
    let minLoss = Infinity;
    let maxLoss = -Infinity;
    let maxStep = 0;

    chartData.forEach(runData => {
        runData.forEach(point => {
            minLoss = Math.min(minLoss, point.loss);
            maxLoss = Math.max(maxLoss, point.loss);
            maxStep = Math.max(maxStep, point.step);
        });
    });

    // Handle edge cases
    if (maxStep === 0) maxStep = 1;
    if (minLoss === Infinity || maxLoss === -Infinity) {
        drawEmptyChart();
        return;
    }

    // Add padding to bounds
    const lossRange = maxLoss - minLoss;
    const padding = lossRange > 0 ? lossRange * 0.1 : 0.1;
    const paddedMinLoss = Math.max(0, minLoss - padding);
    const paddedMaxLoss = maxLoss + padding;

    // Draw grid and axes
    drawAxes(paddedMinLoss, paddedMaxLoss, maxStep);

    // Draw data for each run
    chartData.forEach((runData, runIndex) => {
        if (runData.length === 0) return;
        const run = runs[runIndex];
        drawRun(runData, run.color, paddedMinLoss, paddedMaxLoss, maxStep);
    });
}

function drawEmptyChart() {
    ctx.save();
    ctx.translate(chartPadding.left, chartPadding.top);

    // Draw axes
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 1;

    // X axis
    ctx.beginPath();
    ctx.moveTo(0, chartHeight);
    ctx.lineTo(chartWidth, chartHeight);
    ctx.stroke();

    // Y axis
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, chartHeight);
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#b0b0b0';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Steps', chartWidth / 2, chartHeight + 35);
    
    ctx.save();
    ctx.translate(-45, chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();

    ctx.restore();
}

function drawAxes(minLoss, maxLoss, maxStep) {
    ctx.save();
    ctx.translate(chartPadding.left, chartPadding.top);

    // Draw grid
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 1;

    // Horizontal grid lines (loss)
    const numHLines = 5;
    for (let i = 0; i <= numHLines; i++) {
        const y = (chartHeight / numHLines) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(chartWidth, y);
        ctx.stroke();
    }

    // Vertical grid lines (steps)
    const numVLines = 5;
    for (let i = 0; i <= numVLines; i++) {
        const x = (chartWidth / numVLines) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, chartHeight);
        ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 2;

    // X axis
    ctx.beginPath();
    ctx.moveTo(0, chartHeight);
    ctx.lineTo(chartWidth, chartHeight);
    ctx.stroke();

    // Y axis
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, chartHeight);
    ctx.stroke();

    // Y axis labels (loss)
    ctx.fillStyle = '#b0b0b0';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    const lossRange = maxLoss - minLoss;
    for (let i = 0; i <= numHLines; i++) {
        const loss = maxLoss - lossRange * (i / numHLines);
        const y = (chartHeight / numHLines) * i;
        ctx.fillText(loss.toFixed(2), -10, y + 4);
    }

    // X axis labels (steps)
    ctx.textAlign = 'center';
    for (let i = 0; i <= numVLines; i++) {
        const step = Math.floor((maxStep / numVLines) * i);
        const x = (chartWidth / numVLines) * i;
        ctx.fillText(step.toString(), x, chartHeight + 20);
    }

    // Axis titles
    ctx.fillStyle = '#b0b0b0';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Steps', chartWidth / 2, chartHeight + 35);
    
    ctx.save();
    ctx.translate(-45, chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();

    ctx.restore();
}

function drawRun(runData, color, paddedMinLoss, paddedMaxLoss, maxStep) {
    if (runData.length === 0) return;

    ctx.save();
    ctx.translate(chartPadding.left, chartPadding.top);

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;

    const lossRange = paddedMaxLoss - paddedMinLoss;

    // Draw line
    ctx.beginPath();
    runData.forEach((point, idx) => {
        const x = (point.step / maxStep) * chartWidth;
        const y = chartHeight - ((point.loss - paddedMinLoss) / lossRange) * chartHeight;
        
        if (idx === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    // Draw points
    runData.forEach(point => {
        const x = (point.step / maxStep) * chartWidth;
        const y = chartHeight - ((point.loss - paddedMinLoss) / lossRange) * chartHeight;
        
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
    });

    ctx.restore();
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

function updateHistory() {
    historyBody.innerHTML = '';

    runs.forEach(run => {
        if (run.finalLoss === undefined) return;

        const row = document.createElement('tr');
        
        const colorCell = document.createElement('td');
        const colorIndicator = document.createElement('span');
        colorIndicator.className = 'color-indicator';
        colorIndicator.style.backgroundColor = run.color;
        colorCell.appendChild(colorIndicator);
        colorCell.appendChild(document.createTextNode(` Run ${run.id}`));
        row.appendChild(colorCell);

        const lrCell = document.createElement('td');
        lrCell.textContent = run.learningRate.toFixed(3);
        row.appendChild(lrCell);

        const stepsCell = document.createElement('td');
        stepsCell.textContent = run.steps;
        row.appendChild(stepsCell);

        const lossCell = document.createElement('td');
        lossCell.textContent = run.finalLoss.toFixed(4);
        row.appendChild(lossCell);

        historyBody.appendChild(row);
    });
}

// ---------------------------------------------------------------------------
// UI Updates
// ---------------------------------------------------------------------------

function updateUI() {
    if (!trainButton || !learningRateSlider || !numStepsSlider || !temperatureSlider || !generateButton) return;
    
    trainButton.disabled = isTraining;
    learningRateSlider.disabled = isTraining;
    numStepsSlider.disabled = isTraining;
    temperatureSlider.disabled = isTraining;

    generateButton.disabled = isTraining || runs.length === 0 || runs[runs.length - 1].finalLoss === undefined;

    const hasSamples = replayState.samples.length > 0;
    const hasTrace = replayState.samples.some((s) => s.trace && s.trace.length > 0);
    if (sampleSelect) sampleSelect.disabled = !hasSamples;
    if (replayPrevButton) replayPrevButton.disabled = !hasTrace;
    if (replayNextButton) replayNextButton.disabled = !hasTrace;
    if (replayPlayButton) replayPlayButton.disabled = !hasTrace;
    if (cinemaPrevButton) cinemaPrevButton.disabled = !hasTrace || !cinemaEnabled;
    if (cinemaNextButton) cinemaNextButton.disabled = !hasTrace || !cinemaEnabled;
    if (cinemaPlayButton) cinemaPlayButton.disabled = !hasTrace || !cinemaEnabled;
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

// Wait for DOM to load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
