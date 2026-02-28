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

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------

let lossChart, ctx, learningRateSlider, learningRateValue, numStepsSlider, numStepsValue;
let temperatureSlider, temperatureValue, trainButton, generateButton, statusDiv;
let summaryDiv, historyBody, samplesContainer, currentTemperatureSpan;

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
    const rect = lossChart.getBoundingClientRect();
    if (lossChart.width !== rect.width || lossChart.height !== rect.height) {
        lossChart.width = rect.width || 600;
        lossChart.height = rect.height || 300;
    }
    
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
    statusDiv = document.getElementById('status');
    summaryDiv = document.getElementById('summary');
    historyBody = document.getElementById('historyBody');
    samplesContainer = document.getElementById('samplesContainer');
    currentTemperatureSpan = document.getElementById('currentTemperature');

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
    if (generateButton) {
        generateButton.addEventListener('click', generateSamples);
    }

    // Initial chart render
    drawChart();
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
            break;

        case 'error':
            if (statusDiv) {
                statusDiv.textContent = `Error: ${data.message}`;
            }
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
            break;
    }
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
        updateUI();
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

function generateSamples() {
    if (isTraining || !worker) return;

    const temperature = parseFloat(temperatureSlider.value);
    samplesContainer.innerHTML = '<p style="color: #666;">Generating samples...</p>';

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
