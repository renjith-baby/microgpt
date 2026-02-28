# GPT Training Visualizer

A browser-based visualization tool for training and inference with a minimal GPT model. Watch the model learn in real-time as it trains on a dataset of names.

## Features

- **Real-time Training Visualization**: Watch loss decrease as the model trains
- **Interactive Controls**: Adjust learning rate, number of steps, and temperature
- **Multiple Training Runs**: Compare different training runs with color-coded loss curves
- **Inference Generation**: Generate samples and see per-token probability distributions
- **No Dependencies**: Pure vanilla JavaScript, no frameworks or bundlers required

## How to Run

### Option 1: Direct File Opening
Simply double-click `index.html` to open it in your browser.

**Note**: Some browsers may block Web Workers when opening files directly. If you see errors, use Option 2.

### Option 2: Local Server (Recommended)

Using Python 3:
```bash
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

Using Python 2:
```bash
python -m SimpleHTTPServer 8000
```

Using Node.js (with `http-server`):
```bash
npx http-server
```

## Usage

### Training

1. Adjust the sliders:
   - **Learning Rate**: Controls how fast the model learns (0.001 - 0.1)
   - **Number of Steps**: How many training iterations to run (100 - 2000)
   - **Temperature**: Controls randomness in generation (0.1 - 2.0)

2. Click **Train** to start training

3. Watch the loss chart update in real-time

4. After training completes, see the summary comparing to the previous run

5. Each training run is shown in a different color on the chart

### Inference

1. After at least one training run completes, click **Generate**

2. The model will generate 10 sample names

3. Below each generated name, see the probability distribution for each token:
   - Green bars indicate the selected token
   - Blue bars show alternative tokens the model considered
   - Percentages show the model's confidence

### Multiple Training Runs

- Weights persist across training runs within a session
- Each run continues from where the previous one left off
- Compare runs visually using the color-coded loss chart
- View run history in the table below the chart

## Model Architecture

- **Layers**: 1 transformer layer
- **Embedding Dimension**: 16
- **Attention Heads**: 4
- **Context Length**: 16 characters
- **Parameters**: ~4,192

## Technical Details

- Model runs in a Web Worker to keep UI responsive
- Uses automatic differentiation (autograd) for backpropagation
- Adam optimizer with adaptive learning rates
- Character-level tokenization
- Dataset: Loaded from `input.txt` (full file, one name per line). The bundled file currently has 32,032 names.

## Files

- `index.html` - Main HTML layout and styling
- `app.js` - UI logic, worker communication, chart rendering
- `worker.js` - Model implementation, training, and inference
- `microgpt.js` - Original compiled model (not used in browser version)

## Browser Compatibility

Works in modern browsers that support:
- Web Workers
- Canvas API
- ES6 JavaScript

Tested in Chrome, Firefox, Safari, and Edge.

## Troubleshooting

**Worker errors when opening directly:**
- Use a local server (Option 2 above)

**Chart not updating:**
- Check browser console for errors
- Ensure JavaScript is enabled

**Model not training:**
- Check that worker.js is in the same directory
- Verify browser console for error messages

