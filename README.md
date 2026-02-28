# MicroGPT Developer Studio

Interactive playground for learning how a small GPT-style model trains and generates tokens.

Live site: https://renjith-baby.github.io/microgpt/

## What this project is

MicroGPT Developer Studio is a browser-first educational tool that lets developers:
- train a tiny character-level model in the browser,
- inspect training behavior in real time,
- replay inference token-by-token,
- view activation flow and attention snapshots,
- optionally run a cinematic stage walkthrough.

Compute runs in a Web Worker so UI interactions remain responsive.

## Key features

- Real-time loss curve during training
- Presets (`Safe`, `Balanced`, `Experimental`)
- Live network flow visualization
- Inference replay studio with token timeline
- Top-k token probabilities + what-if temperature
- Attention head view (latest layer)
- Opt-in cinematic mode for guided stage-by-stage explanation

## Project structure

- `frontend/` React + Vite + Tailwind UI (primary app)
- `frontend/public/worker.js` Worker used by the frontend
- `frontend/public/input.txt` dataset loaded by the worker
- `worker.js` canonical worker source in repo root
- `input.txt` canonical dataset in repo root
- `run-ui.sh` one-line local launch script
- `.github/workflows/pages.yml` GitHub Pages deployment workflow

## Local development

### One-line run (recommended)

```bash
./run-ui.sh
```

This script installs frontend dependencies (if needed), syncs worker/data into `frontend/public/`, and starts Vite dev server.

### Manual run

```bash
cd frontend
npm install
npm run sync-worker
npm run dev
```

## Build and verify

```bash
cd frontend
npm run sync-worker
npm run build
```

## Deployment (GitHub Pages)

This repo deploys automatically from `main` using GitHub Actions.

- Workflow builds `frontend/dist`
- Pages serves the app at:
  - `https://renjith-baby.github.io/microgpt/`

If GitHub Pages is already configured for GitHub Actions, pushing to `main` triggers deployment.

## Usage flow

1. Choose a preset or tune knobs
2. Click `Train`
3. After training completes, click `Generate`
4. Explore replay timeline and per-step values
5. Enable cinematic mode only when you want guided walkthrough animation

## Troubleshooting

### Stuck on `Booting worker...`

Usually indicates worker path/deploy mismatch.

- Ensure latest deployment is live
- Hard refresh the page (`Ctrl+Shift+R`)
- Confirm `worker.js` is reachable under `/microgpt/worker.js`

### JS/CSS 404 under `/assets/...`

This means wrong Vite base path. Current config uses project-site base:
- `frontend/vite.config.js` -> `base: '/microgpt/'`

### Replay controls have no data

Replay is populated only after `Generate` completes.

## Links

- Repository: https://github.com/renjith-baby/microgpt
- README: https://github.com/renjith-baby/microgpt#readme
