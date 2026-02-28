#!/usr/bin/env bash
set -euo pipefail
npm --prefix frontend install --no-audit --no-fund
npm --prefix frontend run sync-worker
npm --prefix frontend run dev
