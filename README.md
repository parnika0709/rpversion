# Brain Tumor MRI Analysis — Research Prototype

> **Research/educational prototype.** Not a medical device. Not validated for clinical or surgical use. Performance figures reflect a public challenge dataset (BraTS), not real-world clinical data.

A modular, monorepo-based platform for brain MRI tumor segmentation, quantitative analysis, and 3D visualization. Built for a national-level student hackathon.

## Architecture at a Glance

```
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  apps/web        │ ──HTTP─▶│  apps/api        │ ──API──▶│  packages/ml     │
│  React + Three   │         │  FastAPI         │         │  PyTorch + MONAI │
└──────────────────┘         └──────────────────┘         └──────────────────┘
                                      │
                                      ▼
                              ┌──────────────────┐
                              │  SQLite + Disk   │
                              └──────────────────┘
```

The ML inference layer (`packages/ml`) is exposed through a stable `Predictor` interface so models can be swapped (2D Attention U-Net → nnU-Net → 3D U-Net) without touching the backend or frontend.

## Repository Layout

```
brain-tumor-mri/
├── apps/
│   ├── api/          # FastAPI backend
│   └── web/          # React + TypeScript frontend
├── packages/
│   └── ml/           # MONAI/PyTorch training, inference, features
├── configs/          # YAML configs (model, training, eval)
├── docs/             # Architecture, dataset, decisions
├── scripts/          # One-off utilities
├── reports/          # Generated case reports
├── data/             # gitignored — raw, processed, external
└── pyproject.toml    # Workspace root
```

## Quick Start

```bash
# Install workspace
pip install -e packages/ml -e apps/api

# Run backend
uvicorn app.main:app --reload --app-dir apps/api

# Run frontend
cd apps/web && npm install && npm run dev
```

See [`docs/architecture.md`](docs/architecture.md), [`docs/dataset.md`](docs/dataset.md), and [`docs/decisions.md`](docs/decisions.md) for design rationale.

## Scope

**In scope:** segmentation (BraTS WT/TC/ET), quantitative feature extraction, 2D slice viewer, 3D mesh visualization, explainability (uncertainty + saliency), HTML case reports, Anatomical Proximity Analysis (exploratory, atlas-based).

**Out of scope:** tumor type/grade classification (no reliable labels), surgical risk prediction, clinical validation, regulatory clearance, DICOM-to-report autonomous pipelines.

See [`docs/clinical_limitations.md`](docs/clinical_limitations.md) for the complete disclaimer policy.
