# ai-erbil-golden-zone-inconsistency-model

**The Psychological Effects of Inconsistent Architectural Styles in Erbil: Causes, Perceptions, and AI-Supported Solutions**

## Overview

This repository studies how inconsistent architectural styles in Erbil's Golden Zone affect human perception and psychological comfort, and validates those perceptual findings against an objective, computer-vision measure of architectural inconsistency. The project combines a **structured perception survey** with a **DINOv2-based facade analysis pipeline**, allowing subjective human ratings to be tested against quantitative visual inconsistency scores derived directly from street-level imagery.

The workflow spans two complementary tracks that feed into a unified analysis:

1. **Survey Track** — 98 respondents rate 8 Golden Zone streetscapes across six psychological constructs, generating human-perception baseline data.
2. **AI Track** — DINOv2 (self-supervised ViT-S/14) embeds each facade photograph, computes global descriptors, and derives per-segment *AI architectural inconsistency* scores (mean intra-segment cosine distance of embeddings).

Both merge on segment/scene to enable regression, mediation, and correlation analysis linking visual inconsistency → psychological perception → emotional/cognitive outcomes.

---

## Project Components

### 1. Perception Survey & Statistical Analysis  
**Status: ✅ Fully Implemented**

A 98-respondent Likert-7 survey across 8 streetscape images, measuring six psychological constructs:

| Code | Construct | Notes |
|------|-----------|-------|
| A | Visual Coherence | how unified/ordered the facades appear |
| B | Architectural Inconsistency | degree of style mismatch (reversed in index) |
| C | Emotional Comfort | how pleasant/safe the space feels |
| D | Cognitive Load | how much effort to process the scene (reversed in index) |
| E | Sense of Place | identity & distinctiveness |
| F | Aesthetic Preference | beauty & appeal |

**File:** `golden_zone_analysis_R2.py`

**Features:**

- **Data Preparation** (Section 0–1)
  - Automatic item-code parsing from column headers (`[A1.1]` → `A1_1`)
  - Straightline detection (respondents with SD < 0.30 flagged)
  - Reverse-recoding (construct B, D reversed only for the overall index)
  - Descriptive frequency tables
  
- **Construct Scoring** (Section 2)
  - Per-construct mean scores across all respondents
  - Per-respondent mean scores across all images
  - Per-segment (image) mean scores
  - Overall Positive Perception Index (reverse-coded B + D)

- **Descriptive Analysis** (Section 3)
  - Ranking of all segments by construct
  - Radar charts (one per construct)
  - Heatmaps (construct × segment)
  - Histograms per construct

- **Reliability & Validity** (Section 4)
  - Cronbach's α and McDonald's ω per construct
  - Item-total correlations
  - Alpha-if-deleted analysis
  - Exploratory Factor Analysis (EFA) with Kaiser varimax rotation
  - Kaiser-Meyer-Olkin (KMO) and Bartlett sphericity tests
  
- **Within-Image Variation** (Section 5)
  - Repeated-measures ANOVA (F-test, partial η²)
  - Friedman test + Kendall's W
  - Pairwise Wilcoxon tests (Holm-corrected)
  - Effect-size visualization

- **Demographic / Moderator Analysis** (Section 6)
  - Gender: t-test, Mann-Whitney U, Cohen's d
  - Age & Education: ANOVA, Kruskal-Wallis, eta²
  - Familiarity: Spearman rank correlation

**Outputs:** `analysis_outputs/` folder containing 15+ CSV tables and PNG figures.

---

### 2. DINOv2 Facade Feature Pipeline  
**Status: ✅ Fully Implemented**

Walks a folder tree of facade photographs, extracts **DINOv2 ViT-S/14** visual embeddings, and produces the *AI architectural inconsistency* variable that unlocks Sections 8–11 of the survey analysis.

**File:** `golden_zone_dinov2.py`

**What it does:**

1. **Feature Extraction** (one forward pass per image)
   - CLS token global descriptor (one vector per facade)
   - Patch tokens local feature grid (P × D dense grid)

2. **Visualization: PCA→RGB**
   - Per-image or global SVD on patch tokens → 3 principal components
   - Render as RGB grid (upscaled to 384×384 by nearest-neighbour)
   - Optional side-by-side original | PCA composite

3. **Embeddings & Similarity**
   - Save global descriptors as CSV + NumPy archive
   - Cosine distance matrix (N × N)
   - Per-image distinctiveness (mean distance to all other facades)

4. **AI Architectural Inconsistency**
   - **Key output:** mean intra-segment cosine distance
   - Interpreted as: facades within segment *s* have high visual heterogeneity if score is high
   - Direct AI analogue of survey Construct B (Architectural Inconsistency)

5. **2D Embedding Map**
   - Dimensionality reduction (UMAP → t-SNE → PCA-2D fallback)
   - Scatter plot coloured by segment
   - CSV with (x, y, segment) per image

**Outputs:**

```
PCA/
├── <relative_path>_pca.png              # upscaled PCA->RGB block
├── <relative_path>_compare.png          # facade | PCA side-by-side
├── pca_manifest.csv                     # per-image log (path, status, output files)
└── _features/
    ├── embeddings.csv                   # (image, segment, feature_1, ..., feature_384)
    ├── embeddings.npz                   # NumPy archive of same
    ├── cosine_distance_matrix.csv       # N × N pairwise distances
    ├── cosine_distance_heatmap.png      # heatmap visualisation
    ├── per_image_distinctiveness.csv    # (image, mean_distance_to_all_others)
    ├── segment_ai_inconsistency.csv     # ⭐ **THE KEY FILE** (segment, n_facades, ai_inconsistency)
    ├── segment_ai_inconsistency.png     # bar chart
    ├── embedding_map_2d.csv             # (image, segment, x, y, method)
    └── embedding_map_2d.png             # scatter plot
```

---

## Connecting Survey ↔ AI: The Merged Analysis

Once both pipelines run, merge the outputs on segment/scene:

```python
import pandas as pd

survey = pd.read_csv("analysis_outputs/construct_scores.csv")  # survey data (per-segment)
ai     = pd.read_csv("PCA/_features/segment_ai_inconsistency.csv")  # AI data

merged = survey.merge(ai[["segment", "ai_inconsistency_mean_intra_cosine_distance"]],
                      on="segment", how="left")
```

**Now the unified analysis runs:**
- Sections 8–10: Regression with AI inconsistency as a predictor of emotional comfort, cognitive load, aesthetic preference
- Section 9: Mixed-effects models (segment-level AI, respondent-level human perception)
- Section 11: Mediation analysis (visual inconsistency → perception → psychological outcome)

---

## Configuration & Customization

### Survey Analysis (`golden_zone_analysis_R2.py`)

Top of file, marked `# CONFIG`:

```python
INPUT_CSV           = "golden_zone_synthetic_98.csv"  # path to survey data
OUTPUT_DIR          = "analysis_outputs"              # where to write results
SCALE_MIN, SCALE_MAX = 1, 7                           # Likert range (1–7)
STRAIGHTLINE_SD     = 0.30                            # flag respondents below this
```

Constructs (A–F) and reverse-coding targets are hardcoded in `CONSTRUCT_NAME` and `REVERSE_FOR_INDEX` dicts.

### DINOv2 Analysis (`golden_zone_dinov2.py`)

Top of file, marked `# CONFIG`:

```python
BASE_DIR            = r"d:\msc"                     # root dir; images/ and PCA/ resolve from here
IMAGES_DIR          = os.path.join(BASE_DIR, "images")  # walked recursively
PCA_DIR             = os.path.join(BASE_DIR, "PCA")
FEATURES_DIR        = os.path.join(PCA_DIR, "_features")

MODEL_NAME          = "dinov2_vits14"               # dinov2_vits14, vitb14, vitl14, vitg14
IMG_SIZE            = 224                           # resize + centre-crop
GLOBAL_DESCRIPTOR   = "cls"                         # "cls" or "patch_mean"

# --- PCA->RGB image output ---
SAVE_PCA_IMAGES     = True
PCA_MODE            = "per_image"                   # "per_image" or "global"
SAVE_COMPARISON     = True                          # facade | PCA side-by-side
SAVE_RAW_PCA        = True
RAW_UPSCALE         = 384                           # pixel size of upscaled block

# --- feature/cross-image analysis ---
SAVE_EMBEDDINGS               = True
COMPUTE_SIMILARITY            = True
COMPUTE_SEGMENT_INCONSISTENCY = True
COMPUTE_EMBED_MAP             = True
```

**Image organization** — DINOv2 reads from `BASE_DIR/images/` recursively:

```
images/
├── segment_A/
│   ├── facade_01.jpg
│   ├── facade_02.jpg
│   └── ...
├── segment_B/
│   └── ...
└── flat_image.jpg            # no subfolder → segment "all"
```

Segment is inferred from the first path component; images directly under `images/` are labeled as segment `"all"`.

---

## Installation & Usage

### Survey Analysis Only

```bash
pip install pandas numpy scipy statsmodels matplotlib
python golden_zone_analysis_R2.py
```

Outputs: `./analysis_outputs/` (tables + charts).

### DINOv2 Analysis Only

```bash
pip install torch torchvision pillow numpy matplotlib
# optional, for better 2D embedding map:
pip install umap-learn    # preferred
# or
pip install scikit-learn   # t-SNE fallback
python golden_zone_dinov2.py
```

Outputs: `./PCA/` and `./PCA/_features/` (embeddings, heatmaps, AI inconsistency scores).

### Full Pipeline (Survey + AI)

```bash
# Install both sets of dependencies
pip install pandas numpy scipy statsmodels matplotlib torch torchvision pillow
pip install umap-learn  # optional

# 1. Run survey analysis
python golden_zone_analysis_R2.py

# 2. Run DINOv2 feature extraction
python golden_zone_dinov2.py

# 3. Merge outputs
# (use script snippet above or integrate into unified analysis script)
```

---

## File Manifest

### Input Data

| File | Purpose |
|------|---------|
| `golden_zone_synthetic_98.csv` | 98 respondents × 48 Likert items (6 constructs × 8 scenes) + demographics |
| `images/` folder tree | Facade photographs (structured by segment) |

### Analysis Scripts

| File | Role |
|------|------|
| `golden_zone_analysis_R2.py` | Survey perception data → 6 psychological construct scores + demographic contrasts |
| `golden_zone_dinov2.py` | Facade images → DINOv2 embeddings → per-segment AI inconsistency |

### Generated Outputs

| Folder | Contents |
|--------|----------|
| `analysis_outputs/` | Survey descriptive tables, reliability stats, effect-size charts |
| `PCA/` | Per-image PCA→RGB visualisations + manifest |
| `PCA/_features/` | Embeddings, cosine distance tables, **AI inconsistency scores**, 2D map |

---

## Current Status

| Component | Status |
|-----------|--------|
| Survey screening & scoring | ✅ Implemented |
| Descriptive statistics | ✅ Implemented |
| Reliability (α, ω, EFA) | ✅ Implemented |
| Within-image ANOVA/Friedman | ✅ Implemented |
| Demographic analysis | ✅ Implemented |
| **DINOv2 feature extraction** | ✅ **Implemented** |
| **Facade embeddings & similarity** | ✅ **Implemented** |
| **AI inconsistency scoring** | ✅ **Implemented** |
| **2D embedding map** | ✅ **Implemented** |
| AI–survey correlation (Section 8) | ⏳ Ready to run (merge + `pd.corr()`) |
| Regression with AI predictor (Section 9) | ⏳ Ready to scaffold |
| Mixed-effects models (Section 10) | ⏳ Ready to scaffold |
| Mediation / path analysis (Section 11) | ⏳ Scaffolded (needs larger N) |
| Spatial analysis (Moran's I) | ⏳ Awaits scene coordinates |

---

## Requirements

### Core Dependencies

```
Python 3.9+
pandas, numpy, scipy, statsmodels   # survey track
torch, torchvision, pillow          # AI track
matplotlib                          # both
```

### Optional

```
umap-learn        # high-quality 2D embedding map (preferred)
scikit-learn      # t-SNE fallback
```

GPU (CUDA) is not required but speeds up DINOv2 inference substantially.

---

## References

**DINOv2:**
> Oquab, M., Darcet, T., Moutakanni, T., *et al.* (2023). **DINOv2: Learning Robust Visual Features without Supervision.** *Transactions on Machine Learning Research.*

**Survey Analysis Methods:**
- Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. *Psychometrika*, 16(3), 297–334.
- McDonald, R. P. (1999). Test theory: A unified treatment. Lawrence Erlbaum Associates.
- Kaiser, H. F. (1974). An index of factorial simplicity. *Psychometrika*, 39(1), 31–36.

---

## Citation / Context

This repository supports the research project *"The Psychological Effects of Inconsistent Architectural Styles in Erbil: Causes, Perceptions, and AI-Supported Solutions,"* which examines the Golden Zone district as a case study in urban architectural coherence and its psychological impact on residents and visitors.

**Key Question:** How well does machine-vision derived architectural inconsistency (via DINOv2) predict human-reported psychological comfort and cognitive load?
