#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv2 facade feature pipeline - the AI half the survey script never built.

golden_zone_statistics.py stops at the human data and leaves a note to itself:
sections 7-11 (AI / DINOv2 facade embeddings, cosine distance, UMAP, an "AI
inconsistency" predictor) were never written because there were no images on
hand. This is that section. It is NOT just the PCA picture - the PCA->RGB grid
is only a visualisation OF the patch features. The features themselves, and
everything downstream of them, are the point.

What it does, walking BASE_DIR/images recursively:

  1  extract DINOv2 features for every facade
       - x_norm_clstoken  -> one global descriptor per image  (the embedding)
       - x_norm_patchtokens -> the dense local feature grid
  2  PCA->RGB visualisation of the patch tokens          -> PCA/<rel>_pca.png
                                                          -> PCA/<rel>_compare.png
  3  save the global embeddings                          -> PCA/_features/embeddings.{csv,npz}
  4  facade-to-facade cosine distance matrix             -> PCA/_features/cosine_distance_*.{csv,png}
       + per-image distinctiveness (mean distance to all others)
  5  segment-level AI architectural inconsistency        -> PCA/_features/segment_ai_inconsistency.{csv,png}
       (mean intra-segment cosine distance == the AI analogue of survey construct B)
  6  2D embedding map, UMAP -> t-SNE -> PCA-2D fallback   -> PCA/_features/embedding_map_2d.{csv,png}

That segment_ai_inconsistency.csv is precisely the per-segment AI variable that
sections 8-11 of the statistics script were waiting on. Merge it on segment/scene
and the regression / mediation half becomes runnable.

pip install torch torchvision pillow numpy matplotlib
optional, for a nicer 2D map:  pip install umap-learn   (or scikit-learn for t-SNE)
python golden_zone_dinov2.py
"""

import os
import csv
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ------------------------------- CONFIG -------------------------------
BASE_DIR     = r"d:\msc"                               # folder that contains "images"
IMAGES_DIR   = os.path.join(BASE_DIR, "images")       # walked recursively
RESULTS_DIR  = os.path.join(BASE_DIR, "Results")      # top-level output folder
PCA_DIR      = os.path.join(RESULTS_DIR, "PCA")       # per-image PCA->RGB images go here
FEATURES_DIR = os.path.join(RESULTS_DIR, "Features")  # embeddings + cross-image tables

MODEL_NAME = "dinov2_vits14"
IMG_SIZE   = 224
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# --- PCA->RGB picture options ---
SAVE_PCA_IMAGES = True
PCA_MODE        = "per_image"   # "per_image" (each image its own SVD) or "global" (shared, consistent colours)
SAVE_COMPARISON = True          # facade | PCA->RGB side-by-side
SAVE_RAW_PCA    = True          # standalone PCA->RGB block (nearest-neighbour upscaled)
RAW_UPSCALE     = 384

# --- feature / cross-image analysis options ---
GLOBAL_DESCRIPTOR             = "cls"   # "cls" (CLS token) or "patch_mean" (mean-pooled patches)
SAVE_EMBEDDINGS               = True
COMPUTE_SIMILARITY            = True
COMPUTE_SEGMENT_INCONSISTENCY = True
COMPUTE_EMBED_MAP             = True


# ------------------------------- IO HELPERS ---------------------------
def find_images(root):
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                hits.append(os.path.join(dirpath, f))
    return sorted(hits)


def segment_of(rel):
    """First path component under images/ -> the segment label. Flat -> 'all'."""
    parts = rel.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else "all"


def build_transform():
    return T.Compose([
        T.Resize(IMG_SIZE + 32),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def load_model(device):
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ------------------------------- FEATURES -----------------------------
@torch.no_grad()
def dino_features(model, x):
    """
    One forward pass -> (cls, patch).
      cls   : (D,)    global image descriptor (x_norm_clstoken)
      patch : (P, D)  dense local features (x_norm_patchtokens)
    Falls back to mean-pooled patches if a variant doesn't expose a CLS token.
    """
    out = model.forward_features(x)
    patch = out["x_norm_patchtokens"].squeeze(0).float().cpu().numpy()
    cls = out.get("x_norm_clstoken")
    cls = cls.squeeze(0).float().cpu().numpy() if cls is not None else patch.mean(0)
    return cls, patch


def global_descriptor(cls, patch):
    return patch.mean(0) if GLOBAL_DESCRIPTOR == "patch_mean" else cls


# ------------------------------- PCA -> RGB ---------------------------
def pca_rgb_from_tokens(tok, basis=None, mean_vec=None, gmin=None, grange=None):
    P = tok.shape[0]
    side = int(round(P ** 0.5))
    if side * side != P:
        raise ValueError(f"patch grid isn't square (P={P}); check IMG_SIZE / patch size")
    if basis is None:                              # per-image PCA (the original snippet)
        mv = tok.mean(0, keepdims=True)
        tc = tok - mv
        _, _, vt = np.linalg.svd(tc, full_matrices=False)
        proj = tc @ vt[:3].T
        proj = (proj - proj.min(0)) / (np.ptp(proj, axis=0) + 1e-9)
    else:                                          # shared/global basis
        proj = (tok - mean_vec) @ basis.T
        proj = (proj - gmin) / (grange + 1e-9)
        proj = np.clip(proj, 0.0, 1.0)
    return proj.reshape(side, side, 3)


def fit_global_basis(token_list):
    allt = np.concatenate(token_list, axis=0)
    mean_vec = allt.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(allt - mean_vec, full_matrices=False)
    basis = vt[:3].copy()
    for k in range(3):                             # pin sign -> reproducible colours
        if basis[k][np.argmax(np.abs(basis[k]))] < 0:
            basis[k] = -basis[k]
    proj_all = (allt - mean_vec) @ basis.T
    gmin = proj_all.min(0)
    grange = proj_all.max(0) - gmin
    return basis, mean_vec, gmin, grange


def save_pca_images(stem, src_path, pil, grid):
    written = {}
    if SAVE_RAW_PCA:
        rgb8 = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
        block = Image.fromarray(rgb8, "RGB").resize((RAW_UPSCALE, RAW_UPSCALE), Image.NEAREST)
        p = os.path.join(PCA_DIR, stem + "_pca.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        block.save(p)
        written["pca_png"] = os.path.relpath(p, PCA_DIR)
    if SAVE_COMPARISON and HAVE_MPL:
        if pil is None:
            pil = Image.open(src_path).convert("RGB")
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(pil); ax[0].set_title("facade", fontsize=8); ax[0].axis("off")
        ax[1].imshow(grid); ax[1].set_title("DINOv2 patch features (PCA->RGB)", fontsize=8)
        ax[1].axis("off")
        p = os.path.join(PCA_DIR, stem + "_compare.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written["compare_png"] = os.path.relpath(p, PCA_DIR)
    return written


# ------------------------- CROSS-IMAGE ANALYSIS -----------------------
def l2norm(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def save_embeddings(rels, segs, emb):
    os.makedirs(FEATURES_DIR, exist_ok=True)
    np.savez(os.path.join(FEATURES_DIR, "embeddings.npz"),
             images=np.array(rels), segments=np.array(segs), embeddings=emb)
    p = os.path.join(FEATURES_DIR, "embeddings.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "segment"] + [f"f{j:03d}" for j in range(emb.shape[1])])
        for r, s, row in zip(rels, segs, emb):
            w.writerow([r, s] + [f"{v:.6f}" for v in row])
    print(f"   embeddings -> {p}   ({emb.shape[0]} x {emb.shape[1]})")


def similarity_analysis(rels, segs, emb):
    En = l2norm(emb)
    D = 1.0 - (En @ En.T)
    np.fill_diagonal(D, 0.0)
    n = len(rels)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    with open(os.path.join(FEATURES_DIR, "cosine_distance_matrix.csv"),
              "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["image"] + rels)
        for r, row in zip(rels, D):
            w.writerow([r] + [f"{v:.5f}" for v in row])

    mean_d = D.sum(1) / max(n - 1, 1)
    with open(os.path.join(FEATURES_DIR, "per_image_distinctiveness.csv"),
              "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "segment", "mean_cosine_distance_to_others"])
        for r, s, v in sorted(zip(rels, segs, mean_d), key=lambda t: -t[2]):
            w.writerow([r, s, f"{v:.5f}"])

    if HAVE_MPL and n <= 400:
        order = np.argsort(np.array(segs), kind="stable")
        Dord = D[np.ix_(order, order)]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(Dord, cmap="magma", vmin=0)
        ax.set_title("Facade dissimilarity  (1 - cosine, DINOv2)")
        ax.set_xlabel("facade"); ax.set_ylabel("facade  (grouped by segment)")
        if n <= 40:
            labs = [rels[i] for i in order]
            ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=90, fontsize=5)
            ax.set_yticks(range(n)); ax.set_yticklabels(labs, fontsize=5)
        else:
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, label="cosine distance")
        fig.tight_layout()
        fig.savefig(os.path.join(FEATURES_DIR, "cosine_distance_heatmap.png"), dpi=130)
        plt.close(fig)
    print("   cosine distance matrix + per-image distinctiveness saved")
    return D


def segment_inconsistency(rels, segs, D):
    """Mean pairwise cosine distance inside each segment = AI 'inconsistency'."""
    from collections import defaultdict
    idx = defaultdict(list)
    for i, s in enumerate(segs):
        idx[s].append(i)

    rows = []
    for s in sorted(idx):
        ii = idx[s]
        if len(ii) >= 2:
            sub = D[np.ix_(ii, ii)]
            iu = np.triu_indices(len(ii), k=1)
            rows.append((s, len(ii), float(sub[iu].mean())))
        else:
            rows.append((s, len(ii), float("nan")))

    os.makedirs(FEATURES_DIR, exist_ok=True)
    with open(os.path.join(FEATURES_DIR, "segment_ai_inconsistency.csv"),
              "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["segment", "n_facades", "ai_inconsistency_mean_intra_cosine_distance"])
        for s, k, v in rows:
            w.writerow([s, k, "" if v != v else f"{v:.5f}"])

    valid = [(s, v) for s, k, v in rows if v == v]
    if HAVE_MPL and valid:
        valid.sort(key=lambda t: t[1])
        xs = list(range(len(valid)))
        fig, ax = plt.subplots(figsize=(max(6, len(valid) * 0.5), 4.5))
        ax.bar(xs, [v for _, v in valid], color="#c44e52")
        ax.set_ylabel("mean intra-segment cosine distance")
        ax.set_title("AI architectural inconsistency by segment (DINOv2)")
        ax.set_xticks(xs)
        ax.set_xticklabels([s for s, _ in valid], rotation=40, ha="right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FEATURES_DIR, "segment_ai_inconsistency.png"), dpi=130)
        plt.close(fig)
    print("   segment AI inconsistency saved  (the variable sections 8-11 wanted)")


def embedding_map(rels, segs, emb):
    if len(rels) < 3:
        print("   embedding map skipped (need >= 3 images)")
        return
    Xs = emb - emb.mean(0)
    method, coords = "PCA-2D", None
    try:
        import umap
        nn = max(2, min(15, len(rels) - 1))
        coords = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=0).fit_transform(Xs)
        method = "UMAP"
    except Exception:
        try:
            from sklearn.manifold import TSNE
            per = max(5, min(30, (len(rels) - 1) // 3))
            coords = TSNE(n_components=2, perplexity=per, init="pca",
                          random_state=0).fit_transform(Xs)
            method = "t-SNE"
        except Exception:
            _, _, vt = np.linalg.svd(Xs, full_matrices=False)
            coords = Xs @ vt[:2].T
            method = "PCA-2D"

    os.makedirs(FEATURES_DIR, exist_ok=True)
    with open(os.path.join(FEATURES_DIR, "embedding_map_2d.csv"),
              "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["image", "segment", "x", "y", "method"])
        for r, s, (x, y) in zip(rels, segs, coords):
            w.writerow([r, s, f"{x:.4f}", f"{y:.4f}", method])

    if HAVE_MPL:
        uniq = sorted(set(segs))
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 1)))
        col = {s: palette[i % len(palette)] for i, s in enumerate(uniq)}
        fig, ax = plt.subplots(figsize=(8, 6))
        for s in uniq:
            m = [i for i, ss in enumerate(segs) if ss == s]
            ax.scatter(coords[m, 0], coords[m, 1], s=42, label=s,
                       color=col[s], edgecolor="k", linewidth=0.4)
        ax.set_title(f"DINOv2 facade embedding map ({method})")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        if len(uniq) <= 20:
            ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(FEATURES_DIR, "embedding_map_2d.png"), dpi=130)
        plt.close(fig)
    print(f"   2D embedding map saved ({method})")


def write_manifest(rows):
    path = os.path.join(PCA_DIR, "pca_manifest.csv")
    keys = ["image", "segment", "status", "pca_png", "compare_png"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"   manifest -> {path}")


# -------------------------------- MAIN --------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  |  model: {MODEL_NAME}  |  descriptor: {GLOBAL_DESCRIPTOR}"
          f"  |  pca mode: {PCA_MODE}")

    if not os.path.isdir(IMAGES_DIR):
        raise SystemExit(f"images folder not found: {IMAGES_DIR}")
    images = find_images(IMAGES_DIR)
    if not images:
        raise SystemExit(f"no images under {IMAGES_DIR} (looked for {sorted(EXTS)})")
    print(f"found {len(images)} image(s) under {IMAGES_DIR}")

    os.makedirs(PCA_DIR, exist_ok=True)
    tfm = build_transform()
    model = load_model(device)

    # ---- optional pass 1: shared PCA basis (only when drawing global-colour PCA) ----
    feat_cache = {}
    basis = mean_vec = gmin = grange = None
    if SAVE_PCA_IMAGES and PCA_MODE == "global":
        print("pass 1/2 - collecting patch tokens for a shared PCA basis ...")
        stack = []
        for i, path in enumerate(images, 1):
            rel = os.path.relpath(path, IMAGES_DIR)
            try:
                pil = Image.open(path).convert("RGB")
                x = tfm(pil).unsqueeze(0).to(device)
                cls, patch = dino_features(model, x)
                feat_cache[path] = (cls, patch)
                stack.append(patch)
                print(f"   [{i}/{len(images)}] {rel}")
            except Exception as e:
                print(f"   skip (read/infer failed): {rel}  -> {e}")
        if not stack:
            raise SystemExit("global PCA mode needs at least one readable image; none worked.")
        basis, mean_vec, gmin, grange = fit_global_basis(stack)
        print("   shared basis ready.")

    # ---- main pass: features for every image, PCA picture, accumulate embeddings ----
    where = "pass 2/2 - " if (SAVE_PCA_IMAGES and PCA_MODE == "global") else ""
    print(f"{where}extracting features"
          + (f" + writing PCA images into {PCA_DIR}" if SAVE_PCA_IMAGES else ""))

    rels, segs, gdesc, manifest = [], [], [], []
    for i, path in enumerate(images, 1):
        rel = os.path.relpath(path, IMAGES_DIR)
        seg = segment_of(rel)
        stem = os.path.splitext(rel)[0]
        try:
            pil = None
            if path in feat_cache:
                cls, patch = feat_cache[path]
            else:
                pil = Image.open(path).convert("RGB")
                x = tfm(pil).unsqueeze(0).to(device)
                cls, patch = dino_features(model, x)

            out_paths = {}
            if SAVE_PCA_IMAGES:
                grid = pca_rgb_from_tokens(patch, basis, mean_vec, gmin, grange)
                out_paths = save_pca_images(stem, path, pil, grid)

            rels.append(rel); segs.append(seg)
            gdesc.append(global_descriptor(cls, patch))
            manifest.append({"image": rel, "segment": seg, "status": "ok", **out_paths})
            print(f"   [{i}/{len(images)}] ok   {rel}")
        except Exception as e:
            manifest.append({"image": rel, "segment": seg, "status": f"error: {e}"})
            print(f"   [{i}/{len(images)}] FAIL {rel}  -> {e}")

    write_manifest(manifest)

    if not gdesc:
        raise SystemExit("no features extracted; nothing to analyse.")
    emb = np.vstack(gdesc).astype(np.float64)

    # ---- the 'features etc' the survey script was missing ----
    if SAVE_EMBEDDINGS:
        save_embeddings(rels, segs, emb)
    D = None
    if COMPUTE_SIMILARITY or COMPUTE_SEGMENT_INCONSISTENCY:
        D = similarity_analysis(rels, segs, emb)
    if COMPUTE_SEGMENT_INCONSISTENCY and D is not None:
        segment_inconsistency(rels, segs, D)
    if COMPUTE_EMBED_MAP:
        embedding_map(rels, segs, emb)

    ok = sum(m["status"] == "ok" for m in manifest)
    print(f"\ndone. {ok}/{len(images)} images featurised.")
    print(f"   PCA images   -> {PCA_DIR}")
    print(f"   features     -> {FEATURES_DIR}")


if __name__ == "__main__":
    main()
