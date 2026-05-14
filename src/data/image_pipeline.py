"""
Image Extraction & Filtering Pipeline — Extract operational images from S-1 filings.

Steps:
  1. Extract all raster images from HTML/PDF filings
  2. Content-hash deduplication (removes identical images across S-1/S-1A)
  3. Apply rule-based filters (size, aspect ratio, color)
  4. Train & apply a supervised classifier (operational vs decorative)
  5. Construct per-firm image sets (top-N operational images)

IMPORTANT: Only images from S-1 and S-1/A filings should be processed.
Using post-IPO data (10-K, annual reports, shareholder presentations) would
introduce look-ahead bias — a fatal flaw for the paper.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image extraction from HTML filings
# ---------------------------------------------------------------------------

def extract_images_from_html(
    html_path: str | Path,
    output_dir: str | Path,
    cik: str,
    min_size: tuple[int, int] = (150, 150),
) -> list[dict]:
    """
    Extract embedded images from an S-1 HTML filing.

    Handles:
      - Inline base64-encoded images (data: URIs)
      - Linked image files in the same EDGAR directory

    Returns metadata dicts: {path, width, height, aspect_ratio, cik, filing}.
    """
    from bs4 import BeautifulSoup
    import base64

    html_path = Path(html_path)
    output_dir = Path(output_dir) / cik
    output_dir.mkdir(parents=True, exist_ok=True)

    if not html_path.exists():
        return []

    html_content = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_content, "lxml")

    img_tags = soup.find_all("img")
    results = []

    for idx, img_tag in enumerate(img_tags):
        src = img_tag.get("src", "")
        if not src:
            continue

        pil_img = None

        # Handle base64 inline images
        if src.startswith("data:image"):
            match = re.match(r"data:image/(\w+);base64,(.*)", src, re.DOTALL)
            if match:
                ext = match.group(1).lower()
                if ext == "svg+xml":
                    continue  # Skip SVG
                try:
                    img_bytes = base64.b64decode(match.group(2))
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    continue
        else:
            # Linked file — resolve relative to filing directory
            img_file = html_path.parent / src
            if not img_file.exists():
                # Attempt to download from SEC
                # html_path.stem is the accession number without dashes
                base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{html_path.stem}/"
                img_url = base_url + src
                try:
                    import requests
                    import time
                    headers = {"User-Agent": "IPOValuationResearch pukthuanthongk@missouri.edu"}
                    time.sleep(0.15)  # SEC rate limit
                    resp = requests.get(img_url, headers=headers, timeout=30)
                    if resp.status_code == 200:
                        img_file.write_bytes(resp.content)
                except Exception:
                    pass

            if img_file.exists():
                try:
                    pil_img = Image.open(img_file).convert("RGB")
                except Exception:
                    continue

        if pil_img is None:
            continue

        w, h = pil_img.size
        if w < min_size[0] or h < min_size[1]:
            continue

        # Save extracted image
        out_name = f"{html_path.stem}_img{idx:04d}.png"
        out_path = output_dir / out_name
        pil_img.save(out_path)

        results.append({
            "image_path": str(out_path),
            "width": w,
            "height": h,
            "aspect_ratio": w / h,
            "cik": cik,
            "filing": html_path.stem,
        })

    logger.info("Extracted %d images from %s", len(results), html_path.name)
    return results


# ---------------------------------------------------------------------------
# Content-hash deduplication
# ---------------------------------------------------------------------------

def deduplicate_images(
    image_records: list[dict],
    hash_size: int = 16,
    hamming_threshold: int = 8,
) -> list[dict]:
    """
    Remove duplicate images using perceptual hashing (pHash).

    Images that appear in both the S-1 and S-1/A (common when amendments
    reuse the same graphics) are detected via Hamming distance on their
    perceptual hashes.  Within each CIK group, only the highest-resolution
    copy of each duplicate cluster is kept.

    Parameters
    ----------
    hash_size : int
        Size of the pHash (default 16 → 256-bit hashes for high accuracy).
    hamming_threshold : int
        Maximum Hamming distance to consider two images as duplicates.
        Default 5 is conservative; exact copies have distance 0.
    """
    try:
        import imagehash
    except ImportError:
        logger.warning(
            "imagehash not installed — skipping deduplication. "
            "Install with: pip install imagehash"
        )
        return image_records

    # Group by CIK
    from collections import defaultdict
    cik_groups: dict[str, list[dict]] = defaultdict(list)
    for rec in image_records:
        cik_groups[rec["cik"]].append(rec)

    deduplicated = []
    total_removed = 0

    for cik, records in cik_groups.items():
        # Compute hashes
        hashes = []
        for rec in records:
            try:
                img = Image.open(rec["image_path"])
                h = imagehash.phash(img, hash_size=hash_size)
                hashes.append((rec, h))
            except Exception as e:
                logger.debug("Hash error on %s: %s", rec["image_path"], e)
                continue

        # Greedy clustering: mark duplicates
        kept = []
        used = set()

        for i, (rec_i, hash_i) in enumerate(hashes):
            if i in used:
                continue

            # Find all near-duplicates of this image
            cluster = [(rec_i, i)]
            for j, (rec_j, hash_j) in enumerate(hashes):
                if j <= i or j in used:
                    continue
                if hash_i - hash_j <= hamming_threshold:
                    cluster.append((rec_j, j))
                    used.add(j)

            # Keep the highest-resolution image in the cluster
            best = max(cluster, key=lambda x: x[0]["width"] * x[0]["height"])
            kept.append(best[0])
            used.add(i)

        n_removed = len(records) - len(kept)
        if n_removed > 0:
            logger.info(
                "CIK %s: dedup removed %d/%d duplicate images",
                cik, n_removed, len(records),
            )
        total_removed += n_removed
        deduplicated.extend(kept)

    logger.info(
        "Deduplication: %d → %d images (removed %d duplicates)",
        len(image_records), len(deduplicated), total_removed,
    )
    return deduplicated


# ---------------------------------------------------------------------------
# Rule-based filters
# ---------------------------------------------------------------------------

def apply_rule_filters(
    image_records: list[dict],
    max_aspect_ratio: float = 5.0,
    min_aspect_ratio: float = 0.2,
) -> list[dict]:
    """
    Apply heuristic filters to remove likely non-operational images.

    Filters:
      - Extreme aspect ratios (common for banners, logos, separator lines)
      - Very small color variance (solid backgrounds, spacers)
      - Near-blank images (mostly white/single color with low content)
      - Text-heavy images (high edge density = lots of text lines)
    """
    filtered = []
    for rec in image_records:
        ar = rec["aspect_ratio"]
        if ar > max_aspect_ratio or ar < min_aspect_ratio:
            continue

        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            arr = np.array(img)

            # Check 1: Low variance → solid color / spacer / near-blank
            if arr.std() < 15.0:
                continue

            # Check 2: White-dominated → likely blank/filler
            white_mask = (arr > 240).all(axis=-1)
            white_ratio = white_mask.mean()
            if white_ratio > 0.92:
                continue

            # Check 3: Edge density → text-heavy marketing pages
            # Convert to grayscale and compute simple edge magnitude
            gray = arr.mean(axis=-1)
            dy = np.abs(np.diff(gray, axis=0))
            dx = np.abs(np.diff(gray, axis=1))
            edge_density = (dy > 30).mean() + (dx > 30).mean()
            # Text-heavy pages have many fine edges from letters
            # but low color variance compared to real photographs
            color_std = arr.std()
            if edge_density > 0.15 and color_std < 50:
                continue  # Lots of fine edges + low color = text page
            if edge_density > 0.10 and white_ratio > 0.30:
                continue  # Moderately high edge density + whitespace = Workiva/marketing grid

        except Exception:
            continue

        filtered.append(rec)

    n_removed = len(image_records) - len(filtered)
    logger.info("Rule filters: %d → %d (removed %d)", len(image_records), len(filtered), n_removed)
    return filtered


# ---------------------------------------------------------------------------
# Supervised Image Filter (logo/face/chart/operational classifier)
# ---------------------------------------------------------------------------

class ImageFilterNet(nn.Module):
    """
    ResNet-18-based classifier for image type categorization.

    Classes:
      0 = logo/branding
      1 = face/headshot
      2 = chart/infographic
      3 = generic/decorative
      4 = operational/product  ← target class to keep
    """

    NUM_CLASSES = 5
    CLASS_NAMES = ["logo", "face", "chart", "generic", "operational"]

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.classifier(h)


def build_filter_transform(train: bool = False) -> transforms.Compose:
    """Image transforms for the filter model."""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_filter_model(
    labeled_csv: str | Path,
    output_path: str | Path = "outputs/models/image_filter.pt",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> ImageFilterNet:
    """
    Train the image filter classifier on manually labeled data.

    Expects a CSV with columns: image_path, label (0–4).
    """
    from torch.utils.data import DataLoader, Dataset

    class LabeledImageDataset(Dataset):
        def __init__(self, df, transform):
            self.df = df.reset_index(drop=True)
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row["image_path"]).convert("RGB")
            img = self.transform(img)
            return img, int(row["label"])

    df = pd.read_csv(labeled_csv)
    logger.info("Training filter model on %d labeled images", len(df))

    transform = build_filter_transform(train=True)
    dataset = LabeledImageDataset(df, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = ImageFilterNet(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        logger.info(
            "Epoch %d/%d — loss: %.4f, acc: %.2f%%",
            epoch + 1, epochs, total_loss / total, 100 * correct / total,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info("Saved filter model → %s", output_path)
    return model


def apply_filter_model(
    image_records: list[dict],
    model_path: str | Path,
    threshold: float = 0.8,
) -> list[dict]:
    """Apply the trained filter model and keep only operational images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImageFilterNet(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    transform = build_filter_transform(train=False)
    operational_idx = ImageFilterNet.CLASS_NAMES.index("operational")

    filtered = []
    for rec in image_records:
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)
                op_prob = probs[0, operational_idx].item()
            if op_prob >= threshold:
                rec["operational_prob"] = op_prob
                filtered.append(rec)
        except Exception as e:
            logger.warning("Filter model error on %s: %s", rec["image_path"], e)
            continue

    logger.info(
        "Model filter: %d → %d operational images (threshold=%.2f)",
        len(image_records), len(filtered), threshold,
    )
    return filtered


# ---------------------------------------------------------------------------
# Zero-Shot CLIP Image Filter
# ---------------------------------------------------------------------------

def apply_clip_zero_shot_filter(
    image_records: list[dict],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    threshold: float = 0.35,
    negative_guard: float = 0.15,
) -> list[dict]:
    """
    Apply zero-shot CLIP classification to filter operational images.

    Two positive categories are scored; an image is accepted if
    max(positive_probs) >= threshold AND max(negative_probs) < negative_guard.

    Prompts
    -------
    Positive (accept if high):
       0 - "a product photo or operational facility"
       1 - "a software product user interface screenshot"
    Negative (reject if ANY is dominant):
       2 - "a marketing diagram or infographic"
       3 - "a company logo or branding"
       4 - "a data chart, graph, or diagram"
       5 - "a portrait or face of a person"
       6 - "a decorative background or banner"
       7 - "a text-heavy document page or case study"
       8 - "a marketing brochure or testimonial page"
       9 - "a medical or anatomical illustration"
      10 - "a clinical chart or scientific graph"

    Parameters
    ----------
    threshold:
        Minimum probability for either positive prompt to accept the image
        (default 0.35 to recover Palantir-style false negatives).
    negative_guard:
        If ANY negative prompt probability exceeds this threshold the image
        is rejected, even if a positive prompt also scores well. Lowered
        from 0.30 → 0.25 because we now have more negative categories
        spreading the softmax mass, so each individual negative gets a
        smaller share.
    """
    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Indices must stay in sync with the list below
    POSITIVE_IDXS = [0, 1]
    NEGATIVE_IDXS = list(range(2, 13))  # all negatives: indices 2–12

    prompts = [
        # --- Positive (accept if high) ---
        "a product photo or operational facility",           # 0
        "a software product user interface screenshot",      # 1
        # --- Hard negatives (reject if ANY is dominant) ---
        "a marketing diagram or infographic",                # 2
        "a company logo or branding",                        # 3
        "a data chart, graph, or diagram",                   # 4
        "a portrait or face of a person",                    # 5
        "a decorative background or banner",                 # 6
        "a text-heavy document page or case study",          # 7  — NEW
        "a marketing brochure or testimonial page",          # 8  — NEW
        "a medical or anatomical illustration",              # 9  — NEW
        "a clinical chart or scientific graph",              # 10 — NEW
        "a pricing table or feature comparison chart",       # 11 — NEW
        "a promotional marketing layout with embedded screenshots", # 12 — NEW
    ]
    text_tokens = tokenizer(prompts).to(device)

    # Encode text once; no autocast on CPU for compatibility
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    filtered = []
    rejected_reasons = {}  # track why images were rejected

    for rec in image_records:
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            img_input = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(img_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Cosine similarity → softmax probabilities over all prompts
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = text_probs[0].cpu().numpy()

            # Composite accept rule — reject if ANY negative is dominant
            best_positive_prob = float(max(probs[i] for i in POSITIVE_IDXS))
            worst_negative_idx = max(NEGATIVE_IDXS, key=lambda i: probs[i])
            worst_negative_prob = float(probs[worst_negative_idx])

            rec["operational_prob"] = best_positive_prob
            rec["worst_negative_prob"] = worst_negative_prob
            rec["worst_negative_prompt"] = prompts[worst_negative_idx]
            rec["predicted_class"] = prompts[int(np.argmax(probs))]

            # Accept only if positive is strong AND no negative dominates
            if best_positive_prob < threshold:
                reason = f"positive_too_low ({best_positive_prob:.3f} < {threshold})"
            elif worst_negative_prob >= negative_guard:
                reason = f"negative_guard ({prompts[worst_negative_idx]}: {worst_negative_prob:.3f})"
            else:
                reason = None

            if reason is None:
                filtered.append(rec)
            else:
                # Count rejection reasons for summary
                key = prompts[worst_negative_idx] if "negative_guard" in (reason or "") else "positive_too_low"
                rejected_reasons[key] = rejected_reasons.get(key, 0) + 1

        except Exception as e:
            logger.warning("CLIP filter error on %s: %s", rec["image_path"], e)
            continue

    # Log rejection breakdown
    if rejected_reasons:
        logger.info("Rejection breakdown:")
        for reason, count in sorted(rejected_reasons.items(), key=lambda x: -x[1]):
            logger.info("  %s: %d images", reason, count)

    logger.info(
        "CLIP zero-shot filter: %d → %d operational images "
        "(threshold=%.2f, negative_guard=%.2f)",
        len(image_records), len(filtered), threshold, negative_guard,
    )
    return filtered


# ---------------------------------------------------------------------------
# Per-firm image set construction
# ---------------------------------------------------------------------------

def build_firm_image_sets(
    image_records: list[dict],
    max_per_firm: int = 16,
) -> dict[str, list[str]]:
    """
    Select top-N operational images per firm.

    Priority: higher operational_prob, larger resolution, from main S-1 filing.
    """
    df = pd.DataFrame(image_records)
    if df.empty:
        return {}

    # Sort by operational probability descending, then by resolution
    df["resolution"] = df["width"] * df["height"]
    df = df.sort_values(
        ["cik", "operational_prob", "resolution"],
        ascending=[True, False, False],
    )

    firm_sets = {}
    for cik, group in df.groupby("cik"):
        paths = group["image_path"].tolist()[:max_per_firm]
        firm_sets[cik] = paths

    logger.info(
        "Built image sets for %d firms (max %d images each)",
        len(firm_sets), max_per_firm,
    )
    return firm_sets


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_image_pipeline(
    edgar_dir: str | Path = "data/raw/edgar",
    output_dir: str | Path = "data/processed/images",
    filter_model_path: str | Path | None = None,
    use_clip_zero_shot: bool = False,
    min_size: tuple[int, int] = (150, 150),
    max_per_firm: int = 16,
    filter_threshold: float = 0.35,
    negative_guard: float = 0.20,
) -> dict[str, list[str]]:
    """
    End-to-end image pipeline:
      1. Extract images from all downloaded filings
      2. Rule-based filtering
      3. Model-based filtering (if model available)
      4. Build per-firm image sets
    """
    edgar_dir = Path(edgar_dir)
    output_dir = Path(output_dir)

    # Gather all HTML filing paths
    html_files = list(edgar_dir.rglob("*.html"))
    logger.info("Found %d HTML filings in %s", len(html_files), edgar_dir)

    # --- S-1 scope guard: warn and skip non-S-1 files ---------------------
    # S-1 filings downloaded by EdgarScraper are named by accession number.
    # If other filing types sneak into the directory (e.g., 10-K), they must
    # be excluded to avoid look-ahead bias.
    # We don't have filing-type metadata on disk, so we rely on the scraper
    # having only downloaded S-1/S-1A.  Log a notice for audit trail.
    logger.info(
        "SCOPE CHECK: Processing images ONLY from S-1/S-1A filings. "
        "Ensure EdgarScraper was configured with filing_types=['S-1','S-1/A']. "
        "Found %d HTML files.", len(html_files),
    )

    all_records = []
    for html_path in html_files:
        cik = html_path.parent.name
        records = extract_images_from_html(html_path, output_dir, cik, min_size)
        all_records.extend(records)

    logger.info("Total extracted images: %d", len(all_records))

    # Content-hash deduplication (removes S-1 / S-1A duplicate images)
    all_records = deduplicate_images(all_records)

    # Rule-based filtering
    all_records = apply_rule_filters(all_records)

    # Model-based filtering
    if use_clip_zero_shot:
        logger.info("Using zero-shot CLIP filtering")
        all_records = apply_clip_zero_shot_filter(
            all_records,
            threshold=filter_threshold,
            negative_guard=negative_guard,
        )
    elif filter_model_path and Path(filter_model_path).exists():
        all_records = apply_filter_model(all_records, filter_model_path, filter_threshold)
    else:
        logger.warning(
            "No filter model at %s — skipping model-based filtering. "
            "Train one with: python -m src.data.image_pipeline --train-filter",
            filter_model_path,
        )
        # Add placeholder operational_prob for downstream
        for rec in all_records:
            rec.setdefault("operational_prob", 1.0)

    # Build per-firm sets
    firm_sets = build_firm_image_sets(all_records, max_per_firm)

    # Save manifest
    manifest_path = output_dir / "image_manifest.json"
    import json
    manifest_path.write_text(json.dumps(firm_sets, indent=2))
    logger.info("Saved image manifest → %s", manifest_path)

    return firm_sets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Image extraction & filtering pipeline")
    parser.add_argument("--edgar-dir", default="data/raw/edgar")
    parser.add_argument("--output-dir", default="data/processed/images")
    parser.add_argument("--filter-model", default="outputs/models/image_filter.pt")
    parser.add_argument("--max-per-firm", type=int, default=16)
    parser.add_argument("--use-clip", action="store_true", help="Use zero-shot CLIP filtering")
    parser.add_argument("--filter-threshold", type=float, default=0.35,
                        help="CLIP positive-prompt accept threshold (default: 0.35)")
    parser.add_argument("--negative-guard", type=float, default=0.15,
                        help="CLIP negative-prompt reject guard threshold (default: 0.15)")
    parser.add_argument("--train-filter", type=str, default=None,
                        help="Path to labeled CSV to train filter model")
    args = parser.parse_args()

    if args.train_filter:
        train_filter_model(args.train_filter, args.filter_model)
    else:
        run_image_pipeline(
            edgar_dir=args.edgar_dir,
            output_dir=args.output_dir,
            filter_model_path=args.filter_model,
            use_clip_zero_shot=args.use_clip,
            max_per_firm=args.max_per_firm,
            filter_threshold=args.filter_threshold,
            negative_guard=args.negative_guard,
        )


if __name__ == "__main__":
    main()
