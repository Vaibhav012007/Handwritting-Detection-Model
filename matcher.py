"""
matcher.py — Medicine dictionary matching and confidence scoring.

Provides:
  MedicineMatcher.match(raw_text, ocr_conf) -> MatchResult
  tune_threshold(val_preds, val_labels) -> best threshold (uses val set only)
  run_ablation(test_preds, test_labels, threshold) -> ablation table dict
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_ROOT, RESULTS_DIR, load_brand_generic_map

from rapidfuzz import fuzz
import Levenshtein as lev_lib
from dataclasses import dataclass, asdict


# ─── Load dictionary ──────────────────────────────────────────────────────────
def build_medicine_dict():
    """Build {brand: generic} from training labels CSV."""
    return load_brand_generic_map()


@dataclass
class MatchResult:
    raw_text: str
    matched_brand: str
    generic_name: str
    lev_score: float       # Levenshtein similarity [0,1]
    fuzzy_score: float     # RapidFuzz partial ratio [0,1]
    match_score: float     # blended match score [0,1]
    ocr_confidence: float  # from TrOCR beam search [0,1]
    confidence: float      # final blended confidence [0,1]
    status: str            # "accepted" | "uncertain"


class MedicineMatcher:
    def __init__(self, brand_generic_map: dict, accept_threshold: float = 0.5):
        self.brand_generic_map = brand_generic_map
        self.brands = list(brand_generic_map.keys())
        self.accept_threshold = accept_threshold

    def _levenshtein_sim(self, a: str, b: str) -> float:
        """Normalised Levenshtein similarity in [0,1]."""
        dist = lev_lib.distance(a.lower(), b.lower())
        max_len = max(len(a), len(b), 1)
        return 1.0 - dist / max_len

    def _fuzzy_sim(self, a: str, b: str) -> float:
        """RapidFuzz token_sort_ratio in [0,1]."""
        return fuzz.token_sort_ratio(a.lower(), b.lower()) / 100.0

    def match(self, raw_text: str, ocr_conf: float = 1.0) -> MatchResult:
        raw_text = raw_text.strip()
        best_brand = None
        best_match_score = -1.0
        best_lev, best_fuzzy = 0.0, 0.0

        for brand in self.brands:
            ls = self._levenshtein_sim(raw_text, brand)
            fs = self._fuzzy_sim(raw_text, brand)
            ms = 0.5 * ls + 0.5 * fs
            if ms > best_match_score:
                best_match_score = ms
                best_brand = brand
                best_lev = ls
                best_fuzzy = fs

        generic = self.brand_generic_map.get(best_brand, "")
        # Exact override: if raw_text matches a brand exactly, score = 1.0
        if raw_text in self.brand_generic_map:
            best_brand = raw_text
            best_match_score = 1.0
            generic = self.brand_generic_map[raw_text]

        # Final confidence: equal blend of OCR confidence and match score
        conf = 0.5 * ocr_conf + 0.5 * best_match_score
        status = "accepted" if conf >= self.accept_threshold else "uncertain"

        return MatchResult(
            raw_text=raw_text,
            matched_brand=best_brand or "",
            generic_name=generic,
            lev_score=best_lev,
            fuzzy_score=best_fuzzy,
            match_score=best_match_score,
            ocr_confidence=ocr_conf,
            confidence=conf,
            status=status,
        )


# ─── Threshold tuning (VAL ONLY) ──────────────────────────────────────────────
def tune_threshold(val_preds: list, val_ocr_confs: list, val_gt_brands: list,
                   brand_generic_map: dict) -> float:
    """
    Grid-search over thresholds [0.1..0.9] to maximise balanced accuracy among
    accepted items on the VALIDATION set.
    Returns the best threshold.
    """
    thresholds = np.arange(0.1, 0.95, 0.05)
    matcher_temp = MedicineMatcher(brand_generic_map, accept_threshold=0.0)

    results = []
    for thresh in thresholds:
        accepted_correct = 0
        accepted_total = 0
        uncertain_total = 0
        for raw, conf, gt in zip(val_preds, val_ocr_confs, val_gt_brands):
            mr = matcher_temp.match(raw, conf)
            if mr.confidence >= thresh:
                accepted_total += 1
                if mr.matched_brand == gt:
                    accepted_correct += 1
            else:
                uncertain_total += 1
        acc_among_accepted = accepted_correct / max(accepted_total, 1)
        coverage = accepted_total / max(len(val_preds), 1)
        # F-measure of accuracy and coverage (favour coverage ≥ 50%)
        f_score = (2 * acc_among_accepted * coverage) / max(acc_among_accepted + coverage, 1e-9)
        results.append((thresh, acc_among_accepted, coverage, f_score))
        print(
            f"  thresh={thresh:.2f}  acc_accepted={acc_among_accepted:.3f}  "
            f"coverage={coverage:.3f}  F={f_score:.3f}"
        )

    # Best threshold maximises F-score
    best = max(results, key=lambda x: x[3])
    print(f"\nBest threshold (val): {best[0]:.2f}  (F={best[3]:.3f})")
    return float(best[0])


# ─── Ablation (TEST ONLY, called once from evaluate.py) ───────────────────────
def run_ablation(test_preds: list, test_ocr_confs: list, test_gt_brands: list,
                 test_gt_generics: list, brand_generic_map: dict, threshold: float):
    """
    Compare raw TrOCR vs TrOCR + dictionary matching on test set.
    Returns dict with full ablation stats and also saves plots/tables.
    """
    # ── Raw TrOCR metrics ──
    raw_brand_correct = sum(p == g for p, g in zip(test_preds, test_gt_brands))
    raw_generic_correct = sum(
        brand_generic_map.get(p, "?") == brand_generic_map.get(g, "!")
        for p, g in zip(test_preds, test_gt_brands)
    )
    n = len(test_preds)

    # ── With matching ──
    matcher = MedicineMatcher(brand_generic_map, accept_threshold=threshold)
    accepted_correct_brand = 0
    accepted_correct_generic = 0
    accepted_n = 0
    uncertain_correct_brand = 0
    uncertain_n = 0
    all_confs = []

    for raw, conf, gt_b, gt_g in zip(test_preds, test_ocr_confs, test_gt_brands, test_gt_generics):
        mr = matcher.match(raw, conf)
        all_confs.append(mr.confidence)
        if mr.status == "accepted":
            accepted_n += 1
            if mr.matched_brand == gt_b:
                accepted_correct_brand += 1
            if mr.generic_name == gt_g:
                accepted_correct_generic += 1
        else:
            uncertain_n += 1
            if mr.matched_brand == gt_b:
                uncertain_correct_brand += 1

    # ── Precision / coverage trade-off curve ──
    thresholds = np.arange(0.0, 1.01, 0.02)
    precisions, coverages = [], []
    for t in thresholds:
        ac, an = 0, 0
        for raw, conf, gt_b in zip(test_preds, test_ocr_confs, test_gt_brands):
            mr = matcher.match(raw, conf)
            if mr.confidence >= t:
                an += 1
                if mr.matched_brand == gt_b:
                    ac += 1
        precisions.append(ac / max(an, 1))
        coverages.append(an / n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverages, precisions, marker=".", color="#4a90d9")
    ax.axvline(x=accepted_n / n, color="#e85d5d", linestyle="--", label=f"chosen threshold={threshold:.2f}")
    ax.set_xlabel("Coverage (fraction of test accepted)")
    ax.set_ylabel("Precision (brand accuracy among accepted)")
    ax.set_title("Precision vs Coverage Trade-off (Test Set)")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    tc_path = os.path.join(RESULTS_DIR, "threshold_curve.png")
    plt.savefig(tc_path, dpi=120)
    plt.close()
    print(f"Saved: {tc_path}")

    # ── Ablation table ──
    ablation = {
        "raw_trocr": {
            "brand_acc": raw_brand_correct / n,
            "generic_acc": raw_generic_correct / n,
            "n": n,
        },
        "trocr_with_matching": {
            "accept_threshold": threshold,
            "accepted_n": accepted_n,
            "uncertain_n": uncertain_n,
            "acceptance_rate": accepted_n / n,
            "brand_acc_among_accepted": accepted_correct_brand / max(accepted_n, 1),
            "generic_acc_among_accepted": accepted_correct_generic / max(accepted_n, 1),
            "brand_acc_among_uncertain": uncertain_correct_brand / max(uncertain_n, 1),
        },
    }

    # Markdown table
    md = "# Ablation Table\n\n"
    md += "## Raw TrOCR vs TrOCR + Dictionary Matching (Test Set)\n\n"
    md += f"| Metric | Raw TrOCR | + Dict Matching |\n"
    md += f"|---|---|---|\n"
    md += f"| Brand Exact-Match Acc | {raw_brand_correct/n:.4f} | {accepted_correct_brand/max(accepted_n,1):.4f} (accepted only) |\n"
    md += f"| Generic Exact-Match Acc | {raw_generic_correct/n:.4f} | {accepted_correct_generic/max(accepted_n,1):.4f} (accepted only) |\n"
    md += f"| N total | {n} | {n} |\n"
    md += f"| N accepted | — | {accepted_n} ({accepted_n/n:.1%}) |\n"
    md += f"| N uncertain | — | {uncertain_n} ({uncertain_n/n:.1%}) |\n"
    md += f"| Brand acc among uncertain | — | {uncertain_correct_brand/max(uncertain_n,1):.4f} |\n"

    abl_path = os.path.join(RESULTS_DIR, "ablation_table.md")
    with open(abl_path, "w") as f:
        f.write(md)
    print(f"Saved: {abl_path}")
    print(md)

    return ablation


if __name__ == "__main__":
    # Quick smoke test
    bgm = build_medicine_dict()
    m = MedicineMatcher(bgm, accept_threshold=0.5)
    result = m.match("Aceta", ocr_conf=0.9)
    print(result)
