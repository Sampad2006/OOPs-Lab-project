"""
Arena Runner — Batch benchmarking engine for the Arena view.

Orchestrates running multiple extraction strategies against a
dataset folder of image + ground-truth text pairs, then aggregates
similarity scores into a comparative report.

Usage:
    runner = ArenaRunner(
        dataset_path="./test_data",
        strategies=[ocr_strategy, api_strategy],
    )
    pairs = runner.scan_dataset()
    results_df = runner.run(progress_callback=my_callback)
    summary_df = ArenaRunner.compute_arena_scores(results_df)
"""

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

import config
from models.extraction.base import ExtractionStrategy
from models.similarity.base import SimilarityMetric
from models.similarity.aggregator import SimilarityAggregator
from utils.preprocessor import TextPreprocessor


class ArenaRunner:
    """
    Batch benchmarking engine that races extraction strategies
    against a ground-truth dataset.

    Expected dataset structure:
        dataset_folder/
            doc1.jpg          (image)
            doc1_gt.txt       (ground-truth text)
            doc2.png
            doc2_gt.txt
            ...
    """

    def __init__(
        self,
        dataset_path: str,
        strategies: List[ExtractionStrategy],
        metrics: Optional[List[SimilarityMetric]] = None,
    ):
        """
        Initialize the arena runner.

        Args:
            dataset_path: Path to the dataset directory.
            strategies:   List of ExtractionStrategy instances to benchmark.
            metrics:      Optional similarity metrics. Defaults to all three.
        """
        self._dataset_path = dataset_path
        self._strategies = strategies
        self._metrics = metrics or self._default_metrics()
        self._aggregator = SimilarityAggregator(self._metrics)

    @staticmethod
    def _default_metrics() -> List[SimilarityMetric]:
        """Create the default set of similarity metrics."""
        from models.similarity.edit_distance import EditDistanceSimilarity
        from models.similarity.tfidf_similarity import TFIDFSimilarity
        from models.similarity.embedding_similarity import EmbeddingSimilarity

        return [
            EditDistanceSimilarity(),
            TFIDFSimilarity(),
            EmbeddingSimilarity(),
        ]

    def scan_dataset(self) -> List[Tuple[str, str]]:
        """
        Scan the dataset directory for image + ground-truth pairs.

        Looks for files matching the pattern:
            <name>.<img_ext>  +  <name>_gt.txt

        Returns:
            List of (image_path, gt_text_path) tuples.

        Raises:
            FileNotFoundError: If the dataset path doesn't exist.
            ValueError: If no valid pairs are found.
        """
        if not os.path.isdir(self._dataset_path):
            raise FileNotFoundError(
                f"Dataset directory not found: '{self._dataset_path}'"
            )

        pairs = []
        files = set(os.listdir(self._dataset_path))

        for filename in sorted(files):
            name, ext = os.path.splitext(filename)
            if ext.lower() not in config.ARENA_SUPPORTED_EXTENSIONS:
                continue

            # Look for matching ground-truth file
            gt_filename = f"{name}_gt.txt"
            if gt_filename in files:
                img_path = os.path.join(self._dataset_path, filename)
                gt_path = os.path.join(self._dataset_path, gt_filename)
                pairs.append((img_path, gt_path))

        if not pairs:
            raise ValueError(
                f"No valid image + ground-truth pairs found in "
                f"'{self._dataset_path}'. "
                f"Expected: <name>.<img_ext> + <name>_gt.txt"
            )

        return pairs

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict]:
        """
        Run the benchmark: extract text from each image with each strategy,
        compare against ground truth, and collect results.

        Args:
            progress_callback: Optional callback(current, total, status_msg)
                               for progress updates.

        Returns:
            List of result dicts with keys:
                File, Strategy, Edit Similarity, TF-IDF Similarity,
                Embedding Similarity, Final Similarity, Time (s), Word Count
        """
        pairs = self.scan_dataset()
        total_jobs = len(pairs) * len(self._strategies)
        current_job = 0

        results = []

        for img_path, gt_path in pairs:
            # Load ground-truth text
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_text = f.read().strip()

            # Load image
            image = Image.open(img_path).convert("RGB")
            file_name = os.path.basename(img_path)

            for strategy in self._strategies:
                current_job += 1
                status = f"[{current_job}/{total_jobs}] {strategy.name} → {file_name}"

                if progress_callback:
                    progress_callback(current_job, total_jobs, status)

                try:
                    # Extract text
                    start_time = time.time()
                    extracted = strategy.extract_text(
                        images=[image], doc_type="printed"
                    )
                    elapsed = time.time() - start_time

                    # Clean extracted text
                    extracted = TextPreprocessor.clean_ocr_output(extracted)
                    word_count = len(extracted.split()) if extracted else 0

                    # Compare against ground truth
                    scores = self._aggregator.compute_all(extracted, gt_text)

                    result = {
                        "File": file_name,
                        "Strategy": strategy.name,
                        "Edit Similarity": scores.get("Edit Similarity", 0.0),
                        "TF-IDF Similarity": scores.get("TF-IDF Similarity", 0.0),
                        "Embedding Similarity": scores.get("Embedding Similarity", 0.0),
                        "Final Similarity": scores.get("Final Similarity", 0.0),
                        "Time (s)": round(elapsed, 2),
                        "Word Count": word_count,
                        "Extracted Text": extracted[:200],  # Preview
                    }

                except Exception as e:
                    result = {
                        "File": file_name,
                        "Strategy": strategy.name,
                        "Edit Similarity": 0.0,
                        "TF-IDF Similarity": 0.0,
                        "Embedding Similarity": 0.0,
                        "Final Similarity": 0.0,
                        "Time (s)": 0.0,
                        "Word Count": 0,
                        "Extracted Text": f"ERROR: {str(e)}",
                    }

                results.append(result)

        return results

    @staticmethod
    def compute_arena_scores(results: List[Dict]) -> List[Dict]:
        """
        Aggregate per-strategy average scores from raw results.

        Args:
            results: List of result dicts from run().

        Returns:
            List of summary dicts with average scores per strategy,
            sorted by Final Similarity descending (best first).
        """
        from collections import defaultdict

        strategy_scores = defaultdict(lambda: {
            "count": 0,
            "Edit Similarity": 0.0,
            "TF-IDF Similarity": 0.0,
            "Embedding Similarity": 0.0,
            "Final Similarity": 0.0,
            "Total Time (s)": 0.0,
            "Total Words": 0,
        })

        for r in results:
            s = strategy_scores[r["Strategy"]]
            s["count"] += 1
            s["Edit Similarity"] += r["Edit Similarity"]
            s["TF-IDF Similarity"] += r["TF-IDF Similarity"]
            s["Embedding Similarity"] += r["Embedding Similarity"]
            s["Final Similarity"] += r["Final Similarity"]
            s["Total Time (s)"] += r["Time (s)"]
            s["Total Words"] += r["Word Count"]

        summary = []
        for strategy_name, data in strategy_scores.items():
            n = data["count"]
            summary.append({
                "Strategy": strategy_name,
                "Files Processed": n,
                "Avg Edit Similarity": round(data["Edit Similarity"] / n, 4),
                "Avg TF-IDF Similarity": round(data["TF-IDF Similarity"] / n, 4),
                "Avg Embedding Similarity": round(data["Embedding Similarity"] / n, 4),
                "Arena Score (Avg Final)": round(data["Final Similarity"] / n, 4),
                "Total Time (s)": round(data["Total Time (s)"], 2),
                "Avg Word Count": round(data["Total Words"] / n),
            })

        # Sort by Arena Score descending
        summary.sort(key=lambda x: x["Arena Score (Avg Final)"], reverse=True)
        return summary
