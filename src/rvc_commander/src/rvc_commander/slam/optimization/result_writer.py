import csv
from dataclasses import asdict
from optimizer import RankedRun


class ResultWriter:
    @staticmethod
    def write_ranked_runs_csv(path: str, ranked_runs: list[RankedRun]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "score",
                "tag",
                "max_corr_dist",
                "neighbors_pca",
                "occ_thres",
                "delta_r",
                "mean_translation_error",
                "rmse_translation_error",
                "mean_rotation_error",
                "fallback_count",
                "mean_icp_iterations",
            ])

            for r in ranked_runs:
                writer.writerow([
                    r.score,
                    r.params.tag,
                    r.params.icp.max_correspondence_distance,
                    r.params.icp.neighbors_pca,
                    r.params.scan_matcher.occ_thres,
                    r.params.scan_matcher.delta_r,
                    r.summary["mean_translation_error"],
                    r.summary["rmse_translation_error"],
                    r.summary["mean_rotation_error"],
                    r.summary["fallback_count"],
                    r.summary["mean_icp_iterations"],
                ])