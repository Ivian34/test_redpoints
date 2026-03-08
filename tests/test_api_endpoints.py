import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from src import api


class _FakePipeline:
    def __init__(self, pred: int, proba_positive: float) -> None:
        self._pred = pred
        self._proba_positive = proba_positive
        self.classes_ = [0, 1]

    def predict(self, _titles):
        return [self._pred]

    def predict_proba(self, _titles):
        return [[1.0 - self._proba_positive, self._proba_positive]]


class _FakeSimilarityEngine:
    def __init__(self, top_score: float = 0.42) -> None:
        self._top_score = top_score

    def query(self, _title: str, top_k: int = 3):
        return [
            {
                "reference_id": 1,
                "reference_title": "ref one",
                "similarity_score": self._top_score,
            }
        ] + [
            {
                "reference_id": i + 2,
                "reference_title": f"ref {i + 2}",
                "similarity_score": max(self._top_score - 0.05 * (i + 1), 0.0),
            }
            for i in range(max(top_k - 1, 0))
        ]


class ApiEndpointsTest(unittest.TestCase):
    def test_health_returns_expected_keys(self):
        response = api.health()
        expected = {
            "status",
            "stage1_model_path",
            "stage1_model_exists",
            "stage2_model_path",
            "stage2_model_exists",
            "similarity_model_path",
            "similarity_model_exists",
            "reference_data_path",
            "reference_data_exists",
        }
        self.assertTrue(expected.issubset(set(response.keys())))

    def test_analyze_listing_runs_stage2_for_asset(self):
        stage1 = _FakePipeline(pred=1, proba_positive=0.73)
        stage2 = _FakePipeline(pred=1, proba_positive=0.81)
        sim_engine = _FakeSimilarityEngine(top_score=0.55)

        with patch.object(api, "_load_stage1_pipeline", return_value=stage1), patch.object(
            api, "_load_stage2_pipeline", return_value=stage2
        ), patch.object(api, "_load_similarity_engine", return_value=sim_engine), patch.object(
            api, "insert_analysed_listing"
        ) as insert_mock:
            result = api.analyze_listing(
                api.AnalyzeRequest(title="iphone 14 pro max", top_k=3)
            )

        self.assertTrue(result.is_asset)
        self.assertTrue(result.stage_2_ran)
        self.assertTrue(result.suspicion_flag)
        self.assertAlmostEqual(result.asset_score, 0.73, places=6)
        self.assertAlmostEqual(result.suspicion_score or 0.0, 0.81, places=6)
        self.assertAlmostEqual(result.similarity_score, 0.55, places=6)
        insert_mock.assert_called_once()

    def test_analyze_listing_skips_stage2_for_non_asset(self):
        stage1 = _FakePipeline(pred=0, proba_positive=0.12)
        sim_engine = _FakeSimilarityEngine(top_score=0.33)

        with patch.object(api, "_load_stage1_pipeline", return_value=stage1), patch.object(
            api, "_load_similarity_engine", return_value=sim_engine
        ), patch.object(api, "_load_stage2_pipeline") as stage2_loader, patch.object(
            api, "insert_analysed_listing"
        ):
            result = api.analyze_listing(api.AnalyzeRequest(title="not an asset", top_k=2))

        self.assertFalse(result.is_asset)
        self.assertFalse(result.stage_2_ran)
        self.assertIsNone(result.suspicion_flag)
        self.assertIsNone(result.suspicion_score)
        stage2_loader.assert_not_called()

    def test_analyze_listing_empty_title_raises_400(self):
        stage1 = _FakePipeline(pred=1, proba_positive=0.6)
        sim_engine = _FakeSimilarityEngine(top_score=0.2)

        with patch.object(api, "_load_stage1_pipeline", return_value=stage1), patch.object(
            api, "_load_similarity_engine", return_value=sim_engine
        ):
            with self.assertRaises(HTTPException) as exc:
                api.analyze_listing(api.AnalyzeRequest(title="   ", top_k=1))
        self.assertEqual(exc.exception.status_code, 400)

    def test_get_analyzed_listings_by_threshold_pass_through(self):
        expected = [{"id": 1}, {"id": 2}]
        with patch.object(
            api, "get_analysed_listings_above_threshold", return_value=expected
        ) as mocked:
            got = api.get_analyzed_listings_by_threshold(threshold=0.8, stage=1)
        self.assertEqual(got, expected)
        mocked.assert_called_once()

    def test_get_analyzed_listings_by_threshold_raises_400_on_value_error(self):
        with patch.object(
            api, "get_analysed_listings_above_threshold", side_effect=ValueError("bad stage")
        ):
            with self.assertRaises(HTTPException) as exc:
                api.get_analyzed_listings_by_threshold(threshold=0.4, stage=1)
        self.assertEqual(exc.exception.status_code, 400)

    def test_get_last_n_listings_pass_through(self):
        expected = [{"id": 7}]
        with patch.object(api, "get_last_n_analysed_listings_db", return_value=expected):
            got = api.get_last_n_analysed_listings(n=1)
        self.assertEqual(got, expected)

    def test_model_metadata_returns_stage_specific_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            asset_path = tmp / "asset.json"
            suspicion_path = tmp / "suspicion.json"
            asset_payload = {"kind": "asset", "metrics": {"f1": 0.9}}
            suspicion_payload = {"kind": "suspicion", "metrics": {"f1": 0.8}}
            asset_path.write_text(json.dumps(asset_payload), encoding="utf-8")
            suspicion_path.write_text(json.dumps(suspicion_payload), encoding="utf-8")

            with patch.object(api, "ASSET_METADATA_PATH", asset_path), patch.object(
                api, "SUSPICION_METADATA_PATH", suspicion_path
            ):
                stage1 = api.get_model_metadata(stage=1)
                stage2 = api.get_model_metadata(stage=2)

        self.assertEqual(stage1["stage"], 1)
        self.assertEqual(stage1["metadata"], asset_payload)
        self.assertEqual(stage2["stage"], 2)
        self.assertEqual(stage2["metadata"], suspicion_payload)

    def test_model_metadata_returns_404_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            missing = tmp / "missing.json"
            with patch.object(api, "ASSET_METADATA_PATH", missing):
                with self.assertRaises(HTTPException) as exc:
                    api.get_model_metadata(stage=1)
        self.assertEqual(exc.exception.status_code, 404)

    def test_model_metadata_returns_500_when_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bad = tmp / "bad.json"
            bad.write_text("{this is not valid json", encoding="utf-8")
            with patch.object(api, "ASSET_METADATA_PATH", bad):
                with self.assertRaises(HTTPException) as exc:
                    api.get_model_metadata(stage=1)
        self.assertEqual(exc.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()
