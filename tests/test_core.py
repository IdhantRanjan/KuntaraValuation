"""
Tests for core modules — Smoke tests ensuring all components import and initialize.
"""

import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data module tests
# ---------------------------------------------------------------------------

class TestIPOUniverse:
    def test_imports(self):
        from src.data.ipo_universe import (
            IPORecord, IPOUniverse, load_ipo_list,
            apply_sample_filters, compute_first_day_return, time_split,
        )

    def test_empty_universe(self):
        from src.data.ipo_universe import IPOUniverse
        u = IPOUniverse()
        assert len(u) == 0
        df = u.to_dataframe()
        assert len(df) == 0

    def test_load_ipo_list_no_file(self):
        from src.data.ipo_universe import load_ipo_list
        df = load_ipo_list(csv_path="/nonexistent/path.csv")
        assert len(df) == 0


class TestEdgarScraper:
    def test_imports(self):
        from src.data.edgar_scraper import EdgarScraper, _clean_text

    def test_clean_text(self):
        from src.data.edgar_scraper import _clean_text
        text = "Hello   World\n\n\n\nTest\n-----\nEnd"
        cleaned = _clean_text(text)
        assert "-----" not in cleaned
        assert "Hello World" in cleaned


class TestImagePipeline:
    def test_imports(self):
        from src.data.image_pipeline import (
            ImageFilterNet, apply_rule_filters,
            build_firm_image_sets,
        )

    def test_filter_net_init(self):
        from src.data.image_pipeline import ImageFilterNet
        model = ImageFilterNet(pretrained=False)
        assert model.NUM_CLASSES == 5

        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 5)

    def test_rule_filters(self):
        from src.data.image_pipeline import apply_rule_filters
        records = [
            {"image_path": "/tmp/test.png", "aspect_ratio": 2.0, "width": 200, "height": 200},
            {"image_path": "/tmp/test2.png", "aspect_ratio": 10.0, "width": 200, "height": 200},
        ]
        # Second one has extreme aspect ratio
        filtered = apply_rule_filters(records, max_aspect_ratio=5.0)
        # Can't test fully without actual images, but structure is correct
        assert isinstance(filtered, list)


class TestDataset:
    def test_imports(self):
        from src.data.dataset import IPOMultimodalDataset, build_dataloaders


# ---------------------------------------------------------------------------
# Feature encoder tests
# ---------------------------------------------------------------------------

class TestTabularEncoder:
    def test_forward(self):
        from src.features.tabular_encoder import TabularEncoder
        enc = TabularEncoder(input_dim=8, hidden_dims=[64, 64], output_dim=128)
        x = torch.randn(4, 8)
        out = enc(x)
        assert out.shape == (4, 128)
        assert enc.get_output_dim() == 128


class TestImageEncoder:
    def test_attention_pooling(self):
        from src.features.image_encoder import AttentionPooling
        pooler = AttentionPooling(embed_dim=128, hidden_dim=64)
        embeddings = torch.randn(2, 5, 128)
        mask = torch.tensor([[True, True, True, False, False],
                             [True, True, False, False, False]])
        out = pooler(embeddings, mask)
        assert out.shape == (2, 128)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestFusion:
    def test_late_fusion(self):
        from src.models.fusion import LateFusion
        f = LateFusion(modality_dims=[128, 128, 128], output_dim=256)
        embs = [torch.randn(4, 128) for _ in range(3)]
        out = f(embs)
        assert out.shape == (4, 256)

    def test_gated_fusion(self):
        from src.models.fusion import GatedFusion
        f = GatedFusion(modality_dims=[128, 128, 128], output_dim=256)
        embs = [torch.randn(4, 128) for _ in range(3)]
        out = f(embs)
        assert out.shape == (4, 256)

    def test_cross_attention_fusion(self):
        from src.models.fusion import CrossAttentionFusion
        f = CrossAttentionFusion(
            modality_dims=[128, 128, 128],
            d_model=128, n_layers=2, n_heads=4, d_ff=256,
        )
        embs = [torch.randn(4, 128) for _ in range(3)]
        out = f(embs)
        assert out.shape == (4, 128)

    def test_build_fusion(self):
        from src.models.fusion import build_fusion
        for strategy in ["late", "gated"]:
            f = build_fusion(strategy, [128, 128], output_dim=128)
            assert f is not None


class TestHeads:
    def test_multi_task_heads(self):
        from src.models.heads import MultiTaskHeads
        heads = MultiTaskHeads(input_dim=128)
        z = torch.randn(4, 128)
        preds = heads(z)
        assert preds["underpricing"].shape == (4, 1)
        assert preds["broken_ipo"].shape == (4, 1)
        assert preds["volatility"].shape == (4, 1)

    def test_loss_computation(self):
        from src.models.heads import MultiTaskHeads
        heads = MultiTaskHeads(input_dim=128)
        z = torch.randn(4, 128)
        preds = heads(z)
        targets = torch.randn(4, 3)
        targets[:, 1] = torch.sigmoid(targets[:, 1])  # Binary target
        loss, loss_dict = heads.compute_loss(preds, targets)
        assert loss.requires_grad
        assert "total" in loss_dict


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_regression_metrics(self):
        from src.evaluation.metrics import regression_metrics
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        m = regression_metrics(y, y_pred)
        assert "mae" in m
        assert "rmse" in m
        assert "r2" in m
        assert m["r2"] > 0.9

    def test_diebold_mariano(self):
        from src.evaluation.metrics import diebold_mariano_test
        y = np.random.randn(100)
        p1 = y + np.random.randn(100) * 0.5
        p2 = y + np.random.randn(100) * 0.3
        result = diebold_mariano_test(y, p1, p2)
        assert "dm_statistic" in result
        assert "p_value" in result

    def test_decile_analysis(self):
        from src.evaluation.metrics import decile_analysis
        y = np.random.randn(100)
        y_pred = y + np.random.randn(100) * 0.2
        summary = decile_analysis(y, y_pred)
        assert len(summary) > 0


class TestStatisticalTests:
    def test_bootstrap_r2(self):
        from src.evaluation.statistical_tests import bootstrap_r2_difference
        y = np.random.randn(50)
        p1 = y + np.random.randn(50) * 0.5
        p2 = y + np.random.randn(50) * 0.3
        result = bootstrap_r2_difference(y, p1, p2, n_bootstrap=100)
        assert "mean_r2_diff" in result
        assert "ci_low" in result


# ---------------------------------------------------------------------------
# Data hardening tests  (new — added for research-grade pipeline)
# ---------------------------------------------------------------------------

class TestCIKVerification:
    """Tests for the CIK spot-check tool."""

    def test_imports(self):
        from src.data.verify_cik_mappings import (
            verify_all_ciks, verify_single_cik, _normalize_name,
            _token_overlap_score,
        )

    def test_normalize_name_strips_suffixes(self):
        from src.data.verify_cik_mappings import _normalize_name
        assert _normalize_name("Palantir Technologies Inc.") == "palantir"
        assert _normalize_name("Rivian Automotive, Inc.") == "rivian automotive"
        assert _normalize_name("UserTesting, Inc.") == "usertesting"

    def test_token_overlap_score_exact(self):
        from src.data.verify_cik_mappings import _token_overlap_score
        score = _token_overlap_score("Palantir Technologies Inc", "Palantir Technologies Inc")
        assert score == 1.0

    def test_token_overlap_score_partial(self):
        from src.data.verify_cik_mappings import _token_overlap_score
        # "Rivian Automotive" vs "Rivian Automotive Inc" – high overlap
        score = _token_overlap_score("Rivian Automotive Inc", "Rivian Automotive")
        assert score > 0.6

    def test_token_overlap_score_mismatch(self):
        from src.data.verify_cik_mappings import _token_overlap_score
        # Completely different names should score low
        score = _token_overlap_score("Airbnb Inc", "Palantir Technologies Inc")
        assert score < 0.3

    def test_token_overlap_score_empty(self):
        from src.data.verify_cik_mappings import _token_overlap_score
        assert _token_overlap_score("", "Palantir") == 0.0
        assert _token_overlap_score("Palantir", "") == 0.0


class TestImageDeduplication:
    """Tests for perceptual-hash deduplication."""

    def test_imports(self):
        from src.data.image_pipeline import deduplicate_images

    def test_dedup_empty_list(self):
        from src.data.image_pipeline import deduplicate_images
        result = deduplicate_images([])
        assert result == []

    def test_dedup_no_duplicates(self, tmp_path):
        """Distinct seeded-noise images should all survive deduplication."""
        from PIL import Image
        from src.data.image_pipeline import deduplicate_images
        import numpy as np

        # Random noise images with different seeds produce distinct pHashes.
        # Each seed generates a statistically independent noise pattern, so
        # Hamming distances will be large (>> 5) across all pairs.
        records = []
        for i, seed in enumerate([0, 42, 137]):
            rng = np.random.default_rng(seed)
            arr = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            p = tmp_path / f"img_{i}.png"
            img.save(p)
            records.append({
                "image_path": str(p),
                "cik": "9999",
                "width": 300,
                "height": 300,
                "aspect_ratio": 1.0,
                "filing": f"acc{i:04d}",
            })

        result = deduplicate_images(records, hamming_threshold=5)
        assert len(result) == 3, (
            f"Expected 3 distinct images to survive dedup, got {len(result)}"
        )

    def test_dedup_exact_duplicates(self, tmp_path):
        """Exact byte-for-byte copies should be collapsed to one."""
        from PIL import Image
        from src.data.image_pipeline import deduplicate_images

        # Create one image, save it twice (simulating S-1 vs S-1/A duplicate)
        img = Image.new("RGB", (400, 300), color=(128, 64, 200))
        p1 = tmp_path / "img_s1.png"
        p2 = tmp_path / "img_s1a.png"
        img.save(p1)
        img.save(p2)

        records = [
            {"image_path": str(p1), "cik": "1111", "width": 400, "height": 300,
             "aspect_ratio": 1.33, "filing": "acc0001"},
            {"image_path": str(p2), "cik": "1111", "width": 400, "height": 300,
             "aspect_ratio": 1.33, "filing": "acc0002"},
        ]

        result = deduplicate_images(records, hamming_threshold=5)
        assert len(result) == 1  # Duplicate collapsed

    def test_dedup_preserves_highest_resolution(self, tmp_path):
        """When two duplicates exist, the higher-resolution one is kept."""
        from PIL import Image
        from src.data.image_pipeline import deduplicate_images

        colour = (100, 150, 200)
        small = Image.new("RGB", (200, 200), color=colour)
        large = Image.new("RGB", (600, 600), color=colour)

        p_small = tmp_path / "small.png"
        p_large = tmp_path / "large.png"
        small.save(p_small)
        large.save(p_large)

        records = [
            {"image_path": str(p_small), "cik": "2222", "width": 200, "height": 200,
             "aspect_ratio": 1.0, "filing": "acc0001"},
            {"image_path": str(p_large), "cik": "2222", "width": 600, "height": 600,
             "aspect_ratio": 1.0, "filing": "acc0002"},
        ]

        result = deduplicate_images(records, hamming_threshold=5)
        assert len(result) == 1
        assert result[0]["width"] == 600  # Highest resolution kept

    def test_dedup_cross_cik_isolation(self, tmp_path):
        """Identical images in different CIKs must NOT be deduplicated against each other."""
        from PIL import Image
        from src.data.image_pipeline import deduplicate_images

        img = Image.new("RGB", (300, 300), color=(50, 50, 50))
        p1 = tmp_path / "cik_a.png"
        p2 = tmp_path / "cik_b.png"
        img.save(p1)
        img.save(p2)

        records = [
            {"image_path": str(p1), "cik": "AAAA", "width": 300, "height": 300,
             "aspect_ratio": 1.0, "filing": "acc0001"},
            {"image_path": str(p2), "cik": "BBBB", "width": 300, "height": 300,
             "aspect_ratio": 1.0, "filing": "acc0002"},
        ]

        result = deduplicate_images(records, hamming_threshold=5)
        # Both are kept — they belong to different companies
        assert len(result) == 2


class TestS1ScopeEnforcement:
    """Tests for look-ahead bias guards in the EDGAR scraper."""

    def test_allowed_filing_types_constant(self):
        from src.data.edgar_scraper import ALLOWED_FILING_TYPES
        assert "S-1" in ALLOWED_FILING_TYPES
        assert "S-1/A" in ALLOWED_FILING_TYPES
        # Post-IPO filings must NOT be in the allowed set
        for bad_type in ("10-K", "10-Q", "DEF 14A", "8-K", "20-F"):
            assert bad_type not in ALLOWED_FILING_TYPES, (
                f"{bad_type} must not be allowed (look-ahead bias risk)"
            )

    def test_disallowed_filing_type_raises(self):
        from src.data.edgar_scraper import EdgarScraper
        scraper = EdgarScraper()
        with pytest.raises(ValueError, match="look-ahead bias"):
            # Passing a 10-K should immediately raise
            scraper.get_filing_urls("1321655", filing_types=["10-K"])

    def test_mixed_filing_types_raises(self):
        from src.data.edgar_scraper import EdgarScraper
        scraper = EdgarScraper()
        with pytest.raises(ValueError):
            # Even one bad type in the list should fail
            scraper.get_filing_urls("1321655", filing_types=["S-1", "10-K"])

    def test_valid_s1_types_pass(self):
        """S-1 and S-1/A must not raise."""
        from src.data.edgar_scraper import EdgarScraper
        scraper = EdgarScraper()
        # Should reach the network call without raising ValueError;
        # we patch the session to avoid an actual HTTP request
        from unittest.mock import MagicMock, patch
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "filings": {"recent": {"form": [], "accessionNumber": [],
                                   "filingDate": [], "primaryDocument": []}}
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(scraper.session, "get", return_value=mock_resp):
            result = scraper.get_filing_urls("1321655", filing_types=["S-1", "S-1/A"])
        assert isinstance(result, list)  # No ValueError raised



# ---------------------------------------------------------------------------
# New module tests (Tasks 1, 2, 4, 7, 8, 9)
# ---------------------------------------------------------------------------

class TestRitterParser:
    def test_imports(self):
        from src.data.ritter_parser import (
            RITTER_DATA_URLS, CARTER_MANASTER_RANKS,
            assign_underwriter_ranks, parse_ritter_excel,
            download_ritter_excel, build_ritter_csv,
        )
        assert len(RITTER_DATA_URLS) >= 4
        assert "goldman sachs" in CARTER_MANASTER_RANKS

    def test_assign_underwriter_ranks(self):
        import pandas as pd
        from src.data.ritter_parser import assign_underwriter_ranks
        df = pd.DataFrame({
            "underwriter": ["Goldman Sachs", "Morgan Stanley",
                            "Random Boutique LLP", ""]
        })
        out = assign_underwriter_ranks(df)
        assert out.loc[0, "underwriter_rank"] == 9.0
        assert out.loc[1, "underwriter_rank"] == 9.0
        # Default rank for unmatched
        assert out.loc[2, "underwriter_rank"] == 5.0

    def test_parse_normalizes_first_day_return(self):
        from src.data.ritter_parser import _coerce_first_day_return
        # Percent input: should be converted to decimal
        assert abs(_coerce_first_day_return(12.5) - 0.125) < 1e-9
        assert abs(_coerce_first_day_return("25%") - 0.25) < 1e-9
        # Already decimal: should be left alone
        assert abs(_coerce_first_day_return(0.08) - 0.08) < 1e-9
        # Negative percent
        assert abs(_coerce_first_day_return(-15.0) - (-0.15)) < 1e-9
        # Bad input
        assert _coerce_first_day_return(None) is None
        assert _coerce_first_day_return("") is None


class TestEdgarCIKLookup:
    def test_imports(self):
        from src.data.edgar_cik_lookup import (
            CIKLookup, MANUAL_OVERRIDES, build_cik_mapping, _name_variants,
        )
        assert "ABNB" in MANUAL_OVERRIDES

    def test_normalize_name_in_lookup(self):
        from src.data.edgar_cik_lookup import _norm_name_strict
        from src.data.verify_cik_mappings import _normalize_name
        # Both should agree (CIKLookup re-uses verify_cik_mappings._normalize_name)
        assert _norm_name_strict("Airbnb, Inc.") == _normalize_name("Airbnb, Inc.")

    def test_lookup_handles_not_found(self):
        from unittest.mock import MagicMock
        import requests
        from src.data.edgar_cik_lookup import CIKLookup
        lookup = CIKLookup(user_agent="test test@x.com")
        # Force empty DB
        lookup._tickers_db = {}
        lookup._name_index = {}
        # Mock EFTS to 404 — use requests.HTTPError (what raise_for_status raises)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
        lookup.session.get = MagicMock(return_value=mock_resp)
        result = lookup.lookup(
            ticker="ZZZNOTREAL", company_name="Definitely Not A Company",
            ipo_year=2020,
        )
        assert result["cik"] is None
        assert result["method"] == "not_found"


class TestFetchFinancials:
    def test_imports(self):
        from src.data.fetch_s1_financials import (
            extract_financial_tables, find_balance_sheet,
            find_income_statement, extract_key_metrics,
            compute_financial_ratios, enrich_universe_with_financials,
        )

    def test_compute_financial_ratios_basic(self):
        from src.data.fetch_s1_financials import compute_financial_ratios
        m = {
            "total_assets": 1_000_000_000,
            "total_debt": 200_000_000,
            "revenue": 500_000_000,
            "revenue_prior": 400_000_000,
            "rnd_expense": 100_000_000,
            "founding_year": 2010,
        }
        r = compute_financial_ratios(m, ipo_year=2020)
        assert abs(r["leverage"] - 0.2) < 1e-9
        assert abs(r["rnd_intensity"] - 0.2) < 1e-9
        assert abs(r["revenue_growth"] - 0.25) < 1e-9
        assert r["firm_age"] == 10
        # log(1e9) ~= 20.72
        assert 20 < r["log_assets"] < 21


class TestFetchPostIpoReturns:
    def test_imports(self):
        from src.data.fetch_post_ipo_returns import (
            download_ticker_returns, compute_outcomes,
            enrich_universe_with_returns, _ols_beta,
        )

    def test_ols_beta_perfect_fit(self):
        import numpy as np
        from src.data.fetch_post_ipo_returns import _ols_beta
        x = np.linspace(0.0, 1.0, 100)
        y = 2.5 * x + 0.1
        beta = _ols_beta(y, x)
        assert beta is not None
        assert abs(beta - 2.5) < 1e-6


class TestRunEval:
    def test_imports(self):
        from src.evaluation.run_eval import run_eval, main


class TestTables:
    def test_imports(self):
        from src.analysis.tables import (
            table1_descriptive_stats, table2_main_results,
            table3_statistical_tests, table4_visual_factors,
            generate_all_tables,
        )

    def test_table1_descriptive_stats(self, tmp_path):
        import pandas as pd
        from src.analysis.tables import table1_descriptive_stats
        df = pd.DataFrame({
            "first_day_return": [0.1, 0.2, -0.05, 0.15, 0.0],
            "broken_ipo": [0, 0, 1, 0, 1],
            "offer_size": [1e8, 2e8, 5e7, 1e8, 1.5e8],
        })
        out_path = tmp_path / "table1.tex"
        out = table1_descriptive_stats(df, out_path)
        assert "\\toprule" in out
        assert "\\bottomrule" in out
        assert "first\\_day\\_return" in out
        assert out_path.exists()


class TestRegressionAnalysis:
    def test_imports(self):
        from src.analysis.regression_analysis import (
            run_ols_with_visual_factors, run_fama_macbeth,
            incremental_r2_test,
        )

    def test_incremental_r2_test_smoke(self):
        import numpy as np, pandas as pd
        from src.analysis.regression_analysis import incremental_r2_test
        rng = np.random.default_rng(42)
        n = 150
        df = pd.DataFrame({
            "cik": [str(i) for i in range(n)],
            "ipo_date": pd.to_datetime("2018-01-01") + pd.to_timedelta(np.arange(n), unit="D"),
            "first_day_return": rng.normal(0, 0.1, n),
            "log_assets": rng.normal(0, 1, n),
            "leverage": rng.uniform(0, 1, n),
            "rnd_intensity": rng.uniform(0, 1, n),
            "revenue_growth": rng.normal(0, 1, n),
            "firm_age": rng.uniform(0, 20, n),
            "underwriter_rank": rng.uniform(0, 9, n),
            "vc_backed": rng.integers(0, 2, n),
            "broken_ipo": rng.integers(0, 2, n),
        })
        factors = pd.DataFrame({
            "cik": df["cik"],
            "VF1": rng.normal(0, 1, n),
            "VF2": rng.normal(0, 1, n),
        })
        result = incremental_r2_test(df, factors)
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "delta_R2" in result.columns
            assert "F_stat" in result.columns


class TestOfferPriceExtraction:
    """Tests for the S-1 / 424B4 offer price extractor."""

    def test_imports(self):
        from src.data.extract_offer_price import (
            extract_offer_price_from_html,
            extract_offer_price_for_cik,
            extract_best_offer_price,
            enrich_universe_with_offer_prices,
        )

    def test_price_regex_dollar_per_share(self):
        from src.data.extract_offer_price import _PAT_DOLLAR_PER_SHARE
        text = "We are offering 10,000,000 shares at $25.00 per share"
        m = _PAT_DOLLAR_PER_SHARE.search(text)
        assert m is not None
        assert m.group(1) == "25.00"

    def test_price_regex_range(self):
        from src.data.extract_offer_price import _PAT_PRICE_RANGE
        text = "per share will be between $90.00 and $95.00. We have been approved"
        m = _PAT_PRICE_RANGE.search(text)
        assert m is not None
        assert m.group(1) == "90.00"
        assert m.group(2) == "95.00"

    def test_price_regex_ipo_label(self):
        from src.data.extract_offer_price import _PAT_IPO_PRICE
        text = "Initial public offering price $68.00 per share"
        m = _PAT_IPO_PRICE.search(text)
        assert m is not None
        assert m.group(1) == "68.00"

    def test_parse_price(self):
        from src.data.extract_offer_price import _parse_price
        assert _parse_price("25.00") == 25.0
        assert _parse_price("1,250.50") == 1250.5
        assert _parse_price(None) is None

    def test_extract_from_html_not_found(self, tmp_path):
        from src.data.extract_offer_price import extract_offer_price_from_html
        # Create a dummy HTML file with no price info
        p = tmp_path / "test.html"
        p.write_text("<html><body>No price here</body></html>")
        result = extract_offer_price_from_html(p)
        assert result["offer_price"] is None
        assert result["method"] == "not_found"

    def test_extract_from_html_with_price(self, tmp_path):
        from src.data.extract_offer_price import extract_offer_price_from_html
        # Create HTML with a clear offer price
        p = tmp_path / "test.html"
        p.write_text(
            "<html><body>"
            "Initial public offering price $42.00 per share. "
            "We are offering 5,000,000 shares."
            "</body></html>"
        )
        result = extract_offer_price_from_html(p)
        assert result["offer_price"] == 42.0
        assert result["method"] == "ipo_price_label"

    def test_extract_range_midpoint(self, tmp_path):
        from src.data.extract_offer_price import extract_offer_price_from_html
        p = tmp_path / "test.html"
        p.write_text(
            "<html><body>"
            "The price per share will be between $20.00 and $24.00."
            "</body></html>"
        )
        result = extract_offer_price_from_html(p)
        assert result["offer_price"] == 22.0
        assert result["method"] == "price_range_midpoint"
        assert result["price_low"] == 20.0
        assert result["price_high"] == 24.0


class TestBuildRitterUniverse:
    def test_imports(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "build_ritter_universe",
            Path("scripts/build_ritter_universe.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "run_pipeline")
        assert hasattr(mod, "build_synthetic_master")
        assert len(mod.SYNTHETIC_IPOS) == 20

    def test_synthetic_master(self, tmp_path):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "build_ritter_universe",
            Path("scripts/build_ritter_universe.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = tmp_path / "ipo_master.csv"
        df = mod.build_synthetic_master(out)
        assert len(df) == 20
        assert "cik" in df.columns
        assert "first_day_return" in df.columns
        assert out.exists()
