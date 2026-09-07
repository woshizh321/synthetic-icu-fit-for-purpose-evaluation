import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import jsonschema

from portable_qualification.algebra import (
    model_based_tolerance_probability,
    qualify_summary,
    realization_quantities,
    target_a_variance,
    target_b_variance,
    variance_equivalence_count,
)
from portable_qualification.cli import _fit_summary
from portable_qualification.constants import U4_CALIBRATION_BOUNDARY
from portable_qualification.io import read_summary
from portable_qualification.schema_validation import validate_evidence_card


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "portable_demo/example"
ALGEBRA_TOLERANCE = 1e-12
FIT_TOLERANCE = 1e-6


def fitted_fixture_summary():
    return _fit_summary(
        FIXTURE / "fixture_cells.csv",
        FIXTURE / "fixture_sampling_covariance.csv",
        FIXTURE / "fixture_reliability.json",
    )


def assert_nested_close(testcase, observed, expected, tolerance):
    if isinstance(expected, dict):
        testcase.assertEqual(set(observed), set(expected))
        for key in expected:
            assert_nested_close(testcase, observed[key], expected[key], tolerance)
    elif isinstance(expected, list):
        testcase.assertEqual(len(observed), len(expected))
        for left, right in zip(observed, expected):
            assert_nested_close(testcase, left, right, tolerance)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        testcase.assertLessEqual(abs(float(observed) - float(expected)), tolerance)
    else:
        testcase.assertEqual(observed, expected)


class AlgebraTests(unittest.TestCase):
    def test_target_bridge_and_locked_formulas(self):
        result = realization_quantities(1, 0.1, 0.2, 0.3, 0.4)
        self.assertAlmostEqual(result["V_het_m"], 0.9, delta=ALGEBRA_TOLERANCE)
        self.assertAlmostEqual(result["V_pred_m"], 0.91, delta=ALGEBRA_TOLERANCE)
        self.assertAlmostEqual(result["G_m"], 2 / 3, delta=ALGEBRA_TOLERANCE)
        self.assertAlmostEqual(
            result["V_pred_m"], target_b_variance(0.1, 0.2, 0.3, 0.4),
            delta=ALGEBRA_TOLERANCE,
        )
        self.assertAlmostEqual(target_a_variance(0.1, 0.3), 0.31, delta=ALGEBRA_TOLERANCE)

    def test_variance_equivalence_standard_and_edge_cases(self):
        self.assertEqual(variance_equivalence_count(0.2, 0.3, 0.4), (2, "FINITE_POINT_ESTIMATE"))
        self.assertEqual(
            variance_equivalence_count(0.2, 0.0, 0.4),
            ("INF", "HOSPITAL_VARIANCE_ZERO_GENERATOR_VARIANCE_POSITIVE"),
        )
        self.assertEqual(variance_equivalence_count(0.0, 0.0, 0.0), (1, "NO_TRUE_HETEROGENEITY"))
        self.assertEqual(
            variance_equivalence_count(0.0, 0.3, 0.0),
            (1, "NO_GENERATOR_RELATED_HETEROGENEITY"),
        )

    def test_tolerance_probability_and_delta_monotonicity(self):
        self.assertAlmostEqual(model_based_tolerance_probability(0.0, 0.0, 1.0), 0.5, delta=ALGEBRA_TOLERANCE)
        values = [model_based_tolerance_probability(delta, 0.1, 0.25) for delta in (-0.2, 0.1, 0.4)]
        self.assertLessEqual(values[0], values[1])
        self.assertLessEqual(values[1], values[2])
        self.assertEqual(model_based_tolerance_probability(0.1, 0.1, 0.0), 1.0)
        self.assertEqual(model_based_tolerance_probability(0.0, 0.1, 0.0), 0.0)

    def test_variance_monotonicity_and_constancy(self):
        changing = [realization_quantities(m, 0.1, 0.2, 0.3, 0.4) for m in range(1, 21)]
        self.assertTrue(all(changing[i + 1]["V_het_m"] < changing[i]["V_het_m"] for i in range(19)))
        self.assertTrue(all(changing[i + 1]["V_pred_m"] < changing[i]["V_pred_m"] for i in range(19)))
        constant = [realization_quantities(m, 0.1, 0.0, 0.3, 0.0) for m in range(1, 21)]
        self.assertEqual(len({row["V_het_m"] for row in constant}), 1)
        self.assertEqual(len({row["V_pred_m"] for row in constant}), 1)

    def test_q_has_no_assumed_monotonic_direction_in_m(self):
        lower_tail = [model_based_tolerance_probability(0.0, 0.1, v) for v in (1.0, 0.5, 0.25)]
        upper_tail = [model_based_tolerance_probability(0.2, 0.1, v) for v in (1.0, 0.5, 0.25)]
        self.assertGreater(lower_tail[0], lower_tail[-1])
        self.assertLess(upper_tail[0], upper_tail[-1])


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = fitted_fixture_summary()
        cls.card, _ = qualify_summary(cls.summary, m_max=20, delta=None)

    def test_valid_card_and_null_delta(self):
        validate_evidence_card(self.card)
        self.assertIsNone(self.card["user_tolerance_delta"])
        self.assertIsNone(self.card["model_based_tolerance_probability_by_m"])
        self.assertEqual(self.card["u4_calibration_boundary"], U4_CALIBRATION_BOUNDARY)

    def test_canonical_u4_boundary_is_synchronized(self):
        schema = json.loads((ROOT / "schema/evidence_card.schema.json").read_text())
        self.assertEqual(schema["properties"]["u4_calibration_boundary"]["const"], U4_CALIBRATION_BOUNDARY)
        self.assertIn(U4_CALIBRATION_BOUNDARY, (ROOT / "README_PORTABLE.md").read_text())
        self.assertIn(U4_CALIBRATION_BOUNDARY, (FIXTURE / "README_FIXTURE.md").read_text())
        expected = json.loads((FIXTURE / "fixture_expected_qualification.json").read_text())
        self.assertEqual(expected["expected_evidence_card"]["u4_calibration_boundary"], U4_CALIBRATION_BOUNDARY)

    def test_missing_mandatory_field_fails(self):
        invalid = copy.deepcopy(self.card)
        del invalid["u4_calibration_boundary"]
        with self.assertRaises(jsonschema.ValidationError):
            validate_evidence_card(invalid)

    def test_wrong_type_fails(self):
        invalid = copy.deepcopy(self.card)
        invalid["se_mu"] = "not-a-number"
        with self.assertRaises(jsonschema.ValidationError):
            validate_evidence_card(invalid)

    def test_one_row_csv_summary_is_supported(self):
        scalar_summary = {
            key: value for key, value in self.summary.items()
            if not isinstance(value, (dict, list))
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(scalar_summary))
                writer.writeheader()
                writer.writerow(scalar_summary)
            card, _ = qualify_summary(read_summary(path), m_max=3, delta=None)
            validate_evidence_card(card)
            self.assertEqual(card["realization_grid"], [1, 2, 3])

    def test_malformed_realization_maps_fail(self):
        invalid = copy.deepcopy(self.card)
        del invalid["G_by_m"]["20"]
        with self.assertRaises(ValueError):
            validate_evidence_card(invalid)

    def test_prohibited_decision_field_fails(self):
        invalid = copy.deepcopy(self.card)
        invalid["recommended"] = True
        with self.assertRaises((jsonschema.ValidationError, ValueError)):
            validate_evidence_card(invalid)


class FabricatedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit = fitted_fixture_summary()
        cls.expected_fit = json.loads((FIXTURE / "fixture_expected_fit.json").read_text())["expected_fit"]
        cls.expected_qualification = json.loads(
            (FIXTURE / "fixture_expected_qualification.json").read_text()
        )["expected_evidence_card"]

    def test_fixture_dimensions_and_provenance(self):
        with (FIXTURE / "fixture_cells.csv").open(newline="") as handle:
            cells = list(csv.DictReader(handle))
        self.assertEqual(len(cells), 96)
        self.assertEqual(len({row["seed_id"] for row in cells}), 8)
        self.assertEqual(len({row["hospital_id"] for row in cells}), 12)
        provenance = json.loads((FIXTURE / "fixture_reliability.json").read_text())["fixture_provenance"]
        self.assertEqual(provenance["FIXTURE_CONTAINS_REAL_CLINICAL_DATA"], "NO")
        self.assertEqual(provenance["FIXTURE_DERIVED_FROM_REAL_PATIENT_ROWS"], "NO")
        self.assertEqual(provenance["FIXTURE_DERIVED_FROM_EMPIRICAL_HOSPITAL_ESTIMATES"], "NO")
        self.assertEqual(provenance["FIXTURE_PUBLIC_RELEASE_SAFE"], "YES")

    def test_full_covariance_fit_reproduction(self):
        for key in ("mu_eul", "se_mu", "sigma2_seed", "sigma2_hospital", "sigma2_seed_hospital",
                    "target_A_PI_lower", "target_A_PI_upper", "target_B_PI_lower", "target_B_PI_upper"):
            self.assertAlmostEqual(self.fit[key], self.expected_fit[key], delta=FIT_TOLERANCE)
        for key in ("converged", "hospital_n", "seed_n", "cell_n", "method"):
            self.assertEqual(self.fit[key], self.expected_fit[key])

    def test_end_to_end_qualification_reproduction(self):
        observed, _ = qualify_summary(self.fit, m_max=20, delta=0.15)
        validate_evidence_card(observed)
        assert_nested_close(self, observed, self.expected_qualification, FIT_TOLERANCE)
        self.assertEqual(observed["m_eq"], self.expected_qualification["m_eq"])
        for m in observed["realization_grid"]:
            row = realization_quantities(
                m, observed["se_mu"], observed["sigma2_seed"],
                observed["sigma2_hospital"], observed["sigma2_seed_hospital"]
            )
            self.assertAlmostEqual(row["V_het_m"], observed["V_het_by_m"][str(m)], delta=ALGEBRA_TOLERANCE)
            self.assertAlmostEqual(row["V_pred_m"], observed["V_pred_by_m"][str(m)], delta=ALGEBRA_TOLERANCE)
            self.assertAlmostEqual(row["G_m"], observed["G_by_m"][str(m)], delta=ALGEBRA_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
