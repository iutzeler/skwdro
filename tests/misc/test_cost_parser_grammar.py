# Author: Claude Sonnet 4.5 (Flo.V. vibe-coding purposeless coverage test)

import pytest
from skwdro.base.cost_decoder import (
    cost_from_str,
    ParsedCost,
    parse_code_torch,
    cost_from_parse_torch,
    DEFAULT_KAPPA,
)
from skwdro.base.costs_torch import NormCost, NormLabelCost


class TestCostFromStr:
    def test_norm_cost_basic(self):
        cost = cost_from_str("t-NC-2.0-2.0")
        assert isinstance(cost, NormCost)
        assert cost.p == 2.0
        assert cost.power == 2.0

    def test_norm_cost_with_custom_values(self):
        cost = cost_from_str("t-NC-1.5-3.0")
        assert isinstance(cost, NormCost)
        assert cost.power == 1.5
        assert cost.p == 3.0

    def test_norm_label_cost_with_default_kappa(self):
        cost = cost_from_str("t-NLC-2.0-2.0")
        assert isinstance(cost, NormLabelCost)
        assert cost.p == 2.0
        assert cost.power == 2.0
        assert cost.kappa == DEFAULT_KAPPA

    def test_norm_label_cost_with_custom_kappa(self):
        cost = cost_from_str("t-NLC-2.0-2.0-5000.0")
        assert isinstance(cost, NormLabelCost)
        assert cost.p == 2.0
        assert cost.power == 2.0
        assert cost.kappa == 5000.0

    def test_invalid_id_raises_error(self):
        with pytest.raises(ValueError, match="Cost code invalid"):
            cost_from_str("t-INVALID-2.0-2.0")

    def test_invalid_engine_raises_error(self):
        with pytest.raises(ValueError, match="Cost code invalid"):
            cost_from_str("x-NC-2.0-2.0")


class TestParsedCost:
    def test_init(self):
        parsed = ParsedCost("t", "NC", 2.0, 2.0, DEFAULT_KAPPA)
        assert parsed.engine == "t"
        assert parsed.id == "NC"
        assert parsed.power == 2.0
        assert parsed.type == 2.0
        assert parsed.kappa == DEFAULT_KAPPA

    def test_can_imp_samp_true(self):
        parsed = ParsedCost("t", "NC", 2.0, 2.0, DEFAULT_KAPPA)
        assert parsed.can_imp_samp() is True

    def test_can_imp_samp_false_power(self):
        parsed = ParsedCost("t", "NC", 1.5, 2.0, DEFAULT_KAPPA)
        assert parsed.can_imp_samp() is False

    def test_can_imp_samp_false_type(self):
        parsed = ParsedCost("t", "NC", 2.0, 3.0, DEFAULT_KAPPA)
        assert parsed.can_imp_samp() is False

    def test_iter(self):
        parsed = ParsedCost("t", "NC", 2.0, 2.0, DEFAULT_KAPPA)
        values = list(parsed)
        assert values == ["t", "NC", 2.0, 2.0, DEFAULT_KAPPA]


class TestParseCodeTorch:
    def test_none_without_labels(self):
        parsed = parse_code_torch(None, has_labels=False)
        assert parsed.engine == "t"
        assert parsed.id == "NC"
        assert parsed.power == 2.0
        assert parsed.type == 2.0
        assert parsed.kappa == DEFAULT_KAPPA

    def test_none_with_labels(self):
        parsed = parse_code_torch(None, has_labels=True)
        assert parsed.engine == "t"
        assert parsed.id == "NLC"
        assert parsed.power == 2.0
        assert parsed.type == 2.0
        assert parsed.kappa == DEFAULT_KAPPA

    def test_four_part_code(self):
        parsed = parse_code_torch("t-NC-1.5-3.0")
        assert parsed.engine == "t"
        assert parsed.id == "NC"
        assert parsed.power == 1.5
        assert parsed.type == 3.0
        assert parsed.kappa == DEFAULT_KAPPA

    def test_three_part_code(self):
        parsed = parse_code_torch("NC-1.5-3.0")
        assert parsed.engine == "t"
        assert parsed.id == "NC"
        assert parsed.power == 1.5
        assert parsed.type == 3.0
        assert parsed.kappa == DEFAULT_KAPPA

    def test_five_part_code(self):
        parsed = parse_code_torch("t-NLC-1.5-3.0-5000.0")
        assert parsed.engine == "t"
        assert parsed.id == "NLC"
        assert parsed.power == 1.5
        assert parsed.type == 3.0
        assert parsed.kappa == 5000.0

    def test_invalid_length_raises_error(self):
        with pytest.raises(ValueError, match="Cost code invalid"):
            parse_code_torch("t-NC")

    def test_four_part_code_wrong_engine_raises_error(self):
        with pytest.raises(AssertionError):
            parse_code_torch("x-NC-1.5-3.0")


class TestCostFromParseTorch:
    def test_norm_cost(self):
        parsed = ParsedCost("t", "NC", 2.0, 2.0, DEFAULT_KAPPA)
        cost = cost_from_parse_torch(parsed)
        assert isinstance(cost, NormCost)
        assert cost.p == 2.0
        assert cost.power == 2.0

    def test_norm_label_cost(self):
        parsed = ParsedCost("t", "NLC", 2.0, 2.0, 5000.0)
        cost = cost_from_parse_torch(parsed)
        assert isinstance(cost, NormLabelCost)
        assert cost.p == 2.0
        assert cost.power == 2.0
        assert cost.kappa == 5000.0

    def test_invalid_id_raises_error(self):
        parsed = ParsedCost("t", "INVALID", 2.0, 2.0, DEFAULT_KAPPA)
        with pytest.raises(ValueError, match="Cost code invalid"):
            cost_from_parse_torch(parsed)
