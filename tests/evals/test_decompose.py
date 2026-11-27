"""decompose: a reply split into claims in code, the Input numbered as facts, and the pre-match that
only ever rules `supported` -- each rule both ways."""

from __future__ import annotations

import json

from llm_ports.evals import decompose as d

INVOICE = "Vendor: Acme Supplies\nSubtotal: 90.00\nTax: 10.00\nTotal due: 100.00\nTerms: net 30"


class TestLeaves:
    def test_each_json_scalar_is_a_claim_at_its_path(self) -> None:
        reply = json.dumps({"vendor": "Acme Supplies", "lines": [{"sku": "A1", "qty": 2}], "paid": False})
        got = {lf.path: lf.raw for lf in d.leaves(reply)}
        assert got == {"vendor": "Acme Supplies", "lines[0].sku": "A1", "lines[0].qty": 2, "paid": False}

    def test_an_empty_container_or_null_is_one_claim_that_nothing_applies(self) -> None:
        lf = {x.path: x for x in d.leaves(json.dumps({"notes": [], "po": None, "meta": {}, "memo": ""}))}
        assert all(x.empty for x in lf.values()) and set(lf) == {"notes", "po", "meta", "memo"}
        assert not d.Leaf("qty", "0", 0).empty and not d.Leaf("paid", "false", False).empty

    def test_the_field_drops_list_indices(self) -> None:
        assert d.Leaf("lines[3].items[0].sku", "A1", "A1").field == "lines[].items[].sku"

    def test_prose_is_one_claim_per_sentence_and_a_decimal_is_not_a_boundary(self) -> None:
        got = [lf.value for lf in d.leaves("The total is 100.00 due. Tax was 2.5 percent!\nPaid on time.")]
        assert got == ["The total is 100.00 due.", "Tax was 2.5 percent!", "Paid on time."]
        assert [lf.path for lf in d.leaves("One. Two.")] == ["sentence[0]", "sentence[1]"]

    def test_a_json_scalar_or_broken_json_is_prose(self) -> None:
        assert [lf.value for lf in d.leaves('"just a string"')] == ['"just a string"']
        assert [lf.value for lf in d.leaves('{"a": ')] == ['{"a":']
        assert d.leaves("") == ()


class TestFacts:
    def test_lines_are_numbered_blank_and_repeated_lines_dropped(self) -> None:
        facts = d.facts_from_text("Subtotal: 90.00\n\n  Subtotal:   90.00 \nTax: 10.00")
        assert [(f.id, f.text) for f in facts] == [("F1", "Subtotal: 90.00"), ("F2", "Tax: 10.00")]

    def test_a_long_line_is_split_at_sentence_ends_else_at_spaces(self) -> None:
        long = "First sentence here. Second sentence here. Third one."
        assert [f.text for f in d.facts_from_text(long, cap=25)] == [
            "First sentence here.",
            "Second sentence here.",
            "Third one.",
        ]
        words = [f.text for f in d.facts_from_text("aaaa bbbb cccc dddd", cap=9)]
        assert words == ["aaaa bbbb", "cccc dddd"] and all(len(w) <= 9 for w in words)
        assert d.facts_from_text("x", prefix="L")[0].id == "L1"


class TestOccurs:
    def test_whole_words_only(self) -> None:
        assert d.occurs("Acme Supplies", INVOICE)
        assert not d.occurs("Acme Supp", INVOICE)

    def test_a_short_value_is_a_symbol_and_matched_as_written(self) -> None:
        assert not d.occurs("AI", "she said so") and not d.occurs("ai", "she said so")
        assert d.occurs("ACME", "vendor ACME ltd") and not d.occurs("acme", "vendor ACME ltd")

    def test_a_long_value_ignores_case_curly_quotes_and_spacing(self) -> None:
        assert d.occurs("acme   supplies", INVOICE)
        assert d.occurs("the “net 30” terms", 'per the "net 30" terms')
        assert not d.occurs("", INVOICE) and not d.occurs("   ", INVOICE)


class TestPrematch:
    def test_an_extract_found_verbatim_is_supported(self) -> None:
        pm = d.prematch(d.Leaf("vendor", "Acme Supplies", "Acme Supplies"), "extract", input_text=INVOICE)
        assert pm.ruled and pm.label == "supported"

    def test_an_extract_not_found_is_left_to_the_judge_with_a_hint(self) -> None:
        pm = d.prematch(d.Leaf("vendor", "Globex", "Globex"), "extract", input_text=INVOICE)
        assert not pm.ruled and "the value is not in the Input verbatim" in pm.hints

    def test_a_free_value_found_is_a_hint_never_a_ruling(self) -> None:
        pm = d.prematch(d.Leaf("summary", "net 30", "net 30"), "free", input_text=INVOICE)
        assert not pm.ruled and "the value occurs in the Input verbatim" in pm.hints

    def test_a_classify_value_found_is_a_hint_never_a_ruling(self) -> None:
        pm = d.prematch(d.Leaf("status", "due", "due"), "classify", input_text=INVOICE, allowed=("due", "paid"))
        assert not pm.ruled
        assert pm.hints == ("the value is one of the field's allowed values",)
        assert d.prematch(d.Leaf("t", "Terms", "Terms"), "classify", input_text=INVOICE).hints == (
            "the value's words occur in the Input",
        )

    def test_a_licensed_figure_is_supported_and_an_unlicensed_one_is_a_hint(self) -> None:
        assert d.prematch(d.Leaf("total", "100.00", 100.0), "free", input_text=INVOICE).label == "supported"
        assert d.prematch(d.Leaf("total", "100.00", "100.00"), "extract", input_text=INVOICE).ruled
        off = d.prematch(d.Leaf("total", "250", 250), "extract", input_text=INVOICE)
        assert not off.ruled and off.hints[0].startswith("figure(s) the Input does not show")

    def test_a_classify_score_and_a_bool_are_never_ruled_as_figures(self) -> None:
        assert not d.prematch(d.Leaf("score", "10", 10), "classify", input_text=INVOICE).ruled
        assert not d.prematch(d.Leaf("paid", "true", True), "free", input_text=INVOICE).ruled

    def test_an_empty_claim_is_never_found(self) -> None:
        assert not d.prematch(d.Leaf("po", "null", None), "extract", input_text="po: null").ruled


class TestEnums:
    def test_enums_consts_refs_items_and_combinators_are_read(self) -> None:
        schema = {
            "$defs": {"Status": {"enum": ["due", "paid"]}},
            "properties": {
                "status": {"$ref": "#/$defs/Status"},
                "lines": {"items": {"properties": {"unit": {"anyOf": [{"const": "ea"}, {"enum": ["kg", None]}]}}}},
                "note": {"type": "string"},
            },
        }
        assert d.enums(schema) == {"status": ("due", "paid"), "lines[].unit": ("ea", "kg")}

    def test_a_schema_it_cannot_read_yields_nothing(self) -> None:
        assert d.enums(None) == {} and d.enums({"properties": {"a": {"$ref": "#/$defs/Missing"}}}) == {}
