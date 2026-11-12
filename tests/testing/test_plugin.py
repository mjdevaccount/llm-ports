"""The llm_ports pytest plugin: live requests blocked both ways, the opt-in marker, the fixtures'
teardown checks, and that `import llm_ports` never loads it."""

from __future__ import annotations

import subprocess
import sys

import httpx
import httpx2
import pydantic_ai.messages  # noqa: F401  # loaded once here: pytester's in-process runs would re-import it
import pytest

from llm_ports.testing._live import LiveRequestBlocked, block_live_requests, is_loopback


async def test_this_suite_blocks_a_live_host_by_default() -> None:
    # The plugin is on in this very session: an unmarked test cannot reach a live host.
    with pytest.raises(LiveRequestBlocked) as e:
        async with httpx2.AsyncClient() as c:
            await c.get("https://api.example.invalid/v1/models")
    assert e.value.host == "api.example.invalid"
    with pytest.raises(LiveRequestBlocked):
        httpx.get("https://api.example.invalid/")


async def test_a_mock_transport_is_never_blocked() -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _r: httpx2.Response(204))) as c:
        assert (await c.get("https://api.example.invalid/")).status_code == 204


def test_release_restores_the_transports_and_both_ways_block_again() -> None:
    before = httpx.HTTPTransport.handle_request
    release = block_live_requests()
    assert httpx.HTTPTransport.handle_request is not before
    release()
    assert httpx.HTTPTransport.handle_request is before


@pytest.mark.parametrize(
    ("host", "local"),
    [
        ("localhost", True),
        ("127.0.0.1", True),
        ("127.9.9.9", True),
        ("[::1]", True),
        ("10.0.0.1", False),
        ("x.test", False),
    ],
)
def test_only_loopback_is_the_test_s_own(host: str, local: bool) -> None:
    assert is_loopback(host) is local


ASYNC = ("-o", "asyncio_mode=auto", "-o", "asyncio_default_fixture_loop_scope=function")

LIVE_TEST = """
import httpx
import pytest
from llm_ports.testing import LiveRequestBlocked

@pytest.mark.llm_live
def test_live():
    with pytest.raises(httpx.ConnectError):  # reached the network: `.invalid` never resolves
        httpx.get("https://api.example.invalid/")

def test_unmarked():
    with pytest.raises(LiveRequestBlocked):
        httpx.get("https://api.example.invalid/")
"""


def test_a_live_test_is_skipped_without_the_flag(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(LIVE_TEST)
    pytester.runpytest_inprocess("-p", "no:cacheprovider", *ASYNC).assert_outcomes(passed=1, skipped=1)


def test_both_ways_a_live_test_runs_unblocked_with_the_flag(pytester: pytest.Pytester) -> None:
    # A subprocess: in process, THIS session's block (this test is unmarked) would still be on.
    pytester.makepyfile(LIVE_TEST)
    pytester.runpytest_subprocess("-p", "no:cacheprovider", "--llm-live").assert_outcomes(passed=2)


def test_the_stub_generator_fixture_fails_a_script_left_unplayed(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
from llm_ports.testing import Reply

def test_unused(stub_generator):
    stub_generator([Reply.ok("never asked")])
"""
    )
    res = pytester.runpytest_inprocess("-p", "no:cacheprovider", *ASYNC)
    res.assert_outcomes(passed=1, errors=1)
    res.stdout.fnmatch_lines(["*script_unused*"])


def test_llm_replay_fails_a_partly_played_cassette_and_records_v2(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
import json
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from llm_ports.adapters.pydantic_ai import PydanticAIGenerator
from llm_ports.ports import GenRequest

def live(_m, _i):
    return ModelResponse(parts=[TextPart("recorded")])

async def test_record(llm_replay, tmp_path):
    model = llm_replay(tmp_path / "c.json", FunctionModel(live, model_name="m"))
    res = await PydanticAIGenerator(model).generate(GenRequest(model="m", prompt="a"), str)
    assert res.value == "recorded"
"""
    )
    pytester.runpytest_inprocess("-p", "no:cacheprovider", *ASYNC, "--llm-record").assert_outcomes(passed=1)
    pytester.makepyfile(
        test_partly="""
import json
from llm_ports.adapters.pydantic_ai.replay import key_parts

def test_partly(llm_replay, tmp_path):
    p = tmp_path / "c.json"
    parts = dict(key_parts([], model="m", system="", settings=None, output_schema=None, attempt=0))
    p.write_text(json.dumps({"version": 2, "entries": {"k:0": {"text": "t", "parts": parts}}}))
    llm_replay(p)
"""
    )
    res = pytester.runpytest_inprocess("-p", "no:cacheprovider", *ASYNC, "test_partly.py")
    res.assert_outcomes(passed=1, errors=1)
    res.stdout.fnmatch_lines(["*cassette_unplayed*k:0*"])


def test_import_llm_ports_loads_neither_pytest_nor_the_testing_package() -> None:
    code = "import sys, llm_ports; print(sorted(m for m in sys.modules if m == 'pytest' or m.startswith('llm_ports.testing')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


def test_a_request_with_no_url_is_judged_unknown_and_blocked() -> None:
    from llm_ports.testing import _live

    assert _live._blocked_host(object()) == "<unknown>"


def test_a_url_the_transport_cannot_send_reaches_the_transport() -> None:
    # Not http(s): the transport's own refusal is the test's, never `LiveRequestBlocked`.
    with pytest.raises(httpx.UnsupportedProtocol):
        httpx.get("ftp://files.example.invalid/x")


def test_a_loopback_request_reaches_the_sync_transport() -> None:
    with pytest.raises(httpx.ConnectError):  # nothing listens on port 9 here: reached, not blocked
        httpx.get("http://127.0.0.1:9/")


def test_a_transport_that_is_not_installed_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_ports.testing import _live

    monkeypatch.setattr(_live, "_TARGETS", (("no_such_module_xyz", "T", "send", _live._guard_sync),))
    _live.block_live_requests()()  # nothing to patch, nothing to undo


def test_the_live_opt_in_runs_in_process_with_the_flag(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
import pytest

@pytest.mark.llm_live
def test_live():
    pass
"""
    )
    pytester.runpytest_inprocess("-p", "no:cacheprovider", "--llm-live").assert_outcomes(passed=1)


def test_the_tracer_fixture_and_a_record_without_a_live_model(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
import pytest
from llm_ports.testing import ContractViolation, RecordingTracer

def test_tracer(recording_tracer):
    assert isinstance(recording_tracer, RecordingTracer) and recording_tracer.generations == []

def test_record_needs_live(llm_replay, tmp_path):
    with pytest.raises(ContractViolation, match="record_needs_live"):
        llm_replay(tmp_path / "c.json")
"""
    )
    pytester.runpytest_inprocess("-p", "no:cacheprovider", *ASYNC, "--llm-record").assert_outcomes(passed=2)


async def test_a_loopback_request_reaches_the_async_transport() -> None:
    async with httpx2.AsyncClient() as c:
        with pytest.raises(httpx2.ConnectError):  # reached, not blocked
            await c.get("http://127.0.0.1:9/")
