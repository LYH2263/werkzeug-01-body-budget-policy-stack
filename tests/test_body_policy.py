from __future__ import annotations

import contextvars
import io
import json
import sys
import threading
import time

import pytest

from werkzeug import body_budget
from werkzeug import BodyBudgetError
from werkzeug import BodyBudgetFrame
from werkzeug import BudgetMerge
from werkzeug import FormQuotaProfile
from werkzeug import pop_body_budget
from werkzeug import push_body_budget
from werkzeug import StreamGate
from werkzeug.body_policy import current_body_budget_frames
from werkzeug.body_policy import resolve_body_budget
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.formparser import FormDataParser
from werkzeug.formparser import parse_form_data
from werkzeug.test import create_environ
from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Request

MULTIPART = (
    b"--foo\r\nContent-Disposition: form-field; name=foo\r\n\r\n"
    b"Hello World\r\n"
    b"--foo\r\nContent-Disposition: form-field; name=bar\r\n\r\n"
    b"bar=baz\r\n--foo--"
)
MULTIPART_TYPE = "multipart/form-data; boundary=foo"
URLENCODED = b"foo=Hello+World&bar=baz"
URLENCODED_TYPE = "application/x-www-form-urlencoded"


def raw_environ(
    data: bytes = b"",
    *,
    content_type: str | None = None,
    content_length: int | None | object = 0,
    terminated: bool = False,
    method: str = "POST",
) -> dict[str, object]:
    """Build a minimal WSGI environ. Pass ``content_length=...`` sentinel to
    omit CONTENT_LENGTH entirely (a length-unknown streaming request)."""

    environ: dict[str, object] = {
        "REQUEST_METHOD": method,
        "SCRIPT_NAME": "",
        "PATH_INFO": "/",
        "QUERY_STRING": "",
        "SERVER_NAME": "localhost",
        "SERVER_PORT": "80",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": "http",
        "wsgi.input": io.BytesIO(data),
        "wsgi.errors": sys.stderr,
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
    }

    if content_type is not None:
        environ["CONTENT_TYPE"] = content_type

    if content_length is not ...:
        environ["CONTENT_LENGTH"] = str(content_length)

    if terminated:
        environ["wsgi.input_terminated"] = True

    return environ


def baseline_view() -> object:
    return resolve_body_budget(
        max_content_length=1000,
        max_form_memory_size=500_000,
        max_form_parts=1000,
    )


# ---------------------------------------------------------------------------
# Empty stack parity and frame validation
# ---------------------------------------------------------------------------


def test_empty_stack_matches_static_attributes() -> None:
    view = resolve_body_budget(
        max_content_length=1000,
        max_form_memory_size=500_000,
        max_form_parts=1000,
    )
    assert view.max_content_length == 1000
    assert view.max_form_memory_size == 500_000
    assert view.max_form_parts == 1000
    assert view.stream_gate is StreamGate.TERMINATED
    assert view.form_quota_profile is FormQuotaProfile.MULTIPART_TRIPLE
    assert current_body_budget_frames() == ()


def test_request_body_budget_uses_attributes_without_frames() -> None:
    req = Request(raw_environ(b"abc", content_type="text/plain", content_length=3))
    req.max_content_length = 42
    req.max_form_memory_size = 7
    req.max_form_parts = 2
    view = req.body_budget
    assert view.max_content_length == 42
    assert view.max_form_memory_size == 7
    assert view.max_form_parts == 2
    assert view.stream_gate is StreamGate.TERMINATED
    assert view.form_quota_profile is FormQuotaProfile.MULTIPART_TRIPLE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_content_length": -1},
        {"max_content_length": "10"},
        {"max_content_length": True},
        {"max_form_memory_size": -1},
        {"max_form_parts": 1.5},
        {"budget_merge": "strict_min"},
        {"stream_gate": "terminated"},
        {"form_quota_profile": "opaque_body"},
    ],
)
def test_invalid_frame_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        BodyBudgetFrame(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Numeric merge modes
# ---------------------------------------------------------------------------


def test_strict_min_takes_stricter_side() -> None:
    with body_budget(max_content_length=100):
        assert baseline_view().max_content_length == 100
    with body_budget(max_content_length=10_000):
        # Cannot widen the baseline.
        assert baseline_view().max_content_length == 1000
    # Explicit unlimited cannot widen either.
    with body_budget(max_content_length=None):
        assert baseline_view().max_content_length == 1000


def test_strict_min_applies_per_limit_independently() -> None:
    with body_budget(max_form_memory_size=5, max_form_parts=2):
        view = baseline_view()
        assert view.max_content_length == 1000
        assert view.max_form_memory_size == 5
        assert view.max_form_parts == 2


def test_active_frame_recognizes_only_top_numbers() -> None:
    with body_budget(max_content_length=5_000, budget_merge=BudgetMerge.ACTIVE_FRAME):
        assert baseline_view().max_content_length == 5_000

    # Explicit None removes the cap, falling back to nothing; an unset limit
    # falls back to the Request baseline.
    with body_budget(max_form_memory_size=5, budget_merge=BudgetMerge.ACTIVE_FRAME):
        view = baseline_view()
        assert view.max_content_length == 1000
        assert view.max_form_memory_size == 5

    with body_budget(max_content_length=None, budget_merge=BudgetMerge.ACTIVE_FRAME):
        assert baseline_view().max_content_length is None


def test_inherit_only_allows_narrowing_blocks_widening() -> None:
    with body_budget(max_content_length=500, budget_merge=BudgetMerge.INHERIT_ONLY):
        assert baseline_view().max_content_length == 500

    # An unlimited outer budget may be narrowed by an inherit_only frame.
    with body_budget(max_content_length=500, budget_merge=BudgetMerge.INHERIT_ONLY):
        view = resolve_body_budget(
            max_content_length=None,
            max_form_memory_size=None,
            max_form_parts=None,
        )
        assert view.max_content_length == 500

    with body_budget(max_content_length=5_000, budget_merge=BudgetMerge.INHERIT_ONLY):
        with pytest.raises(BodyBudgetError, match="[Ww]iden"):
            baseline_view()

    with body_budget(max_content_length=None, budget_merge=BudgetMerge.INHERIT_ONLY):
        with pytest.raises(BodyBudgetError):
            baseline_view()

    # Widening is also blocked relative to an outer frame, not just baseline.
    with body_budget(max_content_length=500):
        with body_budget(max_content_length=600, budget_merge=BudgetMerge.INHERIT_ONLY):
            with pytest.raises(BodyBudgetError):
                baseline_view()


def test_gate_and_profile_inherit_unless_inner_overrides() -> None:
    with body_budget(
        stream_gate=StreamGate.EMPTY_IF_UNKNOWN,
        form_quota_profile=FormQuotaProfile.OPAQUE_BODY,
    ):
        view = baseline_view()
        assert view.stream_gate is StreamGate.EMPTY_IF_UNKNOWN
        assert view.form_quota_profile is FormQuotaProfile.OPAQUE_BODY

        # A frame that only tightens numbers must not silently reopen the
        # stricter outer gate or relax the outer profile.
        with body_budget(max_content_length=10):
            view = baseline_view()
            assert view.max_content_length == 10
            assert view.stream_gate is StreamGate.EMPTY_IF_UNKNOWN
            assert view.form_quota_profile is FormQuotaProfile.OPAQUE_BODY

            # An inner frame may still restore an axis explicitly.
            with body_budget(
                stream_gate=StreamGate.TERMINATED,
                form_quota_profile=FormQuotaProfile.MULTIPART_TRIPLE,
            ):
                view = baseline_view()
                assert view.stream_gate is StreamGate.TERMINATED
                assert view.form_quota_profile is (FormQuotaProfile.MULTIPART_TRIPLE)

            view = baseline_view()
            assert view.stream_gate is StreamGate.EMPTY_IF_UNKNOWN
            assert view.form_quota_profile is FormQuotaProfile.OPAQUE_BODY


# ---------------------------------------------------------------------------
# Push / pop mechanics and exception cleanup
# ---------------------------------------------------------------------------


def test_context_manager_pops_on_exception() -> None:
    with pytest.raises(ValueError, match="boom"):
        with body_budget(max_content_length=10):
            assert len(current_body_budget_frames()) == 1
            raise ValueError("boom")

    assert current_body_budget_frames() == ()


def test_nested_pop_restores_outer_frame_not_global() -> None:
    with body_budget(max_content_length=500):
        assert baseline_view().max_content_length == 500

        with body_budget(max_content_length=100):
            assert baseline_view().max_content_length == 100

        # Back to the outer user frame, not an empty/global state.
        assert len(current_body_budget_frames()) == 1
        assert baseline_view().max_content_length == 500

    assert current_body_budget_frames() == ()
    assert baseline_view().max_content_length == 1000


def test_unbalanced_pop_is_rejected() -> None:
    outer = push_body_budget(BodyBudgetFrame(max_content_length=500))
    inner = push_body_budget(BodyBudgetFrame(max_content_length=100))

    with pytest.raises(BodyBudgetError):
        pop_body_budget(outer)

    pop_body_budget(inner)
    assert pop_body_budget(outer).max_content_length == 500
    assert current_body_budget_frames() == ()


def test_invalid_pop_token_type() -> None:
    with pytest.raises(TypeError):
        pop_body_budget("not a token")  # type: ignore[arg-type]


def test_push_frame_or_kwargs_but_not_both() -> None:
    with pytest.raises(TypeError):
        push_body_budget(BodyBudgetFrame(), max_content_length=1)


def test_copy_context_isolates_stacks() -> None:
    token = push_body_budget(BodyBudgetFrame(max_content_length=1))

    try:
        ctx = contextvars.copy_context()

        def run_in_ctx() -> None:
            assert len(current_body_budget_frames()) == 1
            with body_budget(max_content_length=2):
                assert len(current_body_budget_frames()) == 2

        ctx.run(run_in_ctx)
        # The frame pushed in the copied context did not escape, and the
        # outer context frame is untouched.
        assert len(current_body_budget_frames()) == 1
        assert current_body_budget_frames()[0].max_content_length == 1
    finally:
        pop_body_budget(token)


# ---------------------------------------------------------------------------
# Request.stream: gate x (Content-Length, wsgi.input_terminated) matrix
# ---------------------------------------------------------------------------


def test_stream_known_content_length_preemptive_413() -> None:
    req = Request(
        raw_environ(b"x" * 100, content_type="text/plain", content_length=100)
    )

    with body_budget(max_content_length=10):
        with pytest.raises(RequestEntityTooLarge):
            req.stream.read()

    # The frame raised before reading anything; after pop the baseline allows
    # the full advertised body again.
    assert req.stream.read() == b"x" * 100


def test_stream_known_content_length_limited_read() -> None:
    environ = raw_environ(b"0123456789", content_type="text/plain", content_length=10)
    req = Request(environ)
    req.max_content_length = 10
    assert req.stream.read(4) == b"0123"
    assert req.stream.read() == b"456789"


def test_stream_no_content_length_no_marker_is_empty_for_every_gate() -> None:
    for gate in (
        StreamGate.TERMINATED,
        StreamGate.CONTENT_LENGTH,
        StreamGate.EMPTY_IF_UNKNOWN,
    ):
        environ = raw_environ(b"abc", content_type="text/plain", content_length=...)
        req = Request(environ)

        with body_budget(max_content_length=3, stream_gate=gate):
            assert req.stream.read() == b"", gate


def test_stream_terminated_marker_caps_unknown_length() -> None:
    environ = raw_environ(
        b"abcdef", content_type="text/plain", content_length=..., terminated=True
    )
    req = Request(environ)

    with body_budget(max_content_length=3, stream_gate=StreamGate.TERMINATED):
        stream = req.stream
        assert stream.read(3) == b"abc"

        with pytest.raises(RequestEntityTooLarge):
            stream.read(1)


def test_stream_terminated_marker_no_cap_passes_through() -> None:
    environ = raw_environ(
        b"abcdef", content_type="text/plain", content_length=..., terminated=True
    )
    req = Request(environ)
    assert req.stream is environ["wsgi.input"]
    assert req.stream.read() == b"abcdef"


def test_stream_empty_if_unknown_ignores_marker() -> None:
    environ = raw_environ(
        b"abcdef", content_type="text/plain", content_length=..., terminated=True
    )
    req = Request(environ)

    with body_budget(max_content_length=3, stream_gate=StreamGate.EMPTY_IF_UNKNOWN):
        assert req.stream.read() == b""


def test_stream_content_length_gate_ignores_marker_for_cap() -> None:
    environ = raw_environ(
        b"abcdef", content_type="text/plain", content_length=..., terminated=True
    )
    req = Request(environ)
    req.max_content_length = 3

    with body_budget(stream_gate=StreamGate.CONTENT_LENGTH):
        # The marker still guarantees EOF, so the raw stream is returned, but
        # max_content_length is not enforced read-time without a CL.
        assert req.stream is environ["wsgi.input"]
        assert req.stream.read() == b"abcdef"

    # The same gate still enforces advertised content length preemptively.
    environ = raw_environ(b"abcdef", content_type="text/plain", content_length=6)
    req = Request(environ)
    req.max_content_length = 3

    with body_budget(stream_gate=StreamGate.CONTENT_LENGTH):
        with pytest.raises(RequestEntityTooLarge):
            req.stream.read()


def test_stream_cached_until_effective_identity_changes() -> None:
    environ = raw_environ(b"0123456789", content_type="text/plain", content_length=10)
    req = Request(environ)
    req.max_content_length = 10
    first = req.stream
    assert req.stream is first

    # A frame that resolves to the same effective numbers keeps the object.
    with body_budget(max_content_length=10):
        assert req.stream is first

    # A different effective gate invalidates the cached guarded stream; the
    # numeric cap is unchanged so the advertised body is still readable.
    with body_budget(stream_gate=StreamGate.EMPTY_IF_UNKNOWN):
        assert req.stream is not first


def test_stream_rebuild_after_pop_continues_partial_read_under_outer_budget() -> None:
    # Length-unknown terminated stream: the inner cap applies read-time.
    environ = raw_environ(
        b"0123456789", content_type="text/plain", content_length=..., terminated=True
    )
    req = Request(environ)
    req.max_content_length = 10

    with body_budget(max_content_length=3):
        inner = req.stream
        assert inner.read(3) == b"012"

        with pytest.raises(RequestEntityTooLarge):
            inner.read(1)

    # Pop: the first-frame limits are not silently retained. The stream is
    # rebuilt over wsgi.input at its current position with the outer cap.
    outer = req.stream
    assert outer is not inner
    assert outer.read() == b"3456789"


def test_stream_known_cl_rebuild_after_tight_frame_raised() -> None:
    environ = raw_environ(b"0123456789", content_type="text/plain", content_length=10)
    req = Request(environ)

    with body_budget(max_content_length=3):
        with pytest.raises(RequestEntityTooLarge):
            req.stream.read()

    stream = req.stream
    assert stream.read() == b"0123456789"


def test_stream_delete_and_reassign() -> None:
    environ = raw_environ(b"abc", content_type="text/plain", content_length=3)
    req = Request(environ)
    assert req.stream.read() == b"abc"
    del req.stream
    assert "stream" not in req.__dict__
    replacement = io.BytesIO(b"zz")
    req.stream = replacement
    assert req.stream is replacement


# ---------------------------------------------------------------------------
# Test client (no wsgi.input_terminated) vs real server (marker for chunked)
# ---------------------------------------------------------------------------


def test_environ_builder_does_not_set_terminated() -> None:
    builder = EnvironBuilder(method="POST", data=b"abc")
    environ = builder.get_environ()
    builder.close()

    assert "wsgi.input_terminated" not in environ
    assert environ["CONTENT_LENGTH"] == "3"


def test_test_client_unknown_length_is_empty_without_marker() -> None:
    builder = EnvironBuilder(
        method="POST", data=b"abc", content_type="application/octet-stream"
    )
    environ = builder.get_environ()
    builder.close()
    # The test client always sets a Content-Length; remove it to mimic a
    # streaming request arriving without one and without the server marker.
    del environ["CONTENT_LENGTH"]

    req = Request(environ)

    with body_budget(max_content_length=1_000):
        assert req.stream.read() == b""

    # A real server (chunked support) sets the marker, which unlocks capped
    # reading for the same length-unknown request.
    environ["wsgi.input_terminated"] = True
    environ["wsgi.input"] = io.BytesIO(b"abc")

    with body_budget(max_content_length=2):
        req = Request(environ)
        stream = req.stream
        assert stream.read(2) == b"ab"

        with pytest.raises(RequestEntityTooLarge):
            stream.read(1)


# ---------------------------------------------------------------------------
# form_quota_profile: multipart_triple / urlencoded_content_only / opaque_body
# ---------------------------------------------------------------------------


class TestMultipartTriple:
    def test_multipart_uses_all_three_limits(self) -> None:
        req = Request(
            raw_environ(
                MULTIPART, content_type=MULTIPART_TYPE, content_length=len(MULTIPART)
            )
        )

        with body_budget(max_form_memory_size=7):
            with pytest.raises(RequestEntityTooLarge):
                req.form["foo"]

    def test_multipart_content_limit(self) -> None:
        req = Request(
            raw_environ(
                MULTIPART, content_type=MULTIPART_TYPE, content_length=len(MULTIPART)
            )
        )

        with body_budget(max_content_length=10):
            with pytest.raises(RequestEntityTooLarge):
                req.form["foo"]

    def test_multipart_parts_limit(self) -> None:
        req = Request(
            raw_environ(
                MULTIPART, content_type=MULTIPART_TYPE, content_length=len(MULTIPART)
            )
        )

        with body_budget(max_form_parts=1):
            with pytest.raises(RequestEntityTooLarge):
                req.form["foo"]

    def test_multipart_parses_inside_generous_frame(self) -> None:
        req = Request(
            raw_environ(
                MULTIPART, content_type=MULTIPART_TYPE, content_length=len(MULTIPART)
            )
        )

        with body_budget(
            max_content_length=10_000,
            max_form_memory_size=10_000,
            max_form_parts=100,
        ):
            assert req.form["foo"] == "Hello World"

    def test_strict_min_frame_cannot_relax_default_memory(self) -> None:
        req = Request(
            raw_environ(
                MULTIPART, content_type=MULTIPART_TYPE, content_length=len(MULTIPART)
            )
        )

        # Baseline default is 500kB; a huge strict frame keeps the baseline,
        # but a 7-byte frame narrows it.
        with body_budget(max_form_memory_size=10_000_000):
            assert req.body_budget.max_form_memory_size == 500_000


class TestUrlencodedContentOnly:
    def test_urlencoded_ignores_memory_and_parts(self) -> None:
        req = Request(
            raw_environ(
                URLENCODED,
                content_type=URLENCODED_TYPE,
                content_length=len(URLENCODED),
            )
        )

        with body_budget(max_form_memory_size=1, max_form_parts=1):
            assert req.form["foo"] == "Hello World"
            assert req.form["bar"] == "baz"

    def test_urlencoded_still_content_limited(self) -> None:
        req = Request(
            raw_environ(
                URLENCODED,
                content_type=URLENCODED_TYPE,
                content_length=len(URLENCODED),
            )
        )

        with body_budget(max_content_length=4):
            with pytest.raises(RequestEntityTooLarge):
                req.form["foo"]

    def test_multipart_not_parsed_under_profile(self) -> None:
        req = Request(
            raw_environ(
                MULTIPART, content_type=MULTIPART_TYPE, content_length=len(MULTIPART)
            )
        )

        with body_budget(form_quota_profile=FormQuotaProfile.URLENCODED_CONTENT_ONLY):
            assert req.form == {}
            assert req.files == {}
            # The stream passes through untouched; no multipart limits fire.
            assert req.stream.read() == MULTIPART

    def test_urlencoded_parses_under_profile(self) -> None:
        req = Request(
            raw_environ(
                URLENCODED,
                content_type=URLENCODED_TYPE,
                content_length=len(URLENCODED),
            )
        )

        with body_budget(form_quota_profile=FormQuotaProfile.URLENCODED_CONTENT_ONLY):
            assert req.form["foo"] == "Hello World"


class TestOpaqueBody:
    def test_form_entries_raise(self) -> None:
        req = Request(
            raw_environ(
                URLENCODED,
                content_type=URLENCODED_TYPE,
                content_length=len(URLENCODED),
            )
        )

        with body_budget(form_quota_profile=FormQuotaProfile.OPAQUE_BODY):
            with pytest.raises(BodyBudgetError):
                _ = req.form
            with pytest.raises(BodyBudgetError):
                _ = req.files
            with pytest.raises(BodyBudgetError):
                _ = req.values

            assert "form" not in req.__dict__

    def test_get_data_and_get_json_allowed(self) -> None:
        payload = json.dumps({"x": 1}).encode()
        req = Request(
            raw_environ(
                payload,
                content_type="application/json",
                content_length=len(payload),
            )
        )

        with body_budget(form_quota_profile=FormQuotaProfile.OPAQUE_BODY):
            assert req.get_json() == {"x": 1}
            assert req.get_data(cache=False) == payload
            # The data descriptor routes through get_data(parse_form_data=True)
            assert req.data == payload
            assert "form" not in req.__dict__

    def test_get_data_parse_form_data_does_not_enter_parser(self) -> None:
        req = Request(
            raw_environ(
                URLENCODED,
                content_type=URLENCODED_TYPE,
                content_length=len(URLENCODED),
            )
        )

        def boom() -> None:
            raise AssertionError("form parser must not run")

        req._load_form_data = boom  # type: ignore[method-assign]

        with body_budget(form_quota_profile=FormQuotaProfile.OPAQUE_BODY):
            assert req.get_data(parse_form_data=True) == URLENCODED

    def test_form_works_after_pop(self) -> None:
        req = Request(
            raw_environ(
                URLENCODED,
                content_type=URLENCODED_TYPE,
                content_length=len(URLENCODED),
            )
        )

        with body_budget(form_quota_profile=FormQuotaProfile.OPAQUE_BODY):
            with pytest.raises(BodyBudgetError):
                _ = req.form
            req.get_data()

        assert req.form["foo"] == "Hello World"


# ---------------------------------------------------------------------------
# FormDataParser bypasses read the current stack
# ---------------------------------------------------------------------------


def test_parse_from_environ_resolves_pushed_content_limit() -> None:
    environ = raw_environ(
        URLENCODED, content_type=URLENCODED_TYPE, content_length=len(URLENCODED)
    )
    parser = FormDataParser(max_content_length=10_000)

    _, form, _ = parser.parse_from_environ(environ)
    assert form["foo"] == "Hello World"

    environ["wsgi.input"] = io.BytesIO(URLENCODED)

    with body_budget(max_content_length=4):
        with pytest.raises(RequestEntityTooLarge):
            FormDataParser(max_content_length=10_000).parse_from_environ(environ)


def test_parse_from_environ_resolves_gate_and_opaque_profile() -> None:
    # Gate from the stack: length-unknown terminated stream capped read-time.
    environ = raw_environ(
        b"abcdef", content_type="text/plain", content_length=..., terminated=True
    )

    with body_budget(max_content_length=3):
        stream, _, _ = FormDataParser().parse_from_environ(environ)
        assert stream.read(3) == b"abc"

        with pytest.raises(RequestEntityTooLarge):
            stream.read(1)

    # Opaque profile blocks the standalone bypass as well.
    environ = raw_environ(
        URLENCODED, content_type=URLENCODED_TYPE, content_length=len(URLENCODED)
    )

    with body_budget(form_quota_profile=FormQuotaProfile.OPAQUE_BODY):
        with pytest.raises(BodyBudgetError):
            parse_form_data(environ)


def test_parser_direct_parse_uses_configured_profile() -> None:
    stream = io.BytesIO(URLENCODED)
    parser = FormDataParser(form_quota_profile=FormQuotaProfile.OPAQUE_BODY)

    with pytest.raises(BodyBudgetError):
        parser.parse(stream, URLENCODED_TYPE, len(URLENCODED))


# ---------------------------------------------------------------------------
# Baseline attributes stay effective; frames take priority
# ---------------------------------------------------------------------------


def test_baseline_attribute_still_enforced_without_frame() -> None:
    req = Request(
        raw_environ(b"x" * 100, content_type="text/plain", content_length=100)
    )
    req.max_content_length = 5
    with pytest.raises(RequestEntityTooLarge):
        req.stream.read()


def test_frame_narrows_baseline_instance_attribute() -> None:
    req = Request(
        raw_environ(b"x" * 100, content_type="text/plain", content_length=100)
    )
    req.max_content_length = 50

    with body_budget(max_content_length=10):
        assert req.body_budget.max_content_length == 10
        with pytest.raises(RequestEntityTooLarge):
            req.stream.read()


# ---------------------------------------------------------------------------
# Concurrency: concurrent requests push different budgets without crossing
# ---------------------------------------------------------------------------


def test_concurrent_requests_do_not_share_stacks() -> None:
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker(cap: int) -> None:
        try:
            with body_budget(max_content_length=cap):
                barrier.wait()

                for _ in range(50):
                    view = resolve_body_budget(
                        max_content_length=10_000,
                        max_form_memory_size=500_000,
                        max_form_parts=1000,
                    )
                    assert view.max_content_length == cap
                    assert current_body_budget_frames()[-1].max_content_length == cap
                    time.sleep(0)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t1 = threading.Thread(target=worker, args=(10,))
    t2 = threading.Thread(target=worker, args=(20,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []
    # The frames never escaped into the main thread.
    assert current_body_budget_frames() == ()
    view = resolve_body_budget(
        max_content_length=99, max_form_memory_size=None, max_form_parts=None
    )
    assert view.max_content_length == 99


def test_frame_does_not_leak_between_sequential_requests() -> None:
    environ = raw_environ(b"abc", content_type="text/plain", content_length=3)

    with body_budget(max_content_length=1):
        with pytest.raises(RequestEntityTooLarge):
            Request(environ).stream.read()

    # New request on the same thread after the frame was popped sees baseline.
    environ = raw_environ(b"abc", content_type="text/plain", content_length=3)
    req = Request(environ)
    assert req.body_budget.max_content_length is None
    assert req.stream.read() == b"abc"


# ---------------------------------------------------------------------------
# create_environ based sanity for the documented integration path
# ---------------------------------------------------------------------------


def test_middleware_style_temporary_tightening() -> None:
    environ = create_environ(
        method="POST",
        data=b"x" * 50,
        content_type="application/octet-stream",
    )
    req = Request(environ)

    # Outer section allows the body.
    assert req.get_data(cache=False) == b"x" * 50

    environ = create_environ(
        method="POST",
        data=b"x" * 50,
        content_type="application/octet-stream",
    )
    req = Request(environ)

    # Nested section tightens the total budget; baseline stays unchanged.
    with body_budget(max_content_length=10):
        with pytest.raises(RequestEntityTooLarge):
            req.get_data(cache=False)

    assert req.max_content_length is None
