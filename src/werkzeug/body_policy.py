"""Request-scoped body parsing and length budget policy stack.

This module implements a nestable, restorable stack of body budget frames
that sits on top of the static :attr:`~werkzeug.wrappers.Request.max_content_length`,
:attr:`~werkzeug.wrappers.Request.max_form_memory_size`, and
:attr:`~werkzeug.wrappers.Request.max_form_parts` configuration. It allows
middleware or nested calls to temporarily tighten (or, depending on the frame,
replace) the request body budget without mutating class or instance attributes.

The stack is deliberately decoupled from :class:`~werkzeug.local.LocalStack`
and from any request context object. It is stored in a
:class:`~contextvars.ContextVar`, which is a single shared mount point that
keeps per-execution-context state. Concurrent requests handled in separate
threads (or async tasks) therefore see independent stacks even though they
enter the same mount point.

A frame expresses three things:

-   How its numeric limits combine with the outer effective budget and the
    :class:`~werkzeug.wrappers.Request` baseline (:class:`BudgetMerge`).
-   How a request without an advertised ``Content-Length`` may be read
    (:class:`StreamGate`).
-   Which body parsing entry points are allowed in the frame
    (:class:`FormQuotaProfile`).

When the stack is empty, every observation is identical to relying only on the
existing ``max_*`` attributes. Popping an inner frame restores the exact outer
frame; it never falls back to a process-wide shared state.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import enum
import typing as t

__all__ = [
    "BodyBudgetError",
    "BodyBudgetFrame",
    "BodyBudgetToken",
    "BudgetMerge",
    "EffectiveBodyBudget",
    "FormQuotaProfile",
    "StreamGate",
    "body_budget",
    "current_body_budget_frames",
    "pop_body_budget",
    "push_body_budget",
    "resolve_body_budget",
]


class BudgetMerge(enum.Enum):
    """How a frame's numeric limits combine with the budget already in effect
    from the outer frames and the :class:`~werkzeug.wrappers.Request`
    baseline.

    ``None`` always means "unlimited" when explicitly passed as a limit.
    """

    STRICT_MIN = "strict_min"
    """Each limit becomes the smaller of the frame value and the outer
    effective value. A frame can only narrow, never widen. A limit the frame
    does not set is inherited unchanged."""

    ACTIVE_FRAME = "active_frame"
    """Only the limit numbers on this frame are recognized. A limit set on the
    frame (including an explicit ``None`` for unlimited) replaces the outer
    effective value; a limit the frame does not set falls back to the
    :class:`~werkzeug.wrappers.Request` baseline."""

    INHERIT_ONLY = "inherit_only"
    """Like :attr:`STRICT_MIN`, but the frame may only narrow the outer
    effective budget. Attempting to widen it (larger limit or removing a
    limit) raises :class:`BodyBudgetError` when the budget is resolved."""


class StreamGate(enum.Enum):
    """How the guarded :attr:`~werkzeug.wrappers.Request.stream` may read a
    request whose length is not advertised with ``Content-Length``.

    Requests with a known ``Content-Length`` are always bounded by that value
    and checked against the effective ``max_content_length`` regardless of the
    gate; these modes only differ for length-unknown (streaming) requests.
    """

    CONTENT_LENGTH = "content_length"
    """Only an advertised ``Content-Length`` bounds the stream. The
    ``wsgi.input_terminated`` marker is not used to enforce
    ``max_content_length``: a length-unknown stream is returned raw if the
    server still marks it terminated, otherwise the safe empty fallback
    applies."""

    TERMINATED = "terminated"
    """A length-unknown stream may be read only when the WSGI server sets
    ``wsgi.input_terminated``. It is then wrapped with the effective
    ``max_content_length`` (or passed through when there is no limit). Without
    the marker the safe empty fallback applies. This is the historical
    behavior and the default when no frame is active."""

    EMPTY_IF_UNKNOWN = "empty_if_unknown"
    """A length-unknown request always yields an empty stream, even when
    ``wsgi.input_terminated`` is set. This is the strictest gate and matches
    the historical safe default for environments without the marker."""


class FormQuotaProfile(enum.Enum):
    """Which form parsing entry points the frame allows."""

    MULTIPART_TRIPLE = "multipart_triple"
    """``multipart/form-data`` is parsed with the effective
    ``max_content_length``, ``max_form_memory_size``, and ``max_form_parts``
    limits. ``application/x-www-form-urlencoded`` is parsed with the content
    limit only; the memory and parts limits are never applied to it. This is
    the historical behavior and the default."""

    URLENCODED_CONTENT_ONLY = "urlencoded_content_only"
    """Only ``application/x-www-form-urlencoded`` form data is parsed, bounded
    solely by the effective ``max_content_length``. ``max_form_memory_size``
    is never applied to urlencoded data. Multipart data is not parsed through
    the form entry; the stream passes through with empty ``form`` and
    ``files``."""

    OPAQUE_BODY = "opaque_body"
    """Only whole-body reads through ``get_data`` / ``get_json`` are allowed.
    Accessing the form parsing entry points (``form``, ``files`` and anything
    that triggers them) raises :class:`BodyBudgetError`; no form parser runs."""


class BodyBudgetError(RuntimeError):
    """Raised when the body budget stack is misused (unbalanced pop, widening
    an :attr:`BudgetMerge.INHERIT_ONLY` budget) or when a body read violates
    the active :class:`FormQuotaProfile`."""


class _Unset:
    """Sentinel for a budget number that a frame does not set.

    This is distinct from ``None``, which explicitly means unlimited.
    """

    _instance: t.ClassVar[_Unset | None] = None

    def __new__(cls) -> _Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()

# Type of a budget number: unset sentinel, unlimited (None), or bytes.
_BudgetNumber = t.Union[int, None, _Unset]


@dataclasses.dataclass(frozen=True)
class BodyBudgetFrame:
    """One frame of body budget policy pushed onto the stack.

    Every numeric limit defaults to the :data:`UNSET` sentinel, meaning the
    frame does not mention that limit. Passing ``None`` explicitly means
    "unlimited". Only non-negative integers, ``None``, and :data:`UNSET` are
    valid. The gate and profile likewise default to :data:`UNSET`, meaning the
    frame does not change the outer effective gate or profile; pass the enum
    member explicitly to set (or restore) one.

    :param max_content_length: Total request body size budget in bytes.
    :param max_form_memory_size: In-memory size budget for multipart text
        parts in bytes.
    :param max_form_parts: Maximum number of multipart parts.
    :param budget_merge: How the frame's numbers combine with the outer
        effective budget.
    :param stream_gate: How length-unknown request streams may be read. When
        unset, the effective gate is inherited from the nearest outer frame
        that sets one, or :attr:`StreamGate.TERMINATED` when no frame does.
    :param form_quota_profile: Which form parsing entry points are allowed.
        When unset, the effective profile is inherited from the nearest outer
        frame that sets one, or :attr:`FormQuotaProfile.MULTIPART_TRIPLE` when
        no frame does.
    """

    max_content_length: _BudgetNumber = UNSET
    max_form_memory_size: _BudgetNumber = UNSET
    max_form_parts: _BudgetNumber = UNSET
    budget_merge: BudgetMerge = BudgetMerge.STRICT_MIN
    stream_gate: StreamGate | _Unset = UNSET
    form_quota_profile: FormQuotaProfile | _Unset = UNSET

    def __post_init__(self) -> None:
        for name in (
            "max_content_length",
            "max_form_memory_size",
            "max_form_parts",
        ):
            value = getattr(self, name)

            if value is UNSET or value is None:
                continue

            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    f"{name} must be a non-negative int, None, or UNSET, not {value!r}"
                )

        if not isinstance(self.budget_merge, BudgetMerge):
            raise TypeError("budget_merge must be a BudgetMerge member")

        if self.stream_gate is not UNSET and not isinstance(
            self.stream_gate, StreamGate
        ):
            raise TypeError("stream_gate must be a StreamGate member or UNSET")

        if self.form_quota_profile is not UNSET and not isinstance(
            self.form_quota_profile, FormQuotaProfile
        ):
            raise TypeError(
                "form_quota_profile must be a FormQuotaProfile member or UNSET"
            )


@dataclasses.dataclass(frozen=True)
class EffectiveBodyBudget:
    """The budget in effect after folding the whole stack onto a
    :class:`~werkzeug.wrappers.Request` baseline. The numeric limits are
    concrete (``None`` means unlimited), and the gate/profile come from the
    innermost frame that sets each axis, or the historical defaults when no
    frame sets them."""

    max_content_length: int | None
    max_form_memory_size: int | None
    max_form_parts: int | None
    stream_gate: StreamGate
    form_quota_profile: FormQuotaProfile


@dataclasses.dataclass(frozen=True)
class BodyBudgetToken:
    """Opaque token returned by :func:`push_body_budget` and required by
    :func:`pop_body_budget`. It pins the exact frame pushed so pops can be
    checked for LIFO balance."""

    _context_token: contextvars.Token[tuple[BodyBudgetFrame, ...] | None]
    _frame: BodyBudgetFrame


# The single shared mount point. Each thread/async task context gets its own
# value, so concurrent requests never share a stack, but they all enter the
# same context variable.
_body_budget_stack: contextvars.ContextVar[tuple[BodyBudgetFrame, ...] | None] = (
    contextvars.ContextVar("werkzeug_body_budget_stack", default=None)
)


def current_body_budget_frames() -> tuple[BodyBudgetFrame, ...]:
    """Return the frames currently pushed in this execution context, from the
    outermost frame to the top frame."""

    return _body_budget_stack.get() or ()


def _min_limit(a: int | None, b: int | None) -> int | None:
    """Return the smaller of two limits, treating ``None`` as unlimited."""

    if a is None:
        return b

    if b is None:
        return a

    return min(a, b)


def _merge_limit(
    candidate: _BudgetNumber,
    effective: int | None,
    mode: BudgetMerge,
    name: str,
) -> int | None:
    if candidate is UNSET:
        return effective

    if mode is BudgetMerge.STRICT_MIN:
        return _min_limit(effective, t.cast("int | None", candidate))

    if mode is BudgetMerge.ACTIVE_FRAME:
        return t.cast("int | None", candidate)

    # INHERIT_ONLY: only narrowing is permitted.
    if candidate is None:
        if effective is None:
            return None

        raise BodyBudgetError(
            f"Cannot remove the {name} limit in an 'inherit_only' frame, the"
            f" outer budget limits it to {effective} bytes."
        )

    candidate_int = t.cast(int, candidate)

    if effective is not None and candidate_int > effective:
        raise BodyBudgetError(
            f"Cannot widen {name} from {effective} to {candidate_int} in an"
            " 'inherit_only' frame; inner frames may only narrow the outer"
            " effective budget."
        )

    return candidate_int


def resolve_body_budget(
    *,
    max_content_length: int | None,
    max_form_memory_size: int | None,
    max_form_parts: int | None,
) -> EffectiveBodyBudget:
    """Resolve the effective budget for one request.

    The three arguments are the :class:`~werkzeug.wrappers.Request` baseline
    values (class or instance attributes). The active frames are folded onto
    them in push order. With an empty stack the returned view is exactly the
    baseline with the historical gate and profile, so observations match the
    plain ``max_*`` configuration.
    """

    frames = _body_budget_stack.get()

    if not frames:
        return EffectiveBodyBudget(
            max_content_length=max_content_length,
            max_form_memory_size=max_form_memory_size,
            max_form_parts=max_form_parts,
            stream_gate=StreamGate.TERMINATED,
            form_quota_profile=FormQuotaProfile.MULTIPART_TRIPLE,
        )

    content = max_content_length
    memory = max_form_memory_size
    parts = max_form_parts
    # The gate and profile are not numeric budgets, so strict_min/active_frame
    # do not apply to them. A frame inherits whichever axis it leaves unset,
    # so a frame that only tightens the numbers cannot silently reopen a
    # stricter outer gate or relax an outer profile. The innermost frame that
    # explicitly sets an axis wins.
    gate: StreamGate = StreamGate.TERMINATED
    profile: FormQuotaProfile = FormQuotaProfile.MULTIPART_TRIPLE

    for frame in frames:
        content = _merge_limit(
            frame.max_content_length, content, frame.budget_merge, "max_content_length"
        )
        memory = _merge_limit(
            frame.max_form_memory_size,
            memory,
            frame.budget_merge,
            "max_form_memory_size",
        )
        parts = _merge_limit(
            frame.max_form_parts, parts, frame.budget_merge, "max_form_parts"
        )

        if frame.stream_gate is not UNSET:
            gate = t.cast(StreamGate, frame.stream_gate)

        if frame.form_quota_profile is not UNSET:
            profile = t.cast(FormQuotaProfile, frame.form_quota_profile)

    return EffectiveBodyBudget(
        max_content_length=content,
        max_form_memory_size=memory,
        max_form_parts=parts,
        stream_gate=gate,
        form_quota_profile=profile,
    )


def push_body_budget(
    frame: BodyBudgetFrame | None = None, /, **options: t.Any
) -> BodyBudgetToken:
    """Push a :class:`BodyBudgetFrame` onto the stack of the current execution
    context and return a token for :func:`pop_body_budget`.

    Either pass a fully constructed frame or the same keyword arguments as
    :class:`BodyBudgetFrame`. Prefer the :func:`body_budget` context manager,
    which guarantees the frame is popped even when an exception escapes.
    """

    if frame is None:
        frame = BodyBudgetFrame(**options)
    elif options:
        raise TypeError("Pass either a frame or keyword arguments, not both.")
    elif not isinstance(frame, BodyBudgetFrame):
        raise TypeError("frame must be a BodyBudgetFrame instance")

    stack = _body_budget_stack.get()
    token = _body_budget_stack.set((stack or ()) + (frame,))
    return BodyBudgetToken(_context_token=token, _frame=frame)


def pop_body_budget(token: BodyBudgetToken) -> BodyBudgetFrame:
    """Pop the frame represented by ``token`` and restore the previous stack
    state. Raises :class:`BodyBudgetError` if the token does not match the
    current top frame, so nested pushes and pops cannot unbalance each other.
    """

    if not isinstance(token, BodyBudgetToken):
        raise TypeError("token must be the object returned by push_body_budget")

    stack = _body_budget_stack.get()

    if not stack or stack[-1] is not token._frame:
        raise BodyBudgetError(
            "Unbalanced body budget pop: the token does not match the current"
            " top frame. An inner frame is still active or was already popped."
        )

    _body_budget_stack.reset(token._context_token)
    return token._frame


@contextlib.contextmanager
def body_budget(
    frame: BodyBudgetFrame | None = None, /, **options: t.Any
) -> t.Iterator[BodyBudgetToken]:
    """Context manager pushing a body budget frame for the duration of the
    ``with`` block. The frame is popped on every exit path, including when an
    exception escapes, so outer policy (or the empty stack) is always
    restored::

        with body_budget(max_content_length=1024):
            request.get_data()  # bounded by the stricter of 1024 and baseline
    """

    token = push_body_budget(frame, **options)

    try:
        yield token
    finally:
        # On every normal exit path this frame is still the top, so the pop
        # restores the exact outer state. An unbalanced manual push/pop inside
        # the block is a programming error and must surface instead of being
        # silently left on the stack.
        pop_body_budget(token)
