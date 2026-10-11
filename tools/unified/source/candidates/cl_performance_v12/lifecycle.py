"""Preserve failure evidence without retaining exported buffers in tracebacks."""
import traceback


def record_failure(error):
    """Record the full exception chain, then clear its finished frame locals.

    A traceback can own NumPy views into an anonymous mmap through locals in
    install/sampling helpers.  Those frames must not keep an otherwise released
    buffer exported during cleanup.  Active frames are deliberately left alone
    by traceback.clear_frames; their owners are cleared explicitly by callers.
    No allocation or registration is released here.
    """
    text = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
    return text


def cleanup_failed(event, failure, cleanup_failure):
    """Best-effort evidence; reporting must never replace a cleanup failure."""
    try:
        event('lifecycle_cleanup_failed', operation_error=failure,
              cleanup_error=cleanup_failure, lifecycle_complete=False)
    except BaseException:
        pass


def raise_failures(failure, cleanup_failure):
    if failure or cleanup_failure:
        parts = []
        if failure:
            parts.append('Original operation failed:\n' + failure)
        if cleanup_failure:
            parts.append('Resource cleanup failed; ownership was not released:\n' + cleanup_failure)
        error = RuntimeError('\n'.join(parts))
        error.original_traceback = failure
        error.cleanup_traceback = cleanup_failure
        raise error from None
