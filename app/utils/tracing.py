"""
Private Tracing setup and utility functions for OpenTelemetry.
Exposes setup_tracing, get_tracer, trace_span, trace_span_async and traced decorator.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from inspect import signature
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_tracer: trace.Tracer | None = None


def _build_tracer_provider(
    service_name: str = "dotsocr",
    otel_exporter_otlp_traces_endpoint: Optional[str] = None,
    otel_exporter_otlp_traces_timeout: Optional[int] = None,
) -> trace.TracerProvider:
    if otel_exporter_otlp_traces_endpoint is None:
        return trace.NoOpTracerProvider()

    trace_provider = trace_sdk.TracerProvider(
        resource=Resource(attributes={"service.name": service_name})
    )
    otlp_exporter = OTLPSpanExporter(
        endpoint=otel_exporter_otlp_traces_endpoint,
        timeout=otel_exporter_otlp_traces_timeout,
    )
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace_provider.add_span_processor(span_processor)
    return trace_provider


def setup_tracing(
    app,
    service_name: str = "dotsocr",
    otel_exporter_otlp_traces_endpoint: Optional[str] = None,
    otel_exporter_otlp_traces_timeout: Optional[int] = None,
) -> trace.Tracer:
    """
    Initialize the tracer with OTLP exporter and FastAPI instrumentation.
    """
    global _tracer

    if _tracer is not None:
        return _tracer  # Already initialized

    trace_provider = _build_tracer_provider(
        service_name,
        otel_exporter_otlp_traces_endpoint,
        otel_exporter_otlp_traces_timeout,
    )

    trace.set_tracer_provider(trace_provider)

    _tracer = trace.get_tracer(__name__)

    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    SQLAlchemyInstrumentor().instrument(
        tracer_provider=trace.get_tracer_provider(),
    )

    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()

    FastAPIInstrumentor.instrument_app(
        app=app,
        tracer_provider=trace.get_tracer_provider(),
        exclude_spans=["receive", "send"],
    )
    return _tracer


def get_tracer() -> trace.Tracer:
    """
    Get the initialized tracer instance.
    Raises RuntimeError if tracer has not been set up.
    """
    if _tracer is None:
        raise RuntimeError("Tracer is not initialized. Call setup_tracing first.")
    return _tracer


def start_child_span(name: str, parent_span: Optional[trace.Span] = None) -> trace.Span:
    """
    Start and return a child span of the given parent span.
    If parent_span is None, starts a new span with current context.
    """
    tracer = get_tracer()
    if parent_span is not None and parent_span.get_span_context().is_valid:
        return tracer.start_span(name, context=trace.set_span_in_context(parent_span))
    else:
        return tracer.start_span(name)


@contextmanager
def trace_span(name: str, **attributes):
    """
    Sync context manager for tracing a span.

    - Records CPU time automatically as a span attribute.
    - Records any raised exception into the span.
    """

    tracer = trace.get_tracer(__name__)
    cpu_start = time.process_time()

    with tracer.start_as_current_span(name) as span:
        # Set initial attributes
        for k, v in attributes.items():
            span.set_attribute(k, v)

        try:
            yield span
        except Exception as e:
            # Record the exception and mark span as errored
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            # Always record CPU time
            span.set_attribute(
                "cpu_time_ns", (time.process_time() - cpu_start) * 1_000_000_000
            )


@asynccontextmanager
async def trace_span_async(name: str, **attributes):
    """
    Async context manager for tracing a span.

    - Records CPU time automatically as a span attribute.
    - Records any raised exception into the span.
    """

    tracer = trace.get_tracer(__name__)
    cpu_start = time.process_time()

    with tracer.start_as_current_span(name) as span:
        # Set initial attributes
        for k, v in attributes.items():
            span.set_attribute(k, v)

        try:
            yield span
        except Exception as e:
            # Record the exception and mark span as errored
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            # Always record CPU time
            span.set_attribute(
                "cpu_time_ns", (time.process_time() - cpu_start) * 1_000_000_000
            )


def traced(
    name: Optional[str] = None,
    record_args: Optional[list[str]] = None,
    record_return: bool = False,
):
    """
    Decorator to wrap a function (sync or async) in a tracing span.
    Optionally record only specified function arguments.

    :param name: span name. Defaults to function name if None.
    :param record_args: list of argument names to record. If None, record all.
    :param record_return: whether to record the return value. Defaults to False.
    """

    def decorator(func):
        span_name = name or func.__name__
        sig = signature(func)

        def _build_attributes(args, kwargs):
            """
            Build individual attributes for each selected function argument.
            """
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            attributes = {}
            for arg_name, value in bound.arguments.items():
                if record_args is None or arg_name in record_args:
                    attributes[f"param.{arg_name}"] = str(value)
            return attributes

        def _build_json_attributes(args, kwargs, exclude: list[str] = ["self", "cls"]):
            """
            Build a single JSON string attribute with selected function arguments.
            """
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            data = {}
            for k, v in bound.arguments.items():
                if k in exclude:
                    continue
                if record_args is None or k in record_args:
                    try:
                        data[k] = v
                    except Exception:
                        data[k] = repr(v)
            return {"params": json.dumps(data, default=str)}

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                attrs = _build_json_attributes(args, kwargs)
                async with trace_span_async(span_name, **attrs) as span:
                    result = await func(*args, **kwargs)
                    if record_return:
                        try:
                            span.set_attribute(
                                "return", json.dumps(result, default=str)
                            )
                        except Exception:
                            span.set_attribute("return", repr(result))
                    return result

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                attrs = _build_json_attributes(args, kwargs)
                with trace_span(span_name, **attrs) as span:
                    result = func(*args, **kwargs)
                    if record_return:
                        try:
                            span.set_attribute(
                                "return", json.dumps(result, default=str)
                            )
                        except Exception:
                            span.set_attribute("return", repr(result))
                    return result

            return sync_wrapper

    return decorator
