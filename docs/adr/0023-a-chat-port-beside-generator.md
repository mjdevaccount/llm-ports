---
status: accepted
date: 2026-11-12
---

# A Chat port beside Generator: messages in, events and new messages out, tools run by the adapter

## Context and problem statement

`Generator` is one prompt in, one value out. A conversation that calls tools over several model
requests, and that a person watches arrive token by token, did not fit it: each consumer that
needed one would have written its own loop over a vendor SDK, with its own tool-call parsing,
its own handling of a tool that raises, and its own idea of when the loop stops. Those loops are
where a run that never ends, a tool result pasted into user text, and a failure raised across the
port all come from.

pydantic-ai already runs this loop (`Agent.iter`, tool calls, parallel execution, retries of a
malformed call, streamed deltas). What it does not give is the package's contract: the named
outcome, the usage and cost result, the one retry owner, the total deadline, the limiter, and an
event stream that ends with the result in every case.

## Decision

1. **A second port, `Chat`, beside `Generator`.** `stream(req)` yields `TextDelta`,
   `ReasoningDelta`, `ToolCallStarted`, `ToolCallFinished`, `StepFinished`, and `Finished(result)`
   exactly once and last. `chat(req)` is `stream` drained. The port is stateless: the result
   carries `messages`, the new turns, and the caller appends them to its own history.
2. **Tools are values the adapter runs.** `Tool(name, description, parameters, handler,
   timeout_s, untrusted_output, max_output_chars)` with an async handler returning `ToolOutput`.
   A handler that raises, times out or returns another type is an error result the model sees,
   never an exception. An output over its budget is cut in the middle (`[... N chars omitted]`)
   and marked `truncated`. An untrusted output reaches the model through the provider's
   tool-result channel inside a `<tool_output ... untrusted="true">` marker; it is never
   concatenated into user or system text.
3. **`max_steps` bounds the loop.** It is checked before each model request; a run past it is
   `truncated` with `provider_code="max_steps"`, and its messages keep the tool results made.
4. **ADR-0001 holds for streams.** A model, wire, tool or gate failure ends the stream with
   `Finished` carrying the failure; one after output began has `in_stream=True` and finish
   `interrupted`. Closing the iterator or cancelling the task stops the call (the producer task
   is cancelled and awaited) and `asyncio.CancelledError` propagates.
5. **The reference adapter is pydantic-ai's loop.** `PydanticAIChat` drives `Agent.iter` and
   shares `AdapterCore` with `PydanticAIGenerator`: the wire policy, the deadline clocks, the
   limiter slot, the failure classification and the result contract are one implementation.
6. **The call path is `call.chat` / `call.stream`.** Capability check (`tools`, `json_schema`,
   `vision`), admit gate, cache plan and pre-flight (ADR-0024), one trace observation with a
   child span per tool, graders on an `ok` reply. A conversation has no fallback chain.
7. **Conformance.** `testing.StubChat` plays a script of `Turn`s and really runs the tools;
   `testing.conformance.check_chat` holds the stub and the adapter to the same eleven cases.

## Considered options

- **Extend `Generator` with a message list and tools.** Rejected: every `Generator` would have to
  implement the loop or refuse it, and `generate`'s single-value contract would gain a second
  meaning.
- **Let the consumer run the loop over `Generator`.** Rejected: the loop is the hard part, and
  two consumers would write it twice.
- **Validate tool arguments against the full JSON schema.** Not taken: pydantic-ai's
  `Tool.from_schema` passes arguments through; the adapter checks the schema's `required` keys
  and re-asks the model when one is missing. A handler validates the rest.

## Consequences

- `tool_choice="required"` is not sent (pydantic-ai refuses it statically); the result carries a
  `CallWarning`. `"none"` and a tool name restrict the tools offered.
- `Message.cache` is honoured on user messages only; a cache point elsewhere has no wire form in
  pydantic-ai's model messages.
- The wire capture skips a `text/event-stream` body (reading it would buffer the stream); its
  usage and finish come from pydantic-ai's own parse of the stream.
- Langfuse records tool names and durations in the generation's metadata, not as child spans;
  OTel records an `execute_tool {name}` child span per tool.
