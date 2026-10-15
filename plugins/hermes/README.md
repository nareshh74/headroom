# Hermes Agent Integration

## Native compression (including Copilot subscriptions)

`headroom_compress` uses Hermes's supported `llm_request` middleware. It runs
Headroom's local, lossless `densify()` pipeline on outgoing tool-result text.
Hermes keeps its provider, model, credentials, endpoints, streaming and retries.
No proxy, API key purchase or Hermes source patch is needed. Headroom does not
handle Copilot credentials; Hermes keeps its existing authentication flow.

Install `headroom-ai>=0.41.0` in the Python environment that runs Hermes, then
copy `plugins/hermes/headroom_compress` from this checkout into the active
profile's `plugins` directory. For a default Unix installation:

```bash
mkdir -p "${HERMES_HOME:-$HOME/.hermes}/plugins"
cp -r plugins/hermes/headroom_compress "${HERMES_HOME:-$HOME/.hermes}/plugins/"
hermes plugins enable headroom_compress
hermes plugins doctor headroom_compress
```

For named profiles, use that profile's home and `hermes -p <profile> plugins enable
headroom_compress`. Start a fresh Hermes session. Keep `model.provider: copilot`
and your existing model; no `headroom wrap hermes` command exists or is needed.
Installing the `headroom` CLI in a separate environment does not install the
library in Hermes's environment. Use the package-management workflow for your
Hermes installation to add the dependency; the manifest never auto-installs it.

The plugin supports string tool results in Chat Completions, Responses
(`function_call_output`) and Anthropic Messages (`tool_result`). It preserves
user/assistant/system text, tool calls and IDs, reasoning, images and request
options. Structured/multimodal results and Anthropic error results pass through.
Original session history is unchanged. Unsupported or failed compression also
passes through. Lossless mode performs no model downloads, emits no CCR markers
and needs no retrieval tool. Repeated input produces stable compressed text.
Savings appear as `Headroom saved N tool-result tokens` in Hermes logs.

Requires a Hermes version exposing `ctx.register_middleware("llm_request", ...)`.
Copilot ACP uses an external CLI and is not covered by this HTTP-request plugin.
Disable with `hermes plugins disable headroom_compress`, then start a new session.

## Proxy retrieval (separate, optional integration)

CCR retrieval plugin for [Hermes Agent](https://hermes-agent.nousresearch.com/) (Nous Research). Gives Hermes a native `headroom_retrieve` tool so compression markers produced by the headroom proxy are no longer a black box — the agent can fetch the original content back on demand instead of guessing or re-running commands.

## Why this is needed

When Hermes routes its LLM traffic through `headroom proxy`, large tool outputs get compressed into markers like:

```
[1500 items compressed to 50. Retrieve more: hash=abc123]   # Kompress path
<<ccr:abc123>> / <<ccr:abc123,base64,4.5KB>>                # SmartCrusher opaque-blob path
```

Claude Code users get the `headroom_retrieve` MCP tool injected automatically. Hermes registers its own tools, so without this plugin the markers are irreversible from the agent's point of view — in practice the model either re-runs the original command (wasting tokens/time) or, worse, treats `ccr:abc123` as a file path and tries to `cat` it.

This plugin closes the loop by calling the proxy's `POST /v1/retrieve` HTTP endpoint directly. It is only needed for a separately configured proxy that emits CCR markers, not the native lossless middleware above.

## Install

1. Copy the plugin into Hermes's user plugin directory:

   ```bash
   mkdir -p ~/.hermes/plugins
   cp -r headroom_retrieve ~/.hermes/plugins/
   ```

2. Enable it in `~/.hermes/config.yaml`:

   ```yaml
   toolsets:
     - hermes-cli
     - web
     - headroom        # add this

   plugins:
     enabled:
       - headroom_retrieve
   ```

   > Note: once the `plugins.enabled` key exists it acts as an explicit allowlist — list any other user plugins you already rely on.

3. Restart the Hermes gateway / TUI (plugin discovery is cached per process).

## Recommended proxy configuration

Hermes tool names don't match headroom's built-in `DEFAULT_EXCLUDE_TOOLS` (which protects Claude Code's `Read`/`Grep`/`Edit`/...), so two exclusions are strongly recommended on the proxy side:

```bash
HEADROOM_EXCLUDE_TOOLS=read_file,headroom_retrieve
```

- `read_file` — Hermes's file reads are reference data the agent needs verbatim, same rationale as Claude Code's `Read`.
- `headroom_retrieve` — without this, retrieved originals get re-compressed on the next request, producing an endless marker→retrieve→marker loop.

## Behavior

- Accepts the bare hash or the whole marker — `<<ccr:abc123,base64,4.5KB>>`, `ccr:abc123`, and `hash=abc123` are all normalized to `abc123`.
- Retrieval is by hash and always returns the full original content.
- Clear, actionable errors: expired hash (TTL) and proxy-unreachable cases both tell the model to re-run the original command instead of retrying blindly.

## Requirements

- headroom proxy running on `127.0.0.1:8787` (edit `_PROXY_URL` in `__init__.py` otherwise)
- `httpx` (already a Hermes dependency)

Tested against headroom 0.22.4 and 0.23.0 with Hermes Agent on macOS and Linux.
