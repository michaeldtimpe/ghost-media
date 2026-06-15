#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  VISION BACKENDS
  One interface, several engines, so enrichment isn't welded to a single model.
═══════════════════════════════════════════════════════════════════════════════

The blog's priority hierarchy, adapted:
  - ollama       — local qwen2.5vl, free + offline, the bulk workhorse (default)
  - claude-cli   — Claude via the Claude Code CLI on the user's subscription
                   (zero marginal cost); best for quality / hard-case re-rating
  - anthropic-api— the metered Anthropic SDK; opt-in, only when you want API speed

Every backend exposes the same call:

    text, elapsed, error = backend.analyze_frame(image_path, prompt)

`text` is the raw model response (the caller parses + validates it via
vision_schema.normalize_analysis). `error` is None on success, else a short
string. Cloud backends also defensively detect refusal/permission text returned
*as a successful response* (the blog's "CLI returns refusals as success" gotcha)
and surface it as an error so retry / --reenrich-flagged logic can react.

Backends also expose a text-only variant (same return contract, no image):

    text, elapsed, error = backend.analyze_text(prompt)

used by enrich_audio.py to compose descriptive phrases from audio features.
Not implemented on the mlx backend (its default model is a VLM) — use ollama
or claude-cli for text composition.
"""

import base64
import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request

# Phrases that signal a refusal/permission-block returned as normal text.
_REFUSAL_MARKERS = (
    "i can't", "i cannot", "i'm unable", "i am unable", "i won't", "i will not",
    "i need permission", "i don't have permission", "i'm not able",
    "unable to assist", "can't help with", "cannot help with",
)


def _looks_like_refusal(text):
    """Scan the first ~200 chars for refusal/permission language.

    A substring scan (not startswith) because models sometimes prepend a short
    disclaimer before — or instead of — doing the work.
    """
    if not text:
        return False
    head = text[:200].lower()
    return any(m in head for m in _REFUSAL_MARKERS)


def _read_image_b64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─── Base ────────────────────────────────────────────────────────────────────

class VisionBackend:
    name = "base"
    supports_prompt_cache = False
    supports_batch = False
    max_resolution = None  # px on the long edge, None = backend default

    def __init__(self, model):
        self.model = model

    @property
    def label(self):
        return f"{self.name}:{self.model}"

    def provenance(self):
        return {"backend": self.name, "model": self.model}

    def health_check(self):
        """Return (ok: bool, message: str)."""
        raise NotImplementedError

    def analyze_frame(self, image_path, prompt):
        """Return (text, elapsed_sec, error_or_None)."""
        raise NotImplementedError

    def analyze_text(self, prompt):
        """Text-only call. Return (text, elapsed_sec, error_or_None)."""
        raise NotImplementedError(
            f"backend '{self.name}' does not support text-only calls — "
            "use ollama, claude-cli, or anthropic-api")


# ─── Ollama (local, default) ─────────────────────────────────────────────────

class OllamaBackend(VisionBackend):
    name = "ollama"
    supports_batch = False

    def __init__(self, model="qwen2.5vl:7b", api="http://localhost:11434"):
        super().__init__(model)
        self.api = api

    def health_check(self):
        try:
            req = urllib.request.Request(f"{self.api}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                names = {m["name"] for m in data.get("models", [])}
                base = self.model.split(":")[0]
                ok = self.model in names or any(base in n for n in names)
                return ok, ("ready" if ok else f"model {self.model} not pulled")
        except Exception as e:
            return False, f"ollama unreachable ({str(e)[:60]})"

    def analyze_frame(self, image_path, prompt):
        start = time.time()
        try:
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "images": [_read_image_b64(image_path)],
                "stream": False,
                # Cap context: single-frame description needs ~2.6k image tokens
                # + prompt + 1024 output (~3.9k). Without this Ollama allocates the
                # model's full 128k window (a ~52 GB KV cache for qwen2.5vl:7b),
                # ballooning memory and inference time. 8192 leaves 2x headroom.
                "options": {"num_predict": 1024, "num_ctx": 8192},
            }).encode()
            req = urllib.request.Request(
                f"{self.api}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            return data.get("response", "").strip(), time.time() - start, None
        except Exception as e:
            return None, time.time() - start, str(e)[:200]

    def analyze_text(self, prompt):
        start = time.time()
        try:
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 1024, "num_ctx": 8192},
            }).encode()
            req = urllib.request.Request(
                f"{self.api}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            return data.get("response", "").strip(), time.time() - start, None
        except Exception as e:
            return None, time.time() - start, str(e)[:200]


# ─── Claude via the Claude Code CLI (subscription, preferred cloud path) ──────

class ClaudeCLIBackend(VisionBackend):
    """Drives `claude -p` headlessly. Uses the logged-in Claude subscription
    (Max/Pro) — zero marginal cost — UNLESS ANTHROPIC_API_KEY is set in the
    environment, which would silently route to metered API billing. We warn on
    that in health_check().

    For unattended batches, generate a long-lived token with `claude setup-token`
    and export it as CLAUDE_CODE_OAUTH_TOKEN.
    """
    name = "claude-cli"
    supports_prompt_cache = True  # handled internally by the CLI

    def __init__(self, model="sonnet", binary="claude"):
        super().__init__(model)
        self.binary = binary

    def health_check(self):
        if shutil.which(self.binary) is None:
            return False, f"`{self.binary}` not on PATH"
        warn = ""
        if os.environ.get("ANTHROPIC_API_KEY"):
            warn = " (WARNING: ANTHROPIC_API_KEY set → API billing, not subscription)"
        try:
            r = subprocess.run([self.binary, "--version"],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return True, (r.stdout.strip() or "ready") + warn
            return False, f"`{self.binary} --version` failed"
        except Exception as e:
            return False, f"cli error ({str(e)[:60]})"

    def analyze_frame(self, image_path, prompt):
        start = time.time()
        abs_path = os.path.abspath(image_path)
        full_prompt = (
            f"{prompt}\n\n"
            f"The image to analyze is at this absolute path: {abs_path}\n"
            f"Use the Read tool to view it, then return only the JSON."
        )
        cmd = [
            self.binary, "-p", full_prompt,
            "--allowedTools", "Read",
            "--output-format", "json",
            "--model", self.model,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except Exception as e:
            return None, time.time() - start, f"cli error ({str(e)[:120]})"
        elapsed = time.time() - start

        if r.returncode != 0:
            err = (r.stderr or r.stdout or "non-zero exit").strip()
            return None, elapsed, f"cli exit {r.returncode}: {err[:160]}"

        out = (r.stdout or "").strip()
        # Preferred: --output-format json envelope with a `result` field.
        text = out
        try:
            env = json.loads(out)
            if isinstance(env, dict) and "result" in env:
                text = (env.get("result") or "").strip()
        except json.JSONDecodeError:
            pass

        if not text:
            return None, elapsed, "empty response"
        if _looks_like_refusal(text):
            return None, elapsed, f"refusal/permission: {text[:120]}"
        return text, elapsed, None

    def analyze_text(self, prompt):
        start = time.time()
        cmd = [
            self.binary, "-p", prompt,
            "--allowedTools", "",  # pure text composition: no tools needed
            "--output-format", "json",
            "--model", self.model,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except Exception as e:
            return None, time.time() - start, f"cli error ({str(e)[:120]})"
        elapsed = time.time() - start

        if r.returncode != 0:
            err = (r.stderr or r.stdout or "non-zero exit").strip()
            return None, elapsed, f"cli exit {r.returncode}: {err[:160]}"

        out = (r.stdout or "").strip()
        text = out
        try:
            env = json.loads(out)
            if isinstance(env, dict) and "result" in env:
                text = (env.get("result") or "").strip()
        except json.JSONDecodeError:
            pass

        if not text:
            return None, elapsed, "empty response"
        if _looks_like_refusal(text):
            return None, elapsed, f"refusal/permission: {text[:120]}"
        return text, elapsed, None


# ─── Anthropic API (opt-in, metered) ─────────────────────────────────────────

class AnthropicAPIBackend(VisionBackend):
    """Metered Anthropic SDK. Opt-in only — prefer claude-cli to use the
    subscription. Requires `pip install anthropic` and ANTHROPIC_API_KEY.
    """
    name = "anthropic-api"
    supports_prompt_cache = True

    def __init__(self, model="claude-haiku-4-5"):
        super().__init__(model)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed for this backend
            self._client = anthropic.Anthropic()
        return self._client

    def health_check(self):
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "anthropic SDK not installed (pip install anthropic)"
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY not set"
        return True, "ready (metered API billing)"

    def analyze_frame(self, image_path, prompt):
        start = time.time()
        try:
            client = self._get_client()
            msg = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": _read_image_b64(image_path),
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                            # Static prompt → cache it (no-op if under the
                            # cache-minimum token threshold).
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }],
            )
            elapsed = time.time() - start
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
            if not text:
                return None, elapsed, "empty response"
            if _looks_like_refusal(text):
                return None, elapsed, f"refusal: {text[:120]}"
            return text, elapsed, None
        except Exception as e:
            return None, time.time() - start, str(e)[:200]

    def analyze_text(self, prompt):
        start = time.time()
        try:
            client = self._get_client()
            msg = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.time() - start
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
            if not text:
                return None, elapsed, "empty response"
            if _looks_like_refusal(text):
                return None, elapsed, f"refusal: {text[:120]}"
            return text, elapsed, None
        except Exception as e:
            return None, time.time() - start, str(e)[:200]


# ─── MLX (local Apple-Silicon, via mlx-vlm) ──────────────────────────────────

class MLXVisionBackend(VisionBackend):
    """Local vision on Apple Silicon via mlx-vlm. Loads the model **in-process
    once** and caches it (`self._loaded`) — the `python -m mlx_vlm.generate` CLI
    reloads multi-GB weights on every call, which is catastrophic across thousands
    of frames. Free + offline; the bulk workhorse once the faster M5 Max is back.

    Degrades gracefully: if mlx-vlm isn't installed (the current dev-box state) the
    health check says so and the bench skips this engine without aborting the run.

    NOTE: mlx-vlm's `generate`/`apply_chat_template` signatures have drifted across
    releases; the calls are isolated here and the temperature kwarg is probed so a
    version bump needs at most a one-line tweak. Targets mlx-vlm >= 0.1.
    """
    name = "mlx"
    supports_batch = False
    max_resolution = 1024  # px on the long edge; 4-bit VLMs are memory-sensitive

    def __init__(self, model="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"):
        super().__init__(model)
        self._loaded = None  # (model, processor) cached after first load

    def health_check(self):
        try:
            import mlx_vlm  # noqa: F401
            import mlx.core  # noqa: F401
        except ImportError:
            return False, "mlx-vlm not installed (pip install mlx-vlm)"
        if platform.machine() != "arm64":
            return False, "mlx requires Apple Silicon (arm64)"
        return True, f"ready (model {self.model}; loads on first frame)"

    def _ensure_loaded(self):
        if self._loaded is None:
            from mlx_vlm import load
            self._loaded = load(self.model)  # (model, processor)
        return self._loaded

    def _generate(self, model, processor, formatted, image_path):
        from mlx_vlm import generate
        base = dict(max_tokens=1024, verbose=False)
        # temperature kwarg name changed across versions (temp/temperature/none).
        for temp_kw in ("temperature", "temp", None):
            kw = dict(base)
            if temp_kw:
                kw[temp_kw] = 0.0
            try:
                return generate(model, processor, formatted, [str(image_path)], **kw)
            except TypeError:
                continue
        return generate(model, processor, formatted, [str(image_path)], **base)

    def analyze_frame(self, image_path, prompt):
        start = time.time()
        try:
            from mlx_vlm.prompt_utils import apply_chat_template
            model, processor = self._ensure_loaded()
            cfg = getattr(model, "config", None)
            formatted = apply_chat_template(processor, cfg, prompt, num_images=1)
            result = self._generate(model, processor, formatted, image_path)
            # generate() returns a str (older) or a result object with .text (newer).
            text = result if isinstance(result, str) else getattr(result, "text", str(result))
            elapsed = time.time() - start
            text = (text or "").strip()
            if not text:
                return None, elapsed, "empty response"
            return text, elapsed, None  # local model: no refusal check needed
        except Exception as e:
            return None, time.time() - start, str(e)[:200]


# ─── Factory ─────────────────────────────────────────────────────────────────

_DEFAULT_MODELS = {
    "ollama": "qwen2.5vl:7b",
    "claude-cli": "sonnet",
    "anthropic-api": "claude-haiku-4-5",
    "mlx": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
}

_BACKENDS = {
    "ollama": OllamaBackend,
    "claude-cli": ClaudeCLIBackend,
    "anthropic-api": AnthropicAPIBackend,
    "mlx": MLXVisionBackend,
}


def get_backend(name=None, model=None):
    """Build a backend. `name` defaults to $GHOST_VISION_BACKEND or 'ollama'."""
    name = (name or os.environ.get("GHOST_VISION_BACKEND") or "ollama").lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown backend '{name}' (choose from {', '.join(_BACKENDS)})")
    model = model or _DEFAULT_MODELS[name]
    return _BACKENDS[name](model)
