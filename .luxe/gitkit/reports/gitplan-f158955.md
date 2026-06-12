# Refactor change plan
**Steps: 6** — Decouple bench package from root-level modules

## S1: Create bench/vision.py adapter module
- **Files:** `bench/vision.py`
- **Change:** `extract` — `get_backend`, `SCENE_PROMPT`, `normalize_analysis` — Create bench/vision.py that re-exports get_backend from vision_backends, SCENE_PROMPT and normalize_analysis from vision_schema
- **Rationale:** Multiple bench modules import directly from root-level vision_backends and vision_schema, creating tight coupling
- **Risk:** low
- **Verify:** python -c "from bench.vision import get_backend, SCENE_PROMPT, normalize_analysis; print('ok')"

## S2: Create common.py and move try_parse_json there
- **Files:** `common.py`, `enrich_analyses.py`
- **Change:** `move` — `try_parse_json` — Move try_parse_json function from enrich_analyses.py to common.py, update enrich_analyses.py to import try_parse_json from common
- **Rationale:** try_parse_json is used by multiple bench modules, creating cross-boundary dependencies
- **Risk:** low
- **Verify:** python -c "from common import try_parse_json; print('ok')"

## S3: Move _fields from bench/runner.py to bench/util.py
- **Files:** `bench/runner.py`, `bench/util.py`, `bench/hybrid.py`
- **Change:** `move` — `_fields` — Move _fields function from bench/runner.py to bench/util.py, update bench/runner.py to import _fields from bench.util, update bench/hybrid.py to import _fields from bench.util
- **Rationale:** _fields is used by bench/hybrid.py, creating a cross-module dependency on a private function
- **Risk:** low
- **Verify:** python -c "from bench.util import _fields; print('ok')"

## S4: Update bench/runner.py to use bench.vision and common
- **Files:** `bench/runner.py`
- **Change:** `move` — `get_backend`, `SCENE_PROMPT`, `normalize_analysis`, `try_parse_json` — Replace from vision_backends import get_backend with from bench.vision import get_backend. Replace from vision_schema import SCENE_PROMPT, normalize_analysis with from bench.vision import SCENE_PROMPT, normalize_analysis. Replace from enrich_analyses import try_parse_json with from common import try_parse_json
- **Rationale:** Remove cross-boundary dependencies
- **Risk:** low
- **Verify:** python -c "from bench import runner; print('ok')"
- **Depends on:** S1, S2

## S5: Update bench/judge.py to use common
- **Files:** `bench/judge.py`
- **Change:** `move` — `try_parse_json` — Replace from enrich_analyses import try_parse_json with from common import try_parse_json
- **Rationale:** Remove cross-boundary dependency
- **Risk:** low
- **Verify:** python -c "from bench import judge; print('ok')"
- **Depends on:** S2

## S6: Update bench/hybrid.py to use bench.vision and common
- **Files:** `bench/hybrid.py`
- **Change:** `move` — `get_backend`, `SCENE_PROMPT`, `try_parse_json`, `_fields` — Replace imports from enrich_analyses, vision_backends, vision_schema with common and bench.vision. Replace from bench.runner import _fields with from bench.util import _fields. Fix json.load(open(fp)) to use load_json from bench.util
- **Rationale:** Remove cross-boundary dependencies and fix resource leak
- **Risk:** low
- **Verify:** python -c "from bench import hybrid; print('ok')"
- **Depends on:** S1, S2, S3
