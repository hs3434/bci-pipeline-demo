# AGENTS.md — bci-pipeline-demo

End-to-end EEG decoding pipeline (MNE + PyTorch + PyQt6). This file records
non-obvious facts about the repo that an agent would otherwise have to discover
the hard way.

## Layout

- Package source: `bci/` (single Python package, NOT a `src/` layout).
- Tests live at `bci/tests/` — collectable as `bci/tests/test_*.py`. There is
  no top-level `tests/` dir and no `conftest.py`.
- Specs and implementation plans: `docs/superpowers/specs/`,
  `docs/superpowers/plans/`. Read the spec before doing non-trivial work.
- Config: `pyproject.toml` (hatchling build, `requires-python = ">=3.11"`).
  No `setup.cfg`, no `tox.ini`, no `pytest.ini`, no `ruff`/`black`/`mypy`
  config — none of those tools are wired up.

## Environment

- Package manager is **uv** (`uv.lock` is checked in, no `requirements.txt`).
  The checked-in `.venv/` is a uv venv (`uv = 0.11.18`, Python 3.14.5).
- Install: `uv sync` (or `pip install .` for runtime, `pip install ".[dev]"`
  for tests — dev extras are `ipython`, `pytest`, `pytest-xdist`).
- Console script: `bci` → `bci.main:main`. Also reachable as
  `python -m bci.main`.

## Commands

```bash
# Run the full test suite (153 pass, 8 skipped — see "Skipped tests" below)
.venv/bin/python -m pytest

# A single test file or test
.venv/bin/python -m pytest bci/tests/test_decoder.py -v
.venv/bin/python -m pytest bci/tests/test_decoder.py::TestCNNDecoder::test_fit_trains_model

# Run the CLI on a real file
.venv/bin/python -m bci.main path/to/data.edf --config config.yaml
# or just: bci path/to/data.edf

# Launch the GUI (requires an X server; see "GUI" below)
.venv/bin/python -m bci.main --gui
```

There is **no lint, formatter, or typecheck command** — don't invent one. The
project deliberately ships without them. CI does not run (no `.github/`
directory).

## Architecture (one-screen summary)

- `bci/pipeline/__init__.py` — `BCIPipeline` orchestrates the linear chain
  `load → preprocess → create_epochs → decode`. `run()` does incremental
  re-execution: `_find_invalid_from()` compares per-step param snapshots
  (`_param_snapshot`) against `_states` and skips unchanged upstream steps.
  `load_raw()` must be called before `run()`.
- `bci/source/` — `FileSource.load(path | [paths, ...])` returns an
  `mne.io.Raw`. `StreamSource` wraps a `Raw` and yields fixed-duration
  chunks for the streaming tab. The reader registry
  (`bci/source/readers.py`) is populated by `import bci.source` — the
  `from . import readers` line in `bci/source/__init__.py` is what
  triggers `@register_reader` decorators to run, so do not skip that
  import when wiring up new formats.
- `bci/decoder/` — registry in `bci/decoder/__init__.py:_registry` resolves
  method names to lazy factories (`lda`, `ssvep`, `fbcca`, `cnn`,
  `transformer`, `transformer_bert`, `csp`). `list_methods()` is the single
  source of truth used by the GUI's `DecodePage` dropdown — never hardcode
  the list there. The `decode()` function in that file is the canonical
  entry point for cross-validated decoding.
- `bci/gui/worker.py` — `BaseWorker(QObject)` owns its `QThread` lifecycle
  via `start_in_thread()` / `stop()` / `cleanup()`. **Do not call
  `thread.quit()` / `thread.wait()` from outside the worker** — the
  surrounding tabs (`BatchTab`, `StreamTab`) all delegate teardown to
  `BaseWorker.cleanup()`. This pattern is intentional and load-bearing;
  see commit `d3b1d64` and the 2026-06-12 refactor spec for why.

## Naming conventions (post-refactor)

These were unified in the 2026-06-12 cleanup — do not regress to the old names:

- `filepath` (not `source_path`).
- `n_samples` for the time dimension in decoder code (not `n_times` —
  `mne.io.Raw` keeps `n_times`; decoders use `n_samples`).
- `refresh_chart(data=...)` uniform parameter across widget pages.

## GUI

- Built on PyQt6. The GUI explicitly refuses to start without a display:
  `bci/main.py:91-94` checks `DISPLAY` and exits with an X11 hint. To run
  headlessly, set up `xvfb-run` or X-forwarding (`ssh -Y`).
- `bci/gui/widgets/` contains the per-page widgets used by `BatchTab` and
  `StreamTab` (decode, epoch, info, main, preprocess, result, spectrum,
  step_strip, topomap, waveform).

## Testing quirks

- `bci/tests/test_session.py:182` uses `@pytest.mark.realdata` but that
  mark is **not registered** in `pyproject.toml` — pytest emits a
  `PytestUnknownMarkWarning`. The test is a no-op skip when
  `/data/bci/S001R04.edf` is absent, which is the normal case. Don't try
  to "fix" the warning by registering the mark unless you're prepared to
  add it to a `pytest` config block.
- 8 tests are skipped with explicit reasons like
  `"LoadWorker moveToThread causes Abort in CI"` and
  `"GUI widget access flaky in CI"`. These are
  `@pytest.mark.skip(reason=...)` (not `xfail`) — leave them skipped; the
  design spec marks investigation as deferred.
- Test discovery: pytest's default `test_*.py` glob picks up files under
  `bci/tests/` because that's where they live. Do not move them.

## Lessons learned from the 06-11 / 06-12 refactors

These are the engineering principles the codebase has converged on. The
"Don'ts" below are concrete instances; the principles here are the "why".
Read the spec (`docs/superpowers/specs/`) before doing anything bigger than
a one-file change — they encode the rationale.

### Refactor discipline

- **Pure refactor = zero behavior change.** Every existing non-skipped test
  must keep passing. If a test was `@skip`'d for a real reason, leave it
  skipped — the design spec marks investigation as deferred.
- **Plan structure that pays off:** spec = Goals + Non-Goals + Architecture
  + "Why" for each decision; plan = checkbox tasks with explicit
  `Run: <cmd> — Expected: <output>` verification and a commit per task.
  Copy this template for any multi-file refactor.
- **One concern per commit.** `n_times → n_samples` was a separate commit
  from `source_path → filepath`, which was separate from `eeg → source`.
  Reviewers can bisect, and `git revert` actually works.

### Plan for the whole system, not the call site

The 06-11 `EEGData → mne.io.Raw` refactor shipped 13 commits in one day,
all green. The next day's "cleanup" phase (15 more commits) existed
*only* to fix inconsistencies those 13 left: type annotations loosened
to `Optional[object]`, `source_path` renamed three different ways, the
decoder list re-hardcoded in the GUI, thread teardown duplicated,
`__import__('re')` and empty `TYPE_CHECKING: pass` left in place. Root
cause: each 06-11 commit optimized locally ("this file compiles, this
test passes") without asking what the system-wide design should look
like. This section is the meta-lesson that the bullets above are
tactical consequences of.

- **A focused plan is not a whole-system plan.** "Replace `EEGData` with
  `Raw` in the source layer" is a step, not a goal. The goal is "one
  coherent `Raw`-based data model end-to-end" — and that includes
  annotations at every consumer, naming for any new attribute, registry
  usage in every dropdown, and the protocol/union at every boundary.
  Write the whole-system goal first; the focused steps are how to get
  there.
- **Before changing a name or type, scan the whole repo.** `git grep`
  the old name (including comments, docstrings, README, specs, tests)
  and list every site. Every site that survives the rename is a design
  decision, not an oversight. Don't ship "one file at a time" and
  promise yourself a cleanup pass — the cleanup pass is what the spec
  was supposed to prevent.
- **"Tests pass" is necessary, not sufficient.** Tests don't enforce
  design consistency. A refactor that only checks tests is local
  optimization with extra steps. The spec's "Why" sections are the
  consistency checklist — re-read them before each commit and ask
  whether the change honors the stated principle, not just the
  signature.

### Type honesty

- **No `Optional[object]` placeholders.** Type the actual runtime type
  (`'mne.io.Raw'`, `StreamSource`, `'BCIPipeline'`) or define a Protocol.
  `object` is a code smell that hides the truth and breaks IDE help.
- **String forward refs in GUI code** (`Optional['mne.io.Raw']`), not
  `import mne` at module top. PyQt6 launches on the main thread; forcing a
  multi-second mne import there kills UX. The `from __future__ import
  annotations` + `'...'` quoting pattern is load-bearing.
- **Protocol over Union at module boundaries.** `EEGSource` (mne.io.Raw +
  StreamSource) names the concept and lets consumers accept both without
  repeating `Union[...]` everywhere.
- **`@runtime_checkable` is risky** when a Protocol declares a property
  but an implementer has a plain attribute (or vice versa) — `isinstance`
  silently fails. Drop it (default to typing-only) unless you control both
  sides. See commit `c3b0dd2`.

### Single source of truth

- **Registries return lists, consumers call them.** `list_methods()` is the
  GUI's source of truth for the decoder dropdown. The 06-12 refactor fixed
  exactly this kind of drift — don't reintroduce hardcoded lists.
- **Lazy factories in registries** (`_lazy('bci.decoder.deep', 'CNNDecoder')`)
  defer torch/sklearn/mne import until the method is selected. This is
  why `import bci.decoder` is cheap; preserve it when adding decoders.
- **`sorted(_registry.keys())`** for stable UI ordering — unsorted dict
  iteration breaks deterministic dropdowns across runs.

### Threading / Qt

- **The object that creates a QThread owns its cleanup.** `BaseWorker`
  exposes `stop()` (no-op default, subclasses override) and `cleanup()`
  (idempotent, safe pre-start). Consumer tabs call `w.cleanup()` and
  nothing else — no `thread.quit()` / `thread.wait()` from outside.
- **Connect `finished → _thread.wait` inside `start_in_thread`**, not in
  caller code. This drains the thread before the next `start_in_thread`
  call, preventing `cleanup()` from racing a mid-emit `run()`. The fix
  in commit `d3b1d64` exists because a previous version of this pattern
  blocked the GUI thread on `wait()` from the caller.
- **Properties that satisfy a Protocol** must be `@property`, not plain
  attributes, if the Protocol declares them as properties. `StreamSource`
  exposes `filepath` via a property reading `self._filepath` for exactly
  this reason.

### Dead code & drift

- **Sweep leftovers in dedicated commits** — empty `TYPE_CHECKING: pass`,
  unreachable `if self.raw is None` after a required `__init__` arg,
  `__import__('re')` when `re` is already at top. One concern per commit
  is fine; bundling them with a feature muddies `git blame`.
- **README diagrams that describe code are code.** `BatchWorker (QObject)`
  vs `(QThread)` is a load-bearing claim; update it in the same refactor
  that changes the design, don't leave it lying.
- **Don't try to fix warnings you didn't introduce.** The unregistered
  `@pytest.mark.realdata` mark is a no-op skip in the normal case;
  registering it is out of scope unless a `pytest` config block is being
  added for other reasons.

## Things to avoid

- Don't hardcode the decoder method list in the GUI — use
  `bci.decoder.list_methods()`.
- Don't import MNE/PyTorch/sklearn at module top in decoders — the
  registry in `bci/decoder/__init__.py` uses lazy factories
  (`_lazy(module, cls)`) specifically to defer heavy imports.
- Don't add a `tests/` directory at the repo root — pytest already finds
  `bci/tests/`.
- Don't rename `filepath` back to `source_path` or `n_samples` back to
  `n_times` in decoder code; the refactor just landed and the README/spec
  reflect the new names.

## Skills

The following generalized patterns are codified as self-contained skills
in `.agents/skills/`. Each one is project-agnostic and reusable elsewhere;
this section just signals which ones apply to work in this repo. Use the
`skill` tool by name when its triggering conditions match.

- **`whole-system-refactor-planning`** — Load when planning or executing
  a refactor that touches multiple files, modules, or shared concepts.
  Codifies the meta-lesson behind the 06-11/06-12 refactors (local-only
  commits create global debt).

- **`type-honesty`** — Load when writing or tightening type annotations
  for cross-module boundaries, optional/union types, or shared
  interfaces. Covers `Optional[object]` placeholders, Protocol vs Union,
  deferred import strategies, and `@runtime_checkable` risks.
