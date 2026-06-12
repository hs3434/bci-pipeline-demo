---
name: type-honesty
description: Use when writing or tightening type annotations for cross-module boundaries, optional or union types, or shared interfaces. Triggers include introducing a Protocol, choosing between Protocol or Union, tightening an Optional[object] placeholder, or designing an interface that crosses a module boundary. Skip for trivial in-file hints with no cross-module impact.
---

# Type Honesty

## Overview

Type annotations must express the **actual runtime type** — the truth the reader sees, not the abstraction you wish you had. `Optional[object]` is a lie that hides the truth, breaks IDE help, and silently allows wrong types through type checkers.

**Core principle:** name the real type. If multiple types share a shape, define a Protocol that names the *concept*. Never reach for `object` / `Any` / `dict` as a placeholder.

## When to Use

- Writing or tightening a public function/method/attribute signature
- Designing an interface that crosses a module boundary
- Choosing between Protocol, Union, or import strategy
- Tightening an `Optional[object]` / `Any` / `dict` placeholder
- Reviewing a Protocol that uses `@runtime_checkable`

**Do not use** for in-file hints with no cross-module impact.

## Pre-Annotation Gates

Before writing or accepting an annotation:

1. **Does it name the actual runtime type?** If you wrote `Optional[object]` because "the type might be many things", define a Protocol (rule 2).
2. **Is the import cost acceptable at the import site?** Heavy libraries at the top of a startup-critical module can block startup. Either function-body import (rule 3 case 2) or, for library code, defer via `TYPE_CHECKING` (rule 3 case 1).

## Core Rules

1. **No `Optional[object]` / `Any` / `dict` placeholders.** Name the real type or define a Protocol. `object` is a code smell.

2. **Protocol over Union at boundaries.** When two or more types flow through the same consumer, define a Protocol that names the concept. Survives adding a third type; reads as a noun, not a disjunction.

3. **Default: import directly. Reserve deferred imports for narrow cases.**

   ```python
   # Case 1: library code — TYPE_CHECKING block so external checker sees real names
   from __future__ import annotations
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from heavy_lib import DomainType
   def load(x: DomainType) -> DomainType: ...

   # Case 2: startup-critical, type used in few functions — function-body import
   def render(self):
       from heavy_lib import Renderer
       return Renderer(self.data)
   ```

   For optional features, use `try/except ImportError` so the app runs without the dependency. For registries holding types from heavy libraries, use a lazy factory.

   **Footnote: circular imports.** If two modules need each other's types and a literal import would fail at load time, restructure first (extract shared types to a third module, or merge). If restructuring isn't viable, use `from __future__ import annotations` + string forward refs. This is the only legitimate string-forward-ref use case in application code.

4. **Drop `@runtime_checkable` unless you control both sides.** A Protocol property satisfied by a plain attribute (or vice versa) makes `isinstance(x, Protocol)` silently return `False`. Default to typing-only.

5. **Property in Protocol = property in implementer.** Mismatches break both type checkers and `isinstance`. If you declare `@property`, the implementer must too — even if it just sets `self._foo` in `__init__` and exposes it via `@property`.

6. **Local unpacks follow the library's documented shape.** `(batch, channels, samples) = X.shape` — use the names the library documents. Don't rename locals to align with internal naming; that's the wrong layer.

## Quick Reference

| Pattern | Use when |
|---|---|
| Direct import | Default. Most application code. |
| Function-body lazy import | Heavy library, used in few functions, startup-sensitive. |
| `from __future__` + `TYPE_CHECKING` block | Library code, external type checker needs real names. |
| `try/except` import | Optional features (app runs without the dependency). |
| Lazy factory | Registry holds types from heavy libraries. |

## Red Flags — STOP and Rename

- `Optional[object]` / `Optional[Any]` / `: dict` / `: list` in a public signature
- `@runtime_checkable` on a Protocol whose implementers are external
- Protocol property whose implementer is a plain attribute (or vice versa)
- Heavy library imported at module top of a startup-critical code path
- Union used in 3+ places that could be a Protocol
- `isinstance(x, MyProtocol)` on a Protocol you didn't author or didn't verify
