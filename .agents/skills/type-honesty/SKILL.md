---
name: type-honesty
description: Use when writing or tightening type annotations, or resolving type-check warnings. Triggers include pyright/pylance errors, variance mismatch, Optional passed where non-Optional required, introducing a Protocol, or designing cross-module interfaces. Skip for trivial in-file hints.
---

# Type Honesty

## Overview

Type annotations must express the **actual runtime type**. **Core principle:** name the real type. If multiple types share a shape, define a Protocol. Never reach for `object` / `Any` / `dict` as a placeholder.

## Audit Method

**Run pyright/pylance first, then fix every warning.** Grep-based audits miss call-site mismatches (e.g. `Optional[Raw]` passed where `Raw` required) that only flow analysis catches. Fix each error by root cause (Rule 7). After all errors resolve, scan for style issues (bare `: list`, unused imports).

```bash
pip install pyright && pyright <package>/
```

## Core Rules

1. **No `Optional[object]` / `Any` / `dict` placeholders.** Name the real type or define a Protocol.

2. **Protocol over Union at boundaries.** When 2+ types flow through the same consumer, define a Protocol. Survives adding a third type.

3. **Default: import directly.** Reserve deferred imports for:

   - **TYPE_CHECKING block** — library code where checker needs real names.
   - **Function-body import** — heavy library, few functions, startup-sensitive.

   Optional features: `try/except ImportError`. Registries: lazy factory.

4. **Drop `@runtime_checkable` unless you control both sides.** Property/attribute mismatches make `isinstance` silently return `False`.

5. **Property in Protocol = property in implementer.**

6. **Local unpacks follow the library's documented shape.** Don't rename locals to match internal naming.

7. **Resolve type-check warnings, don't suppress them.** Fix the root cause:

   - **Variance mismatch** → use the correct generic. `List` is invariant; `Sequence` is covariant. Same for `Mapping` vs `dict`, `Iterable` vs `list`.
   - **Library stub too broad** → `cast(TargetType, expr)`. Zero runtime cost.
   - **Optional passed where non-Optional required** → add None guard before call site; flow analysis narrows the type.
   - **`# type: ignore` is a last resort.** Must specify error code (`[attr-defined]`, `[abstract]`, etc.) AND a `# why` comment. Bare `# type: ignore` without both is unacceptable.

## Quick Reference

| Pattern | Use when |
|---|---|
| `pyright` audit | First step. Catches what grep misses. |
| Direct import | Default. Most code. |
| Function-body import | Heavy library, few functions, startup-sensitive. |
| `TYPE_CHECKING` block | Library code, checker needs real names. |
| `try/except` import | Optional dependency. |
| Lazy factory | Registry holds heavy types. |

## Red Flags — STOP and Fix

- `Optional[object]` / `Optional[Any]` / bare `: dict` / `: list` in a public signature
- `# type: ignore` without error code AND reason comment
- Type error "fixed" by widening to `Any` / `object`
- `List` param where `Sequence` would work
- `Optional[T]` passed to `T` param without None guard
- `@runtime_checkable` on a Protocol with external implementers
- Protocol property vs plain attribute mismatch
- Heavy library at module top of startup-critical path
- Union in 3+ places that could be a Protocol
