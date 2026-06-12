---
name: whole-system-refactor-planning
description: Use when planning or executing a refactor that touches multiple files, modules, or shared concepts (types, registries, interfaces). Triggers include renaming a type or attribute used across the codebase, replacing a domain object, changing a registry with multiple consumers, or being asked for a "cleanup" or "consistency" pass (signals a prior incomplete refactor). Skip for single-file changes, bug fixes (use systematic-debugging), or new features (use brainstorming).
---

# Whole-System Refactor Planning

## Overview

A refactor done piecemeal — one file at a time, each commit passing tests — still leaves the system inconsistent. Tests don't enforce design consistency; the inconsistency only shows at the system level. The result: a future "cleanup" or "consistency" pass becomes necessary to fix what the original refactor left behind.

**Core principle:** write the whole-system goal first, then the focused steps. The focused step is *how*; the whole-system goal is *what*. Without the goal, every commit optimizes locally.

## When to Use

- Task is "replace X with Y" or "rename Z" where Z appears in more than 2-3 files
- A registry or interface has multiple consumer modules
- The user mentions "consistency", "cleanup", "naming", or "types" — usually a red flag that an earlier refactor was incomplete

**Do not use** for single-file changes, bug fixes (use systematic-debugging), or new features (use brainstorming).

## Pre-Refactor Gates

Pass both gates before writing any code. Failing a gate means **stop and replan**, not "do it later".

1. **Whole-system goal written.** One paragraph: what does the system look like end-to-end after this refactor? (Annotations, naming, registry, protocol boundaries — not just the focused call site.)
2. **All sites of the old concept git-grep'd.** Across source, comments, docstrings, README, specs, plans, tests. Every site that survives the change is a design decision, not an oversight.

## Core Pattern

1. **Write the whole-system goal before the focused step.** "Replace `OldDomain` with `NewDomain` in the source layer" is a step. The goal is "one coherent `NewDomain`-based data model end-to-end" — including type annotations at every consumer, naming for any new attribute, registry usage in every dropdown, and the protocol or union at every boundary.

2. **Scan the whole repo before changing a name or type.** `git grep` for the old name across source, comments, docstrings, README, specs, plans, tests. List every site.

3. **Tests pass ≠ refactor done.** Tests verify behavior, not design consistency. Add a consistency checklist: same name in every consumer? same type at every boundary? registry vs hardcode? lifecycle ownership?

4. **One concept per commit. Update diagrams, specs, and READMEs in the same refactor.** A README architecture diagram that describes a design with a wrong claim (e.g. "synchronous" when the design is async, "blocking" when it's non-blocking) is load-bearing — leaving it lying defeats the refactor.

## Quick Reference

| Phase | Action |
|---|---|
| Before any commit | Write the whole-system goal in one paragraph |
| Before any rename | `git grep -n "<old_name>"` across the whole repo (incl. docs and tests) |
| Before each commit | Re-read the spec's "Why"; does this commit honor the goal? |
| After each commit | Run tests AND walk the consistency checklist (naming, types, registry, lifecycle) |

## Red Flags — STOP and Replan

- "I'll do it file by file and clean up later" — that's the cleanup pass you just created
- "This file builds, this test passes" — local optimization
- "I'll leave the docstring/spec update for a later pass" — drift is debt
- "The user only asked me to change X, I shouldn't touch Y" — Y is the design decision
- A second "cleanup" PR appears a day after a refactor — the refactor was incomplete
