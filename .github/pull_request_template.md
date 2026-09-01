## What changed


## Why


---

Nothing below is mandatory. They are the conventions this codebase already
follows, listed because they are easier to apply while writing than to retrofit
in review.

- [ ] One logical change. Splitting is cheaper than untangling later.
- [ ] Any test that pins a bug says what the bug was, not just what the
      assertion is. Most tests here exist because something specific went wrong.
- [ ] Any number that gates behaviour has a measurement behind it, and something
      that reproduces the measurement. A threshold nobody can argue with is a
      guess wearing a constant's clothes.
- [ ] Anything deciding whether to write to memory, or to state something to a
      contributor, fails towards doing nothing.
- [ ] `ruff check .` and `ruff format .` are clean, and `pytest -q` passes.
