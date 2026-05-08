---
description: Run a focus-gate inspector subcommand and show the raw output
---

<!--
Resolves the binary via `$PATH`. Install with `go install
github.com/kuandriy/focus-gate/cmd/focus@latest` (or a `go build`
followed by moving the resulting `focus-gate` binary onto your `$PATH`).

If you need to point at a non-installed dev build, either symlink it
into a `$PATH` directory or override here with the absolute path,
e.g. `!`/abs/path/to/focus-gate --cmd $ARGUMENTS``.
-->

!`focus-gate --cmd $ARGUMENTS`
