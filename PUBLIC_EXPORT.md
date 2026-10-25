# Public export and coverage

- Main contains runnable operator/profile code and the offline Lab. Doc adds manuscripts, research, resume, original operator events, profile, source snapshots and exported conversation.
- No credentials, signed log URLs, system/developer instructions or hidden reasoning are published.
- Organization-specific `platform.py` files are omitted from public source snapshots. Their captured hashes remain provenance; core operator/profile files are preserved. Original snapshots were verified against worker hashes before public export.
- Website HTML dumps and long verbatim research extracts are not republished. Research findings, source URLs, dates, configuration facts and original-byte hashes are retained.
- `conversations/COVERAGE.json` records source/export counts and the cutoff. All available public message text is retained, plus tool-call indexes. Cloud tool records contain no result bodies; this is not a complete stdout archive. Operator events and profile files separately preserve engineering evidence.
- Current profile analysis selects CPU user_annotation phase labels; original summaries and the sampled source are preserved to explain the projection change.

- 2026-10-25 training update: both branches contain the same sampled public backend/probe code, pinned public inputs, compact terminal evidence, migration plan, training report and weekly journal. Doc additionally preserves exact per-attempt public source files in `docs/artifacts/training-snapshots/`; the manifest checks every included file against worker-captured SHA256.
- Training worker/launcher, internal asset mounts, raw platform responses, receipts and signed log links are omitted. Sampled worker hashes and original bundle hashes remain provenance. Compact JSON events contain numerical results and task completion, not the raw platform stdout/stderr.
- Published training reports use an explicit torch adapter as the numerical reference. Native default/FLA gradient failures, earlier failed attempts, shared-cache first-execution costs and historical doubling instability are preserved. No mainstream adoption or FLA performance win is claimed.
