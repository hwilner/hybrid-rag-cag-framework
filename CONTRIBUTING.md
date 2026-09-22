# Contributing to Hybrid RAG-CAG Framework

Welcome — contributions are genuinely wanted, including (especially) from
people new to retrieval-augmented generation. This file explains how to pick
work, set up your environment, and follow the project's scientific-integrity
rules.

## Picking an issue

Open issues are written as **task cards**. Each card contains:

- **Size** — XS (typo/small fix, < 1 hour), S (an afternoon), M (a few
  days), L (a week+; talk to the maintainer before starting).
- **Dependencies** — other issues or modules that must land first. Don't
  start a card whose dependencies are open.
- **Acceptance criteria** — the checklist your PR will be reviewed against.
  If you think a criterion is wrong, discuss in the issue *before* coding.
- **Boundary** — what is explicitly *out of scope*. Stay inside it; scope
  creep is a common reason PRs get sent back.

Good first issues are labeled `good first issue` and are usually size XS–S.
Comment on the issue to claim it so two people don't do the same work.

## Development setup

```bash
git clone https://github.com/hwilner/hybrid-rag-cag-framework.git
cd hybrid-rag-cag-framework

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Run the test suite before and after your change:

```bash
pytest
```

To sanity-check the evaluation pipeline end to end (Tier 1 is the fastest):

```bash
python src/train_and_evaluate.py --evaluation_tier 1
```

## Branch and PR conventions

- Branch from `main` as `feature/<short-name>`, `fix/<short-name>`, or
  `docs/<short-name>`.
- One issue per PR. Reference the issue in the PR description
  (`Closes #NN`).
- Fill in the PR template completely, including **test evidence** (commands
  run and their output summary).
- Keep diffs focused. Drive-by refactors belong in their own PR.
- If your change touches the CAG contrastive-selection core in
  `src/hybrid_rag_cag_system.py`, say so explicitly and justify why — the
  core is otherwise treated as immutable (see `docs/ROADMAP.md`).

## Scientific-integrity rules

This project treats evaluation honesty as a hard requirement, not a nicety:

1. **Honest negative results.** If your change makes things worse or does
   nothing, report that in the PR. A well-documented negative result is a
   valuable contribution.
2. **No silent dropping of failures.** Never delete or skip failing test
   cases, evaluation questions, or runs to make numbers look better. If a
   failure is expected, mark it with a documented reason.
3. **Pre-registered metrics.** Before running a benchmark for a claimed
   improvement, state in the issue/PR which metric and which significance
   test you will use (defaults are in `docs/METHODS.md` → Undecided choices).
   Post-hoc metric shopping is not accepted.
4. **Simulated data stays labeled.** Tier-3 "expert" responses are simulated
   in code; never present them as a human study.
5. **Reproducibility.** Record dependency versions, seeds, and hardware for
   any result you report.

## Documentation

Behavior changes must update the relevant doc (`README.md` pointer only if
asked; otherwise the appropriate file under `docs/`). New modules get a
docstring explaining purpose, inputs, outputs, and the paradigm paper they
implement (citations from the numbered list in `docs/INTRODUCTION.md`).

## Questions

Open an issue with the `question` label. There are no stupid questions here —
the extended introduction (`docs/EXTENDED_INTRODUCTION.md`) was written for
exactly this reason.
