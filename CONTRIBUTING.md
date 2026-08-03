# Contributing

Thanks for your interest in improving this project! Contributions of all kinds
are welcome — bug reports, features, docs, and tests.

## Development setup

```bash
git clone https://github.com/NayeonKim925/wm-wafer-map.git
cd wm-wafer-map
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

The test suite runs on a small **synthetic** dataset with `kagglehub` mocked, so
you do not need the real 2 GB download to develop or run CI locally.

## Before you open a PR

```bash
ruff check .      # lint (also runs in CI)
pytest            # full suite (also runs in CI)
```

Please make sure both pass and add tests for any new behaviour.

## Conventions

- **Logging, not `print`.** Library code (`src/**`) uses the `logging` module;
  `print` is reserved for user-facing CLI output in the entry scripts.
- **No hardcoded paths.** Everything is configured in `configs/` and resolved
  with `pathlib`; add new options to the dataclasses in `src/config/config.py`.
- **Extending the project is cheap by design:**
  - *New model:* add a builder in `src/models/` and decorate it with
    `@register_model("my_model")`; select it via `--set model.name=my_model`.
  - *New data representation:* extend `src/data/preprocessing.py`.
  - *New metric or plot:* add it in `src/evaluation/`.
- **Never commit dataset files or training artefacts** (`datasets/`, `outputs/`
  and common data/model extensions are git-ignored).

## Commit / PR style

- Keep PRs focused; describe the motivation and the change.
- Reference any related issue.
- The PR template checklist should be green before requesting review.

## Code of conduct

Be respectful and constructive. By participating you agree to keep the project
a welcoming space for everyone.
