# Convenience targets. Run `make help` for the list.
.DEFAULT_GOAL := help
PYTHON ?= python3

.PHONY: help install install-dev download train train-smoke evaluate test lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## Install runtime + dev dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

download:  ## Download and verify the dataset
	$(PYTHON) scripts/download_dataset.py

train:  ## Train with the default config
	$(PYTHON) train.py

train-smoke:  ## Fast CPU smoke run (small subset, few epochs)
	$(PYTHON) train.py --config configs/smoke.yaml

evaluate:  ## Evaluate a checkpoint (pass CKPT=path/to/best.pt)
	$(PYTHON) evaluate.py --checkpoint $(CKPT)

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Lint the codebase
	$(PYTHON) -m ruff check .

clean:  ## Remove caches and build artefacts (keeps datasets/ and outputs/)
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
