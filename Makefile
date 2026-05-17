.PHONY: help install lint format type test cov pre-commit \
        train-image train-text train-audio train-joint \
        calibrate evaluate bench export demo clean

PY ?= python
PIP ?= $(PY) -m pip

help:
	@echo "Targets:"
	@echo "  install        - install package in editable mode with dev extras"
	@echo "  lint           - run ruff check"
	@echo "  format         - run ruff format"
	@echo "  type           - run mypy on src"
	@echo "  test           - run pytest"
	@echo "  cov            - run pytest with coverage"
	@echo "  pre-commit     - run all pre-commit hooks"
	@echo "  train-image    - memo train image"
	@echo "  train-text     - memo train text"
	@echo "  train-audio    - memo train audio"
	@echo "  train-joint    - memo train joint"
	@echo "  calibrate      - memo calibrate"
	@echo "  evaluate       - memo evaluate"
	@echo "  bench          - memo benchmark"
	@echo "  export         - memo export"
	@echo "  demo           - memo demo (requires .[demo])"
	@echo "  clean          - remove build/, dist/, __pycache__/, .pytest_cache/"

install:
	$(PIP) install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff format src tests

type:
	mypy src

test:
	pytest -q

cov:
	pytest -q --cov=src/memo --cov-report=term-missing

pre-commit:
	pre-commit run --all-files

train-image:
	memo train image $(ARGS)

train-text:
	memo train text $(ARGS)

train-audio:
	memo train audio $(ARGS)

train-joint:
	memo train joint $(ARGS)

calibrate:
	memo calibrate $(ARGS)

evaluate:
	memo evaluate $(ARGS)

bench:
	memo benchmark $(ARGS)

export:
	memo export $(ARGS)

demo:
	memo demo $(ARGS)

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
