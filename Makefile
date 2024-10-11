.PHONY: help  # Show help information about available targets
help:
	@echo "Usage: make [target]"
	@echo
	@echo "Targets:"
	@echo
	@sed -n 's/^.PHONY: \(.*\) # \(.*\)/  \1\t\2/p' $(MAKEFILE_LIST) | \
		column -t -s $$'\t'

.PHONY: install # Install project dependencies
install:
	poetry install

.PHONY: test # Run tests
test:
	poetry run pytest

.PHONY: lint # Run linter
lint:
	poetry run black --check src scripts tests
	poetry run flake8 src scripts tests

.PHONY: format # Format code
format:
	poetry run autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive src scripts tests
	poetry run black src scripts tests

.PHONY: list # List available scripts
list:
	@echo "Available scripts:"
	@echo
	@ls scripts | sed 's/\.py$$//'

.PHONY: run # Run a specific script (usage: make run SCRIPT=script_name)
run:
	@if [ -z "$(SCRIPT)" ]; then \
		echo "Usage: make run SCRIPT=script_name"; \
		echo "To see available scripts, use: make list"; \
	else \
		poetry run python -m dip.scripts.$(SCRIPT); \
	fi

.PHONY: clean # Clean up cache and build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf dist

.PHONY: build # Build the project
build:
	poetry build
