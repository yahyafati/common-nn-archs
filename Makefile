PYTHON := python3
SCRIPT := plugins/toml_to_dataclass.py
INPUT := config.toml
OUTPUT := configs/types.py

.PHONY: generate check-clean

generate: check-clean
	@echo "Generating dataclass schema..."
	$(PYTHON) $(SCRIPT) $(INPUT) $(OUTPUT)
	@echo "Done."

check-clean:
	@echo "Checking git working tree..."
	@git diff --quiet $(OUTPUT) || (echo "ERROR: $(OUTPUT) has uncommitted changes. Commit or stash first." && exit 1)