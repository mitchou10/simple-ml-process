run-test:
	echo "Running tests..."
	PYTHONPATH=$(PWD) uv run pytest -s -v tests/

lint:
	echo "Linting code..."