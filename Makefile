.PHONY: update

update:
	@echo "Updating dependencies..."
	@for pkg in $$(uv pip freeze | grep -v '^-e'); do uv pip install -U $$pkg; done
