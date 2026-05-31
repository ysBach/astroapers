DOCS_DIR := docs/quarto
DOCS_PORT ?= 8000
DOCS_RUSTDOC_SITE := $(CURDIR)/$(DOCS_DIR)/_site/rustdoc
DOCS_ENV = MPLBACKEND=module://matplotlib_inline.backend_inline QUARTO_PYTHON="$$(uv run --extra docs python -c 'import sys; print(sys.executable)')"

.PHONY: docs-api docs-rust docs-rust-copy docs-tutorials docs-render docs-build docs-preview docs-serve docs-clean

docs-api:  ## Regenerate docstring/API reference qmd files only.
	cd $(DOCS_DIR) && uv run --extra docs python -m quartodoc build
	cd $(DOCS_DIR) && uv run --extra docs python _scripts/render_runtime_docstrings.py

docs-rust:  ## Build the Rust API reference with rustdoc.
	cargo doc --no-default-features --no-deps

docs-rust-copy:  ## Copy rustdoc output into the rendered Quarto site.
	test -d target/doc
	rm -rf "$(DOCS_RUSTDOC_SITE)"
	mkdir -p "$(DOCS_RUSTDOC_SITE)"
	cp -R target/doc/. "$(DOCS_RUSTDOC_SITE)"

docs-tutorials:  ## Render tutorial qmd notebooks only, leaving generated API sources alone.
	cd $(DOCS_DIR) && $(DOCS_ENV) quarto render --profile tutorials --no-clean

docs-render:  ## Render the whole Quarto website from existing qmd files.
	cd $(DOCS_DIR) && $(DOCS_ENV) quarto render

docs-build: docs-api docs-rust docs-render docs-rust-copy  ## Build complete Python, Rust, and Quarto docs.

docs-preview: docs-build docs-serve  ## Build the site, then serve docs/quarto/_site locally.

docs-serve:  ## Serve the existing rendered site without rebuilding.
	cd $(DOCS_DIR) && python -m http.server $(DOCS_PORT) -d _site

docs-clean:
	rm -rf docs/quarto/_site docs/quarto/.quarto docs/quarto/api docs/quarto/objects.json
	find docs -name .DS_Store -delete
