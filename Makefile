.PHONY: default clean install install_dev uninstall tests

#################################################################################
# GLOBALS                                                                       #
#################################################################################

#PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Default
default:
	@echo "Local options:"
	@echo "    make clean            # Clean unnecessary files."
	@echo "    make install          # Install library and dependencies."
	@echo "    make install_dev      # Install library and dependencies for development."
	@echo "    make uninstall        # Unistall library."
	@echo "    make tests            # Unitest launching."

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;
	find . -type d -name __pycache__ -exec rm -fr {} \;

## Install package
install:
	pip install -e .

## Install package with extra dependencies for development
install_dev:
	pip install -e .[dev]

## Uninstall packager
uninstall:
	pip uninstall DAT

## Unitests
tests:
	pytest tests