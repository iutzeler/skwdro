# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

SPHINXAPI     ?= sphinx-apidoc
APIDIR        = API
EXDIR         = examples
GENDIR        = generated
PROJDIR       = ../skwdro

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	@rm -rf "$(APIDIR)"/*
	@echo "Removing everything under $(APIDIR)"
	@rm -rf "$(EXDIR)"/*
	@echo "Removing everything under $(EXDIR)"
	@rm -rf "$(GENDIR)"/*
	@echo "Removing everything under $(GENDIR)"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
ifeq (,$(wildcard API/skwdro.distributions.rst))
	@$(SPHINXAPI) -o "$(APIDIR)" "$(PROJDIR)" 
	@echo "   :ignore-module-all:" >> API/skwdro.distributions.rst
else 
	@$(SPHINXAPI) -o "$(APIDIR)" "$(PROJDIR)" 
endif
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
