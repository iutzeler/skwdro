ENV=env-dev
TEST_ENV=env-dev

reset_env:
	@hatch env prune

shell:
	@hatch -v -e $(ENV) shell

styletest:
	@echo "Style test:"
	@hatch -e $(TEST_ENV) run env-test:ruff-test
	@hatch -e $(TEST_ENV) run env-test:pycodestyle-test
	@hatch -e $(TEST_ENV) run env-test:mypy-test

doctest:
	@echo "Style test:"
	@hatch -e $(TEST_ENV) run env-docs:doc-test

test: test_sk test_misc test_gen

lfstests:
	@pytest -v ./tests/torch_tests/test_regularized_linear_exact.py -W ignore::FutureWarning

test_gen:
	@echo "General tests:"
	@git lfs pull
	@hatch -e $(TEST_ENV) run env-test:test-custom

test_sk:
	@echo "Sklearn tests:"
	@hatch -e $(TEST_ENV) run env-test:test-sklearn

test_misc:
	@echo "Solo tests:"
	@hatch -e $(TEST_ENV) run env-test:test-misc

coverage:
	@echo "Converage computation"
	coverage run -m pytest
	coverage report


epsilon_plotting_in_source:
	mv skwdro/solvers/entropic_dual_torch.py skwdro/solvers/entropic_dual_torch.cp.py
	mv skwdro/solvers/__entropic_dual_torch_epsilon.py skwdro/solvers/entropic_dual_torch.py

remove_epsilon_plotting_in_source:
	mv skwdro/solvers/entropic_dual_torch.py skwdro/solvers/__entropic_dual_torch_epsilon.py
	mv skwdro/solvers/entropic_dual_torch.cp.py skwdro/solvers/entropic_dual_torch.py
