ENV=env-dev
TEST_ENV=env-dev

.PHONY: docs test shell reset_env clean

reset_env:
	@hatch env prune

shell:
	@hatch -v -e $(ENV) shell

test: test_gen test_sk test_misc

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

docs:
	@echo "Doc building (no warnings):"
	$(MAKE) -C doc html

clean:
	$(MAKE) -C doc clean

epsilon_plotting_in_source:
	mv skwdro/solvers/entropic_dual_torch.py skwdro/solvers/entropic_dual_torch.cp.py
	mv skwdro/solvers/entropic_dual_torch_epsilon.py skwdro/solvers/entropic_dual_torch.py

remove_epsilon_plotting_in_source:
	mv skwdro/solvers/entropic_dual_torch.py skwdro/solvers/entropic_dual_torch_epsilon.py
	mv skwdro/solvers/entropic_dual_torch.cp.py skwdro/solvers/entropic_dual_torch.py
