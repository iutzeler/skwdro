ENV=env-dev
TEST_ENV=env-dev

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
