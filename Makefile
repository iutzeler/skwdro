ENV=dev
TEST_ENV=dev

reset_env:
	hatch env prune

shell:
	hatch -v -e $(ENV) shell

test: test_gen test_sk test_misc

test_gen:
	@echo "General tests:"
	@hatch -e $(TEST_ENV) run test:test-custom

test_sk:
	@echo "Sklearn tests:"
	@hatch -e $(TEST_ENV) run test:test-sklearn

test_misc:
	@echo "Solo tests:"
	@hatch -e $(TEST_ENV) run test:test-misc
