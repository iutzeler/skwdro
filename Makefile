ENV=dev
TEST_ENV=dev

reset_env:
	hatch env prune

shell:
	hatch -v -e $(ENV) shell

test:
	hatch -e $(TEST_ENV) run test