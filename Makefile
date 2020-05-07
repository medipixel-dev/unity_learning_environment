dev:
	pip install -U -r requirements.txt
	pip install -U git+https://github.com/medipixel/rl_algorithms.git
	pre-commit install

format:
	black . --exclude checkpoint
	isort -y --skip checkpoint

test:
	black . --check
	isort -y --check-only --skip checkpoint
	env PYTHONPATH=. pytest --pylint --flake8 --ignore=checkpoint --cov=tests
