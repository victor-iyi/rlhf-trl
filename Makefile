.PHONY: venv setup setup-dev install install-dev tidy compile clean

all: setup

venv:
	pyenv virtualenv .venv || true
	pyenv activate .venv

install:
	# pip install --upgrade poetry
	# poetry install -E all
	pip install -r requirements.txt
	# Hacky way to make sure HG and WandB cli exists.
	# pip install --upgrade huggingface_hub	wandb
	huggingface-cli login		# login to huggingface
	wandb login							# login to wandb

install-dev:
	# pip install --upgrade poetry
	# poetry install --with dev -E all
	pip install -r requirements-dev.txt
	# Hacky way to make sure HG and WandB cli exists.
	# pip install --upgrade huggingface_hub	wandb
	huggingface-cli login		# login to huggingface
	wandb login							# login to wandb

setup: venv install

setup-dev: venv install-dev

tidy:
	pre-commit run --all-files

compile:
	poetry export -f requirements.txt --output requirements.txt --only main --without-hashes --without-urls
	poetry export -f requirements.txt --output requirements-dev.txt --with dev --without-hashes --without-urls

clean:
	pyenv deactivate
