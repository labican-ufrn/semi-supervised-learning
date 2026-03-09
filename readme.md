# MLabican

This is the Machine Learning library used at Labican.

## Develop

We use Python 3.12 to develop this project and may not work at previous versions
of python. We also use a Python Virtual Environments, so you can use your
favourite tool (e.g. pyenv, poetry, venv, others), but we explain using the venv.

We choose `venv` because it is easy for beginners and it is a built-in package.

```sh
python -m venv .venv
## Linux
source .venv/bin/activate
```

```sh
pip install -r requirements-dev.txt
```

After you run this command, you need to finish the configurations activating the
`pre-commit` hooks. This tool prevent bad/break commits in the repository.

```sh
pre-commit install --install-hooks
```

Finish, when you try to commit now the pre-commit hooks will analyse your files
and will fix some minor problems to improve and standardize the code based on the
patters defined in the `.pyproject` file.
