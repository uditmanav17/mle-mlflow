# Machine Learning Engineering with MLFlow

---

## Technical Requirements

### On Windows
- [Docker](https://docs.docker.com/desktop/install/windows-install/)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)

### On Linux
- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- [Pyenv](https://github.com/pyenv/pyenv)(Optional)

**Note** - Make sure to select `Add conda to PATH` while installing conda.

---

## Start ML FLow server
- Navigate to `mlf_dev_env`, and execute `docker compose up` in terminal
- Above command will spin up ML Flow service with its dependencies

---

## Setting up Conda Environment
- After installing Conda, use `conda_env` to create environment as follows -
```conda env create --name envname --file=conda_env.yaml```

- Refer [Conda cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) for more commands
