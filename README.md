# mapc-mab
MAPC (c-SR) for IEEE 802.11 networks using MAB

## Development

To develop and modify mapc-mab, you need to install
[`hatch`]([https://hatch.pypa.io](https://hatch.pypa.io)), a tool for Python packaging and
dependency management.

To  enter a virtual environment for testing or debugging, you can run:

```bash
hatch shell
```
This will create the default environment and install dependencies.
To use a GPU version of jaxlib one must create `plgvenv` virtual environment - 
**Note** this is configured to be used in pl-grid infrastructure.
The command can create this environment

```
hatch env create plgvenv
hatch -e plgvenv run pip install -e . 
```
Here we use pip install to firstly install env versions of the libraries and later the application dependencies.

### PyCharm

Currently, pycharm does not support hatch, but the default environment has a path set to `./venv` so you can add it manually and use all the tools.
Similarly, this works in vscode.

