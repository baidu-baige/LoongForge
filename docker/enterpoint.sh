#!/bin/bash

set -x

# start sshd
# service ssh restart

# jupyterlab
export JUPYTER_TOKEN=${JUPYTER_TOKEN:-$(pwgen -B -s 32 1)}
export JUPYTERLAB_BASE_URL=${JUPYTERLAB_BASE_URL:-"/jupyter"}
export JUPYTER_PORT=${1:-8600}
export JUPYTER_NOTEBOOK_DIR=${JUPYTER_NOTEBOOK_DIR:-"/workspace"}

JUPYTER_DIR="/root/.jupyter"
mkdir -p "$JUPYTER_DIR"
CONFIG_FILE="$JUPYTER_DIR/jupyter_notebook_config.py"

cat <<EOL > $CONFIG_FILE
import os
c = get_config()
c.NotebookApp.allow_remote_access = True
c.NotebookApp.allow_root = True
c.NotebookApp.terminado_settings = {'shell_command' : ['/bin/bash']}
c.LabApp.base_url = os.environ.get("JUPYTERLAB_BASE_URL", "/jupyter")
c.LabApp.ip = '0.0.0.0'
c.LabApp.port = os.environ.get("JUPYTER_PORT", "8600")
c.LabApp.terminado_settings = {'shell_command' : ['/bin/bash']}
c.NotebookApp.notebook_dir = os.environ.get("JUPYTER_NOTEBOOK_DIR", "/workspace")
EOL

jupyter-lab --no-browser --NotebookApp.token="${JUPYTER_TOKEN}" --port=${JUPYTER_PORT} --ip=0.0.0.0 --NotebookApp.base_url="${JUPYTERLAB_BASE_URL}" --notebook-dir="${JUPYTER_NOTEBOOK_DIR}" --allow-root