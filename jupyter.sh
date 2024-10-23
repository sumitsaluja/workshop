#!/bin/bash
export PORT=8888
echo "set up SSH port  forwarding between the compute resource and your local computer (laptop/desktop)"
echo "ssh -N -L $PORT:[HOST]:$PORT [userID]@hpc.c2b2.columbia.edu" 
echo "Access the Jupyter Notebook via your web browser on your local computer." 
echo "http://127.0.0.1:$PORT/" 


module load conda/3
echo " "
echo "============================================================================="
echo "=== jupyterlab.sh Install and run JupyterLab locally" 
echo "+++ installing jupyter" 
python -m venv .jup
. .jup/bin/activate 
python -m pip install pip --upgrade 
python -m pip install --upgrade jupyterlab 
python -m pip install --upgrade bash_kernel 
python -m bash_kernel.install 
python -m pip install --upgrade jupyterlab-spellchecker 

echo "+++ run jupyter" 


jupyter-lab --ip=0.0.0.0 --port=$PORT --no-browser
