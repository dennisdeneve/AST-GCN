1) First create a new virtual environment

python -m venv env_name

-> Replace env_name with the name you want for the virtual enviornment


2) Activate that virtual environment

source myenv/bin/activate

3) Install the dependencies listed in the 'requirements.txt' file

pip install -r requirements.txt

4) Run the main.py program with the initial command 

## old 
python3 main.py --model_name ast-gcn


## new
python3 main.py --mode config.yaml

-> --model_name flag can be set amongst other options specified in the main.py program, model
can be either tgcn or ast-gcn


5) Lastly deactivate the virtual environment

deactivate

