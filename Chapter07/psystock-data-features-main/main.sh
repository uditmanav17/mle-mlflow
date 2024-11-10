

# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
export MLFLOW_TRACKING_URI=http://localhost:5000
echo $MLFLOW_TRACKING_URI

# activate environment containind dependicies of main.py
echo $(which python)

python ./main.py







