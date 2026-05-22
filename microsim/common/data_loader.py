import json
import re
import os.path
from microsim.regression_models.regression_model import RegressionModel


def get_absolute_datafile_path(filename):
    # data/ lives at the microsim package root, one level up from this common/ module
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_datafile_path = os.path.normpath(os.path.join(package_root, "data", filename))
    return abs_datafile_path


def load_datafile(filename):
    datafile_path = get_absolute_datafile_path(filename)
    with open(datafile_path, "r") as datafile:
        return datafile.read()


def load_model_spec(modelname):
    modelspecnamepattern = r"^[A-Za-z0-9\-]+$"
    if not re.match(modelspecnamepattern, modelname):
        raise ValueError(f"Potentially unsafe model name: {modelname}")
    data = load_datafile(f"{modelname}Spec.json")
    model_spec = json.loads(data)
    return model_spec


def load_regression_model(modelname):
    model_spec = load_model_spec(modelname)
    return RegressionModel(**model_spec)
