#!/bin/bash

# Activate the virtual environment
source /home/admin/NodeRedFiles/WaveFinder/Model/visibility_model_package/venv/bin/activate

# Run the training script with a location argument
python3 /home/admin/NodeRedFiles/WaveFinder/Model/visibility_model_package/predict_visibility_from_db.py "$1"
