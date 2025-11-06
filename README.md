AI/ML Intermediate Project
# Data Self-Driving Car Simulation

## Overview
This project implements a self-driving car simulation using behavioral cloning. It trains a deep learning model to predict steering angles from camera images, replicating human driving behavior. The model is built using TensorFlow and Keras and trained on datasets containing road images and driving logs.

## Project Structure
├── self_driving_car_dataset_jungle/
│   ├── IMG/
│   └── driving_log.csv
├── self_driving_car_dataset_make/
│   ├── IMG/
│   └── driving_log.csv
├── self_driving_car_simulation.ipynb
├── model.h5
└── README.md

## Installation
Clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/self-driving-car-simulation.git
cd self-driving-car-simulation
pip install tensorflow==2.17.0 opencv-python pandas scikit-learn matplotlib imgaug
```

## Dataset
The project uses two compatible datasets:
- `self_driving_car_dataset_jungle`
- `self_driving_car_dataset_make`
Link : https://www.kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning?select=self_driving_car_dataset_jungle

Each dataset contains:
- `IMG/` folder with road images.
- `driving_log.csv` containing steering, throttle, brake, and speed values.

These datasets are automatically merged in the notebook before training.

## How It Works
1. Loads and merges both driving logs.
2. Extracts and preprocesses image data.
3. Trains a Convolutional Neural Network to predict steering angles.
4. Evaluates model performance on validation data.
5. Saves the trained model for simulation or inference.

## Usage
1. Open the Jupyter notebook:
```bash
jupyter notebook self_driving_car_simulation.ipynb
```
2. Run all cells to preprocess data, train the model, and test predictions.
3. The trained model will be saved as `model/model.h5`.

## Requirements
- Python 3.11 or later
- TensorFlow 2.17.0
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- imgaug

## Future Enhancements
- Add Streamlit interface for real-time predictions.
- Experiment with CNN-LSTM for sequential frame prediction.
- Integrate lane detection or object recognition modules.

## Contributing
Contributions are welcome!
Feel free to fork the repository, improve the game, and open a pull request. Let's grow this classic game together!

## License
This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Author
**Aarya Mehta**  
🔗 [GitHub Profile](https://github.com/AaryaMehta2506)



