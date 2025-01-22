# README.md

# Image Classifier App

This project is a mini application built using Tkinter that allows users to classify images into predefined categories. The application displays images one by one and enables users to select the corresponding class for each image. The selected classes are recorded in a DataFrame and can be saved to a CSV file.

## Project Structure

```
image-classifier-app
├── src
│   ├── main.py               # Entry point of the application
│   ├── data_handler.py       # Functions for loading and saving data
│   ├── gui
│   │   ├── __init__.py       # GUI package initializer
│   │   └── image_viewer.py    # Class for displaying images
│   └── utils
│       ├── __init__.py       # Utils package initializer
│       └── file_operations.py # Utility functions for file operations
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Requirements

To run this application, you need to install the following dependencies:

- pandas
- tkinter

You can install the required packages using pip:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```
   git clone <repository-url>
   cd image-classifier-app
   ```

2. Run the application:
   ```
   python src/main.py
   ```

3. Follow the on-screen instructions to classify the images.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.