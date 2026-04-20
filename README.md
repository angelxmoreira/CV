# Billiard Ball Detection and Table Top-View Generator

This repository contains a pipeline for detecting and classifying billiard balls in images and generating a normalized top‑down view of the pool table. The main script (`main.py`) reads a list of image paths from a JSON file, applies preprocessing, performs ball detection and classification, warps the table to a top view, and writes the results to an output JSON file.

## Directory Structure
The script expects the following directories in the project root (adjustable in the script constants):
```
├── development_set/ # Input images locations
├── processed_images/ # Preprocessed images
├── top_view/ # Warped top‑view images
├── input.json # Input for the images
├── main.py # Main python script
├── README.md
├── requirements
```
## Requirements

Install the required Python packages:

```bash
pip install Pillow matplotlib seaborn opencv-python numpy
```
## Usage
```bash
python main.py [<input_json>] [<output_json>] [--debug]
```
### Arguments
Argument || Description || Default
--
`input_json` || Path to a JSON file containing a list of image paths. || `input.json`

--
`ouput_json` || Path where the output JSON will be saved. || `output.json`

--
`--debug` || Enable debug mode.	|| Disabled



