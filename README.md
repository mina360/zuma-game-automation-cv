# Zuma Game Automation using Computer Vision

A computer vision project that automates gameplay decisions in the Zuma game using screen capture, image processing, template matching, color detection, path analysis, and mouse control.

## Project Overview

This project was developed as part of a fourth-year Computer Vision course.

The goal of the project is to analyze the Zuma game screen in real time, detect important game objects, estimate danger levels along the ball path, and make automated shooting decisions based on computer vision techniques.

The system captures the game window, detects the frog shooter, identifies the current and next balls, detects the moving chain of balls, estimates the path risk, and chooses a target based on scoring and safety strategies.

## Features

* Zuma game window detection using template matching.
* Game start detection using level screen analysis.
* Frog detection using multi-angle template matching.
* Ball chain detection using Hough Circle Transform.
* Ball color classification using HSV color ranges.
* Current and next ball detection.
* End/hole/skull detection.
* Motion-based trail detection.
* Risk map generation for the ball path.
* Strategy-based target selection.
* Raycasting-based shooting validation.
* Mouse movement and shooting automation.
* Special effect detection such as slowdown, backward, and accuracy balls.
* Debug output generation for trail and risk visualization.

## Computer Vision Techniques Used

* Screen capture
* Template matching
* HSV color segmentation
* Morphological operations
* Hough Circle Transform
* Connected components analysis
* Edge detection using Canny
* Multi-scale and multi-angle matching
* Motion detection
* Risk map construction
* Raycasting for target validation

## Technologies Used

* Python
* OpenCV
* NumPy
* MSS
* PyAutoGUI
* Keyboard

## Project Structure

```txt
zuma-game-automation-cv/
  src/
    main.py
    balls_detector.py
    ball_color_detector.py
    end_detector.py
    frog_detector.py
    frog_fast.py
    window_detector.py

  templates/
    accuracy_ball.png
    backwards_ball.png
    frog_template.png
    gray.png
    level.png
    skull_mask.png
    slowdown_ball.png
    zuma_window_head.png

  docs/
    zuma.pdf

  README.md
  requirements.txt
  .gitignore
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/mina360/zuma-game-automation-cv.git
cd zuma-game-automation-cv
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment.

On Windows:

```bash
.venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the game locally in windowed mode.

6. Run the automation script:

```bash
python src/main.py
```

## Notes

* This project is an academic computer vision prototype.
* The project does not include the Zuma game itself.
* The game must be running locally for the automation script to capture and analyze the screen.
* The system is designed mainly for a Windows desktop environment.
* Some automation features may require permission to control the mouse and keyboard.
* The code may need minor calibration depending on screen resolution, window size, and game version.

## Documentation

The `docs/zuma.pdf` file contains the original project report explaining the main computer vision steps and strategy design.

## Disclaimer

This project was created for educational purposes as part of a Computer Vision course. It is intended to demonstrate image processing and automation techniques in a controlled local environment.
