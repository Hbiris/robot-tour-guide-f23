import sys
import signal
import time
from joblib import dump, load  # To get the model to the robot

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from mbot_bridge.api import MBot
from utils.camera import CameraHandler
from utils.robot import plan_to_pose, turn_to_theta
from waypoint_writer import read_labels_and_waypoints

# TODO: Update PATH_TO_MODEL.
PATH_TO_MODEL = "model.joblib"

robot = MBot()


def signal_handler(sig, frame):
    print("Stopping...")
    robot.stop()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def main():
  ch = CameraHandler()  # Initialize the camera

  # NOTE: This code will fail if you have not updated PATH_TO_MODEL above.
  assert os.path.exists(PATH_TO_MODEL), f"Model file {PATH_TO_MODEL} does not exist."
  model = load(PATH_TO_MODEL)
  
  # TODO: Either uncomment and modify the hard coded definitions for
  # waypoints and labels below or run waypoint_writer.py to write
  # waypoints and labels for your current map to output/waypoints.txt.
  #
  # You must handle the case where your model detects a label which is not part of the
  # given course.

  labels, waypoints = read_labels_and_waypoints()  # Load from waypoints.txt

  # labels = [0, 1, 2, ...]
  # waypoints = [[0, 0, 0], [9, 2, 8], [3, 6, 4], ...]
  if not labels or not waypoints:
        print("Error: Labels and waypoints are not properly loaded.")
        sys.exit(1)

  while(True):
  # NOTE: To get the cropped image of a post-it from the camera, do:
  #
  #     frame = ch.get_processed_image()
  #
  # If a post-it is not detected, frame will be None. Make sure you check for this
  # case. To save the image to the folder "output", pass `save=True` to this
  # function. It will be the last image labeled 08_bordered_frame_X.jpg. Check out
  # the previous images to understand how the preprocessing step works. This might 
  # slow down execution slightly. 
    # Get processed image from the camera
        frame = ch.get_processed_image(save=True)  # Save the processed frame for debugging
        if frame is None:
            print("No digit detected. Skipping...")
            continue

        # Predict the label using the model
        predicted_label = model.predict([frame])[0]
        print(f"Detected label: {predicted_label}")
  
  # NOTE: To plan a path to a goal position (x and y, in meters), do:
  #
  #     plan_to_pose(x, y, robot)
  #
  # To turn to a given angle (theta, in radians), do:
  #
  #     turn_to_theta(theta, robot)
  #
  # You must turn to the theta after x, y location is reached.
  
  # TODO: Your code here!
  # Write your code to detect the label on a poster at a given waypoint, and use
  # the result to determine which waypoint to visit next. You will need to use the
  # "labels" and "waypoints" variables! When the robot reads a poster with label "0",
  # it should return to the start position (0, 0, 0) and the program should exit.
  # Handle label "0" (return to start and stop)
        if predicted_label == 0:
            print("Returning to start position...")
            plan_to_pose(0, 0, robot)
            turn_to_theta(0, robot)
            print("Task complete. Exiting...")
            break
        
        # Get the corresponding waypoint
        waypoint_index = labels.index(predicted_label)
        target_waypoint = waypoints[waypoint_index]
        x, y, theta = target_waypoint

        # Move to the waypoint
        print(f"Moving to waypoint {target_waypoint}...")
        plan_to_pose(x, y, robot)
        turn_to_theta(theta, robot)

if __name__ == '__main__':
    main()
