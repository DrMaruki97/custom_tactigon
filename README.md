# Custom middleware for the use of [<span style="color:red">Tactigon</span>](https://thetactigon.com/)

## Record.py

- We created a Tester class (That can be found in `customTskin/middleware/itsGesture`) that collects the sesnsory data from the device in the middleware separate process when an `Event` object is set by tapping once with one finger on the touchpad of the tactigon.
  
- To automatically label these data "rows" we exploited the `TwoFingerGesture` class setting a series of consecutive events that get activated on each two-finger tap and that modify the label associated with each recording row in the middleware process.
  
- Once a session of recording is completed (ideally a session consists of a few recordings for each gesture, in our case: up, down, left, right, forward and backward) all the sensory data collected and their respective labels are saved in a csv file in the `movement_data` folder.

## Data processing and ML model creation

- From the data that we saved in our csv, we divide the file for each label, then we transform each gesture (in our case a 2 seconds recording that equates to 100 rows of data) in a single row of a new dataset by putting all the rows of a gesture in a list, then creating a pandas `DataFrame` object from said list and trasposing it from single column to single row.
  
- The new datasets processed as described above are then merged and saved as `final_df.csv`.

- A random forest model is then istantiated as a `MovementClassifier` class object and trained on the data coming from the processed csv.

## Audio processing

For what concers audio recognition, we decided to use the open-source model "Whisper" developed by OpenAI and invoke its `base` form in the `wernicke(self,path,model)` method of the `CustomTskin` class.
The strange name, for those interested, come from [here](https://en.wikipedia.org/wiki/Wernicke%27s_area).

## robot.py

To test our custom classes we integrated them with robot commands that can be found in [tactigon-ironboy](https://pypi.org/project/tactigon-ironboy/) library to impart simple commands to a robot trough either our gestures or trough an audio-to-text-to-action method.

## Next steps

In the following months we aim to improve the overall architecture of this project and implement:

- A new method for gesture recognition based on a Neural Network

- Stop using "whisper" as our main method for parsing audio files and replace it with a custom model
