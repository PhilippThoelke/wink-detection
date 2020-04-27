# wink-detection
The detector uses opencv's Haar cascade classifier to detect a face and eyes within the face. Then it figures out whether an eye is closed or not by looking at the dark regions around the eye (pupil, eyebrows). The classifier was only adapted to light skin and dark eyebrows. The keyboard module can be used to send key press events when winking. Different wink events are generated based on which eye was closed.

You can run the detector with `python wink.py`, the detection parameters at the top of the file might need tweaking for other individuals.
