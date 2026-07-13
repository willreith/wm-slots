# This script shows how to load this in Python using PyNWB and LINDI
# It assumes you have installed PyNWB and LINDI (pip install pynwb lindi)

import pynwb
import lindi

local_cache = lindi.LocalCache()

# Load https://api.dandiarchive.org/api/assets/15d93ab8-e57b-4bd7-b1ac-9ebf8b5da2bc/download/
f = lindi.LindiH5pyFile.from_hdf5_file("https://api.dandiarchive.org/api/assets/15d93ab8-e57b-4bd7-b1ac-9ebf8b5da2bc/download/", local_cache=local_cache)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

nwb.session_description # (str) Data from macaque performing multi-object working memory task. Subject is presented with multiple objects at different locations on a screen. After a delay, the subject is then cued with one of the objects, now displayed at the center of the screen. Subject should respond by saccading to the location of the cued object at its initial presentation.
nwb.identifier # (str) 5738f8a4-4c12-4f2c-b2e0-e93f29cd9461
nwb.session_start_time # (datetime) 2022-05-28T14:08:52-04:00
nwb.file_create_date # (datetime) 2023-12-30T18:31:15.244635-05:00
nwb.timestamps_reference_time # (datetime) 2022-05-28T14:08:52-04:00
nwb.experimenter # (List[str]) ["Watters, Nicholas", "Gabel, John"]
nwb.experiment_description # (str) 
nwb.institution # (str) MIT
nwb.keywords # (List[str]) []
nwb.protocol # (str) 
nwb.lab # (str) Jazayeri
nwb.subject # (Subject)
nwb.subject.age # (str) P10Y
nwb.subject.age__reference # (str) birth
nwb.subject.description # (str) 
nwb.subject.genotype # (str) 
nwb.subject.sex # (str) F
nwb.subject.species # (str) Macaca mulatta
nwb.subject.subject_id # (str) Perle
nwb.subject.weight # (str) 
nwb.subject.date_of_birth # (datetime) 

display = nwb.intervals["display"] # (TimeIntervals) data about each displayed frame
display["closed_loop_eye_position"] # (h5py.Dataset) shape [319172, 2]; dtype <f8 For each frame, the eye position in the close-loop task engine. This was used to for real-time eye position computations, such as saccade detection and reward delivery.
display["fixation_cross_scale"] # (h5py.Dataset) shape [319172]; dtype <f8 For each frame, the scale of the central fixation cross. Fixation cross scale grows as the eye position deviates from the center of the fixation cross, to provide a cue to maintain good fixation.
display["id"] # (h5py.Dataset) shape [319172]; dtype <i8 undefined
display["start_time"] # (h5py.Dataset) shape [319172]; dtype <f8 Start time of epoch, in seconds
display["stop_time"] # (h5py.Dataset) shape [319172]; dtype <f8 Stop time of epoch, in seconds
display["task_phase"] # (h5py.Dataset) shape [319172]; dtype |O The phase of the task for each frame.

trials = nwb.intervals["trials"] # (TimeIntervals) data about each trial
trials["background_indices"] # (h5py.Dataset) shape [1840, 2]; dtype <i8 For each trial, the indices of the background noise pattern patch.
trials["broke_fixation"] # (h5py.Dataset) shape [1840]; dtype |b1 For each trial, whether the subject broke fixation and the trial was aborted
trials["closed_loop_response_position"] # (h5py.Dataset) shape [1840, 2]; dtype <f8 For each trial, the position of the response saccade used by the closed-loop game engine. This is used for determining reward.
trials["closed_loop_response_time"] # (h5py.Dataset) shape [1840]; dtype <f8 For each trial, the time of the response saccade used by the closed-loop game engine. This is used for the timing of reward delivery.
trials["delay_object_blanks"] # (h5py.Dataset) shape [1840]; dtype |b1 For each trial, a boolean indicating whether the objects were rendered as blank discs during the delay phase.
trials["id"] # (h5py.Dataset) shape [1840]; dtype <i8 undefined
trials["phase_cue_time"] # (h5py.Dataset) shape [1840]; dtype <f8 Time of cue phase onset for each trial.
trials["phase_delay_time"] # (h5py.Dataset) shape [1840]; dtype <f8 Time of delay phase onset for each trial.
trials["phase_fixation_time"] # (h5py.Dataset) shape [1840]; dtype <f8 Time of fixation phase onset for each trial.
trials["phase_iti_time"] # (h5py.Dataset) shape [1840]; dtype <f8 Time of inter-trial interval onset for each trial.
trials["phase_response_time"] # (h5py.Dataset) shape [1840]; dtype <f8 Time of response phase onset for each trial.
trials["phase_reveal_time"] # (h5py.Dataset) shape [1840]; dtype <f8 Time of reveal phase onset for each trial.
trials["phase_stimulus_time"] # (h5py.Dataset) shape [1840]; dtype <f8 Time of stimulus phase onset for each trial.
trials["response_position"] # (h5py.Dataset) shape [1840, 2]; dtype <f8 Response position for each trial. This differs from closed_loop_response_position in that this is calculated post-hoc from high-resolution eye tracking data, hence is more accurate. Note that unlike closed_loop_response_position, this may be inconsistent with reward delivery.
trials["response_time"] # (h5py.Dataset) shape [1840]; dtype <f8 Response time for each trial. This differs from closed_loop_response_time in that this is calculated post-hoc from high-resolution eye tracking data, hence is more accurate. Note that unlike closed_loop_response_time, this may be inconsistent with reward delivery.
trials["reward_duration"] # (h5py.Dataset) shape [1840]; dtype <f8 Reward duration for each trial
trials["reward_time"] # (h5py.Dataset) shape [1840]; dtype <f8 Time of reward delivery onset for each trial.
trials["start_time"] # (h5py.Dataset) shape [1840]; dtype <f8 Start time of epoch, in seconds
trials["stimulus_object_identities"] # (h5py.Dataset) shape [1840]; dtype |O For each trial, a serialized list with one element for each object. Each element is the identity symbol (e.g. "a", "b", "c", ...) of the corresponding object.
trials["stimulus_object_positions"] # (h5py.Dataset) shape [1840]; dtype |O For each trial, a serialized list with one element for each object. Each element is the initial (x, y) position of the corresponding object, in coordinates of arena width.
trials["stimulus_object_target"] # (h5py.Dataset) shape [1840]; dtype |O For each trial, a serialized list with one element for each object. Each element is a boolean indicating whether the corresponding object is ultimately the cued target.
trials["stimulus_object_velocities"] # (h5py.Dataset) shape [1840]; dtype |O For each trial, a serialized list with one element for each object. Each element is the initial (dx/dt, dy/dt) velocity of the corresponding object, in units of arena width per display update.
trials["stop_time"] # (h5py.Dataset) shape [1840]; dtype <f8 Stop time of epoch, in seconds

behavior = nwb.processing["behavior"] # (ProcessingModule) Contains behavior, audio, and reward data from experiment.

audio = nwb.processing["behavior"]["audio"] # (LabeledEvents) Audio data representing auditory stimuli events
audio.timestamps # (h5py.Dataset) shape [1854]; dtype <f8

eye_position = nwb.processing["behavior"]["eye_position"] # (SpatialSeries) Eye position data recorded by EyeLink camera
eye_position.data # (h5py.Dataset) shape [3924355, 2]; dtype <f8
eye_position.timestamps # (h5py.Dataset) shape [3924355]; dtype <f8

pupil_size = nwb.processing["behavior"]["pupil_size"] # (TimeSeries) Pupil size data recorded by EyeLink camera
pupil_size.data # (h5py.Dataset) shape [3924180]; dtype <f8
pupil_size.timestamps # (h5py.Dataset) shape [3924180]; dtype <f8

reward_line = nwb.processing["behavior"]["reward_line"] # (LabeledEvents) Reward line data representing events of reward dispenser
reward_line.timestamps # (h5py.Dataset) shape [2343]; dtype <f8
