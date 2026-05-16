
"""
preprocess.py converts the raw data into extracted features, which are to be used
by the models.

The extracted features are log mel spectograms, which are run live during inference.
"""

# preprocesses each batch using the 'processor' as an argument, since it differs between each model. 
def preprocess_batch(audio_arrays: list, processor):
    WHISPER_SR = 16000
    # TODO; might want to resample the audio if we get errors, but I think the audio is already at 16kHz
    input_features = processor(audio_arrays, sampling_rate=WHISPER_SR, return_tensors="pt")
    return input_features["input_features"]
