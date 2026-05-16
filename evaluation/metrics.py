
import json
import os

import jiwer
import pandas as pd

"""
This file contains the simple functions that calclate WER, CER and RTF. 
It is refactored this way so we can import those function to inference.py, 
at the same time being able to run them seperately on a transcripts / results
from the inference. Nice to have this support just in case. 
"""


# functions mainly used in inference 
def compute_wer(references: list[str], hypotheses: list[str]):
    return jiwer.wer(references, hypotheses)
def compute_cer(references: list[str], hypotheses: list[str]):
    return jiwer.cer(references, hypotheses)
def compute_rtf(inference_times: list[float], audio_durations: list[float]):
    return sum(inference_times) / sum(audio_durations)
