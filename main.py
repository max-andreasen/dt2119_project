
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset


processor = AutoProcessor.from_pretrained("KBLab/kb-whisper-large")
model_kb = AutoModelForSpeechSeq2Seq.from_pretrained("KBLab/kb-whisper-large")

processor = AutoProcessor.from_pretrained("openai/whisper-large")
model_whisp = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large")


# options include 'no-both' for close + distant, 'no-close' or 'no-distant' 
ds = load_dataset("NbAiLab/NST", "no-both", split="train")
