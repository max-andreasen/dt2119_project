# dt2119_project

## Set up
To load the models weights and the dataset to cache, run: 
```bash
pip install -r requirements.txt
python main.py # optional
``` 
Make sure to comment out the train split if just running inference. 

### On the KTH server
```bash
source setup_env.sh
python main.py # optional
git lfs install
git lfs pull
```  
setup_env.sh will automatically run the pip install for all requirements. 
python main.py isn't strictly needed, as running e.g. infefence.py loads everything automatically. 
git lfs pull will pull the large adapter weigths-file from the GH repo, in case that is not already present. 

## Repo structure
This repo is mainly built to run inference, or evaluate the models. 

### Testing
To evaluate each model by running inference on the test, first create a config file. 
This is done in evaluation/configs, and by copying inference.yaml (template). 

If the test set was fully donwloaded, it consists of

After this, run evaluation via; 
```bash
cd evaluation
python inference.py --config configs/CONFIG_NAME.yaml
``` 
This will: 
1) Run inference and translate tokens to readable text. 
2) Save the transcription and results.csv.
3) Load the results.csv and transcription automatically to run the metrics. 
4) The metrics calculated, WER, CER and RTF, are all saved into a new results file. 


### Models
No fully built PyTorch models exists here, instead we load the model weights via HuggingFace. 
Also we export the model weights from our fine-tuned KB Whisper and load them here for evaluation. 

### Training
We only train the KB Whisper via LoRA fine-tuning. This we do in a seperate notebook (train.ipynb) on the KTH GPU cluster. 
