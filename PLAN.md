
# Project plan

## Fine-tune KB Whisper
- Implement LoRA on KB Whisper. 
- Create supporting code. 
- Run pilot studies / hyperparam search. 
- Run final tuning → Store model. 
- Testing / gather results (run inference)

## Baseline KB
- Implement support for inference / testing. 
- Create supporting code. 
- Run pilot studies (maybe no param-search, since we are not training / tuning the model). 
- Testing / gather results (run inference). 

## Report
- Short literature study. 
- Written parts (method, experiment, results). 
- Discussion. 


## Code structure
- Use condigs to enter model values for training. 
- Configs are used to easily run as a CLI argument, and can be directly translated to a reuslt file, containing information about the model etc. 
