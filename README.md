- I want to start with only social text.
- Let try with small scale of the language model task: 100K docs, max_length: 100, then increase to 512 tokens. batch_size keeps at 64
    + Check speed training
    + Check maximum batch_size
    + Check maximum size of model
    + Check regularization factor
    
- Let try with small scale: 1M mentions from auto_clf project: positive_class_1 + pool

- Container: `lm_19-08-02_15_48_41` got killed
```yaml
------------------      Evaluation      ------------------                                                                                                                                           
INFO:root:Number of batchs: 313                                                                                                                                                                      
INFO:root:New best score: -3.5943461462331654                                                                                                                                                        
INFO:root:Saved model at /source/main/train/output//saved_models/Model/1.4/210000.pt                                                                                                                 
INFO:root:Current best score: -3.5943461462331654 recorded at step 210000
```