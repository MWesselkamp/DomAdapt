The different model numerations refer to the data that they have been trained and evaluated on.

1 or 2, only one way will finally be used to select models. To discuss with Maria.


0) 6-fold CV. Training data: 5 Years of Profound data (Hyytiala). Validation data: 1 Year of Profound data. ONLY AFTER MODEL SELECTION

1) 6-fold CV. Training data: 5 Years of Profound data (Le Bray). Validation data: 1 Year of Profound data.
2) 5-fold CV. Training data: 4 Years of Profound data (Le Bray). Validation data: 1 Year of Profound data. Test data: 1 Year of Profound data.
3) 5-fold CV. Training data: 4 Years of Profound data (Le Bray). Validation data: 1 Year of Profound data. 
	Test data: 1 Year of Profound as input features and respective Preles simulations as target.
4) 15-fold CV: Training data: 5 Year of Profound from 3 stands, i.e. 15 years. Validation data: 1 Year of Profound of Profound data.
	Test data: 1 Year of Profound data from 3 different stands, i.e. 3 years. 

5) 5-fold CV. Training and Validation data: Simulated Climate. Pretrained for finetuning.

Following models: MLP with additional Input feature Encoder and Adaptive Pooling Layer!! 

6) 5-fold CV. Training and Validation data: Simulated Climate. Pretrained for finetuning.

7) 5-fold CV. Training and Validation data: Simulated Climate and Parameter samples. Pretrained for finetuning and feature extraction.
	a) trained on simulations with uniformly sampled parameter values
	b) trained on simulations with gaussian sampled parameter values
8) 5-fold CV. Training and Validation data: Simulated Climate and Parameter samples. Inc Dropout. Pretrained for finetuning and feature extraction.
	a) trained on simulations with uniformly sampled parameter values
	b) trained on simulations with gaussian sampled parameter values

Stand: 24.10.2020