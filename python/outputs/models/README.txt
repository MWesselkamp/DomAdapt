The different model numerations refer to the data that they have been trained and evaluated on.

1 or 2, only one way will finally be used to select models. To discuss with Maria.

0) 5-fold CV. Training data: 5 Years of Profound data (Hyytiala) (2001, 2003, 2004, 2005, 2006). Test data: 1 Year of Profound data (2008). ONLY AFTER MODEL SELECTION - Best MLP. REFERENCE MODEL
	a) without adaptive pooling
	b) with adaptive pooling

1) 6-fold CV. Training data: 5 Years of Profound data (Le Bray). Validation data: 1 Year of Profound data. NOT USED ANYMORE?

2) 5-fold CV. Training data: 7 Years of Profound data (Bily Kriz) (2001, 2002, 2003, 2004, 2005, 2006, 2007). Test data: one year of profound data (2008). ONLY AFTER MODEL SELECTION - Best MLP. Best LSTM. Best CNN. Best RF.
	a) without adaptive pooling
	b) with adaptive pooling

3) 5-fold CV. Training data: 4 Years of Profound data (Le Bray). Validation data: 1 Year of Profound data. 
	Test data: 1 Year of Profound as input features and respective Preles simulations as target.

4) 5-fold CV: Training data: 7 Years of Profound from 3 stands, i.e. 21 years (Bily_Kriz, Collelongo, Soro) (2001, 2002, 2003, 2004, 2005, 2006, 2007). 
	Test data: 1 Year of Profound data from 3 different stands (2008), i.e. 3 years. 

PRETRAINING ON SIMULATIONS

5) 5-fold CV. Training and Validation on Simulated Climate only. Pretrained for finetuning.
	a) fixedPars: Parammeter values fixed to calibrated values (Elisa Schneider)
	b) normalPars: Parameters sampled from trunc normal
	c) uniformPars: Parameters sampled from uniform

Models > 5: MLP with additional Input feature Encoder and Adaptive Pooling Layer!! 

6) 5-fold CV. Training and Validation on Simulated Climate only. Pretrained for finetuning.
	a) fixedPars: Parammeter values fixed to calibrated values (Elisa Schneider)
	b) normalPars: Parameters sampled from trunc normal
	c) uniformPars: Parameters sampled from uniform

7) 5-fold CV. Training and Validation on Full Simulations. Climate and Parameter samples from normal distribution
	a) dropout
	b) no dropout
8) 5-fold CV. Training and Validation on Full Simulations. Climate and Parameter samples from uniform distribution.
	a) dropout
	b) no dropout

Stand: 07.11.2020