The dev_ files contain functions that were used to set up the network architectures and select models.
Due to necessary adjustments dependent on model type (MLP; CNN; LSTM) there's a seperate file for each model type.
The functions defined in each file are the following.

train_model_CV: trains the model on data in cross validation. 
_selection_parallel: for training different models on multiple cores and returning the output to queue
selected: Trains the model selected from grid search with the used hyperparameters. 
	Saves outputs to folder "models".
	This is the only function that is called from an upper common script (train_selected, train_selected_simulations).
	As such, function arguments need to be the same in all three scripts (CHANGE THIS AND REPLACE BY ONE FUN).