Prerequisite: Scikit Learn

1) create base local directory

$) mkdir <base_directory>


2) start training

$) python fetchr_train.py -input <base_directory> -skip true 

if you already downloaded data-set then set '-skip' to 'false' 
this code generates several files in <base_directory>

- preprocessed_data: succssfully pre-processed data-set and these are the inputs to the training.
- preprocessed_err_data: error samples such as missing delivered coordinates and so on.
- index_#: contents:integer index mapping per each column.
- training_stats: metrics 
  MSE: mean squared error
  MAE: mean absolute error
  R2: R squared, coefficient of determination
  ###_delivered_coords_x: min & max of delivered_coordinates
  ###_scheulded_coords_x: min & max of scheduled_coordinates
- model.pkl: trained model file


3) test/predict 

$) python fetch_predict.py -input <base_directory> -data <sample_data>

this code generates the file 'prediction_result' in <base_directory>
- prediction_result:
  first column: <customer_name>_<customer_phone>
  second_column: predicted coordinates

