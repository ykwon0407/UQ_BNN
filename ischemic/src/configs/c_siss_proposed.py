
CONFIG_DICT={
"name" : __name__.split('.')[-1],
"num_patch" : 600, # Number of patches for each case
"proportion" : 0.5, # Oversampling proportion
"row_size" : 16, # Row size
"channel_size" : 16, # Channel size 
"lr" : 2e-4, # learning rate
"batch_size" : 16, # Batch size
"n_filter" : 16, # number of filters
"n_residual" : 0, # Repeat number of residual block
"filter_size" : 3, # filter_size
"data_dir" : '../input/train_siss_2015', # input directory
"model_name" : 'unet_uncertain_2015', # model name [M,S]

"prediction_dataset" : 'train', # prediction dataset for predict.py
"test_data_dir" : '../input/test_data_2015', # input directory for predict.py

"FIXED_WIDTH" : 100,
"FIXED_DEPTH" : 72,
"TIME_POINT" : 4,
"ROW_STRIDE" : 2,
"CHA_STRIDE" : 2,
"N_EPOCHS" : 300,
"N_REPEAT" : 1,
}


