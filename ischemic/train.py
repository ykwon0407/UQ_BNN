from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import numpy as np
import pandas as pd
np.random.seed(1004)
from glob import glob
import gc, os, click, sys, time, logging, shutil
import settings, models
from utils import *
from data import *

N_ITERS=settings.N_ITERS
N_EPOCHS=settings.N_EPOCHS
N_EPOCHS_FINE=settings.N_EPOCHS_FINE
PATIENCE=settings.PATIENCE
PATIENCE_FINE=settings.PATIENCE_FINE
TIME_POINT=settings.TIME_POINT
ROW_STRIDE=settings.ROW_STRIDE
CHA_STRIDE=settings.CHA_STRIDE
N_REPEAT=settings.N_REPEAT
FIXED_WIDTH=settings.FIXED_WIDTH
FIXED_DEPTH=settings.FIXED_DEPTH

@click.command()
@click.option('--cnf', default='c_model', show_default=True,
              help="Model configuration files")
def main(cnf):
    start = time.time()
    # Load configuration
    CONFIG_DICT=load_module('configs/{}.py'.format(cnf))
    globals().update(CONFIG_DICT)

    # Set logging 
    log_name = name +'_' + '_'.join([str(s) for s in list(time.localtime(time.time())[1:6])])

    if os.path.exists('loggings/{}.log'.format(log_name)) is True:
        os.remove('loggings/{}.log'.format(log_name))
    logging.basicConfig(filename='loggings/{}.log'.format(log_name), \
                                level=logging.INFO, stream=sys.stdout)
    stderrLogger=logging.StreamHandler()
    stderrLogger.setFormatter(\
        logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s'))
    logging.getLogger().addHandler(stderrLogger)
    logging.info(CONFIG_DICT)
    logging.info('File saved: {}'.format(log_name))

    # Cross-validation settings
    if 'siss' in name:
        file_path=data_dir+'/*/*/*DWI*nii'
    else:
        file_path=data_dir+'/*/*/*TTP*nii'
    N_sample=len(glob(file_path))

    if os.path.exists('weights/{}'.format(log_name)) is True:
        shutil.rmtree('weights/{}'.format(log_name), ignore_errors=True)
    os.mkdir('weights/{}'.format(log_name))

    result=np.zeros((N_sample, N_REPEAT))
    for seed in xrange(N_REPEAT):
        logging.info('-'*50)
        logging.info("Seed {}".format(seed+1))
        logging.info('-'*50)
        
        count_folds=0
        kf=KFold(n_splits=5, shuffle=True, random_state=1004+seed)
        for tr_list, te_list in kf.split(np.arange(N_sample)):
            count_folds += 1
            logging.info('-'*50)
            logging.info("Train {}-Fold".format(count_folds))
            logging.info('-'*50)
            
            logging.info("Load Train")
            X_train, X_train_aug, y_train, lesion_indicator_train=extract_multiscale_patches_from_mri( \
                                                        tr_list, data_dir, \
                                                        is_test=False, is_oversampling=True, row_size=row_size, \
                                                         channel_size=channel_size, num_patch=num_patch, \
                                                            proportion=proportion, n_time_point=TIME_POINT, \
                                                            fixed_width=FIXED_WIDTH, fixed_depth=FIXED_DEPTH)

            X_train, X_train_aug, y_train, mean, std, mean_aug, std_aug=preprocess( \
                                        X_train, X_train_aug, y_train, None, None, None, None)
            
            logging.info("Load Validation")
            X_val, X_val_aug, y_val, lesion_indicator_val=extract_multiscale_patches_from_mri( \
                                                        te_list, data_dir, \
                                                        is_test=False, is_oversampling=False, row_size=row_size, \
                                                         channel_size=channel_size, num_patch=num_patch, \
                                                            proportion=proportion, n_time_point=TIME_POINT, \
                                                            fixed_width=FIXED_WIDTH, fixed_depth=FIXED_DEPTH)
            X_val, X_val_aug, y_val = preprocess(X_val, X_val_aug, y_val, mean, std, mean_aug, std_aug)

            logging.info("{} / {}".format(len(X_train), len(X_val)))
            logging.info("P(Y=1) on training : {}".format(np.mean(y_train)))
            logging.info("P(Y=1) on val : {}".format(np.mean(y_val)))
            logging.info("P(lesion indicator=1) on training : {}".format(np.mean(lesion_indicator_train)))
            logging.info("P(lesion indicator=1) on val : {}".format(np.mean(lesion_indicator_val)))

            logging.info("Load Model")
            model_class=find_class_by_name(model_name, models)()
            model=model_class.create_model(channel_size=channel_size, row_size=row_size, \
                n_filter=n_filter, filter_size=filter_size, lr=lr, TIME_POINT=TIME_POINT)
            
            logging.info('-'*50)
            logging.info('Fitting : compile.....')
            logging.info('-'*50)
            
            # Callbacks
            SCHEDULER=lambda epoch:lr*(0.99 ** epoch)
            info_check_string='weights/{}/{}_{}.hdf5'.format(log_name, seed, count_folds)
            early_stopping=EarlyStopping(monitor='val_loss', patience=PATIENCE)
            model_checkpoint=ModelCheckpoint(info_check_string, monitor='loss', save_best_only=True)
            change_lr=LearningRateScheduler(SCHEDULER) 

            b_generator=balance_generator(X_train, X_train_aug, y_train, \
                                         lesion_indicator_train, batch_size, \
                                         0.2)
            N_ITERS = num_patch*len(tr_list)//batch_size
            
            model.fit_generator(b_generator, steps_per_epoch=N_ITERS, epochs=N_EPOCHS, \
             validation_data=({'main_input':X_val, 'aug_input':X_val_aug}, \
              y_val), callbacks=[early_stopping, model_checkpoint, change_lr])

            model.load_weights(info_check_string)
                
            logging.info('-'*50)
            logging.info('Validating')
            logging.info('-'*50)
            
            _, label_list=load_mri_from_directory(te_list, FIXED_WIDTH, FIXED_DEPTH, \
                                                is_test=False, data_dir=data_dir, is_fixed_size=False)
            X_val_patch, X_val_patch_aug, cache=extract_multiscale_patches_from_mri( \
                                                        te_list, data_dir, \
                                                        is_test=True, is_oversampling=False, row_size=row_size, \
                                                         channel_size=channel_size, num_patch=num_patch, \
                                                         patch_r_stride=row_size/ROW_STRIDE, \
                                                         patch_c_stride=channel_size/CHA_STRIDE, \
                                                        proportion=proportion, is_fixed_size=True, \
                                                        n_time_point=TIME_POINT, \
                                                        fixed_width=FIXED_WIDTH, fixed_depth=FIXED_DEPTH)

            N_val = len(X_val_patch)   
            for i in xrange(N_val):
                list_sum_of_GT_by_depth_axis=[]
                X_val_patch_i=np.transpose(X_val_patch[i].reshape(TIME_POINT, -1, \
                    channel_size, row_size, row_size), (1,0,2,3,4) )
                X_val_patch_aug_i=np.transpose(X_val_patch_aug[i].reshape(TIME_POINT, -1, \
                    channel_size, row_size, row_size), (1,0,2,3,4) )
                X_val_patch_i, X_val_patch_aug_i = preprocess(\
                         X_val_patch_i, X_val_patch_aug_i, None, mean, std, mean_aug, std_aug)

                N_uncertain = 5 if 'uncertain' in model_name else 1
                y_val_patch=np.transpose(np.array(label_list[i]), (2,0,1))
                y_val_patch_pred_original_size = np.zeros(y_val_patch.shape)
                for k in xrange(N_uncertain):                    
                    y_val_patch_pred_i=model.predict({'main_input':X_val_patch_i, 'aug_input': X_val_patch_aug_i}, \
                        batch_size=batch_size)
                    if 'kendall' in log_name:
                        y_val_patch_pred_i = 1./(1.+np.exp(-y_val_patch_pred_i))
                    else:
                        pass
                    y_val_patch_pred=make_brain_from_patches(y_val_patch_pred_i, cache[i], patch_r_stride=row_size/ROW_STRIDE, \
                        patch_c_stride=channel_size/CHA_STRIDE)
                    
                    y_val_patch_pred/=((ROW_STRIDE ** 2) * 1.0 * CHA_STRIDE)
                    zoomRate=[float(ai)/bi for ai, bi in zip(y_val_patch.shape, y_val_patch_pred.shape)]
                    y_val_patch_pred_original_size+=transform_shrink(y_val_patch_pred, zoomRate)

                    del y_val_patch_pred, y_val_patch_pred_i
                    gc.collect()
                
                y_val_patch_pred_original_size=(y_val_patch_pred_original_size > N_uncertain*0.5)
                
                logging.info('data:{}, pred: {}, GT: {}'.format( (i+1), \
                    np.mean(y_val_patch_pred_original_size), np.mean(y_val_patch)))
                for j in xrange(y_val_patch_pred_original_size.shape[0]):
                    list_sum_of_GT_by_depth_axis.append( \
                        [np.sum(y_val_patch_pred_original_size[j]), np.sum(y_val_patch[j])])
                logging.info(list_sum_of_GT_by_depth_axis)
                
                
                dice_coef=cal_dice_coef(y_val_patch_pred_original_size.reshape(-1), y_val_patch.reshape(-1))            
                    
                logging.info('Dice Coef: {}'.format(dice_coef))
                result[te_list[i], seed]=dice_coef

                del X_val_patch_i, X_val_patch_aug_i
                gc.collect()

            logging.info("Number of parameters: {}".format(model.count_params()))
            del X_train, X_train_aug, y_train
            del X_val, X_val_aug, y_val
            del X_val_patch, X_val_patch_aug
            del model
            gc.collect()

        logging.info("RESULT: \n {}".format(result[:,seed]))
        logging.info("MEAN: {}".format(np.mean(result[:,seed])))
        logging.info("STD: {}".format(np.std(result[:,seed])))
    
    logging.info("-"*50)
    logging.info("RESULT")
    logging.info("-"*50)
    logging.info("MEAN: {}".format(np.mean(result)))
    logging.info("STD: {}".format(np.std(result)))

    pd.DataFrame(result).to_csv('weights/{}/result.csv'.format(log_name), index=False)
    
    end = time.time()
    logging.info("Elapsed time: {}".format(end-start))
    logging.info('File saved: {}'.format(log_name))
    logging.info(CONFIG_DICT)    
    
if __name__ == "__main__":
    main()


