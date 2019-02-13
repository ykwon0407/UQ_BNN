"""
This file is to support data.py file
"""
import numpy as np
import cv2
import importlib
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def make_range(point, img_size, patch_size, isFront=True, factor=2):
    if isFront == True:
        return max(point-factor*patch_size, 0)
    else:
        return min(point, img_size-factor*patch_size)

def normalize_img(img):
    img = np.array(img, dtype='float32')
    img -= np.mean(img)
    img /= np.std(img)
    return img

def transform_shrink(img, zoomRate=(0.5,0.5,0.5)):
    dst = zoom(img, zoom=zoomRate, order=1, mode='nearest') 
    return dst

def transform_to_fixed_size(img, FIXED_WIDTH, FIXED_DEPTH):
    a = img.shape
    b = (FIXED_WIDTH,FIXED_WIDTH,FIXED_DEPTH)
    zoomFactors = [bi/float(ai) for ai, bi in zip(a, b)]
    dst = zoom(img, zoom=zoomFactors, order=1, mode='nearest') 
    if dst.shape != b:
        print("differnt {}".format(dst.shape))
    return dst
    
def transform_saggital(img):
    dst2 = cv2.flip(img, 0)
    return dst2

def find_class_by_name(name, module_name):
    """
    load model class
    """
    modules = getattr(module_name, name, None)
    if modules is None:
        raise StandardError("Appropriate model name is needed")
    else:
        return modules

def load_module(mod):
    """
    import config files
    """
    config_file=importlib.import_module(mod.replace('/','.').split('.py')[0])
    return config_file.CONFIG_DICT

def choose_index_original(lesion_indicator, prop=0.5):
    """
    Oversample the patches which include at least one lesion voxel.
    """
    one_index=np.argwhere(lesion_indicator==1).reshape(-1)
    zero_index=np.argwhere(lesion_indicator==0).reshape(-1)
    N_index = int(len(one_index)*prop)
    if N_index % 2 != 0:
        N_index -= 1
    index_set=np.append(np.random.choice(one_index, N_index), \
        np.random.choice(zero_index, int(len(zero_index))))
    index_set=np.append(index_set, np.random.choice(zero_index, (len(one_index)-N_index)))
    
    #permutation
    perm=np.random.permutation(len(index_set))
    index_set=index_set[perm]
    return index_set

def choose_index(lesion_indicator, prop=0.5):
    """
    Oversample the patches which include at least one lesion voxel.
    """
    one_index=np.argwhere(lesion_indicator==1).reshape(-1)
    zero_index=np.argwhere(lesion_indicator==0).reshape(-1)
    
    N1_index = len(one_index)
    N0_index = int(len(one_index)*(1-prop)/prop)
    
    index_set=np.append(np.random.choice(one_index, N1_index), \
        np.random.choice(zero_index, N0_index))
    # index_set=np.append(index_set, np.random.choice(zero_index, (len(one_index)-N1_index)))
    
    #permutation
    perm=np.random.permutation(len(index_set))
    index_set=index_set[perm]
    return index_set    
    
def balance_generator_med(X_main, X_aug, y, clinical_info, lesion_indicator, batch_size):
    """
    make input generator
    """
    rate = 1.0
    while True:
        oversampled_index = choose_index(lesion_indicator, rate)
        # rate *= 0.99        
        X_main_batch, X_aug_batch = X_main[oversampled_index], X_aug[oversampled_index]
        y_batch, clinical_info_batch = y[oversampled_index], clinical_info[oversampled_index]
        for i in range(0, len(y_batch), batch_size):
            yield ({'main_input': X_main_batch[i:(i+batch_size)], \
                'aug_input': X_aug_batch[i:(i+batch_size)], \
                'clinical_input': clinical_info_batch[i:(i+batch_size)] }, \
                y_batch[i:(i+batch_size)])

def balance_generator(*args):
    """
    make input generator
    """
    if len(args) == 6:
        X_main, X_aug, y, lesion_indicator, batch_size, rate = args
        while True:
            oversampled_index=choose_index(lesion_indicator, rate) 
            # rate = rate*0.975
            X_main_batch, X_aug_batch=X_main[oversampled_index], X_aug[oversampled_index]
            y_batch=y[oversampled_index]
            for i in range(0, len(y_batch), batch_size):
                yield ({'main_input': X_main_batch[i:(i+batch_size)], \
                    'aug_input': X_aug_batch[i:(i+batch_size)]}, \
                    y_batch[i:(i+batch_size)])
    else:
        X_main, y, lesion_indicator, batch_size, rate = args
        while True:
            oversampled_index=choose_index(lesion_indicator, rate) 
            X_main_batch=X_main[oversampled_index]
            y_batch=y[oversampled_index]
            for i in range(0, len(y_batch), batch_size):
                yield ({'main_input': X_main_batch[i:(i+batch_size)]}, y_batch[i:(i+batch_size)])


def preprocess(*args):
    if len(args) == 4:
        X_main, y, mean, std=args
        if y is not None:
            if mean is None:
                X_main = np.transpose(X_main, (0, 1, 4, 2, 3))
                y = np.transpose(y, (0, 3, 1, 2))
                y = y[:,np.newaxis,:,:,:]
                mean, std=0.0, 1.0
                return X_main, y, mean, std
            else:
                X_main = np.transpose(X_main, (0, 1, 4, 2, 3))
                y = np.transpose(y, (0, 3, 1, 2))
                y = y[:,np.newaxis,:,:,:]
                return X_main, y
        else:
            return X_main
    else:
        X_main, X_aug, y, mean, std, mean_aug, std_aug=args
        if y is not None:
            if mean is None:
                X_main = np.transpose(X_main, (0, 1, 4, 2, 3))
                X_aug = np.transpose(X_aug, (0, 1, 4, 2, 3))
                y = np.transpose(y, (0, 3, 1, 2))
                y = y[:,np.newaxis,:,:,:]
                mean, std, mean_aug, std_aug=0.0, 1.0, 0.0, 1.0
                return X_main, X_aug, y, mean, std, mean_aug, std_aug
            else:
                X_main = np.transpose(X_main, (0, 1, 4, 2, 3))
                X_aug = np.transpose(X_aug, (0, 1, 4, 2, 3))
                y = np.transpose(y, (0, 3, 1, 2))
                y = y[:,np.newaxis,:,:,:]
                return X_main, X_aug, y
        else:
            return X_main, X_aug


def cal_dice_coef(y_pred, y_true):
    return 2.0*sum(y_pred*y_true)/(sum(y_pred)+ sum(y_true))


def cal_kendall_dice_coef(y_pred, y_true):
    y_pred = 1./(1+np.exp(-y_pred))
    return 2.0*sum(y_pred*y_true)/(sum(y_pred)+ sum(y_true))


"""
Below are working code

def transform_elastic(img, alpha= 6, sigma=2.5, alpha_affine=2):
    random_state = np.random.RandomState(None)
    shape = img.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)

"""
