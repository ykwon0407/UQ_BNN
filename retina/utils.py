import numpy as np

def preprocess(ims):
    # convert to float32
    ims = ims.astype(np.float32)

    # scale to 0-1 interval
    if ims.max() > 1.0:
        ims /= 255.
    
    print('min: {}, max: {}, shape: {}, type: {}'.format(ims.min(), ims.max(), ims.shape, ims.dtype))
    return ims


def random_snippet(x, y, size=(48,48), rotate=True, flip=True):
    '''sample snippets from images. return image tuple (real, segmentation) of size `size` '''
    
    assert x.shape[:2] == y.shape[:2]
    
    # get image sample
    sample = np.random.randint(0, x.shape[0])
    # get x and y dimensions
    min_h = np.random.randint(0, x.shape[1]-size[0])
    max_h = min_h+size[0]
    min_w = np.random.randint(0, x.shape[2]-size[1])
    max_w = min_w+size[1]
    # extract snippet
    im_x = x[sample, min_h:max_h, min_w:max_w, :]
    im_y = y[sample, min_h:max_h, min_w:max_w, :]
    
    # rotate
    if rotate:
        k = np.random.randint(0,4)
        im_x = np.rot90(im_x, k=k, axes=(0,1))
        im_y = np.rot90(im_y, k=k, axes=(0,1))
        
    # flip left-right, up-down
    if flip:
        if np.random.random() < 0.5:
            lr_ud = np.random.randint(0,2) # flip up-down or left-right?
            im_x = np.flip(im_x, axis=lr_ud)
            im_y = np.flip(im_y, axis=lr_ud)
    
    return (im_x, im_y)

def get_random_snippets(x, y, number, size):
    snippets = [random_snippet(x, y, size) for i in range(number) ]
     
    ims_x = np.array([i[0] for i in snippets])
    ims_y = np.array([i[1] for i in snippets])
    return (ims_x, ims_y)    


def predict(model, image, gt, T=10):
    
    # add batch dimension
    image = np.expand_dims(image, 0)
    gt = np.expand_dims(gt, 0)
    
    # predict stochastic dropout model T times
    p_hat = []
    for t in range(T):
        p_hat.append( model.predict(image)[0] )
    p_hat = np.array(p_hat)
    
    # mean prediction
    prediction = np.mean(p_hat, axis=0)
    # threshold mean prediction
    #prediction = np.where(prediction > 0.5, 1, 0)
    
    # estimate uncertainties (eq. 4 )
    # eq.4 in https://openreview.net/pdf?id=Sk_P2Q9sG
    # see https://github.com/ykwon0407/UQ_BNN/issues/1
    aleatoric = np.mean(p_hat*(1-p_hat), axis=0)
    epistemic = np.mean(p_hat**2, axis=0) - np.mean(p_hat, axis=0)**2
    
    loss, dice, prob, precision, recall = model.evaluate(image, gt, batch_size=1, verbose=0)

    return np.squeeze(prediction), np.squeeze(aleatoric), np.squeeze(epistemic), (dice, precision, recall)    



    