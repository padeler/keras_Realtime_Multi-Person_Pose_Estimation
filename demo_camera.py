import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from demo_image import predict, skeletonize, visualize

# from model import get_testing_model
from vnect_model import get_testing_model

def showPAFs(PAFs, startIdx=0, endIdx=19):
    allpafs = []
    for idx in range(startIdx, endIdx):
        
        if idx in (9,13):
            continue

        X = PAFs[:,:,idx*2]
        Y = PAFs[:,:,idx*2+1]
        tmp = np.dstack((X, Y, np.zeros_like(X)))
        allpafs.append(tmp)

    pafs = np.mean(allpafs, axis=0)
    pafs = cv2.normalize(pafs,None, 0,255,cv2.NORM_MINMAX,cv2.CV_8UC3)
    cv2.imshow("PAF", pafs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='result.png', help='output image')
    parser.add_argument('--model', type=str, default='training/vnect_pafs_weights.h5', help='path to the weights file')
    # parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    # parser.add_argument('--model', type=str, default='model/keras/weights_epoch29_loss462.h5', help='path to the weights file')

    args = parser.parse_args()
    output = args.output
    keras_weights_file = args.model


    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    print('start loop...')

    # load config
    params, model_params = config_reader()
    k = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.3)

    while k&0xFF != ord('q'):
        ret, frame = cap.read()
        if not ret:
            raise Exception("VideoCapture.read() returned False")

        tic = time.time()
        # generate image with body parts
        hm, pafs, img = predict(frame, params, model_params, model)
        toc = time.time()

        all_peaks, subset, candidate = skeletonize(hm, pafs, img.shape, params['thre1'], params['thre2'])
        viz = visualize(img.copy(), all_peaks, subset, candidate)
        cv2.imshow("VIZ", viz)
        
        # showPAFs(pafs)

        # bg = cv2.normalize(hm[:,:,18], None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # hmnorm = cv2.normalize(np.sum(hm[:,:,:18],axis=2), None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # cv2.imshow("BG", bg)
        # cv2.imshow("HM", hmnorm)

        k = cv2.waitKey(30)
        print 'processing time is %.5f' % (toc - tic)

    cv2.destroyAllWindows()



