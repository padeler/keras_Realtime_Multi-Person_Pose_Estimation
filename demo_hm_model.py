import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from hm_model import get_testing_model

# visualize
colors = np.array([[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]],dtype=np.float32)

colors = colors/255.0


def visualize(hm):
    hmcolor = np.zeros(hm.shape[:2] + (3,), dtype=np.float32)
    for i in range(18):
        h = hm[:, :, i]
        h[h>0.01] = 1.0
        hmcolor += np.dstack((h, h, h)) * colors[i]

    viz = cv2.normalize(hmcolor, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)

    return viz

def predict(oriImg, params, model_params, model):
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
    scale = multiplier[0]

    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                      model_params['padValue'])

    input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

    output_blobs = model.predict(input_img)

    # extract outputs, resize, and remove padding
    heatmap = np.squeeze(output_blobs[0])  # output 0 is heatmaps
    heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

    return heatmap, oriImg



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='result.png', help='output image')
    parser.add_argument('--model', type=str, default='training/hm_model_weights.best.h5', help='path to the weights file')

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
        hm, img = predict(frame, params, model_params, model)
        toc = time.time()

        bg = cv2.normalize(hm[:,:,18], None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        viz = cv2.normalize(np.sum(hm[:,:,:18],axis=2), None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow("BG", bg)

        nose = hm[:,:,0]
        print "nose ",np.min(nose),np.max(nose)
        noseNorm = cv2.normalize(nose,None,0,255,cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow("NOSE", noseNorm)
        # viz = visualize(hm[:,:,:18])

        cv2.imshow("HM", viz)

        k = cv2.waitKey(30)
        print 'time to predict() %.5f' % (toc - tic)

    cv2.destroyAllWindows()



