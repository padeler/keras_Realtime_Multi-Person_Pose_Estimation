import PyMBVCore as Core
import PyMBVAcquisition as Acquisition

from tensorflow import keras
from keras.models import Model
import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter

from cvcommons import image


# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def preprocess(oriImg, model_params):
    scale = float(model_params['boxsize']) / float(oriImg.shape[0])

    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest = imageToTest[:,:,np.newaxis]
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                      model_params['padValue'])

    input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

    return input_img, pad


def postprocess(output_blobs, model_params, pad, oriImgShape, inputImgShape, hm_idx=0):

    # extract outputs, resize, and remove padding
    heatmap = np.squeeze(output_blobs[hm_idx])  # output at hm_idx is heatmaps
    heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:inputImgShape[0] - pad[2], :inputImgShape[1] - pad[3], :]
    heatmap = cv2.resize(heatmap, (oriImgShape[1], oriImgShape[0]), interpolation=cv2.INTER_CUBIC)

    return heatmap



def find_peaks(hm, thre1):

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = hm[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    return all_peaks


def visualize(canvas, all_peaks, part_str):
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
            cv2.putText(canvas, part_str[i], all_peaks[i][j][0:2], 0, 0.5, colors[i])

    return canvas


if __name__ == '__main__':
    # import model
    import vnect_model as model

    keras_weights_file = "training/gray_vnect_weights_2.6k.h5"
    # keras_weights_file = "model/keras/model.h5"

    model = model.get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()
    k = 0

    gr = Acquisition.OpenNI2Grabber(True, True, "", False, False)
    gr.initialize()
    
    print('start loop...')

    while k&0xFF != ord('q'):
        try:
            imgs, clbs = gr.grab()
        except:
            print("Failed to grab")
            break

        depth,ir = imgs
        # image.show("Depth", depth)
        # k = image.show("IR", ir,1)

        gray = cv2.normalize(ir,None,0,255,cv2.NORM_MINMAX, cv2.CV_8UC1)        
        # gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)        
        input_img, pad = preprocess(gray, model_params)

        tic = time.time()
        output_blobs = model.predict(input_img)
        toc = time.time()

        hm = postprocess(output_blobs, model_params, pad, gray.shape, input_img.shape[1:],hm_idx=0)

        bg = cv2.normalize(hm[:,:,18], None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        viz = cv2.normalize(np.sum(hm[:,:,:18],axis=2), None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow("BG", bg)

        # nose = hm[:,:,0]
        # print "nose ", np.min(nose),np.max(nose)
        # noseNorm = cv2.normalize(nose, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # cv2.imshow("NOSE", noseNorm)

        cv2.imshow("HM", viz)

        peaks = find_peaks(hm, params['thre1'])
        bgr = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        viz = visualize(bgr, peaks, model_params["part_str"])
        cv2.imshow("VIZ", viz)


        k = cv2.waitKey(1)
        print 'time to predict() %.5f' % (toc - tic)

    cv2.destroyAllWindows()



