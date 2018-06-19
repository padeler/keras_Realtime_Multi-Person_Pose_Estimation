import cv2
import math
import time
import numpy as np
import util
import os
import glob
from config_reader import config_reader
import json

from demo_image import predict, skeletonize, visualize, colors



coco_part_str = [u'nose', u'leye', u'reye', u'lear', u'rear', u'lshou', u'rshou', u'lelbow', u'relbow', 
                u'lwrist', u'rwrist', u'lhip', u'rhip', u'lknee', u'rknee', u'lankle', u'rankle']

TO_COCO = [0, 15,14,17,16, 5,2,6,3,7,4, 11,8,12,9,13,10]

def ToCOCOResultList(image_id, skeletons, peaks):
    res = []
    # print("PEAKS:\n",peaks)
    for i in range(skeletons.shape[0]): # for each skeleton
        data = {"image_id": image_id, "category_id":1}
        # len(sk) == 20 ([18 joints indices] + [score, valid_joint_count])
        sk = skeletons[i]

        # coco keypoints 2017 has 17 keypoints (no neck, different order)
        coco_joints = [0,0,0] * 17
        # print("SK: ", sk)
        for p, idx in enumerate(sk[TO_COCO]):
            idx = int(idx)
            if idx!=-1:
                # print("IDX",idx)
                X = peaks[idx, 0]
                Y = peaks[idx, 1]
                coco_joints[p*3  ] = int(X)
                coco_joints[p*3+1] = int(Y)
                coco_joints[p*3+2] = 2
            
        data["keypoints"] = coco_joints
        data["score"] = sk[18] # score index TODO check for correct value.
        
        # print("Indices for Skeleton ",i)
        # print(sk)
        res.append(data)
    


    return res


def visualize_results_list(canvas,res, part_str=None):
    for idx,data in enumerate(res):
        joints = data["keypoints"]
        color = colors[idx%len(colors)]
        for i in range(17):
            p = i*3
            pos = tuple(joints[p:p+2])
            if joints[p+2]==2:
                cv2.circle(canvas, pos, 4, color, thickness=-1)
                if part_str is not None:
                    cv2.putText(canvas, part_str[i], pos, 0, 0.5, colors[i])

    
    return canvas

if __name__ == '__main__':

    dataset_path = "/media/storage/home/padeler/work/heatmaps/keras_Realtime_Multi-Person_Pose_Estimation/dataset/val2017"
    max_count = 1000
    thre1 = 0.01
    thre2 = 0.005
    sigma = 3

    params, model_params = config_reader()

    from vnect_model import get_testing_model
    # from model import get_testing_model
    keras_weights_file = "training/vnect_pafs_weights.h5"
    # keras_weights_file = "model/keras/model.h5"

    # out_file = "openpose_1k_r2_val2017.json"
    out_file = "vnect_pafs_1k_8.1k.json"
    
    val_set = glob.glob(dataset_path+os.sep+"*.jpg")
    val_set.sort()
    results = []

    print("Validation set size ",len(val_set))


    # load model
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    print('start loop...')
    # load config

    delay = {
        True: 0,
        False: 5,
    }
    paused = True


    k = 0
    count=0
    skel_count=0
    images_with_skel=0
    all_results = []

    for fname in val_set:
        image_id = int(os.path.os.path.basename(fname)[:-4])

        frame = cv2.imread(fname)

        # generate image with body parts
        hm, pafs, img = predict(frame, params, model_params, model)

        # print("Source Image shape",frame.shape,"HM shape ", hm.shape, " PAFs shape", pafs.shape)
        
        # subset holds the skeletons, candidate holds the peaks
        all_peaks, subset, candidate = skeletonize(hm, pafs, img.shape, thre1, thre2, sigma=sigma)

        # viz = visualize(img.copy(), all_peaks, subset, candidate)
        # cv2.imshow("VIZ", viz)
        # showPAFs(pafs)
        
        res = ToCOCOResultList(image_id, subset, candidate)
        canvas = visualize_results_list(img, res, coco_part_str)

        cv2.imshow("Canvas", canvas)

        if len(res)>0:
            skel_count += len(res)
            images_with_skel += 1
        
        print("[%d/%d] Processed image id %d. People found %d" % (count,max_count,image_id, len(res)))

        all_results += res
            
        count+=1

        k = cv2.waitKey(delay[paused])
        if k&0xFF==ord('q') or count==max_count:
            break
        if k&0xFF==ord('p'):
            paused = not paused

    cv2.destroyAllWindows()


    print("Processed %d images. Total skeletons %d, images with at least one skeleton %d"%(count, skel_count, images_with_skel))

    print("Saving keypoint thre1,2,sigma = {%.3f,%.4f, %d} results to "%(thre1,thre2,sigma), out_file)
    with open(out_file,"w") as f:
        json.dump(all_results, f)
    

    print("Done.")
    




