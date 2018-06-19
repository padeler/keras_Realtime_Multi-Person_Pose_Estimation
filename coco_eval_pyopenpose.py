import cv2
import numpy as np
import os
import glob
import json

import PyOpenPose as OP
from coco_eval import TO_COCO, visualize_results_list, coco_part_str

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]

def ToCOCOResultList(image_id, persons):
    res = []
    if persons is None:
        return res

    # print(persons.shape)
    for i in range(persons.shape[0]): # for each skeleton
        data = {"image_id": image_id, "category_id":1}
        sk = persons[i]

        # coco keypoints 2017 has 17 keypoints (no neck, different order)
        coco_joints = [0,0,0] * 17
        # print("SK: ", sk)
        score_sum = 0
        for idx,joint in enumerate(sk[TO_COCO]):
            x,y,score = joint
            if score>0:
                coco_joints[idx*3  ] = int(x)
                coco_joints[idx*3+1] = int(y)
                coco_joints[idx*3+2] = 2
                score_sum += score
            
        data["keypoints"] = coco_joints
        data["score"] = float(score_sum)
        res.append(data)

    return res



if __name__ == '__main__':

    dataset_path = "/media/storage/home/padeler/work/heatmaps/keras_Realtime_Multi-Person_Pose_Estimation/dataset/val2017"
    max_count = 1000

    download_heatmaps = False
    with_face = with_hands = False
    # op = OP.OpenPose((656, 368), (368, 368), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
    #                  download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)
    op = OP.OpenPose((-1, 368), (240, 240), (-1, -1), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, download_heatmaps)

    out_file = "pyopenpose_1k_val2017.json"
    
    val_set = glob.glob(dataset_path+os.sep+"*.jpg")
    val_set.sort()
    results = []

    print("Validation set size ",len(val_set))
    print('start loop...')


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
        op.detectPose(frame)

        # res = op.render(frame)
        # cv2.imshow("PyOpenpose result", res)
        
        persons = op.getKeypoints(op.KeypointType.POSE)[0]
        
        res = ToCOCOResultList(image_id, persons)
        viz = visualize_results_list(frame, res, coco_part_str)
        cv2.imshow("VIZ", viz)

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

    print("Saving PyOpenpose keypoint results to ", out_file)
    with open(out_file,"w") as f:
        json.dump(all_results, f)
    

    print("Done.")
    




