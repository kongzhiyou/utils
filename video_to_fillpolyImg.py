import cv2
import glob
import os
import numpy as np

root_path = '/Users/peter/data/wljr/data/sld/vedio/1号出库/*.*'
save_dir = '/Users/peter/data/wljr/data/sld/frame/'
video_list = glob.glob(root_path)
print(video_list)

'''
视频抽帧,并且增加了感兴趣区域，不感兴趣区域自动屏蔽
'''


def vedio_switch_to_image():
    for vedio_path in video_list:
        vedio_name = vedio_path.split('.')[0].split('/')[-1]
        vedio_dir = vedio_path.split('.')[0].split('/')[-2]
        print(vedio_dir)
        cap = cv2.VideoCapture(vedio_path)
        # print(cap.read())

        if cap.isOpened():
            success = True
            print('video open successfully')
        else:
            success = False
            print('video open failed')

        frame_index = 0
        interval = 30  # 每隔interval个帧抽一次
        img_list = []

        # fps = 30
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # videoWriter = cv2.VideoWriter(vedio_path[:-4] + '_test_1.avi', fourcc, fps,
        #                               (1920, 1080))  # 最后一个是保存图片的尺寸

        while success:
            success, frame = cap.read()
            if success and frame_index % interval == 0:
                # im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # im_gamma = cv2.convertScaleAbs(np.power(frame / 255, 0.5) * 255)
                # frame = cv2.flip(frame, -1)
                img_list.append(frame)

            frame_index += 1

        cap.release()

        save_path = os.path.join(save_dir, vedio_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for j, img in enumerate(img_list):
            # cv2.flip(img, -1)
            # 右上，右下，左下，左上
            all_poly_points1 = np.array([[1919, 400], [1919, 1079], [0, 1079], [0, 400]])
            # img = cv2.fillPoly(img, np.int32([all_poly_points1]), (0, 0, 0))
            all_poly_points2 = np.array([[1919, 0], [1919, 800], [800, 0]])
            # img = cv2.fillPoly(img,np.int32([all_poly_points2]),(0,0,0))
            cv2.imwrite(os.path.join(save_path, '{:05d}.jpg'.format(j)), img)


if __name__ == '__main__':
    vedio_switch_to_image()
