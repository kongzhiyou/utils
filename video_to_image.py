import cv2
import glob
import os

root_path = r'/Users/peter/data/wljr/test3.mp4'
save_path = r'/Users/peter/data/wljr'

video_list = glob.glob(root_path)

'''
视频抽帧
'''

def vedio_switch_to_image():
    i = 90
    for vedio_path in video_list:
        cap = cv2.VideoCapture(vedio_path)
        print(cap.read())

        if cap.isOpened():
            success = True
            print('video open successfully')
        else:
            success = False
            print('video open failed')

        frame_index = 0
        interval = 1   # 每隔interval个帧抽一次
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
        i = i+1

        if not os.path.exists(save_path+'/%d'%i):
            os.makedirs(save_path+'/%d'%i)

        for j, img in enumerate(img_list):
            # cv2.flip(img, -1)
            cv2.imwrite(save_path+'/'+str(i)+'/'+str(i)+'{:05d}.jpg'.format(j), img)

if __name__ == '__main__':
    vedio_switch_to_image()