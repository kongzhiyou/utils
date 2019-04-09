import cv2

'''
视频格式转换：avi->mp4
'''

vedio_path = r'E:\a_pinlan_data\box\all_1.avi'

def test_vedio(vedio_path):
    cap = cv2.VideoCapture(vedio_path)

    if cap.isOpened():
        success = True
    else:
        success = False

    frame_index = 0
    interval = 1
    img_list = []
    while success:
        success, frame = cap.read()
        if success and frame_index % interval == 0:
            img_list.append(frame)

        frame_index += 1

    cap.release()

    fps = 15
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG') avi格式
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#mp4格式
    videoWriter = cv2.VideoWriter(vedio_path[:-4] + '.mp4', fourcc, fps, (1280, 1024))  # 最后一个是保存图片的尺寸

    for image in img_list:
        videoWriter.write(image)

    videoWriter.release()


if __name__ == '__main__':
    test_vedio(vedio_path)

