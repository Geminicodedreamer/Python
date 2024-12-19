import cv2


START_TIME = 1
END_TIIME = 28800

vidcap = cv2.VideoCapture("C:\\Users\\DELL\\Desktop\\narun.avi")

fps = int(vidcap.get(cv2.CAP_PROP_FPS))
print(fps)

frameToStart = START_TIME*fps
print(frameToStart)
frameToStop = END_TIIME*fps
print(frameToStop)

vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)

print(vidcap.get(cv2.CAP_PROP_POS_FRAMES))

success, image = vidcap.read()

count = 0
while success and frameToStop >= count:
    if count % 1 == 0:
        cv2.imwrite("C:\\Users\\DELL\\Desktop\\action0\\image_%d.png" %
                    (count // 1), image)
        print('Process %dtypeerror: not all arguments converted during string formattingth seconds:' % int(
            count / 1), success)
    success, image = vidcap.read()
    count += 1


print("end!")
