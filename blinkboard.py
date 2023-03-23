import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import FaceMeshModule as fmm
#######################
signal = []
morse_list = []
########################
def preprocess(input_array, flik, leng):
    output = []
    current_chunk = []
    consecutive_zeros = 0

    for i in range(len(input_array)):
        if input_array[i] == 1:
            if consecutive_zeros <= flik:
                for j in range(i - consecutive_zeros, i):
                    input_array[j] = 1
            consecutive_zeros = 0
        elif input_array[i] == 0:
            consecutive_zeros += 1

    for num in input_array:
        if num == 1:
            current_chunk.append(num)
        elif current_chunk:
            if len(current_chunk) > leng:
                output.append(1)
            else:
                output.append(0)
            current_chunk = []
    if current_chunk:
        if len(current_chunk) > leng:
            output.append(1)
        else:
            output.append(0)
    return output
##############################################
MORSE_CODE = {
    "01": "a", "1000": "b", "1010": "c", "100": "d", "00": "e", "0010": "f",
    "110": "g", "0001": "h", "000": "i", "0111": "j", "101": "k", "0100": "l",
    "111": "m", "10": "n", "1111": "o", "0110": "p", "1101": "q", "010": "r",
    "0000": "s", "11": "t", "001": "u", "0101": "v", "011": "w", "1001": "x",
    "1011": "y", "1100": "z", "0011": " ", "1": "__backspace__", "00000": "\n", "000000": "__clearBoard__"
}
########################################################
cap = cv2.VideoCapture(1)

detector = fmm.FaceMeshDetector(minDetectionCon=0.65, maxFaces=1)

while True:
    success, img = cap.read()
    img_f, faces = detector.findFaceMesh(img, draw=False)

    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    if len(faces) != 0:
        lmList = faces[0]


        left_eye_landmarks = []
        right_eye_landmarks = []

        for i in [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]  :
            left_eye_landmarks.append(lmList[i])
        for i in [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  :
            right_eye_landmarks.append(lmList[i])

        cv2.polylines(img, [np.array(left_eye_landmarks, np.int32).reshape((-1,1,2))], True, (255, 0, 0))
        cv2.polylines(img, [np.array(right_eye_landmarks, np.int32).reshape((-1, 1, 2))], True, (0, 255, 0))


        # Modified Eye Aspect Ratio

        right_eye_EAR = 100*((((right_eye_landmarks[4][0] - right_eye_landmarks[12][0]) ** 2 +
                         (right_eye_landmarks[4][1] - right_eye_landmarks[12][1]) ** 2) ** 0.5) +
                        (((right_eye_landmarks[5][0] - right_eye_landmarks[11][0]) ** 2 +
                            (right_eye_landmarks[5][1] - right_eye_landmarks[11][1]) ** 2) ** 0.5) + (((right_eye_landmarks[3][0] - right_eye_landmarks[13][0]) ** 2 +
                            (right_eye_landmarks[3][1] - right_eye_landmarks[13][1]) ** 2) ** 0.5)) / ((((right_eye_landmarks[0][0] - right_eye_landmarks[8][0]) ** 2 +
                         (right_eye_landmarks[0][1] - right_eye_landmarks[8][1]) ** 2) ** 0.5) +
                        (((right_eye_landmarks[1][0] - right_eye_landmarks[7][0]) ** 2 +
                            (right_eye_landmarks[1][1] - right_eye_landmarks[7][1]) ** 2) ** 0.5) + (((right_eye_landmarks[9][0] - right_eye_landmarks[15][0]) ** 2 +
                            (right_eye_landmarks[9][1] - right_eye_landmarks[15][1]) ** 2) ** 0.5))

        left_eye_EAR = 100*((((left_eye_landmarks[4][0] - left_eye_landmarks[12][0]) ** 2 +
                         (left_eye_landmarks[4][1] - left_eye_landmarks[12][1]) ** 2) ** 0.5) +
                        (((left_eye_landmarks[5][0] - left_eye_landmarks[11][0]) ** 2 +
                            (left_eye_landmarks[5][1] - left_eye_landmarks[11][1]) ** 2) ** 0.5) + (((left_eye_landmarks[3][0] - left_eye_landmarks[13][0]) ** 2 +
                            (left_eye_landmarks[3][1] - left_eye_landmarks[13][1]) ** 2) ** 0.5)) / ((((left_eye_landmarks[0][0] - left_eye_landmarks[8][0]) ** 2 +
                         (right_eye_landmarks[0][1] - right_eye_landmarks[8][1]) ** 2) ** 0.5) +
                        (((left_eye_landmarks[1][0] - left_eye_landmarks[7][0]) ** 2 +
                            (left_eye_landmarks[1][1] - left_eye_landmarks[7][1]) ** 2) ** 0.5) + (((left_eye_landmarks[9][0] - left_eye_landmarks[15][0]) ** 2 +
                            (left_eye_landmarks[9][1] - left_eye_landmarks[15][1]) ** 2) ** 0.5))


        img = cv2.flip(img_f, 1)
        plot_img = np.zeros_like(img)


        if left_eye_EAR < 25:   # Write Mode
            cv2.putText(img, f'left eye: close', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            if right_eye_EAR <20:
                cv2.putText(img, f'right eye: close', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                signal.append(1)
            else:
                cv2.putText(img, f'right eye: open', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                signal.append(0)


        else:   # Display Mode
            cv2.putText(img, f'left eye: open', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            # print(signal)
            morse = ''.join(map(str, preprocess(signal, flik=1, leng= 4)))
            if morse in MORSE_CODE.keys():
                # print(morse)
                if morse == "1":
                    if(len(morse_list))>0:
                        morse_list.pop()
                elif morse == "000000":
                    morse_list = []
                else:
                    morse_list.append(morse)
            signal = []



        cv2.putText(img, f'left mear: {int(left_eye_EAR)}', (380, 420), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, f'right mear: {int(right_eye_EAR)}', (380, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
##################################################
    # Graph
    if len(signal) > 30:
        plt.plot(signal[-30:])
    else:
        plt.plot(signal)
    plt.ylim([-0.1, 1.1])
    plt.xlim([0, 30])
    plt.axhline(y=0.5, c="red")
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    plt.draw()
    plot = cv2.cvtColor(np.array(plt.gcf().canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plot = cv2.resize(plot, (img.shape[1], img.shape[0]//3))
    height1, height2 = img.shape[0], plot.shape[0]
    max_width = max(img.shape[1], plot.shape[1])
    cam = np.zeros((height1 + height2, max_width, 3), dtype=np.uint8)
    cam[:height1, :img.shape[1], :] = img
    cam[height1:, :plot.shape[1], :] = plot
    plt.close()
#####################################################
    text = []
    for morse_letter in morse_list:
        text.append(MORSE_CODE[morse_letter])
    text.append('_')
    if len(text)>0:
        y0, dy = 50, 50
        for i, line in enumerate(''.join(text).split('\n')):
            y = y0 + i * dy
            cv2.putText(imgCanvas, line, (20, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        # cv2.putText(imgCanvas, ''.join(text), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)


    cv2.imshow("Cam", cam)
    cv2.imshow("Text", imgCanvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
