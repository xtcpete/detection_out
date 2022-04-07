"""
test script for mobilenet, this script can also used to collect mis-predicted pictures for further training
after running the script, press n to enable picture collection and follow the prompt
a directory named by entered string will be created, pictures will be store under that directory
press s to stop pictures collection
"""

import cv2
from detection_out import *
import os

model_def = ' '  # file path of prototxt
model_weights = ' '  # file path of caffemodel
label_path = ' '  # txt file for labels
n_classes = '' # the number of different classes the model can predict
root = os.path.dirname(os.path.abspath(__file__))

with open(label_path, "r") as file:
    str = file.read()
    CLASSES = str.split('\n')
    CLASSES.pop()
    file.close()
net = cv2.dnn.readNet(model_def, model_weights)  # loading the model
print(CLASSES)

cap = cv2.VideoCapture(1)  # create the video flow
cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
name = None
img_count = 0  # index for picture

if __name__ == '__main__':

    while True:
        ret, image = cap.read()
        image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (416, 416)), 1/255.0, (416, 416), (255.0, 255.0, 255.0))
        # creating blob
        net.setInput(image_blob)
        model_output = net.forward()
        # inference
        output = model_output
        # initialized blob for further processing
        result_blob = blob(output)
        result = set_up_detection_out(result_blob, num_classes=n_classes)
        # calculating the result
        for i in range(result.shape[2]):
            if result[0][0][i][2] >= 0:
                w = image.shape[1]
                h = image.shape[0]
                if result.shape[3] > 0:
                    left = result[0][0][i][3] * w
                    top = result[0][0][i][4] * h
                    right = result[0][0][i][5] * w
                    bot = result[0][0][i][6] * h
                    score = result[0][0][i][2]
                    label = result[0][0][i][1]
                    try:
                        label = '{:s} '.format(CLASSES[int(label)])
                        print(score, label)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        size = cv2.getTextSize(label, font, 0.5, 0)[0]
                        if label != name and name != None:
                            if not os.path.exists(root + '/' + name):
                                os.mkdir(root + '/' + name)
                            cv2.imwrite("./{}/{}_{}.png".format(name, name, img_count), image)
                            img_count += 1
                            # save the picture if predicted label does not match the entered label
                        cv2.rectangle(image, pt1=(int(left), int(top)), pt2=(int(right), int(bot)), color=(0, 255, 0),
                                      thickness=3)
                        cv2.putText(image, label, (int(left - 20), int(top + size[1] - 20)), font, 2, (0, 0, 255),
                                    1)
                        # add the text to the video flow
                    except:
                        pass

        cv2.imshow("Image", image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        elif key == ord('n'):
            name = input("Please enter the correct label：")
        elif key == ord('s'):
            name = None
    cap.release()
    cv2.destroyAllWindows()