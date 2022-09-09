import sys
import numpy as np
import cv2 as cv

def main(argv):

    window_name = ('Prewitt - Simple Edge Detector')
    ddepth = cv.CV_16S

    if len(argv) < 1:
        print('Not enough parameters')
        print('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    

    src = cv.imread(argv[0], cv.IMREAD_COLOR)


    if src is None:
        print('Error opening image: ' + argv[0])
        return -1

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    src = cv.GaussianBlur(src, (3, 3), 0)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    grad_x = cv.filter2D(gray, ddepth, kernelx)
    grad_y = cv.filter2D(gray, ddepth, kernely)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv.imshow(window_name, grad)
    cv.waitKey(0)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])