import cv2 as cv
import argparse

from mog.mog import MOG

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods '
                                                 'provided by OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.',
                        default='data/scene_1.1_reg.avi')
    parser.add_argument('--show', type=str, help='Boolean which says if to show the result.',
                        default=False)
    args = parser.parse_args()

    backSub = MOG()

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
    if not capture.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        if args.show:
            cv.imshow('Frame', frame)
            cv.imshow('FG Mask', fgMask)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
