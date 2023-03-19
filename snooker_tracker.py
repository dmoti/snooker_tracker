import os
import sys
import cv2
import numpy as np


START_FRAME = 0
END_FRAME = 2e10
NO_ACTIVITY_PIXELS_THRESHOLD = 40
BALL_MIN_SIZE = 10
BALL_MAX_SIZE = 35
MIN_RATIO=0.5
MAX_RATIO=1.5
BALL_MOVEMENT_THRESHOLD = 25
IERSECTION_PIXEL = 128
WHITE_PIXEL = 255
WHITE_MEDIAN=np.array([80, 100, 90])
FRAMES_TO_WAIT_WHITE_BALL_STATE = 5
    
visualize_debug = False
global_median_pixel = None

# compute a global median pixel on the first frame and use it as background
def remove_background(img):
    global global_median_pixel
    
    THRESHOLD = 35
    KERNEL = np.ones((5,5), np.uint8)
    
    if global_median_pixel is None:
        global_median_pixel = np.array([np.median(img[:,:,0]), np.median(img[:,:,1]), np.median(img[:,:,2])]).astype(np.float32)

    img = np.subtract(img.astype(np.float32), global_median_pixel)

    img[img < THRESHOLD] = 0
    img[img >= THRESHOLD] = 255

    img = img.astype(np.uint8)

    img = cv2.erode(img, kernel=KERNEL, iterations=1)
    img = cv2.dilate(img, kernel=KERNEL, iterations=1)

    return img 

# read the frame, apply mask and remove background
def read_frame(capture_dev, mask_3c):
    ret, frame = capture_dev.read()
    if ret == False:
        return ret, None, None
    
    frame_rgb_masked = np.multiply(frame, mask_3c)

    frame_rgb_masked_without_background = remove_background(frame_rgb_masked)
    frame_gray_masked_without_background = cv2.cvtColor(frame_rgb_masked_without_background, cv2.COLOR_BGR2GRAY)
    frame_gray_masked_without_background[frame_gray_masked_without_background > 0] = 255
    
    return ret, frame, frame_gray_masked_without_background

def main(video_filename, mask_filename):
    frame_count=0
    detect_white_ball_flag = True
    white_ball_hit_detected = 0
    
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_filename)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file")
        return

    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
    mask_3c = cv2.merge((mask, mask, mask)).astype(np.bool_)

    prev_frame_masked = None

    if START_FRAME > 0:
        cap.set(1, START_FRAME)
        frame_count = START_FRAME
            
    ################################################################################################## main loop
    while (cap.isOpened()) and (frame_count < END_FRAME):
        frame_count += 1
        good_rects = 0
        
        ret, current_frame, current_frame_masked = read_frame(cap, mask_3c)
        if ret != True:
            break

        frame_to_show = current_frame
        
        # first frame
        if prev_frame_masked is None:
            prev_frame_masked = current_frame_masked
            
        current_minus_prev = np.subtract(current_frame_masked.astype(np.float32), prev_frame_masked.astype(np.float32))
        
        # get the intersection between prev and current image, and mark it with 128
        current_prev_intersection = np.bitwise_and(current_frame_masked, prev_frame_masked)
        current_prev_intersection[current_prev_intersection > 0] = IERSECTION_PIXEL
        current_minus_prev[current_minus_prev < 0] = 0
        current_minus_prev = current_minus_prev.astype(np.uint8)

        # if there is no activity skip the algorithm
        none_diff_zeros = np.count_nonzero(current_minus_prev)
        if none_diff_zeros < NO_ACTIVITY_PIXELS_THRESHOLD:
            continue
                
        # merge new movement and intersection
        current_minus_prev = np.bitwise_or(current_minus_prev, current_prev_intersection)
        
        # save the current to be the next iteration's prev
        prev_frame_masked = current_frame_masked.copy()

        # we look for contours on the new movement and the intersection        
        contours, _ = cv2.findContours(current_minus_prev, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # only for display purposes
        current_minus_prev_3c = cv2.merge((current_minus_prev, current_minus_prev, current_minus_prev))

        for contour in contours:

            # get the xmin, ymin, width, and height coordinates from the contours
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # limit the size
            if (w < BALL_MIN_SIZE) or (w > BALL_MAX_SIZE) or (h < BALL_MIN_SIZE) or (h > BALL_MAX_SIZE):
                if visualize_debug:
                    cv2.putText(current_minus_prev_3c, "SIZE " + str(w) + '_'+ str(h), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (209, 80, 0, 255), 1)
                continue

            # we are looking for square bounded moving objects
            if (w/h < MIN_RATIO) or (w/h > MAX_RATIO):
                if visualize_debug:
                    ratio = "{:.2f}".format(w/h)
                    cv2.putText(current_minus_prev_3c, "SQR " + ratio, (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (209, 80, 0, 255), 1)
                continue

            #  count how many white pixels (movement) in each box
            box = current_minus_prev[y:y+h, x:x+w]
            count = np.sum(box == WHITE_PIXEL)
            if count < BALL_MOVEMENT_THRESHOLD:
                if visualize_debug:
                    cv2.putText(current_minus_prev_3c, "MOV " + str(count), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (209, 80, 0, 255), 1)
                continue
            
            good_rects+=1
            
            # detecting white ball movement, onece detected I allow to detect other balls until there is no movement
            if detect_white_ball_flag:
                box = current_frame[y:y+h, x:x+w]
                box_median_pixel = np.array([np.median(box[:,:,0]), np.median(box[:,:,1]), np.median(box[:,:,2])]).astype(np.float32)
                if visualize_debug:
                    cv2.putText(current_frame, str(box_median_pixel) + str(count), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (209, 80, 0, 255), 1)
                
                if np.all(box_median_pixel > WHITE_MEDIAN):
                    white_ball_hit_detected = 1
                    cv2.rectangle(current_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                else:
                    if white_ball_hit_detected >= 1:
                        cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
            if visualize_debug:
                cv2.rectangle(current_minus_prev_3c, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # if white ball movement detected, if no good_rects was detected we reset the white ball movement detection
        if detect_white_ball_flag:
            if white_ball_hit_detected > 0 and good_rects == 0:
                white_ball_hit_detected += 1
                if white_ball_hit_detected > FRAMES_TO_WAIT_WHITE_BALL_STATE:
                    white_ball_hit_detected = 0
 
        if visualize_debug:
            cv2.putText(current_frame, str(frame_count) + "_" + str(none_diff_zeros) + '_' + str(good_rects) + '_' + str(white_ball_hit_detected), (10,50), cv2.FONT_HERSHEY_PLAIN, 2, (209, 80, 0, 255), 2)
            frame_to_show = np.concatenate((current_minus_prev_3c, current_frame), axis=1)

        cv2.imshow('Frame', frame_to_show)
            
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        # Press P to pause
        if cv2.waitKey(25) & 0xFF == ord('p'):
            cv2.waitKey(-1)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: python", sys.argv[0], "<video filename> <mask filename>")
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Video file", sys.argv[1], "does not exist.", )
        sys.exit(1)

    if not os.path.exists(sys.argv[2]):
        print("Mask file", sys.argv[2], "does not exist.")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
