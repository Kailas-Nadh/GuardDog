import numpy as np
import cv2
import picamera
import picamera.array
import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarmsound.wav')

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize the camera and video stream
camera = picamera.PiCamera(resolution=(640, 480), framerate=15)
camera.start_preview()

# Allow the camera to warm up
time.sleep(2)

# Video writer initialization
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

try:
    with picamera.array.PiRGBArray(camera) as stream:
        for frame in camera.capture_continuous(stream, format='bgr', use_video_port=True):
            # Capture frame-by-frame
            image = frame.array

            # Detect people in the image
            boxes, weights = hog.detectMultiScale(image, winStride=(8,8))

            # Draw bounding boxes on the image
            for (xA, yA, xB, yB) in boxes:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

            # Check if any people are detected and play sound if so
            if len(boxes) > 0:
                alarm_sound.play()

            # Write the frame to the output file
            out.write(image)

            # Display the resulting frame
            cv2.imshow('frame', image)

            # Clear the stream for the next frame
            stream.seek(0)
            stream.truncate()

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    # Release resources
    camera.stop_preview()
    out.release()
    cv2.destroyAllWindows()

