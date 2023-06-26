import cv2
import numpy as np

def rotate_triangles():
    # create initial triangles
    img = np.zeros((500, 500, 3), np.uint8)
    triangles = []
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
              (0, 255, 255), (255, 0, 255), (128, 128, 128), (255, 255, 255)]
    thickness = 2
    for i in range(8):
        triangle = np.array([[np.random.randint(0, 500), np.random.randint(0, 500)],
                             [np.random.randint(0, 500), np.random.randint(0, 500)],
                             [np.random.randint(0, 500), np.random.randint(0, 500)]], np.int32)
        triangles.append(triangle)

    # set initial angles and rotation speeds for each triangle
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    rotation_speeds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # display animation
    while True:
        # clear previous frame
        img[:] = 0

        for i in range(8):
            # calculate centroid
            M = cv2.moments(triangles[i])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # rotate triangle around centroid
            angles[i] += rotation_speeds[i]
            R = cv2.getRotationMatrix2D((cx, cy), angles[i], 1)
            triangle_rotated = cv2.transform(np.array([triangles[i]]), R).astype(np.int32)

            # draw rotated triangle
            color = colors[i]
            cv2.drawContours(img, triangle_rotated, 0, color, thickness)

        # display frame and check for key press
        cv2.imshow('Rotating Triangles', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

rotate_triangles()
