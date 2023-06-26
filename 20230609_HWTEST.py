import cv2 as cv
import numpy as np 

fnames = []
fnames.append(r'D:\Lab\111-2\computer_vision_and_image_measurement\chessboard_practice\192739.jpg')
fnames.append(r'D:\Lab\111-2\computer_vision_and_image_measurement\chessboard_practice\192740.jpg')
fnames.append(r'D:\Lab\111-2\computer_vision_and_image_measurement\chessboard_practice\192741.jpg')

allCorners = np.zeros((3, 44, 2), dtype=float)

for i in range(3):
# Read image
    img = cv.imread(fnames[i], cv.IMREAD_GRAYSCALE)

    # Call findChessboardCorners
    w = 4
    h = 11
    psize = (w, h)
    params = cv.SimpleBlobDetector_Params()
    params.maxArea = 10e4
    params.minArea = 10
    params.minDistBetweenBlobs = 5
    blobDetector = cv.SimpleBlobDetector_create(params)
    ret1 = cv.findCirclesGrid(img, (w, h), cv.CALIB_CB_ASYMMETRIC_GRID, blobDetector, None)
    if ret1[0] == True:
        corners = ret1[1].reshape(-1, 2)
        allCorners[i, :, :] = corners
    else:
        print("Error: Could not find all corners.")

# Call drawChessboardCorners
    imgWithCorners = img.copy()
    imgWithCorners = cv.cvtColor(imgWithCorners, 
                                 cv.COLOR_GRAY2BGR)
    cv.drawChessboardCorners(imgWithCorners, 
                             psize, 
                             corners,
                             ret1[0])

    # Resize the image 
    myScreen = (1280, 720)
    ratio_x = myScreen[0] / imgWithCorners.shape[1]
    ratio_y = myScreen[1] / imgWithCorners.shape[0]
    ratio = min(ratio_x, ratio_y)
    dsize = (int(ratio * imgWithCorners.shape[1]), 
             int(ratio * imgWithCorners.shape[0]))
    imgWithCornersResized = cv.resize(
        imgWithCorners, dsize)
    
    # display
    winName = "Found corners of image %d" % (i + 1)
    cv.imshow(winName, imgWithCornersResized)
#    ikey = cv.waitKey(0)
    
ikey = cv.waitKey(0)
cv.destroyAllWindows()
    
# Call cabibrateCamera()
objPoints = np.zeros((3, 44, 3), dtype=float)
dx = 19.5
dy = 39
for i in range(h):
    if i % 2 == 0:
        for j in range(w):
            objPoints[0, i * w + j, 0] = i * dx
            objPoints[0, i * w + j, 1] = j * dy
            objPoints[0, i * w + j, 2] = 0.
    else:
        for j in range(w):
            objPoints[0, i * w + j, 0] = i * dx
            objPoints[0, i * w + j, 1] = j * dy + 0.5 * dy
            objPoints[0, i * w + j, 2] = 0.
for i in range(1, 3):
    objPoints[i, :, :] = objPoints[0, :, :]
imgSize = (img.shape[1], img.shape[0])
cmat = np.array([[2000,0,1000],[0,2000,1000],[0,0,1.]])
dvec = np.zeros((1,8))
ret2 = cv.calibrateCamera(objPoints.astype(np.float32), 
                          allCorners.astype(np.float32), imgSize, cmat, dvec)
