from tkinter import *
from tkinter import filedialog
from matplotlib import pyplot as plt
import cv2
import numpy as np

# 스티칭 할 이미지 네 개를 불러와 저장
root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img1 = cv2.imread(path)
root.withdraw()

root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img2 = cv2.imread(path)
root.withdraw()

root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img3 = cv2.imread(path)
root.withdraw()

root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img4 = cv2.imread(path)
root.withdraw()

# img1이 img3보다 왼쪽 이미지를 더 많이 포함할 수 있으니 img1 왼쪽을 살짝 잘라서 img3 왼쪽에 까만 부분이 나오는 것 방지.
# 또한 img1이 img2,3,4를 스티칭 한 것보다 윗쪽 이미지를 더 많이 포함할 수 있으니
# img1의 윗쪽도 살짝 잘라 img2,3,4를 스티칭 한 것의 위에서 검정 부분 나오는 것 방지.
aimg1 = img1.shape[1]*1//100
bimg1 = img1.shape[0]*1//100
Mm = np.float32([[1,0,-aimg1], [0,1,-bimg1]])
img1 = cv2.warpAffine(img1, Mm, (img1.shape[1]-aimg1,img1.shape[0]-bimg1))
# cv2.imshow("img1", img1), plt.show()

# img2와 img4를 매칭할 때 img2가 오른쪽 뷰를 더 많이 가져 img4의 오른쪽에 까만 부분이 생기는 것을
# 방지하기 위해 img2의 오른쪽 부분을 약간(1/50) 자름.
# (이후 밑에서 img2,4를 스티칭 할 때 width를 img2의 width 크기로 스티칭)
aimg2 = img2.shape[1]*1//100
Mimg2 = np.float32([[1,0,-aimg2], [0,1,0]])
img2 = cv2.warpAffine(img2, Mimg2, (img2.shape[1]-aimg2, img2.shape[0]))
# cv2.imshow("img2", img2), plt.show()

# img24와 img3은 shift된 후 매칭 됨. => img24_shift, img3_shift
# 따라서 img3의 경우 shift이전과 width는 동일하지만 height가 커지고 그만큼 윗 부분에 까만 부분을 가지고 매칭이 됨.
# 이때 img3의 width가 img1의 width보다 크다면 최종 스티칭 후 img1의 오른쪽에 까만 부분이 생길 수 있음.
# 이를 방지하기 위해서 img3의 오른쪽 부분을 약간 자름.
# 또한 img3이 img4보다 아래쪽 뷰를 더 많이 가져 img4의 아래에 까만 부분이 생기는 것을
# 방지하기 위해서 img3의 아래 부분도 약간 자름.
# (이후 밑에서 img2,3,4를 스티칭 할 때 height를 img3의 height 크기로 스티칭)
aimg3 = img3.shape[1]*1//100
bimg3 = img3.shape[0]*1//100
Mimg3 = np.float32([[1,0,0], [0,1,0]])
img3 = cv2.warpAffine(img3, Mimg3, (img3.shape[1]-aimg3, img3.shape[0]-bimg3))
# cv2.imshow("img3", img3), plt.show()

MIN_MATCH_COUNT = 4
sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

## img2와 img4 스티칭 => dst24 생성
kp4, des4 = sift.detectAndCompute(img4, None)
kp2, des2 = sift.detectAndCompute(img2, None)

matches2 = flann.knnMatch(des4, des2, k=2)

good2 = []
for m, n in matches2:
    if m.distance < 0.3*n.distance:
        good2.append(m)
if len(good2)>MIN_MATCH_COUNT:
    src_pts2 = np.float32([kp4[m.queryIdx].pt for m in good2]).reshape(-1, 1, 2)
    dst_pts2 = np.float32([kp2[m.trainIdx].pt for m in good2]).reshape(-1, 1, 2)
    M2, mask2 = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 5.0)
    matchesMask2 = mask2.ravel().tolist()
    h2, w2 = img4.shape[:2]
    pts2 = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape(-1, 1, 2)
    dst2 = cv2.perspectiveTransform(pts2, M2)
else:
    print("Not enough matches are found - %d/%d" % (len(good2), MIN_MATCH_COUNT))
    matchesMask2 = None

width2 = img2.shape[1] # img2의 width를 img24의 width로 사용.
height2 = (img2.shape[0]+img4.shape[0])*9//10 # img2,4의 겹치는 부분 만큼의 크기를 대충 제거(까만 부분 최소화)
# img4를 img2에 대하여 M2를 통해 이동
dst24 = cv2.warpPerspective(img4, M2, (width2, height2))
# M2에따라 이동한 img1위에 img2를 넣음.
dst24[0:img2.shape[0], 0:img2.shape[1]] = img2
# cv2.imshow("dst24", dst24), plt.show()

# dst24와 img3을 그냥 스티칭하면 img3을 추가하는 과정에서 img2 부분이 음수 부분으로 가면서 짤리게 됨.
# 따라서 구한 dst24와 img3을 이동 후 스티칭 (이후 M3 구함)
# 이때 img3와 dst24가 합쳐질 때 잘리는 부분이 dst24의 윗부분이므로 아래쪽으로 img3의 height 정도 이동.
h_img3 = img3.shape[0]
M1 = np.float32([[1,0,0], [0,1,h_img3]])
dst24_shift= cv2.warpAffine(dst24, M1, (width2, height2+h_img3))
img3_shift= cv2.warpAffine(img3, M1, (img3.shape[1], img3.shape[0]+h_img3))
# cv2.imshow("dst24_shift", dst24_shift), plt.show()
# cv2.imshow("img3_shift", img3_shift), plt.show()

## dst24_shift(img2+img4)와 img3_shift 스티칭 => dst234_shift 생성
kp24, des24 = sift.detectAndCompute(dst24_shift, None)
kp3, des3 = sift.detectAndCompute(img3_shift, None)

matches3 = flann.knnMatch(des24, des3, k=2)

good3 = []
for m, n in matches3:
    if m.distance < 0.3*n.distance:
        good3.append(m)
if len(good3)>MIN_MATCH_COUNT:
    src_pts3 = np.float32([kp24[m.queryIdx].pt for m in good3]).reshape(-1, 1, 2)
    dst_pts3 = np.float32([kp3[m.trainIdx].pt for m in good3]).reshape(-1, 1, 2)
    M3, mask3 = cv2.findHomography(src_pts3, dst_pts3, cv2.RANSAC, 5.0)
    matchesMask3 = mask3.ravel().tolist()
    h3, w3 = dst24_shift.shape[:2]
    pts3 = np.float32([[0, 0], [0, h3-1], [w3-1, h3-1], [w3-1, 0]]).reshape(-1, 1, 2)
    dst3 = cv2.perspectiveTransform(pts3, M3)
else:
    print("Not enough matches are found - %d/%d" % (len(good3), MIN_MATCH_COUNT))
    matchesMask3 = None

width3 = (img3_shift.shape[1]+dst24_shift.shape[1])*9//10 # 이미지 3의 width과 dst24의 width의 합보다 조금 작게.
height3 = img3_shift.shape[0] # img3 밑에 까만 부분이 생기는 것을 막기위해 height는 img3_shift와 동일하게 설정.
# dst24_shift를 img3_shift에 대하여 M3를 통해 이동
dst234_shift = cv2.warpPerspective(dst24_shift, M3, (width3, height3))
# 이동한 dst234_shift위에 img3_shift를 넣음.
dst234_shift[0:img3_shift.shape[0], 0:img3_shift.shape[1]] = img3_shift
# cv2.imshow("dst234_shift", dst234_shift), plt.show()

## dst234_shift(img2+img3+img4)와 img1을 스티칭 => dst1234 생성
kp234, des234 = sift.detectAndCompute(dst234_shift, None)
kp1, des1 = sift.detectAndCompute(img1, None)

matches4 = flann.knnMatch(des234, des1, k=2)

good4 = []
for m, n in matches4:
    if m.distance < 0.3*n.distance:
        good4.append(m)
if len(good4)>MIN_MATCH_COUNT:
    src_pts4 = np.float32([kp234[m.queryIdx].pt for m in good4]).reshape(-1, 1, 2)
    dst_pts4 = np.float32([kp1[m.trainIdx].pt for m in good4]).reshape(-1, 1, 2)
    M4, mask4 = cv2.findHomography(src_pts4, dst_pts4, cv2.RANSAC, 5.0)
    matchesMask4 = mask4.ravel().tolist()
    h4, w4 = dst234_shift.shape[:2]
    pts4 = np.float32([[0, 0], [0, h4-1], [w4-1, h4-1], [w4-1, 0]]).reshape(-1, 1, 2)
    dst4 = cv2.perspectiveTransform(pts4, M4)
else:
    print("Not enough matches are found - %d/%d" % (len(good4), MIN_MATCH_COUNT))
    matchesMask4 = None

width4 = img1.shape[1]+img2.shape[1] # 이미지 1,2의 width를 합한 값을 최종 width로 사용
height4 = img1.shape[0]+img3.shape[0] # 이미지 1,3의 height를 합한 값을 최종 height로 사용
# dst234_shift를 img1에 대하여 M4를 통해 이동
dst1234 = cv2.warpPerspective(dst234_shift, M4, (width4, height4))
# 이동한 dst1234위에 img1을 넣음.
dst1234[0:img1.shape[0], 0:img1.shape[1]] = img1
cv2.imshow("dst1234", dst1234), plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()