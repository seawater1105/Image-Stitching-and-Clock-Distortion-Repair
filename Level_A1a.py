from tkinter import *
from tkinter import filedialog
from matplotlib import pyplot as plt
import cv2
import numpy as np

# 비교할 두 개의 이미지를 불러옴.
root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img1 = cv2.imread(path)
root.withdraw()

root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img2 = cv2.imread(path)
root.withdraw()

MIN_MATCH_COUNT = 4
sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 들어온 이미지를 순서대로 img1, img2로 하여 Homography matrix 찾기.(img2를 기준으로 img1 이동)
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.3*n.distance: # 예시에 따라 0.5
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    # 위 과정을 통해 img2위에 img1이 겹치는 부분 만큼의 line이 그려짐.
    # cv2.imshow("img2(+line)", img2), plt.show()

    # 상태 구분하기
    width = (img2.shape[1]+img1.shape[1])//2 # 입력한 두 이미지의 평균 width
    height = (img2.shape[0]+img1.shape[0])//2 # 입력한 두 이미지의 평균 height
    # 두 이미지의 크기가 동일하지 않음.
    # 따라서 width와 height중 더 긴 쪽의 1/20만큼을 img2의 상하좌우에서 제외하고 상태 판단.
    m = max(width,height)//20
    for x in range(m, img2.shape[1]-m): # 좌,우에서 각각 m씩을 제거한 img2의 width(열)
        for y in range(m, img2.shape[0]-m): # 상,하에서 각각 m씩을 제거한 img2의 height(행)
            x1 = x<img2.shape[1]/2 # x가 img2의 width의 반보다 작을 때(width 앞범위)
            x2 = img2.shape[1]/2<x # x가 img2의 width의 반보다 클 때(width의 뒷범위)
            y1 = y<img2.shape[0]/2 # y가 img2의 height의 반보다 작을 때(height 앞범위)
            y2 = img2.shape[0]/2<y # y가 img2의 height의 반보다 클 때(heignt 뒷범위)

            # img2[m:img2.shape[0또는 1]-m]을 계산하려면 시간이 오래 걸림.
            # 따라서 2*m(m:m*2 또는 -m*2:-m)정도의 길이를 슬라이싱해 사용.
            # np.sum을 통해 img2의 특정 범위 내 픽셀의 채널 값의 합을 구함.
            # axis=1을 하면 채널의 각 열에 있는 값들을 다 더한 값을 나타내 줌.
            # 위에서 img2위에 img1이 겹치는 부분의 채널의 값을 출력해본 결과 (255,0,0)이 나옴.
            # 따라서 채널의 각 열에 있는 값들을 다 더하면 255가 되고, 이 255가 연속으로 나오는 부분을 통해 상태 판단.
            arr3 = np.sum(img2[m:m*2, x, :], axis=1)  # x열에서의 (m:m*2)각 행의 채널 값의 합 (height의 앞)
            arr4 = np.sum(img2[-m*2:-m, x, :], axis=1)  # x열에서의 (-m*2:-m) 각 행의 채널 값의 합 (height의 뒤)
            arr5 = np.sum(img2[y, m:m*2, :], axis=1)  # y행에서의 (m:m*2)각 열의 채널 값의 합 (width의 앞)
            arr6 = np.sum(img2[y, -m*2:-m, :], axis=1)  # y행에서의 (-m*2:-m)각 열의 채널 값의 합 (width의 뒤)
            # 세로로 긴 line이 있는 경우(좌,우 or 우,좌)
            if np.all(arr3 == 255) & np.all(arr4 == 255): # arr3과 arr4의 모든 값이 255이면?
                if x1: # 255가 나온 곳의 x가 width/2보다 작으면?
                    print('img1 : 좌, img2 = 우') # 위 조건을 만족하는 경우 좌,우
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
                elif x2: # 255가 나온 곳의 x가 width/2보다 크면?
                    print('img1 : 우, img2 = 좌')  # 위 조건을 만족하는 경우 우,좌
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
            # 가로로 긴 line이 있는 경우(상,하 or 하,상)
            elif np.all(arr5 == 255) & np.all(arr6 == 255): # arr5와 arr6의 모든 값이 255이면?
                if y1:  # 255가 나온 곳의 y가 height/2보다 작으면?
                    print('img1 : 상, img2 = 하') # 위 조건을 만족하는 경우 상,하
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
                elif y2 :  # 255가 나온 곳의 y가 height/2보다 크면?
                    print('img1 : 하, img2 = 상') # 위 조건을 만족하는 경우 하,상
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
            # 왼쪽 상단이나 오른쪽 하단에 네모의 모서리 부분이 존재하는 경우
            elif np.all(arr3 == 255) & y1: # arr3의 모든 값이 255이면서 y가 height/2보다 작으면?
                if np.all(arr5 == 255) & x1: # arr5의 모든 값이 255이면서 x가 width/2보다 작으면?
                    print('img1 : 좌상, img2 = 우하') # 위 조건을 만족하는 경우 좌상,우하
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
                elif np.all(arr6 == 255) & x2: #arr6의 모든 값이 255이면서 x가 width/2보다 크면?
                    print('img1 : 우상, img2 = 좌하') # 위 조건을 만족하는 경우 우상,좌하
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
            # 오른쪽 상단이나 왼쪽 하단에 네모의 모서리 부분이 존재하는 경우
            elif np.all(arr4 == 255) & y2: # arr4의 모든 값이 255이면서 y가 height/2보다 크면?
                if np.all(arr6 == 255) & x2: # arr6의 모든 값이 255이면서 x가 width/2보다 크면?
                    print('img1 : 우하, img2 = 좌상') # 위 조건을 만족하는 경우 우하, 좌상
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
                elif np.all(arr5 == 255) & x1:
                    print('img1 : 좌하, img2 = 우상') # 위 조건을 만족하는 경우 좌하, 우상
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
        else : continue # 지정한 범위 내에서 특정 조건이 충족되지 않은 경우.
                        # 즉 if, elif문 중 어느 하나도 print되지 않았을 때? 바깥 for문으로 다시 나감.
        break # 내가 원하는 것을 찾고 난 후(어떤 것이 print된 후)에는 그냥 종료
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

img3 = cv2.warpPerspective(img1, M, (img1.shape[1]+img2.shape[1], img1.shape[0]+img2.shape[0]))
img3[0:img2.shape[0], 0:img2.shape[1]] = img2
# cv2.imshow("Stiching12", img3), plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()