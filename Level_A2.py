from tkinter import *
from tkinter import filedialog
from matplotlib import pyplot as plt
import cv2
import numpy as np

# 스티칭 할 이미지 네 개를 불러와 저장
root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img11 = cv2.imread(path)
root.withdraw()

root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img22 = cv2.imread(path)
root.withdraw()

root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img33 = cv2.imread(path)
root.withdraw()

root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img44 = cv2.imread(path)
root.withdraw()

# 최종적으로 기존 순서대로 정리한 후 img1,2,3,4에 넣어주기 위해 우선 None으로 둠.
img1 = None
img2 = None
img3 = None
img4 = None

# 나중에 순서를 찾은 후 스티칭 하기 위해 처음 불러온 사진은 그대로 두고,
# 불러온 원본 사진을 따로 copy해서 img1111에 저장해 둔 후 순서를 찾기위한 과정에서 사용.
img1111 = img11.copy()
img2222 = img22.copy()
img3333 = img33.copy()
img4444 = img44.copy() # 사실 img4444는 없어도 상관없는 값

MIN_MATCH_COUNT = 4
sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 무작위로 들어온 네 개의 이미지의 순서를 판단하기 위해서 img11과 img22를 비교하고, img11과 img33을 비교해 최종 순서를 찾음.
# 첫, 두 번째 들어온 이미지를 순서대로 img1111, img2222로 하여 상태 판단하기.
kp111, des111 = sift.detectAndCompute(img1111, None)
kp222, des222 = sift.detectAndCompute(img2222, None)

matches = flann.knnMatch(des111, des222, k=2)

good = []
for m, n in matches:
    if m.distance < 0.3*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp111[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp222[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img11.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2222 = cv2.polylines(img2222, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    # 위 과정을 통해 img22222위에 img11111이 겹치는 부분 만큼의 line이 그려짐.

    # 상태 구분하기
    width = (img22.shape[1] + img11.shape[1]) // 2  # 입력한 두 이미지의 평균 width
    height = (img22.shape[0] + img11.shape[0]) // 2  # 입력한 두 이미지의 평균 height
    # 두 이미지의 크기가 동일하지 않음.
    # 따라서 width와 height중 더 긴 쪽의 1/20만큼을 img2의 상하좌우에서 제외하고 상태 판단.
    ma = max(width, height)//20
    for x in range(ma, img22.shape[1]-ma): # 좌,우에서 각각 ma씩을 제거한 img22의 width(열)
        for y in range(ma, img22.shape[0]-ma): # 상,하에서 각각 ma씩을 제거한 img22의 height(행)
            x1 = x<img22.shape[1]/2 # x가 img2의 width의 반보다 작을 때(width 앞범위)
            x2 = img22.shape[1]/2<x # x가 img2의 width의 반보다 클 때(width의 뒷범위)
            y1 = y<img22.shape[0]/2 # y가 img2의 height의 반보다 작을 때(height 앞범위)
            y2 = img22.shape[0]/2<y # y가 img2의 height의 반보다 클 때(heignt 뒷범위)

            # img2[m:img2.shape[0또는 1]-m]을 계산하려면 시간이 오래 걸림.
            # 따라서 2*m(m:m*2 또는 -m*2:-m)정도의 길이를 슬라이싱해 사용.
            # np.sum을 통해 img2의 특정 범위 내 픽셀의 채널 값의 합을 구함.
            # axis=1을 하면 채널의 각 열에 있는 값들을 다 더한 값을 나타내 줌.
            # 위에서 img2위에 img1이 겹치는 부분을 (255,0,0)으로 만들어 줌.
            # 따라서 채널의 각 열에 있는 값들을 다 더하면 255가 되고, 이 255가 연속으로 나오는 부분을 통해 상태 판단.
            arr3 = np.sum(img2222[ma:ma*2, x, :], axis=1)  # x열에서의 (m:m*2)각 행의 채널 값의 합 (height의 앞)
            arr4 = np.sum(img2222[-ma*2:-ma, x, :], axis=1)  # x열에서의 (-m*2:-m) 각 행의 채널 값의 합 (height의 뒤)
            arr5 = np.sum(img2222[y, ma:ma*2, :], axis=1)  # y행에서의 (m:m*2)각 열의 채널 값의 합 (width의 앞)
            arr6 = np.sum(img2222[y, -ma*2:-ma, :], axis=1)  # y행에서의 (-m*2:-m)각 열의 채널 값의 합 (width의 뒤)
            # axis=1은 np.sum을 할 때 열의 행 축을 따라 합을 계산함을 의미

            # 세로로 긴 line이 있는 경우(좌,우 or 우,좌)
            if np.all(arr3 == 255) & np.all(arr4 == 255): # arr3과 arr4의 모든 값이 255이면?
                if x1: # 255가 나온 곳의 x가 width/2보다 작으면?
                    # print('img1 : 좌, img2 = 우') # 위 조건을 만족하는 경우 좌,우
                    a = 1
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
                elif x2: # 255가 나온 곳의 x가 width/2보다 크면?
                    # print('img1 : 우, img2 = 좌')  # 위 조건을 만족하는 경우 우,좌
                    a = 2
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
            # 가로로 긴 line이 있는 경우(상,하 or 하,상)
            elif np.all(arr5 == 255) & np.all(arr6 == 255): # arr5와 arr6의 모든 값이 255이면?
                if y1:  # 255가 나온 곳의 y가 height/2보다 작으면?
                    # print('img1 : 상, img2 = 하') # 위 조건을 만족하는 경우 상,하
                    a = 3
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
                elif y2 :  # 255가 나온 곳의 y가 height/2보다 크면?
                    # print('img1 : 하, img2 = 상') # 위 조건을 만족하는 경우 하,상
                    a = 4
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
            # 왼쪽 상단이나 오른쪽 하단에 네모의 모서리 부분이 존재하는 경우
            elif np.all(arr3 == 255) & y1: # arr3의 모든 값이 255이면서 y가 height/2보다 작으면?
                if np.all(arr5 == 255) & x1: # arr5의 모든 값이 255이면서 x가 width/2보다 작으면?
                    # print('img1 : 좌상, img2 = 우하') # 위 조건을 만족하는 경우 좌상,우하
                    a = 5
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
                elif np.all(arr6 == 255) & x2: #arr6의 모든 값이 255이면서 x가 width/2보다 크면?
                    # print('img1 : 우상, img2 = 좌하') # 위 조건을 만족하는 경우 우상,좌하
                    a = 6
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
            # 오른쪽 상단이나 왼쪽 하단에 네모의 모서리 부분이 존재하는 경우
            elif np.all(arr4 == 255) & y2: # arr4의 모든 값이 255이면서 y가 height/2보다 크면?
                if np.all(arr6 == 255) & x2: # arr6의 모든 값이 255이면서 x가 width/2보다 크면?
                    # print('img1 : 우하, img2 = 좌상') # 위 조건을 만족하는 경우 우하, 좌상
                    a = 7
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
                elif np.all(arr5 == 255) & x1:
                    # print('img1 : 좌하, img2 = 우상') # 위 조건을 만족하는 경우 좌하, 우상
                    a = 8
                    break # 이미 상태가 결정되었으니 나머지 계산 과정 생략
        else : continue # 지정한 범위 내에서 특정 조건이 충족되지 않은 경우.
                        # 즉 if, elif문 중 어느 하나도 print되지 않았을 때? 바깥 for문으로 다시 나감.
        break # 내가 원하는 것을 찾고 난 후(어떤 것이 print된 후)에는 그냥 종료
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

# 위의 img1과 img2에서의 상태 판단 방법과 똑같게 img1과 img3 사이의 상태 판단.

# 첫, 세 번째로 들어온 이미지를 순서대로 img1111, img3333으로 하여 상태 판단하기.
kp111, des111 = sift.detectAndCompute(img1111, None)
kp333, des333 = sift.detectAndCompute(img3333, None)

matches = flann.knnMatch(des111, des333, k=2)

good = []
for m, n in matches:
    if m.distance < 0.3*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp111[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp333[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img11.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img3333 = cv2.polylines(img3333, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    width = img33.shape[1] + img11.shape[1]
    height = img33.shape[0] + img11.shape[0]
    m = int(max(width, height)*0.05)
    for x in range(m, img33.shape[1]-m):
        for y in range(m, img33.shape[0]-m):
            x1 = x<img33.shape[1]/2
            x2 = img33.shape[1]/2<x
            y1 = y<img33.shape[0]/2
            y2 = img33.shape[0]/2<y
            arr3 = np.sum(img3333[m:m*2, x, :], axis=1)
            arr4 = np.sum(img3333[-m*2:-m, x, :], axis=1)
            arr5 = np.sum(img3333[y, m:m*2, :], axis=1)
            arr6 = np.sum(img3333[y, -m*2:-m, :], axis=1)
            if np.all(arr3 == 255) & np.all(arr4 == 255):
                if x1:
                    # print('img1 : 좌, img2 = 우') # height의 앞, 뒤
                    b = 1
                    break
                elif x2:
                    # print('img1 : 우, img2 = 좌')  # height의 앞, 뒤
                    b = 2
                    break
            elif np.all(arr5 == 255) & np.all(arr6 == 255):
                if y1:
                    # print('img1 : 상, img2 = 하') # width의 앞, 뒤
                    b = 3
                    break
                elif y2 :
                    # print('img1 : 하, img2 = 상') # width의 앞, 뒤
                    b = 4
                    break
            elif np.all(arr3 == 255) & y1:
                if np.all(arr5 == 255) & x1:
                    # print('img1 : 좌상, img2 = 우하')
                    b = 5
                    break
                elif np.all(arr6 == 255) & x2:
                    # print('img1 : 우상, img2 = 좌하')
                    b = 6
                    break
            elif np.all(arr4 == 255) & y2:
                if np.all(arr6 == 255) & x2:
                    # print('img1 : 우하, img2 = 좌상')
                    b = 7
                    break
                elif np.all(arr5 == 255) & x1:
                    # print('img1 : 좌하, img2 = 우상')
                    b = 8
                    break
        else : continue
        break
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

# a, b가 뭐냐에 따라 img1,2,3,4 결정
if a == 1 and b == 5: # 1. 만약 1,2,4,3로 들어오면? (좌, 우/ 좌상, 우하)
    img1 = img11
    img2 = img22
    img3 = img44
    img4 = img33
elif a == 1 and b == 3: # 2. 만약 1,2,3,4순서로 들어오면? (좌, 우/ 상, 하)
    img1 = img11
    img2 = img22
    img3 = img33
    img4 = img44
elif a == 3 and b == 1: # 3. 만약 1,3,2,4순서로 들어오면? (상, 하/ 좌, 우)
    img1 = img11
    img2 = img33
    img3 = img22
    img4 = img44
elif a == 5 and b == 1: # 4. 만약 1,4,2,3순서로 들어오면? (좌상, 우하/ 좌, 우)
    img1 = img11
    img2 = img33
    img3 = img44
    img4 = img22
elif a == 3 and b == 5: # 5. 만약 1,3,4,2순서로 들어오면? (상, 하/ 좌상, 우하)
    img1 = img11
    img2 = img44
    img3 = img22
    img4 = img33
elif a == 5 and b == 3: # 6. 만약 1,4,3,2순서로 들어오면? (좌상, 우하/ 상, 하)
    img1 = img11
    img2 = img44
    img3 = img33
    img4 = img22
elif a == 2 and b == 6: # 7. 만약 2,1,3,4순서로 들어오면? (우, 좌/ 우상, 좌하)
    img1 = img22
    img2 = img11
    img3 = img33
    img4 = img44
elif a == 2 and b == 3: # 8. 만약 2,1,4,3순서로 들어오면? (우, 좌/ 상, 하)
    img1 = img22
    img2 = img11
    img3 = img44
    img4 = img33
elif a == 4 and b == 1: # 9. 만약 3,1,4,2순서로 들어오면? (하, 상/ 좌, 우)
    img1 = img22
    img2 = img44
    img3 = img11
    img4 = img33
elif a == 7 and b == 2: # 10. 만약 4,1,3,2순서로 들어오면? (우하, 좌상/ 우, 좌)
    img1 = img22
    img2 = img44
    img3 = img33
    img4 = img11
elif a == 4 and b == 8: # 11. 만약 3,1,2,4순서로 들어오면? (하, 상/ 좌하, 우상)
    img1 = img22
    img2 = img33
    img3 = img11
    img4 = img44
elif a == 7 and b == 4: # 12. 만약 4,1,2,3순서로 들어오면? (우하, 좌상/ 하, 상)
    img1 = img22
    img2 = img33
    img3 = img44
    img4 = img11
elif a == 6 and b == 2: # 13. 만약 2,3,1,4순서로 들어오면? (우상, 좌하/ 우, 좌)
    img1 = img33
    img2 = img11
    img3 = img22
    img4 = img44
elif a == 3 and b == 2:  # 14. 만약 2,4,1,3순서로 들어오면? (상, 하/ 우, 좌)
    img1 = img33
    img2 = img11
    img3 = img44
    img4 = img22
elif a == 1 and b == 4: # 15. 만약 3,4,1,2순서로 들어오면? (좌, 우/ 하, 상)
    img1 = img33
    img2 = img44
    img3 = img11
    img4 = img22
elif a == 2 and b == 7: # 16. 만약 4,3,1,2순서로 들어오면? (우, 좌/ 우하, 좌상)
    img1 = img33
    img2 = img44
    img3 = img22
    img4 = img11
elif a == 8 and b == 4: # 17. 만약 3,2,1,4순서로 들어오면? (좌하, 우상/ 하, 상)
    img1 = img33
    img2 = img22
    img3 = img11
    img4 = img44
elif a == 4 and b == 7: # 18. 만약 4,2,1,3순서로 들어오면? (하, 상/ 우하, 좌상)
    img1 = img33
    img2 = img22
    img3 = img44
    img4 = img11
elif a == 4 and b == 2: # 19. 만약 4,2,3,1순서로 들어오면? (하, 상/ 우, 좌)
    img1 = img44
    img2 = img22
    img3 = img33
    img4 = img11
elif a == 8 and b == 1: # 20. 만약 3,2,4,1순서로 들어오면? (좌하, 우상/ 좌, 우)
    img1 = img44
    img2 = img22
    img3 = img11
    img4 = img33
elif a == 1 and b == 8: # 21. 만약 3,4,2,1순서로 들어오면? (좌, 우/ 좌하, 우상)
    img1 = img44
    img2 = img33
    img3 = img11
    img4 = img22
elif a == 2 and b == 4: # 22. 만약 4,3,2,1순서로 들어오면? (우, 좌/ 하, 상)
    img1 = img44
    img2 = img33
    img3 = img22
    img4 = img11
elif a == 3 and b == 6: # 23. 만약 2,4,3,1순서로 들어오면? (상, 하/ 우상, 좌하)
    img1 = img44
    img2 = img11
    img3 = img33
    img4 = img22
elif a == 6 and b == 3: # 24. 만약 2,3,4,1순서로 들어오면? (우상, 좌하/ 상, 하)
    img1 = img44
    img2 = img11
    img3 = img22
    img4 = img33

# cv2.imshow("1", img1), plt.show()
# cv2.imshow("2", img2), plt.show()
# cv2.imshow("3", img3), plt.show()
# cv2.imshow("4", img4), plt.show()

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