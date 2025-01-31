from tkinter import *
from tkinter import filedialog
from matplotlib import pyplot as plt
import cv2
import numpy as np

# 분침만 존재하는 참고용 사진 Ref.JPG를 img1으로 사용
# 참고용 사진을 불러와 img1에 저장
root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img1 = cv2.imread(path)
root.withdraw()

# 참고용 사진을 통해 시간을 확인할 분침 사진을 불러와 img2에 저장
root = Tk()
path = filedialog.askopenfilename(initialdir = "C:/data",title = "choose your image", filetypes = (("jpeg files","*.jpg"), ("all files","*.*")))
img2 = cv2.imread(path)
root.withdraw()

# img1, img2의 특징점을 통해 Homography 변환 행렬 M 찾기
MIN_MATCH_COUNT = 4
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.3*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

# 찾은 Homography 변환 행렬로 부터 img1이 img2에 매칭되기 위해 회전한 각도 추출
# 이미지의 3차원 회전이 거의 없기 때문에 affine transformation이라고 가정.
# or 이미지의 3차원 회전이 거의 없고, translation도 거의 없기 때문에 2D in-plane ratation이라고 가정.
# (정확하지는 않지만) M[0, 1]=-sin(세타), M[0, 0]=cos(세타)
# (M[0, 1]은 x축 방향으로의 회전을 나타내며, M[0, 0]은 y 축 방향으로의 회전을 나타냄. 따라서 이 값들을 사용해 회전 각도를 계산할 수 있음.)
# -sin(세타)/cos(세타)=tan(세타)이므로 arctan(-값)를 하면 세타(회전각도)를 구할 수 있음.
# 아크탄젠트 함수(np.arctan2(y, x))는 주어진 좌표 (x=M[0, 0], y=M[0, 1])가 나타내는 라디안 값을 반환.
# 따라서 라디안 값에 *180/np.pi를 해 각도로 변환.

t = np.arctan2(-M[0, 1], M[0, 0])*180/np.pi
# 그냥 np.arctan2()를 계산하면 M[0, 1]이 -sin이기 때문에 반시계 방향으로 회전한 각도가 나옴.
# 따라서 sin 앞에 -를 붙여서 시계방향으로 회전한 각도로 바꿔 줌.
# 15분에 90도니까 1분당 약 6도정도 차지함.
# 따라서 0도를 구분할 때는 -3~3을 기준으로 하며 더 작은 값을 포함.(-3<=0분<3)
# 즉 0분에서 1분이 지날때마다 -3~3에서 각각 6씩 더해지는 것과 같음.
# 180도(30분)를 기준으로 -각도로 계산(부호가 바뀜)
a = 3
# 결과 출력
for i in range(0,59): # 한 시간에는 0~59분이 존재함.
    if i*6-a < t <= i*6+a: # 1분당 약 6도 정도를 차지함.
        print("%d분" % i)
        break # 만족하는 것을 찾고나면 break
    elif -(60-i)*6-a < t <= -(60-i)*6+a:
        print("%d분" % i)
        break # 만족하는 것을 찾고나면 break

img1 = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
cv2.imshow("img1", img1), plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()