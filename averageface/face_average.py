import os
import cv2
import dlib
import numpy
import numpy as np
import math

# 平均脸运算尺寸
w = 2000
h = 2000
# 总图片数
NUM = 0
# 记录图片集总数
Index = 0
# 本地路径
ROOT_DIR = os.getcwd()
# 平均脸
sum_img = np.zeros((h, w, 3), np.float32())


# 统计所有图片数
def getAllNum(path):
    num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                num = num + 1
    return num


# 检测目录下所有图片的所有点 返回点集合
def getPoints(path):
    # 点集
    pointsArray = []
    # 训练数据 需要下载
    PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"

    # dlib自带的frontal_face_detector作为我们的人脸提取器
    detector = dlib.get_frontal_face_detector()

    # 官方提供的模型构建特征提取器
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    class NoFaces(Exception):
        pass

    for filePath in os.listdir(path):
        if filePath.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, filePath))

            # 进行人脸检测 rects为返回的结果
            rects = detector(img, 1)
            if len(rects) >= 1:
                # 保存一幅图像的所有点
                points = []

                # 进行人脸关键点识别
                landmarks = numpy.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
                print(filePath)
                print(landmarks.size)
                for idx, point in enumerate(landmarks):
                    pos = (point[0, 0], point[0, 1])
                    points.append(pos)
                pointsArray.append(points)

    return pointsArray


# 将所有的图片读取到矩阵 数组中
def readImages(path):
    imagesArray = []

    for filePath in os.listdir(path):
        if filePath.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, filePath))
            # 转浮点
            img = np.float32(img) / 255.0
            # 加入集合
            imagesArray.append(img)

    return imagesArray


# 推算相似变换矩阵
def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]

    inPts.append([np.int(xin), np.int(yin)])

    xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]

    outPts.append([np.int(xout), np.int(yout)])

    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)

    return tform


# 检查一个点是否在矩形中
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# delanauy三角计算
def calculateDelaunayTriangles(rect, points):
    # 创建模型
    subdiv = cv2.Subdiv2D(rect)

    # 点加入模型
    for p in points:
        subdiv.insert((p[0], p[1]))

    # 三角形列表
    triangleList = subdiv.getTriangleList();

    # 找出三角形的下标
    delaunayTri = []

    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri


def constrainPoint(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p


# 仿射变换
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
# 经α混合三角地区img1和IMG2 IMG
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


def get_average(path):
    dir_name = os.path.basename(path)

    # 设置全局变量
    global sum_img
    global Index

    # 所有检测点的数组
    allPoints = getPoints(path)

    # 所有图片矩阵的数组
    images = readImages(path)

    # 固定眼角的位置
    eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)), (np.int(0.7 * w), np.int(h / 3))]

    imagesNorm = []
    pointsNorm = []

    # 对三角剖分设置边界点
    boundaryPts = np.array(
        [(0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2), (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2)])

    # 所有点的实际位置全部初始化为0（数目与检测的点数目相同）
    pointsAvg = np.array([(0, 0)] * (len(allPoints[0]) + len(boundaryPts)), np.float32())

    # 图片数
    numImages = len(images)

    # 通过对图像的变形和地标的变形来输出坐标系统，并求出变换后的地标的平均值。
    for i in range(0, numImages):
        points1 = allPoints[i]

        # 获取眼角的位置
        eyecornerSrc = [allPoints[i][36], allPoints[i][45]]

        # 计算变换矩阵
        tform = similarityTransform(eyecornerSrc, eyecornerDst)

        # 应用相似变换
        img = cv2.warpAffine(images[i], tform, (w, h))

        # 计算检测点变换后的位置
        points2 = np.reshape(np.array(points1), (68, 1, 2))

        points = cv2.transform(points2, tform)

        points = np.float32(np.reshape(points, (68, 2)))

        # 附加边界点。将用于Delaunay 三角形划分
        points = np.append(points, boundaryPts, axis=0)

        # 计算平均地标点的位置
        pointsAvg = pointsAvg + points / numImages

        pointsNorm.append(points)
        imagesNorm.append(img)

    # 三角剖分
    rect = (0, 0, w, h)
    dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))

    # 将所有的图片 进行变换 对应于平均图像
    for i in range(0, len(imagesNorm)):
        img = np.zeros((h, w, 3), np.float32())
        # 对当前图像进行对齐操作
        for j in range(0, len(dt)):
            tin = []
            tout = []

            for k in range(0, 3):
                pIn = pointsNorm[i][dt[j][k]]
                pIn = constrainPoint(pIn, w, h)

                pOut = pointsAvg[dt[j][k]]
                pOut = constrainPoint(pOut, w, h)

                tin.append(pIn)
                tout.append(pOut)

            warpTriangle(imagesNorm[i], img, tin, tout)

        Index = Index + 1
        # Add image intensities for averaging
        sum_img = sum_img + img

    # 将叠加的部分图像保存
    tempout = sum_img / NUM

    res = cv2.resize(tempout, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    outpath = os.path.join(ROOT_DIR, 'out.jpg')
    res = res * 255
    cv2.imwrite(outpath, res)
    print(outpath + "--- be saved Successfully")


if __name__ == '__main__':
    # 照片目录
    dir_name = 'photos'

    # 获取总共的图像数
    NUM = getAllNum(dir_name)
    print("照片数：" + str(NUM))

    get_average(os.path.join(ROOT_DIR, dir_name))
