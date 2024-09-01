import cv2
import dlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mediapipe
import numpy as np
import torch
from PIL import Image
from nets.yolo import YoloBody
from utils.utils_bbox import DecodeBox


class Model:
    def __init__(self, front_img_path, left_ear_img_path, l_front_img_path,
                 right_ear_img_path, r_front_img_path,
                 is_invert_left, is_invert_right):
        self.iris_size_mm = 11.3
        self.pixel_size_type = 'mm'

        # 处理正前方图片
        front_pixel_size, self.front_eye_locs = self._get_pixel_size(front_img_path)
        self.head_length, self.head_width, self.head_locs = self._get_head_l_w(front_img_path, front_pixel_size)

        # 处理偏左侧图片
        l_front_pixel_size, self.l_front_eye_locs = self._get_pixel_size(l_front_img_path)
        l_front_label, l_front_boxes = self.yolo_detection(l_front_img_path)
        self.l_front_ear, self.l_front_detection_box = self.crop_image(Image.open(l_front_img_path), l_front_boxes,
                                                                       is_invert_left)
        self.l_front_ear_points = self.point_detection(self.l_front_ear)
        self.l_front_ear_length, self.l_front_ear_length_points = self.get_ear_length(self.l_front_ear_points)
        self.l_front_ear_length = round(self.l_front_ear_length * l_front_pixel_size / 10, 2)  # cm

        # 处理左侧图片
        left_label, left_boxes = self.yolo_detection(left_ear_img_path)
        self.left_ear, self.left_ear_detection_box = self.crop_image(Image.open(left_ear_img_path), left_boxes,
                                                                     is_invert_left)
        self.left_ear_points = self.point_detection(self.left_ear)
        self.left_ear_parameters = self.get_ear_parameters(self.left_ear_points, self.l_front_ear_length)

        # 处理偏右侧图片
        r_front_pixel_size, self.r_front_eye_locs = self._get_pixel_size(r_front_img_path)
        r_front_label, r_front_boxes = self.yolo_detection(r_front_img_path)
        self.r_front_ear, self.r_front_detection_box = self.crop_image(Image.open(r_front_img_path), r_front_boxes,
                                                                       is_invert_right)
        self.r_front_ear_points = self.point_detection(self.r_front_ear)
        self.r_front_ear_length, self.r_front_ear_length_points = self.get_ear_length(self.r_front_ear_points)
        self.r_front_ear_length = round(self.r_front_ear_length * r_front_pixel_size / 10, 2)  # cm

        # 处理右侧图片
        right_label, right_boxes = self.yolo_detection(right_ear_img_path)
        self.right_ear, self.right_ear_detection_box = self.crop_image(Image.open(right_ear_img_path), right_boxes,
                                                                       is_invert_right)
        self.right_ear_points = self.point_detection(self.right_ear)
        self.right_ear_parameters = self.get_ear_parameters(self.right_ear_points, self.r_front_ear_length)

    def get_ear_parameters(self, ear_points, ear_length):
        max_idx = np.argmax(ear_points[:, 1])
        min_idx = np.argmin(ear_points[:, 1])
        max_x, max_y = ear_points[max_idx]
        min_x, min_y = ear_points[min_idx]
        d5_pixel = round(np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2) , 2)
        ear_pixel_scale = ear_length / d5_pixel  # cm

        d1 = round(self.js(39, 20, ear_points, None, 44) * ear_pixel_scale, 2)
        d2 = round(self.js(47, 20, ear_points, None, 45) * ear_pixel_scale, 2)
        d3 = round(self.js(44, 36, ear_points, 43, None) * ear_pixel_scale, 2)
        d4 = round(self.js(48, 24, ear_points, None, 26) * ear_pixel_scale, 2)
        d5 = ear_length
        d6 = round(self.js(22, 9, ear_points, None, 7) * ear_pixel_scale, 2)
        d7 = round(self.js(38, 40, ear_points, None, None) * ear_pixel_scale, 2)
        return [d1, d2, d3, d4, d5, d6, d7]

    @staticmethod
    def js(i, j, points, k=None, l=None):
        if k:
            x1 = abs((points[k][0] - points[i][0]) / 2) + min(points[i][0], points[k][0])
            y1 = abs((points[k][1] - points[i][1]) / 2) + min(points[i][1], points[k][1])
        else:
            x1 = points[i][0]
            y1 = points[i][1]

        if l:
            x2 = abs((points[l][0] - points[j][0]) / 2) + min(points[j][0], points[l][0])
            y2 = abs((points[l][1] - points[j][1]) / 2) + min(points[j][1], points[l][1])
        else:
            x2 = points[j][0]
            y2 = points[j][1]

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def get_ear_length(self,ear_points):
        max_idx = np.argmax(ear_points[:, 1])
        min_idx = np.argmin(ear_points[:, 1])
        max_x, max_y = ear_points[max_idx]
        min_x, min_y = ear_points[min_idx]
        d5 = round(np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2) , 2)
        return d5, [ear_points[max_idx], ear_points[min_idx]]

    @staticmethod
    def crop_image(image, top_boxes, is_invert=0):
        for i, c in list(enumerate(top_boxes)):
            top, left, bottom, right = top_boxes[i]
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            crop_image = image.crop([left, top, right, bottom])
            box = [left, top, right - left, bottom - top]
            if is_invert == 1:
                crop_image = crop_image.transpose(Image.FLIP_LEFT_RIGHT)
            return np.array(crop_image), box

    @staticmethod
    def point_detection(img):
        predictor_path = 'ear_landmark_check_point.dat'
        predictor = dlib.shape_predictor(predictor_path)
        box = dlib.rectangle(left=0, top=0, right=img.shape[1], bottom=img.shape[0])
        shape = predictor(img, box)
        coordinate_points = dlib.points()
        for i, point in enumerate(shape.parts()):
            x = max(0, min(point.x, img.shape[1]))
            y = max(0, min(point.y, img.shape[0]))
            point = dlib.point(x, y)
            coordinate_points.append(point)
        adjusted_shape = dlib.full_object_detection(box, coordinate_points)
        matlab_points = []  # 转为matlab可以使用的格式
        for point in adjusted_shape.parts():
            matlab_points.append([point.x, point.y])
        matlab_points = np.array(matlab_points)
        return matlab_points

    def yolo_detection(self, img_path, confidence=0.4):
        model_path = 'model_checkpoints/yolo_image5.pth'
        classes_path = 'model_config/ear_classes_image5.txt'
        anchors_path = 'model_config/yolo_anchors.txt'
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        backbone = 'cspdarknet'
        phi = 's'
        input_shape = [640, 640]
        confidence = confidence
        nms_iou = 0.3
        letterbox_image = True
        cuda = False
        image = Image.open(img_path)
        # np.shape(image)[0:2]
        image_shape = np.array(np.array(image).shape[0:2])
        image = self.cvt_color(image)
        image_data = self.resize_image(image, (input_shape[1], input_shape[0]), letterbox_image)
        image_data = np.expand_dims(
            np.transpose(self.preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        class_names, num_classes = self.get_classes(classes_path)
        anchors, num_anchors = self.get_anchors(anchors_path)

        bbox_util = DecodeBox(anchors, num_classes, (input_shape[0], input_shape[1]),
                              anchors_mask)

        net = YoloBody(anchors_mask, num_classes, phi, backbone=backbone,
                       input_shape=input_shape)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(torch.load(model_path, map_location=device))
        net = net.eval()
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if cuda:
                images = images.cuda()
            # 1 3 640 640
            outputs = net(images)
            outputs = bbox_util.decode_box(outputs)

            results = bbox_util.non_max_suppression(torch.cat(outputs, 1), num_classes, input_shape,
                                                    image_shape, letterbox_image, conf_thres=confidence,
                                                    nms_thres=nms_iou)

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_boxes = results[0][:, :4]
        return top_label, top_boxes

    @staticmethod
    def _get_head_l_w(front_img_path, pixel_size):
        front_img = cv2.imread(front_img_path)
        w, h = front_img.shape[1], front_img.shape[0]
        mp_face_mesh = mediapipe.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
        front_img_rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(front_img_rgb)
        if res.multi_face_landmarks:
            for facelandmark in res.multi_face_landmarks:
                left = facelandmark.landmark[234].x * w
                right = facelandmark.landmark[447].x * w
                bottom = facelandmark.landmark[152].y * h
                top = facelandmark.landmark[10].y * h
                top = top - (bottom - top) * 0.15
                head_width = round((right - left) * pixel_size * 0.1, 2)
                head_length = round((bottom - top) * pixel_size * 0.1, 2)
                # 左右上下
                head_locs = [
                    [facelandmark.landmark[234].x * w, facelandmark.landmark[234].y * h],
                    [facelandmark.landmark[447].x * w, facelandmark.landmark[447].y * h],
                    [facelandmark.landmark[152].x * w, facelandmark.landmark[152].y * h],
                    [(right - left) / 2 + left, top]
                ]
                print(f'头宽:{head_width}, 头高:{head_length}')
                return head_length, head_width, head_locs

    @staticmethod
    def distance_between_points(loc1, loc2):
        dx = abs(loc1[0] - loc2[0])
        dy = abs(loc1[1] - loc2[1])

        return np.sqrt(dx ** 2 + dy ** 2)

    @staticmethod
    def cvt_color(image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        else:
            image = image.convert('RGB')
            return image

    @staticmethod
    def resize_image(image, size, letterbox_image):
        iw, ih = image.size
        w, h = size
        if letterbox_image:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image

    @staticmethod
    def preprocess_input(image):
        image /= 255.0
        return image

    @staticmethod
    def get_classes(classes_path):
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)

    @staticmethod
    def get_anchors(anchors_path):
        """loads the anchors from a file"""
        with open(anchors_path, encoding='utf-8') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
        return anchors, len(anchors)

    def _get_pixel_size(self, img_path):
        landmarks_eyes = {
            "left": [474, 475, 476, 477],
            "right": [469, 470, 471, 472],
        }
        opposites = [(0, 2), (1, 3), (4, 6), (5, 7)]

        with mediapipe.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
        ) as face_mesh:
            front_img = cv2.imread(img_path)
            w, h = front_img.shape[1], front_img.shape[0]
            res = face_mesh.process(front_img)

            # 获取眼睛所在位置
            if res.multi_face_landmarks:
                face_landmarks = res.multi_face_landmarks[0]
                # 8个坐标，左右眼睛各4个
                eye_locs = []
                for landmarks in landmarks_eyes.values():
                    for landmark in landmarks:
                        x = face_landmarks.landmark[landmark].x * w
                        y = face_landmarks.landmark[landmark].y * h
                        eye_locs.append((x, y))

                distances = []
                for opposite in opposites:
                    loc1 = eye_locs[opposite[0]]
                    loc2 = eye_locs[opposite[1]]
                    dist = self.distance_between_points(loc1, loc2)
                    distances.append(dist)
                iris_size_pixel = np.mean(distances)
                pixel_size = round(self.iris_size_mm / iris_size_pixel, 2)
                return pixel_size, eye_locs

def find_symmetric_point(line_intercept, point):
    x, y = point

    # 计算对称点的坐标
    x_symmetric = 2 * line_intercept - x
    y_symmetric = y

    return x_symmetric, y_symmetric

if __name__ == '__main__':
    n=35
    fp = 'test_img5/{}_front.bmp'.format(n)
    flp = 'test_img5/{}_l_left.bmp'.format(n)
    lp = 'test_img5/{}_left.bmp'.format(n)
    rp = 'test_img5/{}_right.bmp'.format(n)
    frp = 'test_img5/{}_r_right.bmp'.format(n)
    model = Model(fp, lp, flp, rp, frp, is_invert_left=0, is_invert_right=1)
    print(f'左耳参数d1-7:{model.left_ear_parameters} [cm]')
    print(f'右耳参数d1-7:{model.right_ear_parameters} [cm]')
    fm = Image.open(fp)  # 绘制正前方
    _, ax = plt.subplots()
    ax.imshow(np.array(fm))
    ax.plot([model.head_locs[0][0], model.head_locs[1][0]], [model.head_locs[0][1], model.head_locs[1][1]])  # 宽
    ax.plot([model.head_locs[2][0], model.head_locs[3][0]], [model.head_locs[2][1], model.head_locs[3][1]])  # 高
    ax.add_patch(patches.Polygon(np.array(model.front_eye_locs[0:4]), color='green', fill=False))  # 右眼
    ax.add_patch(patches.Polygon(np.array(model.front_eye_locs[4:8]), color='red', fill=False))  # 左眼
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    fm = Image.open(flp)  # 绘制正前方偏左
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(np.array(fm))
    axs[0].add_patch(patches.Polygon(np.array(model.l_front_eye_locs[0:4]), color='green', fill=False,linewidth=2))  # 右眼
    axs[0].add_patch(patches.Polygon(np.array(model.l_front_eye_locs[4:8]), color='red', fill=False,linewidth=2))  # 左眼
    box_x, box_y, box_w, box_h = model.l_front_detection_box
    axs[0].add_patch(patches.Rectangle((box_x, box_y), box_w, box_h, edgecolor='red', fill=False,linewidth=2))
    axs[1].imshow(model.l_front_ear)
    for i in range(55):
        point_x, point_y = model.l_front_ear_points[i]
        axs[1].plot(point_x, point_y, '.', color='red')
        axs[1].text(point_x, point_y, f'{i}')
    axs[1].plot([model.l_front_ear_length_points[0][0], model.l_front_ear_length_points[1][0]],
                [model.l_front_ear_length_points[0][1], model.l_front_ear_length_points[1][1]])
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    fm = Image.open(lp)  # 绘制左侧
    _, axs = plt.subplots(1, 3)
    axs[0].imshow(np.array(fm))
    box_x, box_y, box_w, box_h = model.left_ear_detection_box
    axs[0].add_patch(patches.Rectangle((box_x, box_y), box_w, box_h, edgecolor='red', fill=False,linewidth=3))
    axs[1].imshow(model.left_ear)
    axs[2].imshow(model.left_ear)
    for i in range(55):
        point_x, point_y = model.left_ear_points[i]
        axs[2].plot(point_x, point_y, '.', color='red')
        axs[2].text(point_x, point_y, f'{i}')
    point_x39, point_y39=  model.left_ear_points[39]
    point_x20, point_y20 = model.left_ear_points[20]
    point_x44, point_y44 = model.left_ear_points[44]
    point_x47, point_y47 = model.left_ear_points[47]
    point_x35, point_y35 = model.left_ear_points[35]
    point_x43, point_y43 = model.left_ear_points[43]
    point_x25, point_y25 = model.left_ear_points[25]
    point_x3, point_y3 = model.left_ear_points[3]
    point_x17, point_y17 = model.left_ear_points[17]
    point_x9, point_y9 = model.left_ear_points[9]
    point_x23, point_y23 = model.left_ear_points[23]
    point_x40, point_y40 = model.left_ear_points[40]
    point_x38, point_y38 = model.left_ear_points[38]
    point_xmid=abs((point_x20 - point_x44) / 2) + min(point_x20, point_x44)
    point_ymid =abs((point_y20 - point_y44) / 2) + min(point_y20, point_y44)
    axs[2].plot([point_x39, point_xmid],[point_y39, point_ymid],color='blue')
    axs[2].plot([point_x47, point_xmid], [point_y47, point_ymid], color='black')
    axs[2].plot([point_x35, point_x43], [point_y35, point_y43], color='green')
    axs[2].plot([point_x47, point_x25], [point_y47, point_y25], color='yellow')
    axs[2].plot([point_x17, point_x3], [point_y17, point_y3], color='white')
    axs[2].plot([point_x9, point_x23], [point_y9, point_y23], color='pink')
    axs[2].plot([point_x40, point_x38], [point_y40, point_y38], color='red')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    _, axs = plt.subplots(1, 2)  # 绘制左侧，把点映射到原图上
    axs[0].imshow(np.array(fm))
    box_x, box_y, box_w, box_h = model.left_ear_detection_box
    axs[0].add_patch(patches.Rectangle((box_x, box_y), box_w, box_h, edgecolor='red', fill=False))
    axs[1].imshow(np.array(fm))
    for i in range(55):
        point_x, point_y = model.left_ear_points[i]
        axs[1].plot(point_x + box_x, point_y + box_y, '.', color='red')
        axs[1].text(point_x + box_x, point_y + box_y, f'{i}')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    fm = Image.open(frp)  # 绘制正前方偏右
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(np.array(fm))
    axs[0].add_patch(patches.Polygon(np.array(model.r_front_eye_locs[0:4]), color='red', fill=False))  # 右眼
    axs[0].add_patch(patches.Polygon(np.array(model.r_front_eye_locs[4:8]), color='red', fill=False))  # 左眼
    box_x, box_y, box_w, box_h = model.r_front_detection_box
    axs[0].add_patch(patches.Rectangle((box_x, box_y), box_w, box_h, edgecolor='red', fill=False))
    axs[1].imshow(model.r_front_ear)
    for i in range(55):
        point_x, point_y = model.r_front_ear_points[i]
        axs[1].plot(point_x, point_y, '.', color='red')
        axs[1].text(point_x, point_y, f'{i}')
    axs[1].plot([model.r_front_ear_length_points[0][0], model.r_front_ear_length_points[1][0]],
                [model.r_front_ear_length_points[0][1], model.r_front_ear_length_points[1][1]])
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    fm = Image.open(rp)  # 绘制右侧
    _, axs = plt.subplots(1, 3)
    axs[0].imshow(np.array(fm))
    box_x, box_y, box_w, box_h = model.right_ear_detection_box
    axs[0].add_patch(patches.Rectangle((box_x, box_y), box_w, box_h, edgecolor='red', fill=False))
    axs[1].imshow(model.right_ear)
    axs[2].imshow(model.right_ear)
    for i in range(55):
        point_x, point_y = model.right_ear_points[i]
        axs[2].plot(point_x, point_y, '.', color='red')
        axs[2].text(point_x, point_y, f'{i}')
    point_x39, point_y39 = model.right_ear_points[39]
    point_x20, point_y20 = model.right_ear_points[20]
    point_x44, point_y44 = model.right_ear_points[44]
    point_x47, point_y47 = model.right_ear_points[47]
    point_x35, point_y35 = model.right_ear_points[35]
    point_x43, point_y43 = model.right_ear_points[43]
    point_x25, point_y25 = model.right_ear_points[25]
    point_x3, point_y3 = model.right_ear_points[3]
    point_x17, point_y17 = model.right_ear_points[17]
    point_x9, point_y9 = model.right_ear_points[9]
    point_x23, point_y23 = model.right_ear_points[23]
    point_x40, point_y40 = model.right_ear_points[40]
    point_x38, point_y38 = model.right_ear_points[38]
    point_xmid = abs((point_x20 - point_x44) / 2) + min(point_x20, point_x44)
    point_ymid = abs((point_y20 - point_y44) / 2) + min(point_y20, point_y44)
    axs[2].plot([point_x39, point_xmid], [point_y39, point_ymid], color='blue')
    axs[2].plot([point_x47, point_xmid], [point_y47, point_ymid], color='black')
    axs[2].plot([point_x35, point_x43], [point_y35, point_y43], color='green')
    axs[2].plot([point_x47, point_x25], [point_y47, point_y25], color='yellow')
    axs[2].plot([point_x17, point_x3], [point_y17, point_y3], color='white')
    axs[2].plot([point_x9, point_x23], [point_y9, point_y23], color='pink')
    axs[2].plot([point_x40, point_x38], [point_y40, point_y38], color='red')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    _, axs = plt.subplots(1, 2)
    axs[0].imshow(np.array(fm))
    box_x, box_y, box_w, box_h = model.right_ear_detection_box
    b = box_w / 2  # 截距
    axs[0].add_patch(patches.Rectangle((box_x, box_y), box_w, box_h, edgecolor='red', fill=False))
    axs[1].imshow(np.array(fm))
    for i in range(55):
        point_x, point_y = find_symmetric_point(line_intercept=b, point=model.right_ear_points[i])
        axs[1].plot(point_x + box_x, point_y + box_y, '.', color='red')
        axs[1].text(point_x + box_x, point_y + box_y, f'{i}')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
