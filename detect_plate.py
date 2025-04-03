import torch 
import cv2
import os

class PlateDetector:
    def __init__(self, weights):
        """
        Khởi tạo mô hình YOLOv5 từ tệp trọng số.
        
        Parameters:
        - weights (str): Đường dẫn tới tệp trọng số của mô hình YOLOv5.
        """
        self.names = ['1_line', '2_line']
        self.model = torch.hub.load('yolov5', 'custom', path=weights, source="local")

    def load_image(self, image_path):
        """
        Đọc ảnh từ đường dẫn và kiểm tra tính hợp lệ.
        
        Parameters:
        - image_path (str): Đường dẫn đến ảnh cần xử lý.
        
        Returns:
        - image (numpy.ndarray): Ảnh đã đọc.
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Đường dẫn {image_path} không tồn tại")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh {image_path}")
        
        return image

    def predict(self, image):
        """
        Dự đoán biển số xe trên ảnh sử dụng mô hình YOLOv5.
        
        Parameters:
        - image (numpy.ndarray): Ảnh đầu vào.
        
        Returns:
        - detections (torch.Tensor): Danh sách bounding boxes dự đoán [x1, y1, x2, y2, conf, cls].
        """
        result = self.model(image)
        detections = result.xyxy[0]  # Lấy kết quả dự đoán 

        return detections

    def save_cropped_images(self, image, detections, image_name, is_save=True):
        """
        Cắt và lưu ảnh biển số xe được phát hiện từ kết quả dự đoán.
        
        Parameters:
        - image (numpy.ndarray): Ảnh gốc.
        - detections (torch.Tensor): Kết quả dự đoán từ mô hình.
        - image_name (str): Tên ảnh gốc (không có phần mở rộng).
        - is_save (bool): Nếu True, lưu ảnh đã cắt vào thư mục output.
        
        Returns:
        - cropped_images (list): Danh sách ảnh biển số đã cắt.
        - labels (list): Nhãn của từng biển số ('1_line' hoặc '2_line').
        - cropped_paths (list): Danh sách đường dẫn của ảnh đã lưu.
        """
        output_dir = os.path.join("output", image_name)
        os.makedirs(output_dir, exist_ok=True)
        
        cropped_images = []
        cropped_paths = []
        labels = []
        
        for i, (x1, y1, x2, y2, _, cls) in enumerate(detections):
            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
            cropped_image_path = os.path.join(output_dir, f"{image_name}_{i}.jpg")
            if is_save:
                cv2.imwrite(cropped_image_path, cropped_image)
            cropped_images.append(cropped_image)
            labels.append(self.names[int(cls)])
            cropped_paths.append(cropped_image_path)
        
        return cropped_images, labels, cropped_paths

    def detect_plate(self, image_path, is_save=True):
        """
        Phát hiện biển số xe trên ảnh và lưu ảnh đã cắt nếu cần.
        
        Parameters:
        - image_path (str): Đường dẫn đến ảnh cần xử lý.
        - is_save (bool): Nếu True, lưu ảnh đã cắt vào thư mục output.
        
        Returns:
        - cropped_images (list): Danh sách ảnh biển số đã cắt.
        - labels (list): Nhãn của từng biển số ('1_line' hoặc '2_line').
        - cropped_paths (list): Danh sách đường dẫn của ảnh đã lưu.
        
        Raises:
        - ValueError: Nếu không phát hiện được biển số trong ảnh.
        """
        image = self.load_image(image_path)
        detections = self.predict(image)
        
        if detections is None or len(detections) == 0:
            raise ValueError(f"Không phát hiện biển số trong {image_path}")
        
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        return self.save_cropped_images(image, detections, image_name, is_save)
