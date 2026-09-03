# face_parser.py

import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image, ImageFile
import numpy as np
import os
import time # Thư viện để đo thời gian
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FaceParser:
    def __init__(self, model_name="jonathandinu/face-parsing", device=None):
        """
        Khởi tạo FaceParser và đo thời gian tải mô hình.
        """
        start_time_init = time.time()

        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        self.model_load_time_seconds = 0
        try:
            start_time_model_load = time.time()
            self.image_processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.model_load_time_seconds = time.time() - start_time_model_load
        except Exception as e:
            # Trong trường hợp không có print, lỗi vẫn cần được ném ra
            # để người dùng biết mô hình không tải được.
            raise RuntimeError(f"Lỗi khi tải mô hình Segformer: {e}") from e

        self.FACE_COMPONENT_LABELS = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        ]
        self.initialization_time_seconds = time.time() - start_time_init


    def create_face_mask(self, pil_image_rgb, not_anime, hair_style=False):
        """
        Tạo mặt nạ nhị phân cho khuôn mặt và trả về mặt nạ cùng thời gian xử lý.

        Returns:
            tuple: (np.ndarray or None, float seconds)
                   Mặt nạ nhị phân (0 hoặc 255) hoặc None nếu lỗi, và thời gian xử lý.
        """
        start_time = time.time()

        if pil_image_rgb is None:
            return None, time.time() - start_time

        try:
            original_size = pil_image_rgb.size
            if not_anime == False:
                logger.info(f"This is anime style.")
                 # if anime style, return zero mask
                return np.zeros(original_size[::-1], dtype=np.uint8), processing_time

            inputs = self.image_processor(images=pil_image_rgb, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            upsampled_logits = nn.functional.interpolate(
                logits,
                size=original_size[::-1],
                mode='bicubic',
                align_corners=True
            )
            predicted_labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

            binary_mask = np.zeros(predicted_labels.shape, dtype=np.uint8)
            if hair_style:
                binary_mask[predicted_labels == 13] = 255
            else:
                for face_label_id in self.FACE_COMPONENT_LABELS:
                    binary_mask[predicted_labels == face_label_id] = 255

            processing_time = time.time() - start_time

            if not np.any(binary_mask):
                 # Trả về mặt nạ rỗng nếu không tìm thấy thành phần nào
                logger.info(f"No face is detected in the image.")
                return np.zeros(original_size[::-1], dtype=np.uint8), processing_time
            logger.info(f"Face is detected in the image.")
            return binary_mask, processing_time

        except Exception:
            # Trả về mặt nạ rỗng khi lỗi và thời gian đã trôi qua
            return np.zeros(pil_image_rgb.size[::-1], dtype=np.uint8), time.time() - start_time


    def get_face_with_transparent_background(self, original_pil_image, binary_mask_np=None):
        """
        Tạo ảnh PIL với khuôn mặt và nền trong suốt, trả về ảnh và thời gian xử lý.

        Returns:
            tuple: (PIL.Image or None, float seconds)
                   Ảnh RGBA hoặc None nếu lỗi, và tổng thời gian xử lý.
        """
        start_time_total = time.time()
        mask_creation_time = 0.0

        if original_pil_image is None:
            return None, time.time() - start_time_total

        pil_image_rgb = original_pil_image.convert("RGB")

        if binary_mask_np is None:
            binary_mask_np, mask_creation_time = self.create_face_mask(pil_image_rgb)
            if binary_mask_np is None or not np.any(binary_mask_np):
                img_rgba_empty_mask = original_pil_image.convert("RGBA")
                alpha_channel = Image.new('L', img_rgba_empty_mask.size, 0)
                img_rgba_empty_mask.putalpha(alpha_channel)
                return img_rgba_empty_mask, (time.time() - start_time_total) + mask_creation_time

        image_rgba = original_pil_image.convert("RGBA")
        img_np = np.array(image_rgba)

        if img_np.shape[:2] != binary_mask_np.shape:
            mask_pil = Image.fromarray(binary_mask_np)
            mask_pil_resized = mask_pil.resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST)
            binary_mask_np_resized = np.array(mask_pil_resized)
        else:
            binary_mask_np_resized = binary_mask_np

        if not np.any(binary_mask_np_resized):
            img_np[:, :, 3] = 0 # Toàn bộ trong suốt
            transparent_face_image = Image.fromarray(img_np, 'RGBA')
            total_processing_time = (time.time() - start_time_total) + mask_creation_time
            return transparent_face_image, total_processing_time

        img_np[:, :, 3] = binary_mask_np_resized.astype(np.uint8)
        transparent_face_image = Image.fromarray(img_np, 'RGBA')

        total_processing_time = (time.time() - start_time_total) + mask_creation_time
        return transparent_face_image, total_processing_time

# --- Ví dụ sử dụng lớp FaceParser (đã loại bỏ print, chỉ giữ lại logic và cách lấy thời gian) ---
if __name__ == '__main__':
    # Tạo các thư mục ví dụ một cách "im lặng"
    INPUT_FOLDER_EXAMPLE = "sample indian"
    OUTPUT_MASKS_FOLDER_EXAMPLE = "output_masks_class_timed"
    OUTPUT_TRANSPARENT_FACES_FOLDER_EXAMPLE = "output_transparent_faces_class_timed"

    for folder in [INPUT_FOLDER_EXAMPLE, OUTPUT_MASKS_FOLDER_EXAMPLE, OUTPUT_TRANSPARENT_FACES_FOLDER_EXAMPLE]:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    # Tạo ảnh ví dụ nếu thư mục input rỗng
    example_image_path = os.path.join(INPUT_FOLDER_EXAMPLE, "example_face_timed.png")
    if not os.path.exists(example_image_path):
        try:
            temp_img = Image.new('RGB', (512, 512), color = 'lightgray')
            from PIL import ImageDraw # Chỉ import khi cần thiết
            draw = ImageDraw.Draw(temp_img)
            draw.ellipse((100, 100, 400, 400), fill='peachpuff')
            draw.ellipse((150, 180, 200, 230), fill='white')
            draw.ellipse((300, 180, 350, 230), fill='white')
            temp_img.save(example_image_path)
        except Exception:
            pass # Bỏ qua lỗi tạo ảnh ví dụ một cách im lặng

    all_times = {} # Để lưu trữ thời gian cho báo cáo cuối cùng (nếu muốn)

    try:
        parser = FaceParser()
        # Thời gian tải mô hình và khởi tạo có thể được truy cập từ parser.model_load_time_seconds
        # và parser.initialization_time_seconds
        all_times['model_initialization_seconds'] = parser.initialization_time_seconds
        all_times['model_load_seconds_within_init'] = parser.model_load_time_seconds
        # In thời gian tải mô hình (đây là print duy nhất theo yêu cầu "chỉ tính thời gian")
        print(f"Thời gian khởi tạo FaceParser (bao gồm tải mô hình): {parser.initialization_time_seconds:.4f} giây")
        print(f"Thời gian tải mô hình (trong __init__): {parser.model_load_time_seconds:.4f} giây")


    except RuntimeError as e:
        # Nếu có lỗi nghiêm trọng như không tải được mô hình, in ra và thoát
        print(f"Lỗi nghiêm trọng khi khởi tạo FaceParser: {e}")
        exit()


    image_files = []
    if os.path.exists(INPUT_FOLDER_EXAMPLE):
        image_extensions = ('.png', '.jpg', '.jpeg')
        for ext in image_extensions:
            image_files.extend(
                [os.path.join(INPUT_FOLDER_EXAMPLE, f) for f in os.listdir(INPUT_FOLDER_EXAMPLE) if f.lower().endswith(ext)]
            )

    if not image_files:
        # print(f"Không tìm thấy tệp ảnh nào trong {INPUT_FOLDER_EXAMPLE} để chạy ví dụ thời gian.")
        pass
    else:
        total_mask_creation_time = 0
        total_transparent_face_time = 0
        processed_images_count = 0

        for image_path in image_files:
            try:
                pil_image_original = Image.open(image_path).convert("RGB")

                binary_mask, mask_time = parser.create_face_mask(pil_image_original)
                total_mask_creation_time += mask_time
                # print(f"Thời gian tạo mặt nạ cho {os.path.basename(image_path)}: {mask_time:.4f} giây")


                if binary_mask is not None and np.any(binary_mask):
                    base_filename = os.path.basename(image_path)
                    name_part, _ = os.path.splitext(base_filename)

                    output_mask_filename = f"{name_part}_mask.png"
                    output_mask_path = os.path.join(OUTPUT_MASKS_FOLDER_EXAMPLE, output_mask_filename)
                    Image.fromarray(binary_mask).save(output_mask_path)

                    transparent_face_pil, trans_time = parser.get_face_with_transparent_background(pil_image_original, binary_mask)
                    total_transparent_face_time += trans_time
                    # print(f"Thời gian tạo ảnh trong suốt cho {os.path.basename(image_path)}: {trans_time:.4f} giây")

                    if transparent_face_pil:
                        output_transparent_face_filename = f"{name_part}_transparent_face.png"
                        output_transparent_face_path = os.path.join(OUTPUT_TRANSPARENT_FACES_FOLDER_EXAMPLE, output_transparent_face_filename)
                        transparent_face_pil.save(output_transparent_face_path)
                    processed_images_count+=1
            except Exception:
                # Bỏ qua lỗi xử lý ảnh một cách im lặng trong ví dụ
                pass

        if processed_images_count > 0:
            all_times['avg_mask_creation_seconds_per_image'] = total_mask_creation_time / processed_images_count
            all_times['avg_transparent_face_seconds_per_image'] = total_transparent_face_time / processed_images_count
            # In thời gian xử lý trung bình (print duy nhất theo yêu cầu "chỉ tính thời gian")
            print(f"Thời gian trung bình tạo mặt nạ mỗi ảnh: {all_times['avg_mask_creation_seconds_per_image']:.4f} giây")
            print(f"Thời gian trung bình tạo ảnh trong suốt mỗi ảnh: {all_times['avg_transparent_face_seconds_per_image']:.4f} giây (bao gồm tạo mặt nạ nếu không được cung cấp)")


    # Bạn có thể sử dụng biến `all_times` để làm gì đó khác nếu cần
    # ví dụ: ghi ra file JSON
    # import json
    # with open("face_parser_timing_report.json", "w") as f:
    #     json.dump(all_times, f, indent=4)