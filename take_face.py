from PIL import Image
import time
import os
import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

def segment_face_same_size(image_path_or_pil_image, not_anime):
    """
    Trả về một ảnh mới với khuôn mặt được tách ra (theo kích thước ảnh gốc) và các phần còn lại là đen.
    Nếu không phát hiện được khuôn mặt, trả về ảnh đen.


    Parameters:
    image_path_or_pil_image (str or PIL.Image): Đường dẫn đến ảnh hoặc đối tượng PIL Image

    Returns:
    numpy.ndarray: Ảnh mới với khuôn mặt được tách ra
    """
    overall_start_time = time.perf_counter()
    face_mesh_init_time = 0
    face_mesh_process_time = 0
    face_mesh = None
    original_image_np = None
    base_filename_for_log = "input_image"

    try:
        if isinstance(image_path_or_pil_image, str):
            base_filename_for_log = os.path.basename(image_path_or_pil_image)
            original_image_np = cv2.imread(image_path_or_pil_image)
            if original_image_np is None:
                logger.error(f"Lỗi: Không thể đọc được ảnh từ đường dẫn: {image_path_or_pil_image}")
                return None
        elif isinstance(image_path_or_pil_image, Image.Image):
            base_filename_for_log = "PIL_image_input"
            original_image_np = cv2.cvtColor(np.array(image_path_or_pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        else:
            logger.error("Lỗi: Đầu vào không phải đường dẫn ảnh hoặc đối tượng PIL Image.")
            return None

        mp_face_mesh = mp.solutions.face_mesh

        start_fminit = time.perf_counter()
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.05,
            min_tracking_confidence=0.9
        )
        face_mesh_init_time = time.perf_counter() - start_fminit
        # cv2.imwrite('anh_tang_cuong.jpg', original_image_np)
        # original_image_np = cv2.detailEnhance(original_image_np, sigma_s=10, sigma_r=0.15)
        # cv2.imwrite('anh_tang_cuong2.jpg', original_image_np)
        image_rgb_for_mp = cv2.cvtColor(original_image_np, cv2.COLOR_BGR2RGB)


        #image_rgb_for_mp.save('anh_tang_cuong.jpg')
        image_height, image_width, _ = original_image_np.shape

        output_image_np = np.zeros_like(original_image_np)
        if not_anime == False:
            return output_image_np

        start_fmprocess = time.perf_counter()
        results = face_mesh.process(image_rgb_for_mp)
        face_mesh_process_time = time.perf_counter() - start_fmprocess


        face_processed = False

        if results.multi_face_landmarks:
            logger.info(f"Phát hiện được {len(results.multi_face_landmarks)} khuôn mặt trong {base_filename_for_log}.")
            for face_landmarks in results.multi_face_landmarks:
                face_processed = True
                landmark_points = []
                for i in range(0, 468):
                    pt = face_landmarks.landmark[i]
                    x = int(pt.x * image_width)
                    y = int(pt.y * image_height)
                    landmark_points.append([x, y])

                if not landmark_points:
                    continue

                convex_hull_points = cv2.convexHull(np.array(landmark_points))
                face_mask_cv = np.zeros((image_height, image_width), dtype=np.uint8) # Đổi tên biến để tránh nhầm lẫn
                cv2.fillConvexPoly(face_mask_cv, convex_hull_points, 255)
                segmented_face_part = cv2.bitwise_and(original_image_np, original_image_np, mask=face_mask_cv)
                output_image_np = cv2.bitwise_or(output_image_np, segmented_face_part)

        if not face_processed:
            logger.info(f"Không phát hiện được khuôn mặt nào trong {base_filename_for_log}. Trả về ảnh đen.")

        return output_image_np

    except Exception as e:
        logger.error(f"Đã xảy ra lỗi trong quá trình xử lý segment_face_same_size cho {base_filename_for_log}: {e}", exc_info=True)
        if original_image_np is not None:
            return np.zeros_like(original_image_np)
        return None
    finally:
        if face_mesh:
            face_mesh.close()
        overall_end_time = time.perf_counter()
        logger.debug(f"segment_face_same_size({base_filename_for_log}): Total time: {overall_end_time - overall_start_time:.4f}s "
                    f"(Init FaceMesh: {face_mesh_init_time:.4f}s, Process FaceMesh: {face_mesh_process_time:.4f}s)")

def create_inverse_face_mask_same_size(image_path_or_pil_input, not_anime):
    """
    Tạo mask ngược khuôn mặt có kích thước giống với ảnh gốc
    (mask là một ảnh grayscale, 0 là vùng mặt, 255 là vùng không phải mặt)

    Args:
        image_path_or_pil_input (str or PIL.Image): Đường dẫn đến file ảnh gốc
            hoặc đối tượng PIL Image của ảnh gốc

    Returns:
        PIL.Image: Ảnh mask ngược khuôn mặt có kích thước giống với ảnh gốc
    """
    start_time = time.perf_counter()
    original_pil_image = None
    base_filename_for_log = "input_image_for_mask"

    if isinstance(image_path_or_pil_input, str):
        base_filename_for_log = os.path.basename(image_path_or_pil_input)
        try:
            original_pil_image = Image.open(image_path_or_pil_input).convert("RGB")
        except Exception as e:
            logger.error(f"Lỗi khi mở ảnh gốc từ đường dẫn để lấy kích thước: {image_path_or_pil_input}, {e}")
            return None
    elif isinstance(image_path_or_pil_input, Image.Image):
        original_pil_image = image_path_or_pil_input.convert("RGB")
        base_filename_for_log = "PIL_input_for_mask"
    else:
        logger.error("Đầu vào không phải đường dẫn ảnh hoặc đối tượng PIL Image.")
        return None

    original_width, original_height = original_pil_image.size

    logger.info(f"Bắt đầu segment_face_same_size cho mask generation ({base_filename_for_log}).")
    # segment_face_same_size đã có logging thời gian bên trong nó
    segmented_face_np_bgr = segment_face_same_size(original_pil_image, not_anime)

    if segmented_face_np_bgr is None:
        logger.warning(f"segment_face_same_size trả về None cho {base_filename_for_log}. Tạo mask trắng hoàn toàn.")
        inverse_mask_pil = Image.new("RGB", (original_width, original_height), (255, 255, 255))
    else:
        segmented_face_gray_np = cv2.cvtColor(segmented_face_np_bgr, cv2.COLOR_BGR2GRAY)
        inverse_mask_np_gray = np.where(segmented_face_gray_np > 0, 0, 255).astype(np.uint8)
        inverse_mask_pil = Image.fromarray(inverse_mask_np_gray).convert("RGB")

    end_time = time.perf_counter()
    logger.info(f"create_inverse_face_mask_same_size ({base_filename_for_log}) execution time: {end_time - start_time:.4f} seconds.")
    return inverse_mask_pil
