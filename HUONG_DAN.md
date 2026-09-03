# Hướng dẫn NestAI image backend

Tài liệu này mô tả cách chạy repository đã được clean, cấu hình hai GPU service,
và gọi đúng các luồng Style Match.

## 1. Thành phần hệ thống

```text
Client / Postman
       |
       | Authorization: Api-Key ... hoặc JWT
       v
Django API (port 8000)
       |
       +-- Gemini Vertex AI
       |     +-- phân tích ảnh reference
       |     +-- mở rộng prompt
       |
       +-- FLUX local trên GPU Django
       |     +-- FLUX.1-Depth-dev INT4
       |     +-- FLUX.1-Kontext-dev INT4
       |     +-- FLUX.1-Fill-dev INT4
       |
       +-- SDXL_NEST_URL (private port 8090)
             +-- SDXL base 1.0
             +-- Depth ControlNet
             +-- IP-Adapter ViT-H / InstantStyle
```

Source chính:

- `ai_art/`: Django settings và root URL.
- `deepart/views.py`: endpoint `image_generate` và pipeline dispatch.
- `deepart/nest.py`: Nest routes, `analyze_ref`, Depth/Fill và proxy SDXL.
- `deepart/pipelines.py`: khởi tạo FLUX pipelines dùng chung encoder/VAE.
- `deepart/kontext_swap.py`: Gemini scene analysis cho Kontext.
- `config/nest/`: style, room, color và furniture configuration.
- `services/sdxl-instantstyle/`: SDXL Style Match service tối giản.

## 2. Chuẩn bị máy

Yêu cầu:

- Linux, Python 3.11 và NVIDIA CUDA tương thích PyTorch 2.6.
- Hugging Face account đã chấp nhận license của các FLUX/SDXL model cần dùng.
- Google Cloud project đã bật Vertex AI.
- Không đặt service-account JSON, `.env`, model weights hoặc API key vào Git.

Clone repository và cài Django/FLUX environment:

```bash
git clone <repository-url> nestai-image-backend
cd nestai-image-backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Copy cấu hình mẫu và export bằng shell hoặc process manager:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Các biến bắt buộc:

- `DJANGO_SECRET_KEY`: chuỗi ngẫu nhiên riêng của deployment.
- `DJANGO_ALLOWED_HOSTS`: danh sách hostname phân cách bằng dấu phẩy.
- `GOOGLE_CLOUD_PROJECT`: Vertex AI project.
- `HF_TOKEN`: token có quyền tải model đã gated.
- `SDXL_NEST_URL`: URL private của SDXL service.

Gemini mặc định:

- Prompt expansion: `gemini-2.5-flash-lite`, location `us-central1`.
- Reference analysis: `gemini-3.1-flash-lite`, location `global`.

Có thể dùng Application Default Credentials:

```bash
gcloud auth application-default login
```

Nếu dùng service account, lưu JSON ngoài repository rồi đặt:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/etc/nestai/service-account.json
```

## 3. Khởi tạo Django

```bash
source .venv/bin/activate
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver 0.0.0.0:8000 --noreload
```

Health check:

```bash
curl http://127.0.0.1:8000/
```

Tạo DRF API key bằng Django admin hoặc shell. Raw key chỉ xuất hiện một lần:

```bash
python manage.py shell -c \
  'from rest_framework_api_key.models import APIKey; obj, key = APIKey.objects.create_key(name="postman"); print(key)'
```

## 4. Chạy SDXL InstantStyle

Nên dùng virtual environment riêng vì version `transformers` của SDXL box khác
với Django/FLUX environment.

```bash
cd services/sdxl-instantstyle
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
set -a
source .env
set +a
./run.sh
```

Kiểm tra:

```bash
curl http://127.0.0.1:8090/health
```

Trên máy Django, cấu hình URL private:

```bash
export SDXL_NEST_URL=http://<sdxl-private-host>:8090
```

SDXL service không có authentication riêng. Chỉ mở port `8090` cho Django
host/VPC; không public trực tiếp ra Internet.

## 5. Style Match qua `analyze_ref`

Đây là flow hai request mà app đang sử dụng.

### Bước 1: phân tích ảnh reference

```bash
curl http://127.0.0.1:8000/deepart/analyze_ref \
  -H "Authorization: Api-Key $API_KEY" \
  -F ref=@reference.png
```

Response mẫu:

```json
{
  "style": "Minimalist",
  "room_type": "Bedroom",
  "color_palette": "Beige and warm neutrals",
  "colors": ["beige", "cream", "white", "brown"]
}
```

Model của bước này là Gemini reference-analysis. Ảnh reference được chuyển thành
text metadata; ảnh reference không được gửi trực tiếp vào FLUX Depth.

### Bước 2: áp style lên ảnh input

```bash
curl http://127.0.0.1:8000/deepart/image_generate \
  -H "Authorization: Api-Key $API_KEY" \
  -F control_image=@input.jpg \
  -F nest_route=interior \
  -F prompt="home interior" \
  -F style="Minimalist" \
  -F room="Bedroom" \
  -o output.png
```

Flow này dùng FLUX.1-Depth-dev INT4 để giữ structure của ảnh input. Vì model chỉ
nhận style text từ Gemini nên có thể giữ layout tốt nhưng không sao chép đầy đủ
texture/lighting tinh tế của reference.

## 6. Style Match trực tiếp bằng SDXL

Endpoint Django nhận cả hai ảnh trong một request:

```bash
curl http://127.0.0.1:8000/deepart/nest_stylematch \
  -H "Authorization: Api-Key $API_KEY" \
  -F image=@input.jpg \
  -F ref=@reference.jpg \
  -F roomtype=Bedroom \
  -F seed=1 \
  -o output.png
```

Django chuyển request tới `${SDXL_NEST_URL}/generate` với
`mode=stylematch`. SDXL service xử lý:

1. Depth Anything V2 tạo depth map từ ảnh input.
2. SDXL Depth ControlNet dùng depth map để giữ cấu trúc.
3. IP-Adapter ViT-H nhận trực tiếp ảnh reference.
4. InstantStyle block weighting truyền style vào SDXL.
5. Kết quả được resize về kích thước input đã preprocess và trả PNG.

Header phản hồi:

- `X-Style: stylematch:sdxl-instantstyle`
- `X-IPA: instantstyle` hoặc scalar fallback.

Có thể test thẳng private SDXL service:

```bash
curl http://127.0.0.1:8090/generate \
  -F mode=stylematch \
  -F image=@input.jpg \
  -F ref=@reference.jpg \
  -F roomtype=Bedroom \
  -F seed=1 \
  -o output.png
```

## 7. Các endpoint Django

| Endpoint | Chức năng |
|---|---|
| `GET /` | Health check |
| `POST /deepart/image_generate` | Main generation/dispatch |
| `POST /deepart/analyze_ref` | Gemini phân tích reference |
| `POST /deepart/nest_generate` | Interior Depth restyle |
| `POST /deepart/nest_exterior` | Exterior restyle |
| `POST /deepart/nest_stylematch` | Proxy SDXL Style Match |
| `POST /deepart/nest_retouch` | Retouch bằng mask/Fill |
| `POST /deepart/suggest_brief` | Gemini gợi ý brief |
| `POST /deepart/nest_addpool` | Garden/pool flow |
| `POST /deepart/addpool_submit` | Submit garden/pool async |
| `GET /deepart/addpool_result/<jid>` | Lấy kết quả async |
| `POST /api/token/` | Lấy JWT access/refresh |
| `POST /api/token/refresh/` | Refresh JWT |

## 8. Postman

Import `postman_collection.json`, sau đó đặt collection variables:

- `base_url`: ví dụ `http://127.0.0.1:8000`.
- `api_key`: DRF API key của deployment.

File trong Git không chứa key thật hoặc public EC2 hostname.

## 9. Deploy systemd

Template:

- `deploy/django-marketbot-ai-art.service.example`
- `deploy/sdxl-instantstyle.service.example`
- `deploy/backend.env.example`

Copy và sửa user/path trước khi enable:

```bash
sudo cp deploy/django-marketbot-ai-art.service.example \
  /etc/systemd/system/django-marketbot-ai-art.service
sudo systemctl daemon-reload
sudo systemctl enable --now django-marketbot-ai-art.service
```

Làm tương tự cho SDXL service trên GPU host tương ứng.

## 10. Kiểm tra trước khi push GitHub

```bash
git diff --cached --check
python -m compileall ai_art deepart services/sdxl-instantstyle
git grep -nE 'BEGIN .*PRIVATE KEY|AKIA|AIza|github_pat_|ghp_|sk-'
git status
```

Không push nếu thấy một trong các file sau:

- `.env`
- `db.sqlite3`
- service-account JSON
- `.pem`/private key
- API key thật
- model weights, cache Hugging Face hoặc output images

Nếu credential từng tồn tại trong repository hoặc archive đã chia sẻ, phải
rotate/revoke credential đó trước khi public GitHub repository.
