# NestAI image backend

Django/DRF GPU service for interior and exterior redesign, image retouching,
reference-style analysis, and FLUX Kontext room reorganization.

Xem [HUONG_DAN.md](HUONG_DAN.md) để cài đặt, chạy hai GPU service và test đầy
đủ các flow bằng curl/Postman.

## Architecture

The application starts in `ai_art/urls.py` and exposes the API routes declared
in `deepart/urls.py`.

- `deepart/views.py` owns `/deepart/image_generate` and dispatches to inpaint,
  structural-control, or Kontext generation.
- `deepart/pipelines.py` builds shared-encoder FLUX pipelines with Nunchaku INT4
  transformers.
- `deepart/nest.py` implements Nest-specific routes, reference analysis,
  Depth/Fill generation, and optional Flask fallbacks.
- `deepart/kontext_swap.py` uses Gemini to analyze a room before FLUX Kontext
  reorganizes it.
- `config/nest/` contains the styles, rooms, colors, and furniture data required
  by `deepart/nest.py`.
- `services/sdxl-instantstyle/` is the extracted SDXL Depth ControlNet +
  IP-Adapter service used by one-request Style Match.

Reference-style application is a two-request flow:

1. Send the reference image to `POST /deepart/analyze_ref`.
2. Send the source image to `POST /deepart/image_generate`, passing the returned
   `style` with `nest_route=interior`.

`POST /deepart/nest_stylematch` is a separate one-request route that can proxy
to the configured SDXL InstantStyle service.

## Requirements

- Linux and Python 3.11
- NVIDIA GPU with a CUDA-compatible PyTorch 2.6 runtime
- Hugging Face access to the gated FLUX repositories
- A Google Cloud project with Vertex AI enabled

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Export the values from `.env` through your shell, process manager, or systemd
environment file. Do not commit a service-account JSON file. Prefer Application
Default Credentials; otherwise point `GOOGLE_APPLICATION_CREDENTIALS` to a file
stored outside this repository.

Initialize the API-key/auth database and run the development server:

```bash
python manage.py migrate
python manage.py runserver 0.0.0.0:8000 --noreload
```

Import `postman_collection.json`, set its `base_url` and `api_key` variables,
then choose an endpoint. Model initialization occurs during Django startup and
can take several minutes.

## Configuration

The main environment variables are documented in `.env.example`. Models and
LoRAs are downloaded by Diffusers/Nunchaku at runtime and are intentionally not
stored in Git.

Production deployment should use a proper WSGI/ASGI server and a process manager.
`deploy/django-marketbot-ai-art.service.example` documents the current systemd
shape but must be adapted to the target host.
