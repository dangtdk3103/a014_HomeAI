# SDXL InstantStyle service

Minimal extraction of the SDXL Style Match path used by the Django backend.
It intentionally excludes the retired Qwen code and unrelated FLUX garden,
pool, retouch, and layout modes from the original GPU-box monolith.

The service combines:

- Depth Anything V2 for the source-room control image.
- SDXL Depth ControlNet for structural conditioning.
- IP-Adapter ViT-H with InstantStyle block weighting for the reference style.

Install the service-specific dependencies and run it on a CUDA host:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=your-token
./run.sh
```

The Django process should set `SDXL_NEST_URL=http://<private-host>:8090`.
Keep this service on a private network because it does not implement API
authentication; the public authentication boundary is the Django backend.

Example:

```bash
curl http://127.0.0.1:8090/generate \
  -F mode=stylematch \
  -F image=@input.jpg \
  -F ref=@reference.jpg \
  -F roomtype=Bedroom \
  -F seed=1 \
  -o output.png
```
