jingwenwang@mac api % uv venv --python 3.9 ws_predictor
Using CPython 3.9.6 interpreter at: /Applications/Xcode.app/Contents/Developer/usr/bin/python3
Creating virtual environment at: ws_predictor
Activate with: source ws_predictor/bin/activate
jingwenwang@mac api % source ws_predictor/bin/activate
(ws_predictor) jingwenwang@mac api % pwd
/Users/jingwenwang/CascadeProjects/predictor/api

# Running with Docker

## Build the Docker image

```sh
docker build -t predictor-api .
```

## Run the Docker container

```sh
docker run -p 5000:5000 predictor-api
```

The API will be available at http://localhost:5000

---

**Note:**
- The application expects access to AWS S3 (see `load_from_bucket` in `api.py`). You may need to provide AWS credentials via environment variables or volume mounts for full functionality.
- The default command runs the Flask development server. For production, consider using Gunicorn or a similar WSGI server.
