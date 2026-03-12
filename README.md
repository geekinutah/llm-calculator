# LLM GPU Throughput Calculator

A zero-dependency web calculator for estimating LLM inference throughput using the
roofline model. Supports HuggingFace model auto-fetch, multi-GPU tensor parallelism,
and all major precisions (FP32 → FP4).

## Project Structure

```
llm-calculator/
├── server.py           # Zero-dep Python stdlib HTTP server
├── static/
│   ├── index.html      # HTML structure
│   ├── style.css       # Styles
│   └── app.js          # Calculator logic
├── Dockerfile
├── docker-compose.yml
└── k8s/
    ├── configmap.yaml
    ├── deployment.yaml
    ├── service.yaml
    └── ingress.yaml
```

## Running Locally

### Direct (Python)
```bash
python3 server.py
# Open http://localhost:8080
```

### Docker
```bash
docker build -t llm-calculator .
docker run -p 8080:8080 llm-calculator
```

### Docker Compose
```bash
docker compose up
```

## Deploying to Kubernetes

1. **Build and push your image:**
   ```bash
   docker build -t your-registry/llm-calculator:latest .
   docker push your-registry/llm-calculator:latest
   ```

2. **Update the image reference** in `k8s/deployment.yaml`:
   ```yaml
   image: your-registry/llm-calculator:latest
   ```

3. **Update the hostname** in `k8s/ingress.yaml`:
   ```yaml
   host: llm-calculator.yourdomain.com
   ```

4. **Apply all manifests:**
   ```bash
   kubectl apply -f k8s/
   ```

5. **Check rollout:**
   ```bash
   kubectl rollout status deployment/llm-calculator
   kubectl get pods -l app=llm-calculator
   ```

## Configuration

| Environment Variable | Default | Description          |
|----------------------|---------|----------------------|
| `PORT`               | `8080`  | HTTP listening port  |

## How It Works

The calculator uses the **roofline model** to estimate tokens/second:

- **Memory-bound** (most decode workloads): `TPS = (GPU_BW × N_GPUs) / model_bytes × batch`
- **Compute-bound** (large batch prefill): `TPS = (TFLOPS × N_GPUs) / (FLOPs_per_token / batch)`

The bottleneck is determined by comparing **arithmetic intensity** (ops/byte) against
the GPU's **ridge point** (peak TFLOPS / peak BW). A realistic MFU (40–45%) is applied
to account for real-world efficiency.

### VRAM breakdown
| Segment     | Formula                                          |
|-------------|--------------------------------------------------|
| Weights     | `params × bytes_per_param`                       |
| KV Cache    | `2 × layers × kv_heads × head_dim × seq × batch × 2B` |
| Activations | `hidden × context × batch × 4B` (peak fwd pass) |
| CUDA/FW     | ~1 GB fixed overhead                             |

## GPU Precision Support

| Precision | A100 | H100 | H200 | B200 |
|-----------|------|------|------|------|
| FP32      | ✓    | ✓    | ✓    | ✓    |
| BF16/FP16 | ✓    | ✓    | ✓    | ✓    |
| FP8       | ✗    | ✓    | ✓    | ✓    |
| INT8      | ✓    | ✓    | ✓    | ✓    |
| INT4 *    | ✓    | ✓    | ✓    | ✓    |
| FP4       | ✗    | ✗    | ✗    | ✓    |

\* INT4 dequantizes to INT8/FP16 at runtime — no additional TFLOPS gain over INT8.
