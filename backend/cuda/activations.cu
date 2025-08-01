extern "C" {

__global__ void sigmoid_forward(const float* input, float* output, int n, int inplace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = input[idx];
        float res = 1.0f / (1.0f + expf(-v));
        if (inplace) {
            ((float*)input)[idx] = res;
        } else {
            output[idx] = res;
        }
    }
}

__global__ void tanh_forward(const float* input, float* output, int n, int inplace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = input[idx];
        float res = tanhf(v);
        if (inplace) {
            ((float*)input)[idx] = res;
        } else {
            output[idx] = res;
        }
    }
}

__global__ void leakyrelu_forward(const float* input, float* output, int n, float alpha, int inplace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = input[idx];
        float res = (v > 0) ? v : alpha * v;
        if (inplace) {
            ((float*)input)[idx] = res;
        } else {
            output[idx] = res;
        }
    }
}

__global__ void elu_forward(const float* input, float* output, int n, float alpha, int inplace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = input[idx];
        float res = (v > 0) ? v : alpha * (expf(v) - 1.0f);
        if (inplace) {
            ((float*)input)[idx] = res;
        } else {
            output[idx] = res;
        }
    }
}

__global__ void relu_forward(const float* input, float* output, unsigned char* mask, int n, int inplace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = input[idx];
        float res = fmaxf(0.0f, v);
        if (inplace) {
            ((float*)input)[idx] = res;
        } else {
            output[idx] = res;
        }
        mask[idx] = (v > 0.0f) ? 1 : 0;  
    }
}

void launchSigmoid(const float* input, float* output, int n, int inplace) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoid_forward<<<blocks, threads>>>(input, output, n, inplace);
}

void launchTanh(const float* input, float* output, int n, int inplace) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    tanh_forward<<<blocks, threads>>>(input, output, n, inplace);
}

void launchLeakyReLU(const float* input, float* output, int n, float alpha, int inplace) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    leakyrelu_forward<<<blocks, threads>>>(input, output, n, alpha, inplace);
}

void launchELU(const float* input, float* output, int n, float alpha, int inplace) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    elu_forward<<<blocks, threads>>>(input, output, n, alpha, inplace);
}

void launchReLU(const float* input, float* output, int n, int inplace) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_forward<<<blocks, threads>>>(input, output, n, inplace);
}

} // extern "C"
