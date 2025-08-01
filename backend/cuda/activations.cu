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

__global__ void sigmoid_backward(const float* output, const float* grad_output, float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float y = output[idx];
        grad_input[idx] = grad_output[idx] * y * (1.0f - y);
    }
}

__global__ void tanh_backward(const float* output, const float* grad_output, float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float y = output[idx];
        grad_input[idx] = grad_output[idx] * (1.0f - y * y);
    }
}

__global__ void leakyrelu_backward(const float* input, const float* grad_output, float* grad_input, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        grad_input[idx] = grad_output[idx] * ((x > 0) ? 1.0f : alpha);
    }
}

__global__ void elu_backward(const float* input, const float* grad_output, float* grad_input, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float grad = (x > 0) ? 1.0f : alpha * expf(x);
        grad_input[idx] = grad_output[idx] * grad;
    }
}

__global__ void relu_backward(const float* grad_output, float* grad_input, const unsigned char* mask, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * mask[idx];
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

void launchReLU(const float* input, float* output, unsigned char* mask, int n, int inplace) { 
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_forward<<<blocks, threads>>>(input, output, mask, n, inplace);
}

void launchSigmoidBackward(const float* output, const float* grad_output, float* grad_input, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoid_backward<<<blocks, threads>>>(output, grad_output, grad_input, n);
}

void launchTanhBackward(const float* output, const float* grad_output, float* grad_input, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    tanh_backward<<<blocks, threads>>>(output, grad_output, grad_input, n);
}

void launchLeakyReLUBackward(const float* input, const float* grad_output, float* grad_input, int n, float alpha) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    leakyrelu_backward<<<blocks, threads>>>(input, grad_output, grad_input, n, alpha);
}

void launchELUBackward(const float* input, const float* grad_output, float* grad_input, int n, float alpha) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    elu_backward<<<blocks, threads>>>(input, grad_output, grad_input, n, alpha);
}

void launchReLUBackward(const float* grad_output, float* grad_input, const unsigned char* mask, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_backward<<<blocks, threads>>>(grad_output, grad_input, mask, n);
}

} // extern "C"
