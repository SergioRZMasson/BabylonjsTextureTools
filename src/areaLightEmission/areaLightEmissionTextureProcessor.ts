export interface KernelData {
    kernel: Float32Array;
    kernelSize: number;
    kernelHalfSize: number;
}

const ALPHA_FACTOR = 0.5;
const BLUR_SIZE = 5;

function generateGaussianKernel2(size: number, sigma: number): KernelData {
    if (size % 2 === 0) throw new Error("Kernel size must be odd.");

    const halfSize = Math.floor(size / 2);
    const kernel = new Float32Array(size);

    let sum = 0;

    for (let i = -halfSize; i <= halfSize; ++i) {
        const value = Math.exp(-(i * i) / (2 * sigma * sigma));
        kernel[i + halfSize] = value;
        sum += value;
    }

    if (sum !== 0) {
        for (let i = 0; i < kernel.length; ++i) kernel[i] /= sum;
    }
    return { kernel, kernelSize: size, kernelHalfSize: halfSize };
}

function mirrorIndex(x: number, width: number): number {
    if (x < 0) x = -x;
    if (x >= width) x = 2 * width - 2 - x;
    return x;
}

function transposeImage(
    input: Uint8Array,
    output: Uint8Array,
    width: number,
    height: number,
    channels: number
) {
    for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
            const srcBase = (y * width + x) * channels;
            const dstBase = (x * height + y) * channels;
            for (let c = 0; c < channels; ++c) {
                output[dstBase + c] = input[srcBase + c];
            }
        }
    }
}

function transposeImageBack(
    input: Uint8Array,
    output: Uint8Array,
    width: number,
    height: number,
    channels: number
) {
    for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
            const srcBase = (x * height + y) * channels;
            const dstBase = (y * width + x) * channels;
            for (let c = 0; c < channels; ++c) {
                output[dstBase + c] = input[srcBase + c];
            }
        }
    }
}

function applyGaussianBlurRange(
    input: Uint8Array,
    output: Uint8Array,
    width: number,
    height: number,
    channels: number,
    kernelLibrary: KernelData[]
) {
    const marginStart = Math.floor(width * 0.125);
    const marginEnd = Math.floor(width * 0.875);

    for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
            let targetKernel = 0;
            if (x <= marginStart)
                targetKernel = Math.max(
                    targetKernel,
                    Math.abs(x - marginStart)
                );
            if (y <= marginStart)
                targetKernel = Math.max(
                    targetKernel,
                    Math.abs(y - marginStart)
                );
            if (x >= marginEnd)
                targetKernel = Math.max(targetKernel, Math.abs(x - marginEnd));
            if (y >= marginEnd)
                targetKernel = Math.max(targetKernel, Math.abs(y - marginEnd));

            const kd = kernelLibrary[targetKernel];
            const kHalf = kd.kernelHalfSize;
            const kernel = kd.kernel;

            // copy alpha/unprocessed channel(s) first (if any) so we always have something there
            for (let c = 0; c < channels; ++c) {
                output[(y * width + x) * channels + c] =
                    input[(y * width + x) * channels + c];
            }

            // Process all channels except the last one (mimic C++ behavior: skip alpha)
            for (let c = 0; c < Math.max(0, channels - 1); ++c) {
                let sum = 0;
                for (let kx = -kHalf; kx <= kHalf; ++kx) {
                    const px = mirrorIndex(x + kx, width);
                    const weight = kernel[kx + kHalf];
                    const pixelData = input[(y * width + px) * channels + c];
                    sum += pixelData * weight;
                }
                const val = Math.round(sum);
                output[(y * width + x) * channels + c] =
                    val < 0 ? 0 : val > 255 ? 255 : val;
            }
        }
    }
}

/**
 * processImage - main exported function
 * @param input Uint8Array image data
 * @param width image width
 * @param height image height
 * @returns Uint8Array with blurred image bytes (same channel count)
 */
export function processImage(
    input: Uint8Array,
    width: number,
    height: number
): Uint8Array {
    
    if (!input || width <= 0 || height <= 0) {
        throw new Error("Invalid input");
    }

    const pixelCount = width * height;

    if (input.length % pixelCount !== 0) {
        throw new Error( "Input buffer length is not consistent with width*height");
    }

    const channels = input.length / pixelCount;

    // buffers
    const bufferA = new Uint8Array(input); // working copy
    const bufferB = new Uint8Array(input.length);
    bufferB.fill(255); // match C++ behavior setting output to 255

    // Build kernel library
    const kernelLibrary: KernelData[] = [];
    let kernelSize = BLUR_SIZE;
    let alpha = (kernelSize / 2) * ALPHA_FACTOR;
    kernelLibrary.push(generateGaussianKernel2(kernelSize, alpha));

    for (let i = 1; i < 512; ++i) {
        kernelSize = BLUR_SIZE + i * 2 + 2;
        alpha = (kernelSize / 2) * ALPHA_FACTOR;
        kernelLibrary.push(generateGaussianKernel2(kernelSize, alpha));
    }

    // First pass (X blur)
    applyGaussianBlurRange(
        bufferA,
        bufferB,
        width,
        height,
        channels,
        kernelLibrary
    );

    // Transpose back, blur again, transpose back (mirror C++ flow)
    transposeImageBack(bufferB, bufferA, width, height, channels);
    applyGaussianBlurRange(
        bufferA,
        bufferB,
        width,
        height,
        channels,
        kernelLibrary
    );
    transposeImageBack(bufferB, bufferA, width, height, channels);

    return bufferA;
}
