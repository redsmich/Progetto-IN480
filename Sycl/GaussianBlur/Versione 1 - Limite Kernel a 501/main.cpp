#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define PI 3.141592653589793

#include "include/stb_image.h"
#include "include/stb_image_write.h"

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>

using namespace sycl;

void input(int &kerDim, std::string &pathName, bool &useCpu) {
    int tmp = 0;
    kerDim = 301;
    std::cout << "Inserire la dimensione del kernel: [Premere 0 per il valore di default: " << kerDim << "]: ";
    std::cin >> tmp;
    if (tmp != 0)
    kerDim = tmp;

    std::cout << "Inserire il percorso dell'immagine [Percorso di esempio: images/pappagallo.jpg]: ";
    std::cin >> pathName;

    std::cout << "Vuoi eseguire il codice su GPU o CPU (GPU = 0, CPU = 1): ";
    std::cin >> useCpu;
}

queue choose_and_printDevices(bool useCpu, bool print) {
    queue q;

    std::cout << "Vuoi stampare l'elenco dei dispositivi compatibili? (Si = 1, No = 0): ";
    std::cin >> print;
    
    if (print){
        std::cout << "\n=== Dispositivi SYCL disponibili ===\n";

        //scorri tutte le piattaforme
        int deviceIndex = 0;
        for (const auto &plat : platform::get_platforms()) {
            std::cout << "Piattaforma: " << plat.get_info<info::platform::name>() << "\n";

            //scorri tutti i device di ogni piattaforma
            for (const auto &dev : plat.get_devices()) {
                std::cout << "  [" << deviceIndex++ << "] "
                    << dev.get_info<info::device::name>() << "  ("
                    << dev.get_info<info::device::vendor>() << ")\n";
            }
        }
    }

    if (useCpu){
        q = queue{cpu_selector_v};
    }
    else{
        q = queue{gpu_selector_v};
    }

    std::cout << "\nEseguendo su: "
     << q.get_device().get_info<info::device::name>() << "\n\n";
    
    return q;
}

std::vector<float> createGaussianKernel(int kernelSize, float sigma) {
    std::vector<float> kernel(kernelSize * kernelSize);
    int radius = kernelSize / 2;
    float sum = 0.0f;

    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            float coeff = 1.0f / (2.0f * (float)PI * sigma * sigma);
            float exponent = - (float(x*x + y*y) / (2.0f * sigma * sigma));
            float value = coeff * std::exp(exponent);
            kernel[(x + radius) * kernelSize + (y + radius)] = value;

            sum += value;
        }
    }

    //normalizzazione del kernel
    for (int i = 0; i < kernelSize * kernelSize; i++){
        kernel[i] /= sum;
    }

    return kernel;
}

int main() {
    int kerDim=301;
    std::string pathName;
    bool useCpu = 0;
    bool print = 0;
    int width, height, channels;

    input(kerDim, pathName, useCpu);

    //se il kernel è pari aggiungiamo uno per renderlo dispari
    if (kerDim % 2 == 0) kerDim++;

    const char *path = pathName.c_str();

    unsigned char* img = stbi_load(path, &width, &height, &channels, 0);
    if (!img) {
        throw std::runtime_error("Errore nel caricamento dell'immagine.");
        return 1;
    }

    if (channels != 3 && channels != 4) {
        throw std::runtime_error("L'immagine deve avere 3 (RGB) o 4 (RGBA) canali.");
        stbi_image_free(img);
        return 1;
    }

    std::vector<float> input(width * height * channels);
    std::vector<float> output(width * height * channels, 0.f); 

    for (int i = 0; i < width * height * channels; i++) {
        input[i] = static_cast<float>(img[i]) / 255.f;
    }
    stbi_image_free(img);

    queue q = choose_and_printDevices(useCpu, print);
    
    buffer<float, 1> buf_in(input.data(), range<1>(input.size()));
    buffer<float, 1> buf_out(output.data(), range<1>(output.size()));

    float sigma = static_cast<float>(kerDim) / 6.0f;
    std::vector<float> gaussianKernel = createGaussianKernel(kerDim, sigma);
    buffer<float, 1> buf_kernel(gaussianKernel.data(), range<1>(gaussianKernel.size()));

    int radius = kerDim / 2; 
    auto start = std::chrono::high_resolution_clock::now();

    q.submit([&](handler& h) {
        accessor acc_in(buf_in, h, read_only);
        accessor acc_out(buf_out, h, write_only, no_init);
        accessor acc_kernel(buf_kernel, h, read_only);

        h.parallel_for(range<2>(height, width), [=](id<2> idx) {
        int y = idx[0];
        int x = idx[1];

        /*auto wrap = [](int coord, int maxVal) -> int {
            int modCoord = coord % maxVal;
            return (modCoord < 0) ? modCoord + maxVal : modCoord;
        };

        auto clampCoord = [](int coord, int maxVal) -> int {
        return sycl::clamp(coord, 0, maxVal - 1);
        };*/

        for (int c = 0; c < channels; c++) {
            float sum = 0.f;

            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int ny = sycl::clamp(y + ky, 0, height - 1);
                    int nx = sycl::clamp(x + kx, 0, width - 1);
                    int idx_in = (ny * width + nx) * channels + c;

                    sum = sum + acc_in[idx_in] * acc_kernel[(ky + radius) * kerDim + (kx + radius)];
                }
            }

            int idx_out = (y * width + x) * channels + c;
            acc_out[idx_out] = sum;
        }
    });

    });
    q.wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    host_accessor host_out(buf_out, read_only);
    for (int i = 0; i < width * height * channels; i++) {
        output[i] = host_out[i];
    }
   

    std::vector<unsigned char> out_img(width * height * channels);
    for (int i = 0; i < width * height * channels; i++) {
        float val = std::clamp(output[i], 0.f, 1.f);
        out_img[i] = static_cast<unsigned char>(val * 255.f);
    }

    if (!stbi_write_png("images/output.png", width, height, channels, out_img.data(), width * channels)) {
        throw std::runtime_error("Errore nel salvataggio dell'immagine.\n");
        return 1;
    }

    std::cout << "Immagine salvata come output.png\n";
    std::cout << "Tempo di puro calcolo: " << elapsed.count() << " secondi.\n";
    
    return 0;
}
