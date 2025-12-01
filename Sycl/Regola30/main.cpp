/*                  #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h" 

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <chrono>

using namespace sycl;

void input(int &width, int &height, bool &useCpu) {
    std::cout << "Inserire la dimensione dell'automa: ";
    std::cin >> width;

    int defaultHeight = width / 2;
    std::cout << "Inserire il numero di iterazioni [default: " << defaultHeight << "]: ";
    std::cin >> height;
    
    std::cout << "Vuoi eseguire il codice su GPU o CPU (GPU = 0, CPU = 1): ";
    std::cin >> useCpu;
}

queue choose_and_printDevices(bool useCpu, bool print) {
    queue q;

    std::cout << "Vuoi stampare l'elenco dei dispositivi compatibili? (Si = 1, No = 0): ";
    std::cin >> print;
    
    if (print){
        std::cout << "\n=== Dispositivi SYCL disponibili ===\n";

        //scorri tutte le piattaforme (es. Intel, NVIDIA, CPU host, ecc.)
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

int main() {
    //parametri dell'automa
    int width = 1024, height;
    bool useCpu = 0;
    bool print = 0;

    input(width, height, useCpu);

    
    queue q = choose_and_printDevices(useCpu, print);

    std::vector<uint32_t> hostData(width * height, 0); 
    hostData[width / 2] = 1;

    buffer<uint32_t, 1> gridBuffer(hostData.data(), range<1>(width * height));

    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 1; y < height; y++) {
        
        q.submit([&](handler& h) {
            auto acc = gridBuffer.get_access<access::mode::read_write>(h);

            h.parallel_for(range<1>(width), [=](id<1> idx) {
                int x = idx[0];
                
                int prevRowOffset = (y - 1) * width;
                int curRowOffset  = y * width;

                uint32_t L_val = (x == 0) ? acc[prevRowOffset + width - 1] : acc[prevRowOffset + x - 1];
                uint32_t C_val = acc[prevRowOffset + x];
                uint32_t R_val = (x == width - 1) ? acc[prevRowOffset] : acc[prevRowOffset + x + 1];

                uint32_t newBit = L_val ^ (C_val | R_val);

                acc[curRowOffset + x] = newBit;
            });
        });

        q.wait(); 
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    host_accessor hostAcc(gridBuffer, read_only);

    std::vector<unsigned char> pngData(width * height * 4);

    for (int i = 0; i < width * height; i++) {
        uint32_t stato = hostAcc[i];
        
        unsigned char color = (stato == 1) ? 0 : 255;

        pngData[i*4 + 0] = color; // R
        pngData[i*4 + 1] = color; // G
        pngData[i*4 + 2] = color; // B
        pngData[i*4 + 3] = 255;   // Alpha
    }

    if (!stbi_write_png("output.png", width, height, 4, pngData.data(), width * 4)) {
        throw std::runtime_error("Errore nel salvataggio dell'immagine.\n");
        return 1;
    }

    std::cout << "\nImmagine salvata come output.png\n";
    std::cout << "Tempo di puro calcolo: " << elapsed.count() << " secondi.\n";

    return 0;
}
 */




#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h" 

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <chrono>

using namespace sycl;

void input(int &width, int &height, bool &useCpu) {
    std::cout << "Inserire la dimensione dell'automa: ";
    std::cin >> width;

    int defaultHeight = width / 2;
    std::cout << "Inserire il numero di iterazioni [default: " << defaultHeight << "]: ";
    std::cin >> height;
    
    std::cout << "Vuoi eseguire il codice su GPU o CPU (GPU = 0, CPU = 1): ";
    std::cin >> useCpu;
}

queue choose_and_printDevices(bool useCpu, bool print) {
    queue q;

    std::cout << "Vuoi stampare l'elenco dei dispositivi compatibili? (Si = 1, No = 0): ";
    std::cin >> print;
    
    if (print){
        std::cout << "\n=== Dispositivi SYCL disponibili ===\n";

        //scorri tutte le piattaforme (es. Intel, NVIDIA, CPU host, ecc.)
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

int main() {
    //parametri dell'automa
    int width = 1024, height;
    bool useCpu = 0;
    bool print = 0;

    input(width, height, useCpu);

    
    queue q = choose_and_printDevices(useCpu, print);

    std::vector<unsigned char> hostData((size_t)width * height, 0); 
    hostData[width / 2] = 1;

    buffer<unsigned char, 1> gridBuffer(hostData.data(), range<1>((size_t)width * height));

    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 1; y < height; y++) {
        
        q.submit([&](handler& h) {
            auto acc = gridBuffer.get_access<access::mode::read_write>(h);

            h.parallel_for(range<1>(width), [=](id<1> idx) {
                int x = idx[0];
                
                size_t prevRowOffset = (size_t)(y - 1) * width;
                size_t curRowOffset  = (size_t)y * width;

                unsigned char L_val = (x == 0) ? acc[prevRowOffset + width - 1] : acc[prevRowOffset + x - 1];
                unsigned char C_val = acc[prevRowOffset + x];
                unsigned char R_val = (x == width - 1) ? acc[prevRowOffset] : acc[prevRowOffset + x + 1];

                unsigned char newBit = L_val ^ (C_val | R_val);

                acc[curRowOffset + x] = newBit;
            });
        });

        q.wait(); 
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    host_accessor hostAcc(gridBuffer, read_only);

    size_t num_pixels = (size_t)width * height;
    const size_t MAX_PIXELS_TO_SAVE = 100ULL * 1000 * 1000; 

    if (num_pixels > MAX_PIXELS_TO_SAVE) {
        cout << "\nL'immagine e' troppo grande (" << num_pixels << " pixel). Salvataggio saltato per sicurezza.\n";
    } 
    else {
        std::vector<unsigned char> pngData(num_pixels * 4);

        for (size_t i = 0; i < num_pixels; i++) {
            uint32_t stato = hostAcc[i];
            unsigned char color = (stato == 1) ? 0 : 255;

            pngData[i*4 + 0] = color; // R
            pngData[i*4 + 1] = color; // G
            pngData[i*4 + 2] = color; // B
            pngData[i*4 + 3] = 255;   // Alpha
        }

        if (!stbi_write_png("output.png", width, height, 4, pngData.data(), width * 4)) {
            throw std::runtime_error("Errore nel salvataggio dell'immagine.\n");
            return 1;
        }

        std::cout << "\nImmagine salvata come output.png\n";
    }

    std::cout << "Tempo di puro calcolo: " << elapsed.count() << " secondi.\n";

    return 0;
}
