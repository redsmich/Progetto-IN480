#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image.h"
#include "include/stb_image_write.h"
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    const char* input_path = "images/input.png";
    const char* output_path = "images/output_copy.png";

    int width, height, channels;
    unsigned char* img = stbi_load(input_path, &width, &height, &channels, 0);
    if (!img) {
        std::cerr << "Errore caricamento " << input_path << "\n";
        return 1;
    }

    std::cout << "Caricata immagine: " << width << "x" << height 
              << " canali=" << channels << "\n";

    if (!stbi_write_png(output_path, width, height, channels, img, width * channels)) {
        std::cerr << "Errore salvataggio " << output_path << "\n";
        stbi_image_free(img);
        return 1;
    }

    stbi_image_free(img);
    std::cout << "Immagine copiata con successo in " << output_path << "\n";
    return 0;
}
