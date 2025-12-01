#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define PI 3.141592653589793
#include "stb_image.h"
#include "stb_image_write.h"

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <string>
#include <cstring>
#include <chrono>
#include <cmath> 

using namespace std;

struct BlurInfo {
    int width;
    int height;
    int kerdim;
    int offsetX; 
    int offsetY; 
};

vector<char> loadShader(const string& filename) {
    ifstream file(filename, ios::ate | ios::binary);

    if (!file.is_open()) {
        throw runtime_error("Impossibile aprire lo shader: " + filename);
    }

    size_t size = (size_t)file.tellg();
    vector<char> contents(size);
    file.seekg(0);
    file.read(contents.data(), size);
    file.close();

    return contents;
}

void input(int &kerDim, string &pathName, bool &useCpu) {
    int cpuChoice;
    cout << "Inserire la dimensione del kernel: ";
    cin >> kerDim;

    cout << "Inserire il percorso dell'immagine: ";
    cin >> pathName;

    cout << "Vuoi eseguire il codice su GPU o CPU (GPU = 0, CPU = 1): ";
    cin >> cpuChoice;
    useCpu = (cpuChoice == 1);
}

uint32_t chooseMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw runtime_error("Impossibile trovare il tipo di memoria adatto.");
}

VkInstance initVulkan() {
    VkInstance instance;
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Gaussian Blur";
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare un'istanza di Vulkan.");
    }

    return instance;
}

VkPhysicalDevice choose_and_printDevices(bool useCpu, bool print, const vector<VkPhysicalDevice>& devices) {
    int printChoice;
    cout << "Vuoi stampare l'elenco dei dispositivi compatibili? (si = 1, no = 0): ";
    cin >> printChoice;
    print = (printChoice == 1);

    cout << "Trovati " << devices.size() << " dispositivi Vulkan compatibili \n";

    VkPhysicalDevice chosenDevice = VK_NULL_HANDLE;

    for (uint32_t i = 0; i < devices.size(); i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (print){
            cout << " [" << i << "] " << props.deviceName << " (";
                switch (props.deviceType) {
                    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: cout << "GPU integrata"; break;
                    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:  cout << "GPU discreta"; break;
                    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:   cout << "GPU virtuale"; break;
                    case VK_PHYSICAL_DEVICE_TYPE_CPU:           cout << "CPU"; break;
                    default:                                    cout << "Altro tipo"; break;
        }
        cout << ")\n";

    }
        if (useCpu && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
            chosenDevice = devices[i];
        else if (!useCpu && 
                (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
                 props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU))
            chosenDevice = devices[i];
    }

    // Fallback: se non trova quello specifico, prende il primo disponibile
    if (chosenDevice == VK_NULL_HANDLE && !devices.empty()) {
        chosenDevice = devices[0];
        cout << "Attenzione: Dispositivo richiesto non trovato, uso il primo disponibile.\n";
    } else if (chosenDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("Nessun dispositivo Vulkan compatibile trovato.");
    }

    //stampa del device scelto
    VkPhysicalDeviceProperties chosenProps;
    vkGetPhysicalDeviceProperties(chosenDevice, &chosenProps);
    cout << "\nDispositivo scelto: " << chosenProps.deviceName << "\n";

    return chosenDevice;
}

vector<float> createGaussianKernel(int kernelSize, float sigma) {
    std::vector<float> kernel(kernelSize * kernelSize);
    int radius = kernelSize / 2;
    float sum = 0.0f;

    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            float coeff = 1.0f /(2.0f * (float)PI * sigma * sigma);
            float exponent = - (float(x*x + y*y)/(2.0f*sigma*sigma));
            float value = coeff * exp(exponent);
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
    try {
        // --- 1. SETUP INIZIALE ---
        int imgWidth, imgHeight, imgChannels;
        int kerDim;
        string pathName;
        bool useCpu = false;
        bool print = false;
        input(kerDim, pathName, useCpu);

        const char *path = pathName.c_str();
        unsigned char* pixels = stbi_load(path, &imgWidth, &imgHeight, &imgChannels, STBI_rgb_alpha);
        
        if (!pixels) throw runtime_error("Immagine non caricata o percorso errato.");

        VkDeviceSize imageSize = imgWidth * imgHeight * sizeof(uint32_t);

        // Inizializzazione Vulkan standard
        VkInstance instance = initVulkan();

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) throw runtime_error("Nessuna GPU fisica disponibile.");
        vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        VkPhysicalDevice physicalDevice = choose_and_printDevices(useCpu, print, devices);

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        int queueFamilyIndex = -1;
        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilyIndex = i;
                break;
            }
        }

        if (queueFamilyIndex == -1) throw runtime_error("Non esiste una famiglia di code compatibile.");

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

        VkDevice device;
        if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
            throw runtime_error("Impossibile creare un logical device.");
        }

        VkQueue computeQueue;
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue); 

        // Creazione Buffer
        VkBuffer inputBuffer, outputBuffer;
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = imageSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &inputBuffer) != VK_SUCCESS ||
            vkCreateBuffer(device, &bufferInfo, nullptr, &outputBuffer) != VK_SUCCESS)
        {
            throw runtime_error("Impossibile creare input/output buffer.");
        }

        VkMemoryRequirements memReqInput, memReqOutput;
        vkGetBufferMemoryRequirements(device, inputBuffer, &memReqInput);
        vkGetBufferMemoryRequirements(device, outputBuffer, &memReqOutput);

        VkDeviceMemory inputMemory, outputMemory;
        VkMemoryAllocateInfo allocInfoInput{};
        allocInfoInput.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfoInput.allocationSize = memReqInput.size;
        allocInfoInput.memoryTypeIndex = chooseMemoryType(memReqInput.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, physicalDevice);

        VkMemoryAllocateInfo allocInfoOutput{};
        allocInfoOutput.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfoOutput.allocationSize = memReqOutput.size;
        allocInfoOutput.memoryTypeIndex = chooseMemoryType(memReqOutput.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, physicalDevice);

        if (vkAllocateMemory(device, &allocInfoInput, nullptr, &inputMemory) != VK_SUCCESS ||
            vkAllocateMemory(device, &allocInfoOutput, nullptr, &outputMemory) != VK_SUCCESS) {
            throw runtime_error("Impossibile allocare la memoria.");
        }

        vkBindBufferMemory(device, inputBuffer, inputMemory, 0);
        vkBindBufferMemory(device, outputBuffer, outputMemory, 0);

        // Preparazione immagine
        vector<uint32_t> pixelBuffer(imgWidth * imgHeight);
        for (int i = 0; i < imgWidth * imgHeight; i++) {
            pixelBuffer[i] = 
                (uint32_t(pixels[i*4 + 0]) << 0) |
                (uint32_t(pixels[i*4 + 1]) << 8) |
                (uint32_t(pixels[i*4 + 2]) << 16) |
                (uint32_t(pixels[i*4 + 3]) << 24);
        }

        void* data;
        vkMapMemory(device, inputMemory, 0, imageSize, 0, &data);
        memcpy(data, pixelBuffer.data(), (size_t)imageSize);
        vkUnmapMemory(device, inputMemory); 

        // Preparazione Kernel
        float sigma = static_cast<float>(kerDim) / 6.0f;
        if (sigma < 0.1f) sigma = 0.1f;
        
        std::vector<float> gaussianKernel = createGaussianKernel(kerDim, sigma);
        VkDeviceSize kernelSizeBytes = gaussianKernel.size() * sizeof(float);

        VkBuffer kernelBuffer;
        VkDeviceMemory kernelMemory;

        VkBufferCreateInfo kernelBufferInfo{};
        kernelBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        kernelBufferInfo.size = kernelSizeBytes;
        kernelBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        kernelBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &kernelBufferInfo, nullptr, &kernelBuffer) != VK_SUCCESS) throw std::runtime_error("Impossibile creare buffer kernel.");

        VkMemoryRequirements kernelMemReq;
        vkGetBufferMemoryRequirements(device, kernelBuffer, &kernelMemReq);

        VkMemoryAllocateInfo allocInfoKernel{};
        allocInfoKernel.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfoKernel.allocationSize = kernelMemReq.size;
        allocInfoKernel.memoryTypeIndex = chooseMemoryType(kernelMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, physicalDevice);

        if (vkAllocateMemory(device, &allocInfoKernel, nullptr, &kernelMemory) != VK_SUCCESS) throw std::runtime_error("Impossibile allocare memoria kernel.");

        vkBindBufferMemory(device, kernelBuffer, kernelMemory, 0);

        void* kernelData;
        vkMapMemory(device, kernelMemory, 0, kernelSizeBytes, 0, &kernelData);
        memcpy(kernelData, gaussianKernel.data(), static_cast<size_t>(kernelSizeBytes));
        vkUnmapMemory(device, kernelMemory);

        // Shader e Pipeline
        vector<char> shaderCode = loadShader("shaders/gaussian_blur.spv");
        VkShaderModuleCreateInfo shaderModuleInfo{};
        shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleInfo.codeSize = shaderCode.size();
        shaderModuleInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &shaderModule) != VK_SUCCESS) throw runtime_error("Impossibile creare shader module.");

        VkDescriptorSetLayoutBinding bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
        };

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = bindings;

        VkDescriptorSetLayout descriptorSetLayout;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) throw runtime_error("Impossibile creare descriptor set layout.");

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(BlurInfo);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        VkPipelineLayout pipelineLayout;
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) throw runtime_error("Impossibile creare pipeline layout.");

        VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;

        VkDescriptorPool descriptorPool;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) throw runtime_error("Impossibile creare descriptor pool.");

        VkDescriptorSetAllocateInfo allocInfoDS{};
        allocInfoDS.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfoDS.descriptorPool = descriptorPool;
        allocInfoDS.descriptorSetCount = 1;
        allocInfoDS.pSetLayouts = &descriptorSetLayout;

        VkDescriptorSet descriptorSet;
        if (vkAllocateDescriptorSets(device, &allocInfoDS, &descriptorSet) != VK_SUCCESS) throw runtime_error("Impossibile creare descriptor set.");

        VkDescriptorBufferInfo bufferInfos[] = {
            {inputBuffer, 0, imageSize},
            {outputBuffer, 0, imageSize},
            {kernelBuffer, 0, kernelSizeBytes}
        };

        VkWriteDescriptorSet writeSets[3];
        for(int i=0; i<3; i++) {
            writeSets[i] = {};
            writeSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeSets[i].dstSet = descriptorSet;
            writeSets[i].dstBinding = i;
            writeSets[i].descriptorCount = 1;
            writeSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writeSets[i].pBufferInfo = &bufferInfos[i];
        }
        vkUpdateDescriptorSets(device, 3, writeSets, 0, nullptr);

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shaderModule, "main", nullptr};
        pipelineInfo.layout = pipelineLayout;

        VkPipeline pipeline;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) throw runtime_error("Impossibile creare pipeline.");

        VkCommandPoolCreateInfo cmdPoolInfo{};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Importante per poter resettare il buffer

        VkCommandPool commandPool;
        if (vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool) != VK_SUCCESS) throw runtime_error("Impossibile creare command pool.");

        VkCommandBufferAllocateInfo cmdBufAllocInfo{};
        cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdBufAllocInfo.commandPool = commandPool;
        cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdBufAllocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        if (vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &commandBuffer) != VK_SUCCESS) throw runtime_error("Impossibile creare command buffer.");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence;
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) throw runtime_error("Impossibile creare fence.");

        // --- 2. LOOP DI ESECUZIONE (Tile + Sync) ---
        
        cout << "Inizio elaborazione..." << endl;
        auto start = std::chrono::high_resolution_clock::now();

        // Dimensione del blocco. Puoi provare 256 o 512.
        int TILE_SIZE = 512; 

        // Ciclo esterno: processiamo una striscia orizzontale (riga di tile) alla volta
        for (int y = 0; y < imgHeight; y += TILE_SIZE) {
            
            // 1. Resettiamo il buffer per registrare nuovi comandi
            vkResetCommandBuffer(commandBuffer, 0);

            // 2. Iniziamo la registrazione
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
                throw runtime_error("Impossibile iniziare command buffer.");
            }

            // Bind pipeline e descrittori (va fatto ogni volta che si resetta il buffer)
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

            // Ciclo interno: accumuliamo i dispatch per questa riga
            for (int x = 0; x < imgWidth; x += TILE_SIZE) {
                
                int currentTileW = std::min(TILE_SIZE, imgWidth - x);
                int currentTileH = std::min(TILE_SIZE, imgHeight - y);

                // Push Constants con Offset
                BlurInfo tilePush = { imgWidth, imgHeight, kerDim, x, y };
                vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurInfo), &tilePush);

                uint32_t groupCountX = (currentTileW + 15) / 16;
                uint32_t groupCountY = (currentTileH + 15) / 16;

                vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
            }

            // Barriera di memoria per assicurare che la scrittura sia completata per questo batch
            VkBufferMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barrier.srcQueueFamilyIndex = queueFamilyIndex;
            barrier.dstQueueFamilyIndex = queueFamilyIndex;
            barrier.buffer = outputBuffer;
            barrier.offset = 0;
            barrier.size = imageSize;

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
                throw runtime_error("Impossibile chiudere command buffer.");
            }

            // 3. Invio alla GPU (Submit)
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            // Importante: resettare la fence prima di usarla
            vkResetFences(device, 1, &fence);

            if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
                throw runtime_error("Impossibile inviare alla coda.");
            }

            // 4. Attesa Attiva (Sync)
            // Questo blocca la CPU finché la GPU non finisce questa riga.
            // Impedisce al Watchdog di Linux di credere che la GPU sia bloccata.
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
            
            cout << "Completata riga Y: " << y << " / " << imgHeight << "\r" << flush;
        }
        cout << endl;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // --- 3. LETTURA E SALVATAGGIO ---
        vkMapMemory(device, outputMemory, 0, imageSize, 0, &data);
        vector<uint32_t> outPacked(imgWidth * imgHeight);
        memcpy(outPacked.data(), data, (size_t)imageSize);

        vector<unsigned char> outImage(imgWidth * imgHeight * 4);
        for (int i = 0; i < imgWidth * imgHeight; i++) {
            outImage[i * 4 + 0] = (outPacked[i] >> 0) & 0xFF;
            outImage[i * 4 + 1] = (outPacked[i] >> 8) & 0xFF;
            outImage[i * 4 + 2] = (outPacked[i] >> 16) & 0xFF;
            outImage[i * 4 + 3] = (outPacked[i] >> 24) & 0xFF;
        }

        stbi_write_png("output.png", imgWidth, imgHeight, 4, outImage.data(), imgWidth * 4);
        vkUnmapMemory(device, outputMemory);

        // --- 4. CLEANUP ---
        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        
        vkDestroyBuffer(device, inputBuffer, nullptr);
        vkFreeMemory(device, inputMemory, nullptr);
        vkDestroyBuffer(device, outputBuffer, nullptr);
        vkFreeMemory(device, outputMemory, nullptr);
        vkDestroyBuffer(device, kernelBuffer, nullptr);
        vkFreeMemory(device, kernelMemory, nullptr);

        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);

        stbi_image_free(pixels);
        cout << "Immagine salvata come output.png\n";
        cout << "Tempo totale: " << elapsed.count() << " secondi\n";

    } catch (const std::exception& e) {
        std::cerr << "Errore critico: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}