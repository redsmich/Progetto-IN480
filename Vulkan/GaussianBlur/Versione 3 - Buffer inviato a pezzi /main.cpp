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

    unsigned int size = file.tellg();

    vector<char> contents(size);
    file.seekg(0);
    file.read(contents.data(), size);

    return contents;
}

void input(int &kerDim, string &pathName, bool &useCpu) {
    int tmp = 0;
    kerDim = 301;
    cout << "Inserire la dimensione del kernel: [Premere 0 per il valore di default: " << kerDim << "]: ";
    cin >> tmp;
    if (tmp != 0)
    kerDim = tmp;

    cout << "Inserire il percorso dell'immagine [Percorso di esempio: images/pappagallo.jpg]: ";
    cin >> pathName;

    cout << "Vuoi eseguire il codice su GPU o CPU (GPU = 0, CPU = 1): ";
    cin >> useCpu;
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
    appInfo.pApplicationName = "Gaussian Blur Separabile";
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
    cout << "Vuoi stampare l'elenco dei dispositivi compatibili? (Si = 1, No = 0): ";
    cin >> print;
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

    if (chosenDevice == VK_NULL_HANDLE)
        throw runtime_error("Nessun dispositivo Vulkan compatibile trovato per la scelta richiesta.");

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
    //caricamento immagine
    int imgWidth, imgHeight, imgChannels;

    int kerDim;
    string pathName;
    bool useCpu = 0;
    bool print = 0;

    input(kerDim, pathName, useCpu);

    //se il kernel è pari aggiungiamo uno per renderlo dispari
    if (kerDim % 2 == 0) kerDim++;

    const char *path = pathName.c_str();
    unsigned char* pixels = stbi_load(path, &imgWidth, &imgHeight, &imgChannels, STBI_rgb_alpha);
    
    if (!pixels) {
        throw runtime_error("Immagine non caricata o percorso errato.");
    }

    //calcolo dimensioni buffer
    unsigned int imageSize = imgWidth * imgHeight * sizeof(uint32_t);

    // Inizializzazione Vulkan
    VkInstance instance = initVulkan();

    //scelta del physical device (GPU)
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw runtime_error("Nessuna GPU fisica disponibile per Vulkan.");
    }
    vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    //scelta tra cpu o gpu
    VkPhysicalDevice physicalDevice = choose_and_printDevices(useCpu, print, devices);

    //creazione logical device e compute queue
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

    if (queueFamilyIndex == -1) {
        throw runtime_error("Non esiste una famiglia di code compatibile.");
    }

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

    // Creazione Buffer I/O
    VkBuffer inputBuffer, outputBuffer;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &inputBuffer) != VK_SUCCESS)
    {
        throw runtime_error("Impossibile creare input buffer.");
    }

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &outputBuffer) != VK_SUCCESS)
    {
        throw runtime_error("Impossibile creare output buffer.");
    }

    //allocazione memoria per i buffer
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

    //compatto i pixel per rendere l'immagine compatibile con lo shader
    vector<uint32_t> pixelBuffer(imgWidth * imgHeight);

    for (int i = 0; i < imgWidth * imgHeight; i++) {
        pixelBuffer[i] = 
            (uint32_t(pixels[i*4 + 0]) << 0) |
            (uint32_t(pixels[i*4 + 1]) << 8) |
            (uint32_t(pixels[i*4 + 2]) << 16) |
            (uint32_t(pixels[i*4 + 3]) << 24);
    }

    //copia i pixel in un'area di memoria condivisa da CPU e GPU
    void* data;
    vkMapMemory(device, inputMemory, 0, imageSize, 0, &data);
    memcpy(data, pixelBuffer.data(), imageSize);
    vkUnmapMemory(device, inputMemory); 

    //preparo il kernel
    float sigma = static_cast<float>(kerDim) / 6.0f;
    std::vector<float> gaussianKernel = createGaussianKernel(kerDim, sigma);
    VkDeviceSize kernelSizeBytes = gaussianKernel.size() * sizeof(float);

    VkBuffer kernelBuffer;
    VkDeviceMemory kernelMemory;

    VkBufferCreateInfo kernelBufferInfo{};
    kernelBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    kernelBufferInfo.size = kernelSizeBytes;
    kernelBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    kernelBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &kernelBufferInfo, nullptr, &kernelBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Impossibile creare il buffer per il kernel.");
    }

    VkMemoryRequirements kernelMemReq;
    vkGetBufferMemoryRequirements(device, kernelBuffer, &kernelMemReq);

    VkMemoryAllocateInfo allocInfoKernel{};
    allocInfoKernel.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoKernel.allocationSize = kernelMemReq.size;
    allocInfoKernel.memoryTypeIndex = chooseMemoryType(kernelMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, physicalDevice);

    if (vkAllocateMemory(device, &allocInfoKernel, nullptr, &kernelMemory) != VK_SUCCESS) {
        throw std::runtime_error("Impossibile allocare memoria per il kernel.");
    }

    vkBindBufferMemory(device, kernelBuffer, kernelMemory, 0);

    void* kernelData;
    vkMapMemory(device, kernelMemory, 0, kernelSizeBytes, 0, &kernelData);
    memcpy(kernelData, gaussianKernel.data(), static_cast<size_t>(kernelSizeBytes));
    vkUnmapMemory(device, kernelMemory);

    //caricamento shader
    vector<char> shaderCode = loadShader("shaders/gaussian_blur.spv");
    if (shaderCode.empty()) {
        throw runtime_error("Impossibile caricare lo shader.");
    }

    VkShaderModuleCreateInfo shaderModuleInfo{};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.codeSize = shaderCode.size();
    shaderModuleInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare lo shader module (per la GPU).");
    }

    //creazione descriptor set layout
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
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare un set layout descriptor.");
    }

    //pipeline layout
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
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare un pipeline layout.");
    }

    //descriptor pool
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare un descriptor pool.");
    }

    //allocazione descriptor set
    VkDescriptorSetAllocateInfo allocInfoDS{};
    allocInfoDS.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfoDS.descriptorPool = descriptorPool;
    allocInfoDS.descriptorSetCount = 1;
    allocInfoDS.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    if (vkAllocateDescriptorSets(device, &allocInfoDS, descriptorSets.data()) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare un descriptor set.");
    }

    //link dei buffer al descriptor set
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

    //creazione compute pipeline
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shaderModule, "main", nullptr};
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare una pipeline di calcolo.");
    }

    //creazione command buffer
    VkCommandPoolCreateInfo cmdPoolInfo{};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; //importante per poter resettare il buffer

    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare una command pool.");
    }

    VkCommandBufferAllocateInfo cmdBufAllocInfo{};
    cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocInfo.commandPool = commandPool;
    cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    if (vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &commandBuffer) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare un command buffer.");
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare fence.");
    } 

    auto start = std::chrono::high_resolution_clock::now();

    //dimensione del blocco (256 o 512).
    int TILE_SIZE = 512; 

    //processiamo una striscia orizzontale
    for (int y = 0; y < imgHeight; y += TILE_SIZE) {
        
        vkResetCommandBuffer(commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw runtime_error("Impossibile iniziare command buffer.");
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        //accumuliamo i dispatch per questa riga
        for (int x = 0; x < imgWidth; x += TILE_SIZE) {
            
            int currentTileW = std::min(TILE_SIZE, imgWidth - x);
            int currentTileH = std::min(TILE_SIZE, imgHeight - y);

            BlurInfo tilePush = { imgWidth, imgHeight, kerDim, x, y };
            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurInfo), &tilePush);

            uint32_t groupCountX = (currentTileW + 15) / 16;
            uint32_t groupCountY = (currentTileH + 15) / 16;

            vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
        }

        //barriera per sincronizzare
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

        //esecuzione shader
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        //resettiamo la fence prima di usarla di nuovo
        vkResetFences(device, 1, &fence);

        if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
            throw runtime_error("Impossibile inviare comandi alla coda.");
        }

        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        
    }

    //fine misurazione tempo di esecuzione
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    //leggo output
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

    //pulizia delle variabili 
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
    cout << "Tempo di puro calcolo: " << elapsed.count() << " secondi\n";


    return 0;
}
