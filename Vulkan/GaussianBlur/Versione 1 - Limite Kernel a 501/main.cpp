#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define PI 3.141592653589793
#include "include/stb_image.h"
#include "include/stb_image_write.h"

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <string>
#include <cstring>
#include <chrono>
#include <cmath> // Aggiunto per exp()

using namespace std;

struct BlurInfo {
    int width;
    int height;
    int kerdim;
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
    cout << "Enter kernel size (e.g., 3, 5, 7...): ";
    cin >> kerDim;

    cout << "Enter image path (e.g. images/pappagallo.jpg): ";
    cin >> pathName;

    cout << "Do you want to run on GPU or CPU (GPU = 0, CPU = 1): ";
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
    cout << "Do you want to print the list of compatible devices? (yes = 1, no = 0): ";
    cin >> printChoice;
    print = (printChoice == 1);

    cout << "Found " << devices.size() << " compatible Vulkan devices \n";

    VkPhysicalDevice chosenDevice = VK_NULL_HANDLE;

    for (uint32_t i = 0; i < devices.size(); i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (print){
            cout << " [" << i << "] " << props.deviceName << " (";
                switch (props.deviceType) {
                    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: cout << "Integrated GPU"; break;
                    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:  cout << "Discrete GPU"; break;
                    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:   cout << "Virtual GPU"; break;
                    case VK_PHYSICAL_DEVICE_TYPE_CPU:           cout << "CPU"; break;
                    default:                                    cout << "Other type"; break;
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
    cout << "\nChosen device: " << chosenProps.deviceName << "\n";

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
        //caricamento immagine
        int imgWidth, imgHeight, imgChannels;

        int kerDim;
        string pathName;
        bool useCpu = false;
        bool print = false;
        input (kerDim, pathName, useCpu);

        const char *path = pathName.c_str();
        unsigned char* pixels = stbi_load(path, &imgWidth, &imgHeight, &imgChannels, STBI_rgb_alpha);
        
        if (!pixels) {
            throw runtime_error("Immagine non caricata o percorso errato.");
        }

        //calcolo dimensioni buffer
        VkDeviceSize imageSize = imgWidth * imgHeight * sizeof(uint32_t);

        //inizializzazione Vulkan
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

        //creazione del buffer I/O
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

        //allocazione memoria per i buffer
        VkMemoryRequirements memReqInput, memReqOutput;
        vkGetBufferMemoryRequirements(device, inputBuffer, &memReqInput);
        vkGetBufferMemoryRequirements(device, outputBuffer, &memReqOutput);

        VkDeviceMemory inputMemory, outputMemory;
        VkMemoryAllocateInfo allocInfoInput{};
        allocInfoInput.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfoInput.allocationSize = memReqInput.size;
        allocInfoInput.memoryTypeIndex = chooseMemoryType(memReqInput.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            physicalDevice);

        VkMemoryAllocateInfo allocInfoOutput{};
        allocInfoOutput.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfoOutput.allocationSize = memReqOutput.size;
        allocInfoOutput.memoryTypeIndex = chooseMemoryType(memReqOutput.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            physicalDevice);

        if (vkAllocateMemory(device, &allocInfoInput, nullptr, &inputMemory) != VK_SUCCESS ||
            vkAllocateMemory(device, &allocInfoOutput, nullptr, &outputMemory) != VK_SUCCESS) {
            throw runtime_error("Impossibile allocare la memoria.");
        }

        vkBindBufferMemory(device, inputBuffer, inputMemory, 0);
        vkBindBufferMemory(device, outputBuffer, outputMemory, 0);

        vector<uint32_t> pixelBuffer(imgWidth * imgHeight); // compatto i pixel per rendere l'immagine compatibile con lo shader

        for (int i = 0; i < imgWidth * imgHeight; i++) {
            pixelBuffer[i] = 
                (uint32_t(pixels[i*4 + 0]) << 0) |   // R
                (uint32_t(pixels[i*4 + 1]) << 8) |   // G
                (uint32_t(pixels[i*4 + 2]) << 16) |  // B
                (uint32_t(pixels[i*4 + 3]) << 24);   // A
        }

        //copia i pixel in un'area di memoria condivisa da CPU e GPU
        void* data;
        vkMapMemory(device, inputMemory, 0, imageSize, 0, &data);
        memcpy(data, pixelBuffer.data(), (size_t)imageSize);
        vkUnmapMemory(device, inputMemory); 

        //preparo il kernel 
        float sigma = static_cast<float>(kerDim) / 6.0f;
        if (sigma < 0.1f) sigma = 0.1f; // Evita divisione per zero o sigma troppo piccolo
        
        std::vector<float> gaussianKernel = createGaussianKernel(kerDim, sigma);
        VkDeviceSize kernelSizeBytes = gaussianKernel.size() * sizeof(float);

        VkBuffer kernelBuffer;
        VkDeviceMemory kernelMemory;

        VkBufferCreateInfo kernelBufferInfo{};
        kernelBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        kernelBufferInfo.size = kernelSizeBytes;
        kernelBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        kernelBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &kernelBufferInfo, nullptr, &kernelBuffer) != VK_SUCCESS)
            throw std::runtime_error("Impossibile creare il buffer per il kernel.");

        VkMemoryRequirements kernelMemReq;
        vkGetBufferMemoryRequirements(device, kernelBuffer, &kernelMemReq);

        VkMemoryAllocateInfo allocInfoKernel{};
        allocInfoKernel.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfoKernel.allocationSize = kernelMemReq.size;
        allocInfoKernel.memoryTypeIndex = chooseMemoryType(kernelMemReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            physicalDevice);

        if (vkAllocateMemory(device, &allocInfoKernel, nullptr, &kernelMemory) != VK_SUCCESS)
            throw std::runtime_error("Impossibile allocare memoria per il kernel.");

        vkBindBufferMemory(device, kernelBuffer, kernelMemory, 0);

        void* kernelData;
        vkMapMemory(device, kernelMemory, 0, kernelSizeBytes, 0, &kernelData);
        memcpy(kernelData, gaussianKernel.data(), static_cast<size_t>(kernelSizeBytes));
        vkUnmapMemory(device, kernelMemory);

        //caricamento shader
        vector<char> shaderCode = loadShader("shaders/gaussian_blur.spv");

        VkShaderModuleCreateInfo shaderModuleInfo{};
        shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleInfo.codeSize = shaderCode.size();
        shaderModuleInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw runtime_error("Impossibile creare lo shader module.");
        }

        //creazione descriptor set layout
        VkDescriptorSetLayoutBinding inputBinding{};
        inputBinding.binding = 0;
        inputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        inputBinding.descriptorCount = 1;
        inputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding outputBinding{};
        outputBinding.binding = 1;
        outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        outputBinding.descriptorCount = 1;
        outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding kernelBinding{};
        kernelBinding.binding = 2;
        kernelBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        kernelBinding.descriptorCount = 1;
        kernelBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding bindings[] = { inputBinding, outputBinding, kernelBinding };
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
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 3;

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
        if (vkAllocateDescriptorSets(device, &allocInfoDS, &descriptorSet) != VK_SUCCESS) {
            throw runtime_error("Impossibile creare un descriptor set.");
        }

        // link dei buffer al descriptor set
        VkDescriptorBufferInfo inputBufferInfo{};
        inputBufferInfo.buffer = inputBuffer;
        inputBufferInfo.offset = 0;
        inputBufferInfo.range = imageSize;

        VkDescriptorBufferInfo outputBufferInfo{};
        outputBufferInfo.buffer = outputBuffer;
        outputBufferInfo.offset = 0;
        outputBufferInfo.range = imageSize;

        VkDescriptorBufferInfo kernelBufferInfoDesc{};
        kernelBufferInfoDesc.buffer = kernelBuffer;
        kernelBufferInfoDesc.offset = 0;
        kernelBufferInfoDesc.range = kernelSizeBytes;

        VkWriteDescriptorSet writeSets[3]{};
        writeSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeSets[0].dstSet = descriptorSet;
        writeSets[0].dstBinding = 0;
        writeSets[0].descriptorCount = 1;
        writeSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeSets[0].pBufferInfo = &inputBufferInfo;

        writeSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeSets[1].dstSet = descriptorSet;
        writeSets[1].dstBinding = 1;
        writeSets[1].descriptorCount = 1;
        writeSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeSets[1].pBufferInfo = &outputBufferInfo;

        writeSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeSets[2].dstSet = descriptorSet;
        writeSets[2].dstBinding = 2;
        writeSets[2].descriptorCount = 1;
        writeSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeSets[2].pBufferInfo = &kernelBufferInfoDesc;

        vkUpdateDescriptorSets(device, 3, writeSets, 0, nullptr);

        //creazione compute pipeline
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;

        VkPipeline pipeline;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            throw runtime_error("Impossibile creare una pipeline di calcolo.");
        }

        //creazione command buffer
        VkCommandPoolCreateInfo cmdPoolInfo{};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;

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

        //registro i comandi nel command buffer
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw runtime_error("Impossibile inizializzare un command buffer.");
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        BlurInfo blurInfoPush{ imgWidth, imgHeight, kerDim };
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurInfo), &blurInfoPush);
        
        //suddivisione del dominio in modo opportuno
        uint32_t groupCountX = (imgWidth + 15) / 16;
        uint32_t groupCountY = (imgHeight + 15) / 16;
        vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1); 

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

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            0, nullptr,
            1, &barrier,
            0, nullptr
        );

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw runtime_error("Impossibile registrare il command buffer.");
        }

        //esecuzione shader
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence;
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            throw runtime_error("Impossibile sincronizzare CPU e GPU.");
        }

        //inizio misurazione tempo di calcolo
        auto start = std::chrono::high_resolution_clock::now();

        if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
            throw runtime_error("Impossibile inviare comandi alla coda.");
        }

        // Sostituisci la vecchia riga vkWaitForFences con questo blocco:
VkResult res = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
if (res == VK_ERROR_DEVICE_LOST) {
    throw runtime_error("FATAL: Device Lost! (TDR/Timeout detected)");
} else if (res != VK_SUCCESS) {
    throw runtime_error("Errore inatteso in vkWaitForFences");
}

        //fine misurazione tempo di esecuzione
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        
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
        
        // AGGIUNTA: Pulizia buffer kernel
        vkDestroyBuffer(device, kernelBuffer, nullptr);
        vkFreeMemory(device, kernelMemory, nullptr);

        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);

        stbi_image_free(pixels);
        cout << "Immagine salvata come output.png\n";
        cout << "Tempo di puro calcolo: " << elapsed.count() << " secondi\n";

    } catch (const std::exception& e) {
        std::cerr << "Errore critico: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
