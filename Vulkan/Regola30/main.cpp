#define STB_IMAGE_WRITE_IMPLEMENTATION
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

struct PushConsts {
    int width;
    int height;
    int currentRow;
};

vector<char> loadShader(const string& filename) {
    ifstream file(filename, ios::ate | ios::binary);

    if (!file.is_open()) {
        throw runtime_error("Impossibile aprire lo shader: ");
    }

    unsigned int size = file.tellg();

    vector<char> contents(size);
    file.seekg(0);
    file.read(contents.data(), size);

    return contents;
}

void input(int &width, int &height, bool &useCpu) {
    tmp1 = 1024;
    tmp2 = 512;
    cout << "Inserire la dimensione dell'automa: [Premere 0 per il valore di default: " << width << "]: ";
    cin >> tmp1;

    if (tmp1 != 0)
    width = tmp1;

    int defaultHeight = width / 2;
    cout << "Inserire il numero di iterazioni [Premere 0 per il valore di default: " << height << "]: ";
    cin >> tmp2;

    if (tmp2 != 0)
    height = tmp;

    cout << "Vuoi  eseguire il codice su GPU o CPU (GPU = 0, CPU = 1): ";
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
    appInfo.pApplicationName = "Regola30";
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
        throw std::runtime_error("Nessun dispositivo Vulkan compatibile trovato per la scelta richiesta.");

    //stampa del device scelto
    VkPhysicalDeviceProperties chosenProps;
    vkGetPhysicalDeviceProperties(chosenDevice, &chosenProps);
    cout << "\nDispositivo scelto: " << chosenProps.deviceName << "\n";

    return chosenDevice;
}


int main() {
    //parametri dell'automa
    int width = 1024, height;
    bool useCpu = 0;
    bool print = 0;

    input(width, height, useCpu);
    
    //inizializzazione Vulkan
    VkInstance instance = initVulkan();

    //scelta del physical device (GPU)
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) 
    {
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
    
    //creazione del buffer
    VkDeviceSize bufferSize = width * height * sizeof(uint32_t);
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer gridBuffer;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &gridBuffer) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare il buffer.");
    }

    //allocazione memoria per il buffer
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, gridBuffer, &memReq);
    
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = chooseMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, physicalDevice);
    
    VkDeviceMemory gridMemory;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &gridMemory) != VK_SUCCESS) {
        throw runtime_error("Impossibile allocare memoria.");
    }
    vkBindBufferMemory(device, gridBuffer, gridMemory, 0);

    //inizializzazione del vettore
    vector<uint32_t> hostData(width * height, 0); 
    hostData[width / 2] = 1; // impostiamo solamente il pixel centrale uguale a 1    

    void* data;
    vkMapMemory(device, gridMemory, 0, bufferSize, 0, &data);
    memcpy(data, hostData.data(), bufferSize);
    vkUnmapMemory(device, gridMemory);

    //caricamento shader
    vector<char> shaderCode = loadShader("shaders/regola30.spv");
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
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorCount = 1;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;   
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    VkDescriptorSetLayout descriptorSetLayout;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare un set layout descriptor.");
    }

    //pipeline layout
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PushConsts);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw runtime_error("Impossibile creare un pipeline layout.");
    }

    //descriptor pool
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS){
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

    //link dei buffer al descriptor set
    VkDescriptorBufferInfo descBuffInfo{};
    descBuffInfo.buffer = gridBuffer;
    descBuffInfo.offset = 0;
    descBuffInfo.range = bufferSize;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &descBuffInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    //creazione compute pipeline
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS){
        throw runtime_error("Impossibile creare una pipeline di calcolo.");
    }

    //creazione command buffer
    VkCommandPoolCreateInfo cmdPoolInfo{};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;

    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool) != VK_SUCCESS){
        throw runtime_error("Impossibile creare command pool.");
    }

    VkCommandBufferAllocateInfo cmdBufAlloc{};
    cmdBufAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAlloc.commandPool = commandPool;
    cmdBufAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAlloc.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    if (vkAllocateCommandBuffers(device, &cmdBufAlloc, &commandBuffer) != VK_SUCCESS){

        throw runtime_error("Impossibile creare un command buffer.");
    }

    //registro i comandi nel command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS){
        throw runtime_error("Impossibile inizializzare un command buffer.");
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    //suddivisione del dominio in modo opportuno per ottenere il massimo delle prestazione (come in CUDA)
    uint32_t groupCountX = (width + 255) / 256;

    for (int y = 1; y < height; y++) {
        PushConsts pc = {width, height, y};
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConsts), &pc);

        vkCmdDispatch(commandBuffer, groupCountX, 1, 1);

        //barriera per sincronizzare
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = gridBuffer;
        barrier.offset = 0;
        barrier.size = bufferSize;

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr,
            1, &barrier, 0, nullptr
        );
    }

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
    auto start = chrono::high_resolution_clock::now();

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
        throw runtime_error("Impossibile inviare comandi alla coda.");
    }

    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    
    //fine misurazione tempo di esecuzione
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    size_t num_pixels = (size_t)width * height; 
    
    //limite di sicurezza per evitare crash
    const size_t MAX_PIXELS_TO_SAVE = 100ULL * 1000 * 1000;

    if (num_pixels > MAX_PIXELS_TO_SAVE) {
        cout << "\nL'immagine e' troppo grande (" << num_pixels << " pixel). Salvataggio saltato per sicurezza.\n";
    } 
    else {
        //leggo output
        vkMapMemory(device, gridMemory, 0, bufferSize, 0, &data);
        uint32_t* gpuData = (uint32_t*)data;
        vector<unsigned char> pngData(num_pixels * 4);
        
        for(size_t i = 0; i < num_pixels; i++) {
            //se 1 (vivo) -> Nero (0), se 0 (morto) -> Bianco (255)
            unsigned char color = (gpuData[i] == 1) ? 0 : 255;

            pngData[i*4 + 0] = color;
            pngData[i*4 + 1] = color;
            pngData[i*4 + 2] = color;
            pngData[i*4 + 3] = 255;
        }
        vkUnmapMemory(device, gridMemory);

        stbi_write_png("output.png", width, height, 4, pngData.data(), width * 4);
        cout << "Immagine salvata come output.png\n";
    }

    //pulizia delle variabili 
    vkDestroyFence(device, fence, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyBuffer(device, gridBuffer, nullptr);
    vkFreeMemory(device, gridMemory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    cout << "Tempo di puro calcolo: " << elapsed.count() << " secondi\n";

    return 0;
}