#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>

// OpenCV Headers
#include <opencv2/opencv.hpp>

// SNPE Headers
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/StringList.hpp"

namespace fs = std::filesystem;

const std::string MODEL_PATH = "/opt/models/onnxsim_romnistereo32_v13_bs16_e194_quantized.dlc"; // Sửa tên file DLC của bạn
const std::string DATA_ROOT  = "/opt/omnidata/hyp_01";
const std::string GRID_DIR   = "/opt/omnidata/grids"; // Nơi chứa grid0.raw, grid1.raw...

std::unique_ptr<zdl::DlSystem::ITensor> loadRawInput(const std::string& filePath, const zdl::DlSystem::TensorShape& shape) {
    auto tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(shape);
    if (!tensor) return nullptr;

    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] Cannot open file: " << filePath << std::endl;
        return nullptr;
    }
    file.read(reinterpret_cast<char*>(&tensor->begin()[0]), tensor->getSize() * sizeof(float));
    return tensor;
}


std::unique_ptr<zdl::DlSystem::ITensor> processImage(const std::string& imgPath, const zdl::DlSystem::TensorShape& shape) {

    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << imgPath << std::endl;
        return nullptr;
    }


    const auto* dims = shape.getDimensions();
    size_t rank = shape.rank();
    size_t H = dims[rank - 2]; 
    size_t W = dims[rank - 1]; 
    
    
    // 3. Resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(W, H));


    resized.convertTo(resized, CV_32F);
    

    resized = resized / 255.0f;

    auto tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(shape);
    std::copy(resized.begin<float>(), resized.end<float>(), tensor->begin());
    
    return tensor;
}

int main(int argc, char** argv) {
    
    std::string dlcPath = MODEL_PATH; 
    if (argc > 1) {
        dlcPath = argv[1];
    }
    
    std::cout << "[INFO] Loading DLC from: " << dlcPath << std::endl;
    // 1. SETUP RUNTIME (DSP ONLY)
    zdl::DlSystem::RuntimeList runtimeList;
    runtimeList.add(zdl::DlSystem::Runtime_t::DSP);
    
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP)) {
        std::cerr << "[FATAL] DSP Runtime not available!" << std::endl;
        return -1;
    }

    // 2. LOAD CONTAINER
    auto container = zdl::DlContainer::IDlContainer::open(dlcPath);
    
    if (!container) {
        std::cerr << "[FATAL] Failed to open DLC: " << dlcPath << std::endl;
        return -1;
    }

    // 3. BUILD ENGINE
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    snpeBuilder.setRuntimeProcessorOrder(runtimeList);
    snpeBuilder.setUseUserSuppliedBuffers(false); 
    
    auto snpe = snpeBuilder.build();
    if (!snpe) {
        std::cerr << "[FATAL] Build SNPE Failed: " << zdl::DlSystem::getLastErrorString() << std::endl;
        return -1;
    }
    std::cout << "[INFO] SNPE Engine initialized on DSP." << std::endl;

    // 4. PREPARE STATIC INPUTS (GRIDS)
    
    std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> gridTensors;
    std::vector<std::string> gridNames = {"grid0", "grid1", "grid2"};
    
    
    std::map<std::string, zdl::DlSystem::ITensor*> gridPtrs;

    for (const auto& name : gridNames) {
        
        auto shapeOpt = snpe->getInputDimensions(name.c_str());
        if (shapeOpt) {
            const auto& shape = *shapeOpt; 
            
            std::string path = GRID_DIR + "/" + name + ".raw";
            auto tensor = loadRawInput(path, shape);
            if (!tensor) return -1;
            
            gridPtrs[name] = tensor.get(); 
            gridTensors.push_back(std::move(tensor)); 
            std::cout << "[INFO] Loaded " << name << std::endl;
        } else {
            std::cerr << "[ERROR] Model does not have input: " << name << std::endl;
            return -1;
        }
    }

    
    std::string cam1Dir = DATA_ROOT + "/cam1";
    int frameCount = 0;

    for (const auto& entry : fs::directory_iterator(cam1Dir)) {
        if (entry.path().extension() == ".png") {
            std::string filename = entry.path().filename().string();
            std::cout << "\n--- Processing Frame: " << filename << " ---" << std::endl;

            
            zdl::DlSystem::TensorMap inputMap;

            
            for (auto const& [name, ptr] : gridPtrs) {
                inputMap.add(name.c_str(), ptr);
            }

            
            std::vector<std::string> imgInputs = {"img0", "img1", "img2"};
            std::vector<std::string> imgPaths = {
                DATA_ROOT + "/cam1/" + filename,
                DATA_ROOT + "/cam2/" + filename,
                DATA_ROOT + "/cam3/" + filename
            };
            
            
            std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> imgTensors;

            bool frameReady = true;
            for (size_t i = 0; i < 3; ++i) {
                
                auto shapeOpt = snpe->getInputDimensions(imgInputs[i].c_str());
                if (shapeOpt) {
                    auto tensor = processImage(imgPaths[i], *shapeOpt);
                    if (!tensor) { frameReady = false; break; }
                    
                    inputMap.add(imgInputs[i].c_str(), tensor.get());
                    imgTensors.push_back(std::move(tensor));
                }
            }

            if (!frameReady) continue;

          
            zdl::DlSystem::TensorMap outputMap;
            bool status = snpe->execute(inputMap, outputMap);

            if (status) {
                std::cout << "[SUCCESS] Inference executed on NPU." << std::endl;
                
                auto outNames = outputMap.getTensorNames();
                if (outNames.size() > 0) {
                     auto pOut = outputMap.getTensor(outNames.at(0));
                     std::cout << "Output Tensor: " << outNames.at(0) << " First val: " << pOut->begin()[0] << std::endl;
                }
            } else {
                std::cerr << "[FAIL] Inference Error: " << zdl::DlSystem::getLastErrorString() << std::endl;
            }

            frameCount++;
            if (frameCount >= 5) break; 
        }
    }

    return 0;
}