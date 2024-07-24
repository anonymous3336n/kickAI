#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <curl/curl.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <stdexcept>
#include <iomanip>
#include <ctime>

namespace fs = std::filesystem;
using namespace tensorflow;

class ImageProcessor {
public:
    ImageProcessor(const std::string &model_path, const std::string &log_file)
        : log_file(log_file) {
        loadModel(model_path);
        initializeLog();
    }

    ~ImageProcessor() {
        if (session) {
            session->Close();
        }
    }

    void processImages(const std::string &image_urls_file, const std::string &output_dir) {
        std::ifstream file(image_urls_file);
        if (!file.is_open()) {
            log("Could not open image URLs file.");
            throw std::runtime_error("File open error.");
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string url, label;
            std::getline(iss, url, ',');
            std::getline(iss, label);

            cv::Mat image = loadImageFromUrl(url);
            if (!image.empty()) {
                auto predictions = predictImage(image);
                saveImage(image, output_dir, label, url);
                logPredictions(predictions);
            } else {
                log("Failed to process image from: " + url);
            }
        }
    }

    void log(const std::string &message) {
        std::ofstream logFile(log_file, std::ios_base::app);
        logFile << currentDateTime() << " - " << message << std::endl;
    }

private:
    std::unique_ptr<Session> session;
    std::string log_file;

    void loadModel(const std::string &model_path) {
        Status status = NewSession(SessionOptions(), &session);
        if (!status.ok()) {
            log("Error creating TensorFlow session: " + status.ToString());
            throw std::runtime_error("TensorFlow session error.");
        }

        MetaGraphDef meta_graph_def;
        status = ReadBinaryProto(Env::Default(), model_path, &meta_graph_def);
        if (!status.ok()) {
            log("Error loading model: " + status.ToString());
            throw std::runtime_error("Model load error.");
        }

        status = session->Create(meta_graph_def.graph_def());
        if (!status.ok()) {
            log("Error creating graph: " + status.ToString());
            throw std::runtime_error("Graph creation error.");
        }
    }

    cv::Mat loadImageFromUrl(const std::string &url) {
        CURL *curl;
        CURLcode res;
        cv::Mat image;

        curl = curl_easy_init();
        if (curl) {
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, nullptr);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, nullptr);
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeImageCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &image);
            res = curl_easy_perform(curl);

            if (res != CURLE_OK) {
                log("Failed to download image: " + url);
            }

            curl_easy_cleanup(curl);
        }
        return image;
    }

    static size_t writeImageCallback(void *contents, size_t size, size_t nmemb, cv::Mat *image) {
        size_t totalSize = size * nmemb;
        std::vector<uchar> data(totalSize);
        std::copy(static_cast<uchar*>(contents), static_cast<uchar*>(contents) + totalSize, data.begin());
        image->imdecode(data, cv::IMREAD_COLOR); // Decode image from data
        return totalSize;
    }

    std::vector<std::pair<std::string, float>> predictImage(const cv::Mat &image) {
        // Preprocess image for the model
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(224, 224));
        resized_image.convertTo(resized_image, CV_32F, 1.0 / 255);

        Tensor input_tensor(DT_FLOAT, TensorShape({ 1, 224, 224, 3 }));
        std::copy(resized_image.data, resized_image.data + resized_image.total() * resized_image.elemSize(), input_tensor.flat<float>().data());

        std::vector<Tensor> outputs;
        Status status = session->Run({{"input_1", input_tensor}}, {"PredictionLayer/Softmax"}, {}, &outputs);
        if (!status.ok()) {
            log("Error during prediction: " + status.ToString());
            throw std::runtime_error("Prediction error.");
        }

        // Decode predictions
        std::vector<std::pair<std::string, float>> predictions;
        auto output = outputs[0].flat<float>();
        for (int i = 0; i < output.size(); ++i) {
            predictions.emplace_back("Class " + std::to_string(i), output(i)); // Assuming class indices
        }
        return predictions;
    }

    void saveImage(const cv::Mat &image, const std::string &output_dir, const std::string &label, const std::string &url) {
        std::string label_dir = output_dir + "/" + label;
        fs::create_directories(label_dir);
        std::string image_filename = fs::path(url).filename().string();
        std::string save_path = label_dir + "/" + image_filename;

        cv::imwrite(save_path, image);
        log("Image saved: " + save_path);
    }

    void logPredictions(const std::vector<std::pair<std::string, float>> &predictions) {
        for (const auto &pred : predictions) {
            log(pred.first + ": " + std::to_string(pred.second));
        }
    }

    std::string currentDateTime() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
        return oss.str();
    }

    void initializeLog() {
        std::ofstream logFile(log_file);
        logFile << "Image processing started." << std::endl;
    }
};

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_urls_file> <output_dir> <log_file>" << std::endl;
        return EXIT_FAILURE;
    }

    try {
        std::string model_path = argv[1];
        std::string image_urls_file = argv[2];
        std::string output_dir = argv[3];
        std::string log_file = argv[4];

        ImageProcessor processor(model_path, log_file);
        processor.processImages(image_urls_file, output_dir);

    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
