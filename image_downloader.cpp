#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <sys/stat.h>
#include <vector>
#include <stdexcept>

using json = nlohmann::json;

class ImageDownloader {
public:
    ImageDownloader(const std::string& query, const std::string& saveDir, int numImages = 10)
        : query(query), saveDir(saveDir), numImages(numImages) {
        createDirectory(saveDir);
        fetchImageLinks();
    }

    void downloadImages() {
        for (size_t i = 0; i < imageLinks.size() && i < numImages; ++i) {
            downloadImage(imageLinks[i], query + "_" + std::to_string(i + 1) + ".jpg");
        }
    }

private:
    std::string query;
    std::string saveDir;
    int numImages;
    std::vector<std::string> imageLinks;

    void createDirectory(const std::string& dir) {
        mkdir(dir.c_str(), 0777); // Create directory if it does not exist
    }

    void fetchImageLinks() {
        std::string searchUrl = "https://yandex.com/images/search?text=" + query + "&format=json";
        std::string response = performGetRequest(searchUrl);
        
        try {
            auto jsonResponse = json::parse(response);
            for (const auto& item : jsonResponse["items"]) {
                std::string imageUrl = item["image"]["url"];
                imageLinks.push_back(imageUrl);
            }
            log("Fetched " + std::to_string(imageLinks.size()) + " image links.");
        } catch (const json::parse_error& e) {
            log("JSON parse error: " + std::string(e.what()));
            throw std::runtime_error("Failed to fetch image links.");
        }
    }

    void downloadImage(const std::string& imageUrl, const std::string& outputName) {
        try {
            std::string imageData = performGetRequest(imageUrl);
            std::ofstream outFile(saveDir + "/" + outputName, std::ios::binary);
            outFile.write(imageData.c_str(), imageData.size());
            outFile.close();
            log("Image downloaded: " + outputName);
        } catch (const std::exception& e) {
            log("Error downloading image: " + std::string(e.what()));
        }
    }

    std::string performGetRequest(const std::string& url) {
        CURL* curl;
        CURLcode res;
        std::string response;

        curl = curl_easy_init();
        if (curl) {
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0");
            res = curl_easy_perform(curl);
            curl_easy_cleanup(curl);

            if (res != CURLE_OK) {
                log("Curl error: " + std::string(curl_easy_strerror(res)));
                throw std::runtime_error("Failed to perform GET request.");
            }
        }
        return response;
    }

    static size_t writeCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
        userp->append((const char*)contents, size * nmemb);
        return size * nmemb;
    }

    void log(const std::string& message) {
        std::ofstream logFile("image_downloader.log", std::ios_base::app);
        if (logFile) {
            logFile << message << std::endl;
        }
    }
};

int main() {
    try {
        std::string query = "nature"; // Search query
        std::string saveDirectory = "downloaded_images";

        ImageDownloader downloader(query, saveDirectory);
        downloader.downloadImages();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
