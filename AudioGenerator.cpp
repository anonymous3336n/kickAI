#include <iostream>
#include <vector>
#include <cmath>
#include <sndfile.h>
#include <fstream>
#include <stdexcept>
#include <string>

class AudioGenerator {
public:
    AudioGenerator(double duration, int sampleRate, double frequency)
        : duration(duration), sampleRate(sampleRate), frequency(frequency) {
        generateSignal();
    }

    void saveToWav(const std::string &filename) {
        SF_INFO sfinfo;
        sfinfo.frames = static_cast<sf_count_t>(signal.size());
        sfinfo.samplerate = sampleRate;
        sfinfo.channels = 1; // 单声道
        sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

        SNDFILE *outfile = sf_open(filename.c_str(), SFM_WRITE, &sfinfo, nullptr);
        if (!outfile) {
            throw std::runtime_error("Error opening output file.");
        }
        
        sf_writef_int(outfile, signal.data(), static_cast<sf_count_t>(signal.size()));
        sf_close(outfile);
        log("Audio saved to " + filename);
    }

private:
    double duration;
    int sampleRate;
    double frequency;
    std::vector<int16_t> signal;

    void generateSignal() {
        int totalSamples = static_cast<int>(sampleRate * duration);
        signal.resize(totalSamples);

        for (int i = 0; i < totalSamples; ++i) {
            double t = static_cast<double>(i) / sampleRate;
            signal[i] = static_cast<int16_t>(32767 * sin(2 * M_PI * frequency * t));
        }

        log("Signal generated with frequency " + std::to_string(frequency) + " Hz.");
    }

    void log(const std::string &message) {
        std::ofstream logFile("audio_generation.log", std::ios_base::app);
        if (logFile) {
            logFile << message << std::endl;
        }
    }
};

int main() {
    try {
        double duration = 5.0;  // 音频时长（秒）
        int sampleRate = 44100; // 采样率
        double frequency = 440;  // 正弦波频率（Hz）

        AudioGenerator audioGen(duration, sampleRate, frequency);
        audioGen.saveToWav("output.wav");
        
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
