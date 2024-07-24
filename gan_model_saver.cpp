#include <iostream>
#include <string>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <glog/logging.h>

using namespace tensorflow;

class ModelSaver {
public:
    explicit ModelSaver(const std::string& model_path) : model_path(model_path) {
        // 初始化glog
        google::InitGoogleLogging("ModelSaver");
        LOG(INFO) << "ModelSaver initialized.";
    }

    void createAndSaveModel() {
        try {
            // 创建一个新的 TensorFlow 图
            Scope root = Scope::NewRootScope();
            LOG(INFO) << "Creating TensorFlow graph.";

            // 生成器输入
            auto noise = Placeholder(root.WithOpName("noise"), DT_FLOAT, Placeholder::Shape({-1, 100})); // 100维噪声

            // 生成器
            auto hidden_gen = Dense(root.WithOpName("hidden_gen"), noise, 256, activation::Relu);
            auto output_gen = Dense(root.WithOpName("output_gen"), hidden_gen, 784, activation::Sigmoid); // 28x28 图像展平为 784维
            auto generated_image = Reshape(root.WithOpName("generated_image"), output_gen, {-1, 28, 28, 1});

            // 判别器输入
            auto real_image = Placeholder(root.WithOpName("real_image"), DT_FLOAT, Placeholder::Shape({-1, 28, 28, 1}));

            // 判别器
            auto hidden_disc_real = Dense(root.WithOpName("hidden_disc_real"), Flatten(root.WithOpName("flatten_real"), real_image), 256, activation::Relu);
            auto output_disc_real = Dense(root.WithOpName("output_disc_real"), hidden_disc_real, 1, activation::Sigmoid);

            auto hidden_disc_fake = Dense(root.WithOpName("hidden_disc_fake"), Flatten(root.WithOpName("flatten_fake"), generated_image), 256, activation::Relu);
            auto output_disc_fake = Dense(root.WithOpName("output_disc_fake"), hidden_disc_fake, 1, activation::Sigmoid);

            // 创建会话
            LOG(INFO) << "Creating TensorFlow session.";
            ClientSession session(root);

            // 初始化变量
            TF_CHECK_OK(session.Run({assignVariables(), assignGeneratorVariables()}).Status());

            // 保存模型
            saveModel(root);
        } catch (const std::exception& e) {
            LOG(ERROR) << "Error occurred: " << e.what();
            throw; // 重新抛出异常以供调用者处理
        }
    }

private:
    std::string model_path;

    void saveModel(const Scope& root) {
        MetaGraphDef meta_graph_def;
        meta_graph_def.mutable_graph_def()->Swap(root.ToGraphDef());

        // 保存 GraphDef
        TF_CHECK_OK(WriteBinaryProto(Env::Default(), model_path, meta_graph_def));
        LOG(INFO) << "Model saved to " << model_path;
    }

    Status assignVariables() {
        // 初始化判别器的权重和偏置
        Scope root = Scope::NewRootScope();

        // 随机数生成器
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-0.1, 0.1);

        // 判别器权重和偏置初始化
        Tensor W_disc1(DT_FLOAT, TensorShape({784, 256})); // 输入784维，输出256维
        Tensor b_disc1(DT_FLOAT, TensorShape({256}));

        Tensor W_disc2(DT_FLOAT, TensorShape({256, 1})); // 输入256维，输出1维
        Tensor b_disc2(DT_FLOAT, TensorShape({1}));

        // 填充权重和偏置
        for (int i = 0; i < 784 * 256; ++i) {
            W_disc1.flat<float>()(i) = distribution(generator);
        }
        for (int i = 0; i < 256; ++i) {
            b_disc1.flat<float>()(i) = distribution(generator);
        }
        for (int i = 0; i < 256; ++i) {
            W_disc2.flat<float>()(i) = distribution(generator);
        }
        b_disc2.flat<float>()(0) = distribution(generator);

        // 初始化变量的赋值操作
        TF_RETURN_IF_ERROR(Assign(root.WithOpName("assign_W_disc1"), Variable(root.WithOpName("W_disc1"), {784, 256}, DT_FLOAT), W_disc1).status());
        TF_RETURN_IF_ERROR(Assign(root.WithOpName("assign_b_disc1"), Variable(root.WithOpName("b_disc1"), {256}, DT_FLOAT), b_disc1).status());
        TF_RETURN_IF_ERROR(Assign(root.WithOpName("assign_W_disc2"), Variable(root.WithOpName("W_disc2"), {256, 1}, DT_FLOAT), W_disc2).status());
        TF_RETURN_IF_ERROR(Assign(root.WithOpName("assign_b_disc2"), Variable(root.WithOpName("b_disc2"), {1}, DT_FLOAT), b_disc2).status());

        return Status::OK();
    }

    Status assignGeneratorVariables() {
        // 初始化生成器的权重和偏置
        Scope root = Scope::NewRootScope();

        // 随机数生成器
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-0.1, 0.1);

        // 生成器权重和偏置初始化
        Tensor W_gen1(DT_FLOAT, TensorShape({100, 256})); // 输入100维，输出256维
        Tensor b_gen1(DT_FLOAT, TensorShape({256}));

        Tensor W_gen2(DT_FLOAT, TensorShape({256, 784})); // 输入256维，输出784维
        Tensor b_gen2(DT_FLOAT, TensorShape({784}));

        // 填充权重和偏置
        for (int i = 0; i < 100 * 256; ++i) {
            W_gen1.flat<float>()(i) = distribution(generator);
        }
        for (int i = 0; i < 256; ++i) {
            b_gen1.flat<float>()(i) = distribution(generator);
        }
        for (int i = 0; i < 256 * 784; ++i) {
            W_gen2.flat<float>()(i) = distribution(generator);
        }
        for (int i = 0; i < 784; ++i) {
            b_gen2.flat<float>()(i) = distribution(generator);
        }

        // 初始化变量的赋值操作
        TF_RETURN_IF_ERROR(Assign(root.WithOpName("assign_W_gen1"), Variable(root.WithOpName("W_gen1"), {100, 256}, DT_FLOAT), W_gen1).status());
        TF_RETURN_IF_ERROR(Assign(root.WithOpName("assign_b_gen1"), Variable(root.WithOpName("b_gen1"), {256}, DT_FLOAT), b_gen1).status());
        TF_RETURN_IF_ERROR(Assign(root.WithOpName("assign_W_gen2"), Variable(root.WithOpName("W_gen2"), {256, 784}, DT_FLOAT), W_gen2).status());
        TF_RETURN_IF_ERROR(Assign(root.WithOpName("assign_b_gen2"), Variable(root.WithOpName("b_gen2"), {784}, DT_FLOAT), b_gen2).status());

        return Status::OK();
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string model_path = argv[1];

    try {
        ModelSaver modelSaver(model_path);
        modelSaver.createAndSaveModel();
    } catch (const std::exception& e) {
        std::cerr << "An error occurred in model processing: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    google::ShutdownGoogleLogging();
    return EXIT_SUCCESS;
}
