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

            // 定义输入
            auto input = Placeholder(root.WithOpName("input"), DT_FLOAT, Placeholder::Shape({-1, 28, 28, 1})); // 输入28x28x1的图像

            // 卷积层1
            auto conv1 = Conv2D(root.WithOpName("conv1"), input, Variable(root.WithOpName("W_conv1"), {5, 5, 1, 32}, DT_FLOAT), {1, 1, 1, 1}, "SAME");
            auto relu1 = Relu(root.WithOpName("relu1"), conv1);
            auto pool1 = MaxPool(root.WithOpName("pool1"), relu1, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME");

            // 卷积层2
            auto conv2 = Conv2D(root.WithOpName("conv2"), pool1, Variable(root.WithOpName("W_conv2"), {5, 5, 32, 64}, DT_FLOAT), {1, 1, 1, 1}, "SAME");
            auto relu2 = Relu(root.WithOpName("relu2"), conv2);
            auto pool2 = MaxPool(root.WithOpName("pool2"), relu2, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME");

            // Flatten层
            auto flat = Reshape(root.WithOpName("flat"), pool2, {-1, 7 * 7 * 64});

            // 全连接层
            auto W_fc = Variable(root.WithOpName("W_fc"), {7 * 7 * 64, 1024}, DT_FLOAT);
            auto b_fc = Variable(root.WithOpName("b_fc"), {1024}, DT_FLOAT);
            auto fc = Add(root.WithOpName("fc"), MatMul(root, flat, W_fc), b_fc);
            auto relu_fc = Relu(root.WithOpName("relu_fc"), fc);

            // 输出层
            auto W_output = Variable(root.WithOpName("W_output"), {1024, 10}, DT_FLOAT);
            auto b_output = Variable(root.WithOpName("b_output"), {10}, DT_FLOAT);
            auto output = Add(root.WithOpName("output"), MatMul(root, relu_fc, W_output), b_output);

            // 创建会话
            LOG(INFO) << "Creating TensorFlow session.";
            ClientSession session(root);

            // 初始化变量
            TF_CHECK_OK(session.Run({Assign(root.WithOpName("assign_W_conv1"), Variable(root.WithOpName("W_conv1"), {5, 5, 1, 32}, DT_FLOAT), Const(root, 0.1f * tf::random::Gaussian(32*5*5)) }),
                                      Assign(root.WithOpName("assign_W_conv2"), Variable(root.WithOpName("W_conv2"), {5, 5, 32, 64}, DT_FLOAT), Const(root, 0.1f * tf::random::Gaussian(64*5*5)) ),
                                      Assign(root.WithOpName("assign_W_fc"), W_fc, Const(root, 0.1f * tf::random::Gaussian(1024 * 7 * 7 * 64))),
                                      Assign(root.WithOpName("assign_b_fc"), b_fc, Const(root, 0.1f * tf::random::Gaussian(1024))),
                                      Assign(root.WithOpName("assign_W_output"), W_output, Const(root, 0.1f * tf::random::Gaussian(10))),
                                      Assign(root.WithOpName("assign_b_output"), b_output, Const(root, 0.1f * tf::random::Gaussian(10)))
                                     }).Status());

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
