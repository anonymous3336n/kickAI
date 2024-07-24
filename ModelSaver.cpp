#include <iostream>
#include <fstream>
#include <string>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <glog/logging.h>

using namespace tensorflow;

class ModelSaver {
public:
    ModelSaver(const std::string& model_path) : model_path(model_path) {
        // 初始化glog
        google::InitGoogleLogging("ModelSaver");
        LOG(INFO) << "ModelSaver initialized.";
    }

    void createAndSaveModel() {
        try {
            // 创建一个新的 TensorFlow 图
            Scope root = Scope::NewRootScope();
            LOG(INFO) << "Creating TensorFlow graph.";

            // 定义输入和输出
            auto X = Placeholder(root.WithOpName("X"), DT_FLOAT);
            auto W = Variable(root.WithOpName("W"), {1}, DT_FLOAT);
            auto b = Variable(root.WithOpName("b"), {}, DT_FLOAT);

            auto assign_W = Assign(root.WithOpName("assign_W"), W, Const(root, {0.5f}));
            auto assign_b = Assign(root.WithOpName("assign_b"), b, Const(root, {0.0f}));

            auto Y = Add(root.WithOpName("Y"), MatMul(root, X, W), b);

            // 创建会话
            LOG(INFO) << "Creating TensorFlow session.";
            ClientSession session(root);

            // 初始化变量
            TF_CHECK_OK(session.Run({assign_W, assign_b}).Status());

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
