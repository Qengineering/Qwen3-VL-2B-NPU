#ifndef RK35LLM_H
#define RK35LLM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "rkllm.h"
//----------------------------------------------------------------------------------------
typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    int model_image_token;
    int model_embed_size;
} RKLLM_app_context_t;
//----------------------------------------------------------------------------------------
class RK35llm
{
private:
    bool                Info;
    bool                Silence;
    float*              ImgVec;
    LLMHandle           llmHandle;
    RKLLMParam          param;
    RKLLM_app_context_t rknn_app_ctx;
    RKLLMInput          rkllm_input;
    RKLLMInferParam     rkllm_infer_params;
    // For collecting output
    std::string         responseBuffer_;
    std::mutex          responseMutex_;
    std::condition_variable responseCv_;
    bool                responseReady_ = false;
private:
    void        DumpTensorAttr(rknn_tensor_attr* attr);
    int         InitImgEnc(const char* model_path);
    int         RunImgEnc(void);
    static int  StaticCallback(RKLLMResult* result, void* userdata, LLMCallState state);
    int         InstanceCallback(RKLLMResult* result, LLMCallState state);
    cv::Mat     Expand2Square(const cv::Mat& img, const cv::Scalar& background_color = cv::Scalar(127,127,127));
protected:
    cv::Mat resized_img;
public:
    RK35llm();
    virtual ~RK35llm();

    void SetInfo(bool Info);                        //show model information. Default=false.
    void SetHistory(bool History);                  //remember the conversation (use "clear" to reset). Default=true.
    void SetSilence(bool Silence);                  //show no output on terminal. Default=false.

    bool LoadModel(const std::string& VLMmodel, const std::string& LLMmodel, int32_t NewTokens=2048, int32_t ContextLength=4096);
    void LoadImage(const cv::Mat& img);

    std::string Ask(const std::string& Question);   //ask something and get an answer.
};
//----------------------------------------------------------------------------------------
#endif // RK35LLM_H
