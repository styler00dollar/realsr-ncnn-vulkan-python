//
// Created by archiemeng on 29/3/21.
//

#ifndef REALSR_NCNN_VULKAN_REALSR_WRAPPED_H
#define REALSR_NCNN_VULKAN_REALSR_WRAPPED_H
#include "sr.h"

// wrapper class of ncnn::Mat
typedef struct Image
{
    unsigned char *data;
    int w;
    int h;
    int elempack;
    Image(unsigned char *d, int w, int h, int channels)
    {
        this->data = d;
        this->w = w;
        this->h = h;
        this->elempack = channels;
    }

} Image;

union StringType
{
    std::string *str;
    std::wstring *wstr;
};

class SRWrapped : public SR
{
public:
    SRWrapped(int gpuid, bool tta_mode = false);
    int load(const StringType &parampath, const StringType &modelpath);
    int process(const Image &inimage, Image outimage);
    int cleanup();
};

int get_gpu_count();
uint32_t get_heap_budget(int gpuid);
#endif // REALSR_NCNN_VULKAN_REALSR_WRAPPED_H
