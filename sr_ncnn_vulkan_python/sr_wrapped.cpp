//
// Created by archiemeng on 29/3/21.
//
#include "sr_wrapped.h"

SRWrapped::SRWrapped(int gpuid, bool tta_mode) : SR(gpuid, tta_mode) {}

int SRWrapped::load(const StringType &parampath, const StringType &modelpath)
{
#if _WIN32
    return SR::load(*parampath.wstr, *modelpath.wstr);
#else
    return SR::load(*parampath.str, *modelpath.str);
#endif
}

int SRWrapped::process(const Image &inimage, Image outimage)
{
    int c = inimage.elempack;
    ncnn::Mat inimagemat = ncnn::Mat(inimage.w, inimage.h, (void *)inimage.data, (size_t)c, c);
    ncnn::Mat outimagemat = ncnn::Mat(outimage.w, outimage.h, (void *)outimage.data, (size_t)c, c);
    return SR::process(inimagemat, outimagemat);
};

int SRWrapped::cleanup()
{
    return SR::cleanup();
};

uint32_t get_heap_budget(int gpuid)
{
    return ncnn::get_gpu_device(gpuid)->get_heap_budget();
}

int get_gpu_count()
{
    return ncnn::get_gpu_count();
}
