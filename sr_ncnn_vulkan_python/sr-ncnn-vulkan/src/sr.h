// sr implemented with ncnn library

#ifndef REALSR_H
#define REALSR_H

#include <string>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

class SR
{
public:
    SR(int gpuid, bool tta_mode = false);
    ~SR();

#if _WIN32
    int load(const std::wstring &parampath, const std::wstring &modelpath);
#else
    int load(const std::string &parampath, const std::string &modelpath);
#endif

    int process(const ncnn::Mat &inimage, ncnn::Mat &outimage) const;
    int cleanup() const;

public:
    // sr parameters
    int scale;
    int tilesize;
    int prepadding;

private:
    ncnn::Net net;
    ncnn::Pipeline *sr_preproc;
    ncnn::Pipeline *sr_postproc;
    ncnn::Layer *bicubic_4x;
    bool tta_mode;
};

#endif // REALSR_H
