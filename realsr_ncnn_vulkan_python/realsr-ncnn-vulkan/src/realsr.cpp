// realsr implemented with ncnn library

#include "realsr.h"

#include <algorithm>
#include <vector>
#include <stdexcept>

using namespace std;

static const uint32_t realsr_preproc_spv_data[] = {
#include "realsr_preproc.spv.hex.h"
};
static const uint32_t realsr_preproc_fp16s_spv_data[] = {
#include "realsr_preproc_fp16s.spv.hex.h"
};
static const uint32_t realsr_preproc_int8s_spv_data[] = {
#include "realsr_preproc_int8s.spv.hex.h"
};
static const uint32_t realsr_postproc_spv_data[] = {
#include "realsr_postproc.spv.hex.h"
};
static const uint32_t realsr_postproc_fp16s_spv_data[] = {
#include "realsr_postproc_fp16s.spv.hex.h"
};
static const uint32_t realsr_postproc_int8s_spv_data[] = {
#include "realsr_postproc_int8s.spv.hex.h"
};

static const uint32_t realsr_preproc_tta_spv_data[] = {
#include "realsr_preproc_tta.spv.hex.h"
};
static const uint32_t realsr_preproc_tta_fp16s_spv_data[] = {
#include "realsr_preproc_tta_fp16s.spv.hex.h"
};
static const uint32_t realsr_preproc_tta_int8s_spv_data[] = {
#include "realsr_preproc_tta_int8s.spv.hex.h"
};
static const uint32_t realsr_postproc_tta_spv_data[] = {
#include "realsr_postproc_tta.spv.hex.h"
};
static const uint32_t realsr_postproc_tta_fp16s_spv_data[] = {
#include "realsr_postproc_tta_fp16s.spv.hex.h"
};
static const uint32_t realsr_postproc_tta_int8s_spv_data[] = {
#include "realsr_postproc_tta_int8s.spv.hex.h"
};

RealSR::RealSR(int gpuid, bool _tta_mode)
{
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_int8_storage = true;
    net.opt.use_int8_arithmetic = false;

    net.set_vulkan_device(gpuid);

    realsr_preproc = 0;
    realsr_postproc = 0;
    bicubic_4x = 0;
    tta_mode = _tta_mode;
}

RealSR::~RealSR()
{
    // cleanup preprocess and postprocess pipeline
    {
        delete realsr_preproc;
        delete realsr_postproc;
    }

    bicubic_4x->destroy_pipeline(net.opt);
    delete bicubic_4x;
}

#if _WIN32
int RealSR::load(const std::wstring &parampath, const std::wstring &modelpath)
#else
int RealSR::load(const std::string &parampath, const std::string &modelpath)
#endif
{
#if _WIN32
    {
        FILE *fp = _wfopen(parampath.c_str(), L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", parampath.c_str());
        }

        net.load_param(fp);

        fclose(fp);
    }
    {
        FILE *fp = _wfopen(modelpath.c_str(), L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", modelpath.c_str());
        }

        net.load_model(fp);

        fclose(fp);
    }
#else
    net.load_param(parampath.c_str());
    net.load_model(modelpath.c_str());
#endif

    // initialize preprocess and postprocess pipeline
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        realsr_preproc = new ncnn::Pipeline(net.vulkan_device());
        realsr_preproc->set_optimal_local_size_xyz(32, 32, 3);

        realsr_postproc = new ncnn::Pipeline(net.vulkan_device());
        realsr_postproc->set_optimal_local_size_xyz(32, 32, 3);

        if (tta_mode)
        {
            if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                realsr_preproc->create(realsr_preproc_tta_int8s_spv_data, sizeof(realsr_preproc_tta_int8s_spv_data), specializations);
            else if (net.opt.use_fp16_storage)
                realsr_preproc->create(realsr_preproc_tta_fp16s_spv_data, sizeof(realsr_preproc_tta_fp16s_spv_data), specializations);
            else
                realsr_preproc->create(realsr_preproc_tta_spv_data, sizeof(realsr_preproc_tta_spv_data), specializations);

            if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                realsr_postproc->create(realsr_postproc_tta_int8s_spv_data, sizeof(realsr_postproc_tta_int8s_spv_data), specializations);
            else if (net.opt.use_fp16_storage)
                realsr_postproc->create(realsr_postproc_tta_fp16s_spv_data, sizeof(realsr_postproc_tta_fp16s_spv_data), specializations);
            else
                realsr_postproc->create(realsr_postproc_tta_spv_data, sizeof(realsr_postproc_tta_spv_data), specializations);
        }
        else
        {
            if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                realsr_preproc->create(realsr_preproc_int8s_spv_data, sizeof(realsr_preproc_int8s_spv_data), specializations);
            else if (net.opt.use_fp16_storage)
                realsr_preproc->create(realsr_preproc_fp16s_spv_data, sizeof(realsr_preproc_fp16s_spv_data), specializations);
            else
                realsr_preproc->create(realsr_preproc_spv_data, sizeof(realsr_preproc_spv_data), specializations);

            if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                realsr_postproc->create(realsr_postproc_int8s_spv_data, sizeof(realsr_postproc_int8s_spv_data), specializations);
            else if (net.opt.use_fp16_storage)
                realsr_postproc->create(realsr_postproc_fp16s_spv_data, sizeof(realsr_postproc_fp16s_spv_data), specializations);
            else
                realsr_postproc->create(realsr_postproc_spv_data, sizeof(realsr_postproc_spv_data), specializations);
        }
    }

    // bicubic 4x for alpha channel
    {
        bicubic_4x = ncnn::create_layer("Interp");
        bicubic_4x->vkdev = net.vulkan_device();

        ncnn::ParamDict pd;
        pd.set(0, 3); // bicubic
        pd.set(1, 4.f);
        pd.set(2, 4.f);
        bicubic_4x->load_param(pd);

        bicubic_4x->create_pipeline(net.opt);
    }

    return 0;
}

int RealSR::process(const ncnn::Mat &inimage, ncnn::Mat &outimage) const
{

    const unsigned char *pixeldata = (const unsigned char *)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = w;
    const int TILE_SIZE_Y = h;

    ncnn::VkAllocator *blob_vkallocator = net.vulkan_device()->acquire_blob_allocator();
    ncnn::VkAllocator *staging_vkallocator = net.vulkan_device()->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    try
    {

        // forcing to process one tile
        const int xtiles = 1;
        const int ytiles = 1;

        const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

        //#pragma omp parallel for num_threads(2)
        for (int yi = 0; yi < ytiles; yi++)
        {
            const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

            int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
            int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding, h);

            ncnn::Mat in;
            if (opt.use_fp16_storage && opt.use_int8_storage)
            {
                in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (unsigned char *)pixeldata + in_tile_y0 * w * channels, (size_t)channels, 1);
            }
            else
            {
                if (channels == 3)
                {
#if _WIN32
                    in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR2RGB, w, (in_tile_y1 - in_tile_y0));
#else
                    in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_RGB, w, (in_tile_y1 - in_tile_y0));
#endif
                }
                if (channels == 4)
                {
#if _WIN32
                    in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGRA2RGBA, w, (in_tile_y1 - in_tile_y0));
#else
                    in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_RGBA, w, (in_tile_y1 - in_tile_y0));
#endif
                }
            }

            ncnn::VkCompute cmd(net.vulkan_device());

            // upload
            ncnn::VkMat in_gpu;
            {
                cmd.record_clone(in, in_gpu, opt);

                if (xtiles > 1)
                {
                    cmd.submit_and_wait();
                    cmd.reset();
                }
            }

            int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
            int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

            ncnn::VkMat out_gpu;
            if (opt.use_fp16_storage && opt.use_int8_storage)
            {
                out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t)channels, 1, blob_vkallocator);
            }
            else
            {
                out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t)4u, 1, blob_vkallocator);
            }

            for (int xi = 0; xi < xtiles; xi++)
            {
                const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

                if (tta_mode)
                {
                    // preproc
                    ncnn::VkMat in_tile_gpu[8];
                    ncnn::VkMat in_alpha_tile_gpu;
                    {
                        // crop tile
                        int tile_x0 = xi * TILE_SIZE_X - prepadding;
                        int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding;
                        int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                        int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding;

                        in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                        in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                        in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                        in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                        in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                        in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                        in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                        in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                        if (channels == 4)
                        {
                            in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, blob_vkallocator);
                        }

                        std::vector<ncnn::VkMat> bindings(10);
                        bindings[0] = in_gpu;
                        bindings[1] = in_tile_gpu[0];
                        bindings[2] = in_tile_gpu[1];
                        bindings[3] = in_tile_gpu[2];
                        bindings[4] = in_tile_gpu[3];
                        bindings[5] = in_tile_gpu[4];
                        bindings[6] = in_tile_gpu[5];
                        bindings[7] = in_tile_gpu[6];
                        bindings[8] = in_tile_gpu[7];
                        bindings[9] = in_alpha_tile_gpu;

                        std::vector<ncnn::vk_constant_type> constants(13);
                        constants[0].i = in_gpu.w;
                        constants[1].i = in_gpu.h;
                        constants[2].i = in_gpu.cstep;
                        constants[3].i = in_tile_gpu[0].w;
                        constants[4].i = in_tile_gpu[0].h;
                        constants[5].i = in_tile_gpu[0].cstep;
                        constants[6].i = prepadding;
                        constants[7].i = prepadding;
                        constants[8].i = xi * TILE_SIZE_X;
                        constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                        constants[10].i = channels;
                        constants[11].i = in_alpha_tile_gpu.w;
                        constants[12].i = in_alpha_tile_gpu.h;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = in_tile_gpu[0].w;
                        dispatcher.h = in_tile_gpu[0].h;
                        dispatcher.c = channels;

                        cmd.record_pipeline(realsr_preproc, bindings, constants, dispatcher);
                    }

                    // realsr
                    ncnn::VkMat out_tile_gpu[8];
                    for (int ti = 0; ti < 8; ti++)
                    {
                        ncnn::Extractor ex = net.create_extractor();

                        ex.set_blob_vkallocator(blob_vkallocator);
                        ex.set_workspace_vkallocator(blob_vkallocator);
                        ex.set_staging_vkallocator(staging_vkallocator);

                        ex.input("data", in_tile_gpu[ti]);

                        ex.extract("output", out_tile_gpu[ti], cmd);

                        {
                            cmd.submit_and_wait();
                            cmd.reset();
                        }
                    }

                    ncnn::VkMat out_alpha_tile_gpu;
                    if (channels == 4)
                    {
                        if (scale == 1)
                        {
                            out_alpha_tile_gpu = in_alpha_tile_gpu;
                        }
                        if (scale == 4)
                        {
                            bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                        }
                    }

                    // postproc
                    {
                        std::vector<ncnn::VkMat> bindings(10);
                        bindings[0] = out_tile_gpu[0];
                        bindings[1] = out_tile_gpu[1];
                        bindings[2] = out_tile_gpu[2];
                        bindings[3] = out_tile_gpu[3];
                        bindings[4] = out_tile_gpu[4];
                        bindings[5] = out_tile_gpu[5];
                        bindings[6] = out_tile_gpu[6];
                        bindings[7] = out_tile_gpu[7];
                        bindings[8] = out_alpha_tile_gpu;
                        bindings[9] = out_gpu;

                        std::vector<ncnn::vk_constant_type> constants(13);
                        constants[0].i = out_tile_gpu[0].w;
                        constants[1].i = out_tile_gpu[0].h;
                        constants[2].i = out_tile_gpu[0].cstep;
                        constants[3].i = out_gpu.w;
                        constants[4].i = out_gpu.h;
                        constants[5].i = out_gpu.cstep;
                        constants[6].i = xi * TILE_SIZE_X * scale;
                        constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                        constants[8].i = prepadding * scale;
                        constants[9].i = prepadding * scale;
                        constants[10].i = channels;
                        constants[11].i = out_alpha_tile_gpu.w;
                        constants[12].i = out_alpha_tile_gpu.h;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                        dispatcher.h = out_gpu.h;
                        dispatcher.c = channels;

                        cmd.record_pipeline(realsr_postproc, bindings, constants, dispatcher);
                    }
                }
                else
                {
                    // preproc
                    ncnn::VkMat in_tile_gpu;
                    ncnn::VkMat in_alpha_tile_gpu;
                    {
                        // crop tile
                        int tile_x0 = xi * TILE_SIZE_X - prepadding;
                        int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding;
                        int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                        int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding;

                        in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                        if (channels == 4)
                        {
                            in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, blob_vkallocator);
                        }

                        std::vector<ncnn::VkMat> bindings(3);
                        bindings[0] = in_gpu;
                        bindings[1] = in_tile_gpu;
                        bindings[2] = in_alpha_tile_gpu;

                        std::vector<ncnn::vk_constant_type> constants(13);
                        constants[0].i = in_gpu.w;
                        constants[1].i = in_gpu.h;
                        constants[2].i = in_gpu.cstep;
                        constants[3].i = in_tile_gpu.w;
                        constants[4].i = in_tile_gpu.h;
                        constants[5].i = in_tile_gpu.cstep;
                        constants[6].i = prepadding;
                        constants[7].i = prepadding;
                        constants[8].i = xi * TILE_SIZE_X;
                        constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                        constants[10].i = channels;
                        constants[11].i = in_alpha_tile_gpu.w;
                        constants[12].i = in_alpha_tile_gpu.h;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = in_tile_gpu.w;
                        dispatcher.h = in_tile_gpu.h;
                        dispatcher.c = channels;

                        cmd.record_pipeline(realsr_preproc, bindings, constants, dispatcher);
                    }

                    // realsr
                    ncnn::VkMat out_tile_gpu;
                    {
                        ncnn::Extractor ex = net.create_extractor();

                        ex.set_blob_vkallocator(blob_vkallocator);
                        ex.set_workspace_vkallocator(blob_vkallocator);
                        ex.set_staging_vkallocator(staging_vkallocator);

                        ex.input("data", in_tile_gpu);

                        ex.extract("output", out_tile_gpu, cmd);
                    }

                    ncnn::VkMat out_alpha_tile_gpu;
                    if (channels == 4)
                    {
                        if (scale == 1)
                        {
                            out_alpha_tile_gpu = in_alpha_tile_gpu;
                        }
                        if (scale == 4)
                        {
                            bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                        }
                    }

                    // postproc
                    {
                        std::vector<ncnn::VkMat> bindings(3);
                        bindings[0] = out_tile_gpu;
                        bindings[1] = out_alpha_tile_gpu;
                        bindings[2] = out_gpu;

                        std::vector<ncnn::vk_constant_type> constants(13);
                        constants[0].i = out_tile_gpu.w;
                        constants[1].i = out_tile_gpu.h;
                        constants[2].i = out_tile_gpu.cstep;
                        constants[3].i = out_gpu.w;
                        constants[4].i = out_gpu.h;
                        constants[5].i = out_gpu.cstep;
                        constants[6].i = xi * TILE_SIZE_X * scale;
                        constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                        constants[8].i = prepadding * scale;
                        constants[9].i = prepadding * scale;
                        constants[10].i = channels;
                        constants[11].i = out_alpha_tile_gpu.w;
                        constants[12].i = out_alpha_tile_gpu.h;

                        ncnn::VkMat dispatcher;
                        dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                        dispatcher.h = out_gpu.h;
                        dispatcher.c = channels;

                        cmd.record_pipeline(realsr_postproc, bindings, constants, dispatcher);
                    }
                }

                if (xtiles > 1)
                {
                    cmd.submit_and_wait();
                    cmd.reset();
                }

                // fprintf(stderr, "%.2f%%\n", (float)(yi * xtiles + xi) / (ytiles * xtiles) * 100);
            }

            // download
            {
                ncnn::Mat out;

                if (opt.use_fp16_storage && opt.use_int8_storage)
                {
                    out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char *)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, (size_t)channels, 1);
                }

                cmd.record_clone(out_gpu, out, opt);

                cmd.submit_and_wait();

                if (!(opt.use_fp16_storage && opt.use_int8_storage))
                {
                    if (channels == 3)
                    {
#if _WIN32
                        out.to_pixels((unsigned char *)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGB2BGR);
#else
                        out.to_pixels((unsigned char *)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGB);
#endif
                    }
                    if (channels == 4)
                    {
#if _WIN32
                        out.to_pixels((unsigned char *)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGBA2BGRA);
#else
                        out.to_pixels((unsigned char *)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGBA);
#endif
                    }
                }
            }
        }

        net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
        net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);
    }
    catch (const runtime_error &re)
    {
        net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
        net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);
        throw runtime_error(re.what());
    }
    catch (const exception &ex)
    {
        net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
        net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);
        throw exception(ex.what());
    }
    catch (...)
    {
        net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
        net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);
        throw runtime_error("Unknoqn error occurred in NCNN process");
    }

    return 0;
}

int RealSR::cleanup() const
{
    ncnn::destroy_gpu_instance();
    return 0;
}