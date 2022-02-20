% module realsr_ncnn_vulkan_wrapper

    % include "cpointer.i" % include "carrays.i" % include "std_string.i" % include "std_wstring.i" % include "stdint.i" % include "pybuffer.i"
    // %include "exception.i"

    % pybuffer_mutable_string(unsigned char *d);
% pointer_functions(std::string, str_p);
% pointer_functions(std::wstring, wstr_p);

%
{
#include "realsr.h"
#include "realsr_wrapped.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
    %
}

class RealSR
{
public:
    RealSR(int gpuid, bool tta_mode = false);
    ~RealSR();

    // realsr parameters
    int scale;
    int tilesize;
    int prepadding;
};
% include "realsr_wrapped.h" % exception
{
    try
    {
        $action
    }
    catch (const std::runtime_error &re)
    {
        PyErr_SetString(PyExc_RuntimeError, std::string("Runtime error: ") + std::string(re.what()));
        SWIG_fail;
    }
    catch (const std::exception &ex)
    {
        PyErr_SetString(PyExc_RuntimeError, std::string("Error occurred: ") + std::string(ex.what()));
        SWIG_fail;
    }
    // catch(OutOfMemory)
    // {
    //     SWIG_exception(SWIG_MemoryError, "Error occurred: Out of memory");
    // }
    catch (...)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error occurred: unknown");
        SWIG_fail;
    }
}