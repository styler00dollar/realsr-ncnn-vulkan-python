%module realsr_ncnn_vulkan_wrapper

%include "cpointer.i"
%include "carrays.i"
%include "std_string.i"
%include "std_wstring.i"
%include "stdint.i"
%include "pybuffer.i"
%include "exception.i"   

%pybuffer_mutable_string(unsigned char *d);
%pointer_functions(std::string, str_p);
%pointer_functions(std::wstring, wstr_p);

%{
#include "realsr.h"
#include "realsr_wrapped.h"
#include <iostream>
%}

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
%include "realsr_wrapped.h"
%exception {
    try
    {
        $action
    }
    catch(const std::runtime_error& re)
    {
        SWIG_exception(SWIG_RuntimeError, "Runtime error: " << re.what() << std::endl);
    }
    catch(const std::exception& ex)
    {
        SWIG_exception(SWIG_RuntimeError, "Error occurred: " << ex.what() << std::endl);
    }
    catch(OutOfMemory)
    {
        SWIG_exception(SWIG_MemoryError, "Error occurred: Out of memory");
    }
    catch (...)
    {
        SWIG_exception(SWIG_UnknownError, "Error occurred: unknown");
    }
}