%module sr_ncnn_vulkan_wrapper

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

%allowexception;   
%exception {
    try
    {
        $action
    }
    catch(const std::runtime_error& re)
    {
        PyErr_SetString(PyExc_RuntimeError, re.what());
        SWIG_fail;
    }
    catch(const std::exception& ex)
    {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        SWIG_fail;
    }
    catch (...)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error occurred: unknown");
        SWIG_fail;
    }
}

%{
#include "sr.h"
#include "sr_wrapped.h"
%}

class SR
{
    public:
        SR(int gpuid, bool tta_mode = false);
        ~SR();

        // sr parameters
        int scale;
        int tilesize;
        int prepadding;
};
%include "sr_wrapped.h"