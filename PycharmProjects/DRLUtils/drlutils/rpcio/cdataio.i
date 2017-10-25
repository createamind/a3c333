%module cdataio

%{
    /* the resulting C file should be built as a python extension */
    #define SWIG_FILE_WITH_INIT
    /*  Includes the header in the wrapper code */
    #include "src/stdafx.h"
    #include "src/cdataio.h"
%}

%include "numpy.i"
%init %{
 import_array();
%}


%apply (unsigned char * IN_ARRAY3, int DIM1, int DIM2, int DIM3) { (unsigned char * src, int sh, int sw, int schannel) };
%apply (unsigned char * INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) { (unsigned char * dst, int dh, int dw, int dchannel) };

%include "std_vector.i"
%include "std_map.i"
%include "std_string.i"

namespace std {
%template(_string_list) vector<string>;
%template(_vector_of_vector_string) vector<vector<string>>;
%template(_map_of_string_string) map<string, string>;
%template(_vector_map_of_string_string) vector<map<string, string>>;
}


# %apply (float * INPLACE_ARRAY2, int DIM1, int DIM2) {(float* array, int len1, int len2)}
%apply (float * IN_ARRAY1, int DIM1) { (float * policy, int plen) }

%apply (int * IN_ARRAY2, int DIM1, int DIM2) { (int * actions, int dim0, int dim1) }

%typemap(in, numinputs=0) std::vector<PyArrayObject*> * outputs (
    std::vector<PyArrayObject*> temp) {
  $1 = &temp;
}


%typemap(in) PyObject * py_states_callback {
    $1 = $input;
    Py_XINCREF($1);
}

%typemap(argout) std::vector<PyArrayObject*> * outputs {
   $result = PyList_New($1->size());
   for (size_t i = 0; i < $1->size(); ++i)
   {
       PyList_SET_ITEM($result, i, (PyObject*)$1->at(i));
   }
}

%typemap(out) PyArrayObject * idents {
   $result = (PyObject*)$1;
}

%typemap(out) PyTupleObject * states {
   $result = (PyObject*)$1;
}
%typemap(out) PyArrayObject * last_is_overs {
   $result = (PyObject*)$1;
}


%typemap(in) PyArrayObject * {
   $1 = (PyArrayObject*)$input;
}

%typemap(out) PyArrayObject * {
  $result = (PyObject*)$1;
}

%typemap(in) PyObject * envDataList {
   $1 = (PyObject*)$input;
   Py_XINCREF($1);
}

%typemap(out) PyObject * {
  $result = $1;
}

%include "src/cdataio.h"


%inline %{
%}

