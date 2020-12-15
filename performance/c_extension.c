#include "Python.h"
#include "numpy/arrayobject.h"
#include "stdlib.h"
#include "omp.h"


#define MAX_SIZE 1000*1000
typedef double BASE_TYPE;
BASE_TYPE row_major[MAX_SIZE];
BASE_TYPE column_major[MAX_SIZE];


BASE_TYPE **init_result_array(int total_rows, int total_columns) {
    //creating 2D array for copying Python list object into
    BASE_TYPE **result_array = (BASE_TYPE **)calloc(total_rows, sizeof(BASE_TYPE *));
    for(int row = 0; row < total_rows; row++) {
        result_array[row] = (BASE_TYPE *)calloc(total_columns, sizeof(BASE_TYPE));
    }
    return result_array;
};

BASE_TYPE **convert(PyObject *ndimarray, int rows, int columns) {
    //Unwraps Python list into C pointer to 2D array

    BASE_TYPE **c_array = init_result_array(rows, columns);
    PyObject *current_row;
    for (int i = 0; i < rows; ++i) {
        current_row = PyList_GetItem(ndimarray, i);
        for (int j = 0; j < columns; ++j) {
            c_array[i][j] = (BASE_TYPE )PyLong_AsLong(PyList_GetItem(current_row, j));
        }
    }
    return c_array;
};

void transform_row_major(BASE_TYPE **ndimarray, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            row_major[i * columns + j] = ndimarray[i][j];
        }
    }
};

void transform_column_major(BASE_TYPE **ndimarray, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            column_major[j * rows + i] = ndimarray[i][j];
        }
    }
};

PyObject* build_python_array(BASE_TYPE** result_array, int rows, int columns) {
    // Building Python result object from C 2D array pointer

    PyObject *item;
    PyObject *pyResult = PyList_New(rows);
    for (int i= 0; i< rows; ++i) {
        item = PyList_New(columns);
        for (int j= 0; j< columns; ++j) {
            PyList_SetItem(item, j, PyLong_FromLong(result_array[i][j]));
        }
        PyList_SetItem(pyResult, i, item);
    }
    return pyResult;
};

PyObject* dot_product_optimized_parallel(PyObject* self, PyObject* args) {
    PyObject *mat1;
    PyObject *mat2;

    if (!PyArg_ParseTuple(args, "O|O", &mat1, &mat2)){
        return NULL;
    }
    int mat1_rows, mat1_columns, mat2_rows, mat2_columns;
    mat1_rows = PyObject_Length(mat1);
    mat1_columns = PyObject_Length(PyList_GetItem(mat1, 0));
    mat2_rows = PyObject_Length(mat2);
    mat2_columns = PyObject_Length(PyList_GetItem(mat2, 0));
    BASE_TYPE **mat1_c = convert(mat1, mat1_rows, mat1_columns);
    BASE_TYPE **mat2_c = convert(mat2, mat2_rows, mat2_columns);
    transform_row_major(mat1_c, mat1_rows, mat1_columns);
    transform_column_major(mat2_c, mat2_rows, mat2_columns);
    BASE_TYPE **result = init_result_array(mat1_rows, mat2_columns);
    #pragma omp parallel num_threads(6)
    {
        float tot;
        int iOff, jOff;
        #pragma omp for
        for(int i=0; i < mat1_rows; i++) {
            iOff = i * mat1_columns;
            for(int j=0; j < mat2_columns; j++) {
                tot = 0;
                jOff = j * mat2_rows;
                for(int k=0; k < mat2_rows; k++){
                    tot += (float)row_major[iOff + k] * (float)column_major[jOff + k];
                }
                result[i][j] = tot;
            }
        }
    };
    //test1
    return Py_BuildValue("O", build_python_array(result, mat1_rows, mat2_columns));
};

PyObject* dot_product_optimized(PyObject* self, PyObject* args) {
    PyObject *mat1;
    PyObject *mat2;

    if (!PyArg_ParseTuple(args, "O|O", &mat1, &mat2)){
        return NULL;
    }
    int mat1_rows, mat1_columns, mat2_rows, mat2_columns;
    mat1_rows = PyObject_Length(mat1);
    mat1_columns = PyObject_Length(PyList_GetItem(mat1, 0));
    mat2_rows = PyObject_Length(mat2);
    mat2_columns = PyObject_Length(PyList_GetItem(mat2, 0));

    BASE_TYPE **mat1_c = convert(mat1, mat1_rows, mat1_columns);
    BASE_TYPE **mat2_c = convert(mat2, mat2_rows, mat2_columns);
    transform_row_major(mat1_c, mat1_rows, mat1_columns);
    transform_column_major(mat2_c, mat2_rows, mat2_columns);
    BASE_TYPE **result = init_result_array(mat1_rows, mat2_columns);
    int tot, iOff, jOff;
    for (int i=0; i < mat1_rows; i++) {
        iOff = i * mat1_columns;
        for (int j=0; j < mat2_columns; j++) {
            tot = 0;
            jOff = j * mat2_rows;
            for (int k=0; k < mat2_rows; k++){
                tot += row_major[iOff + k] * column_major[jOff + k];
            }
            result[i][j] = tot;
        }
    }
    return Py_BuildValue("O", build_python_array(result, mat1_rows, mat2_columns));
};



PyObject* dot_product(PyObject* self, PyObject* args) {
    PyObject *mat1, *mat2;
    if(!PyArg_ParseTuple(args, "O|O", &mat1, &mat2)) {
        return NULL;
    }
    //getting dimensions of our lists
    int mat1_rows, mat1_columns, mat2_rows, mat2_columns;
    mat1_rows = PyObject_Length(mat1);
    mat1_columns = PyObject_Length(PyList_GetItem(mat1, 0));
    mat2_rows = PyObject_Length(mat2);
    mat2_columns = PyObject_Length(PyList_GetItem(mat2, 0));
    PyObject *pyResult = PyList_New(mat1_rows);
    PyObject *item;
    PyObject *mat1_current_row;
    int total;
    for(int i = 0; i < mat1_rows; i++) {
        item = PyList_New(mat2_columns);
        mat1_current_row = PyList_GetItem(mat1, i);
        for(int j = 0; j < mat2_columns; j++) {
            total = 0;
            for (int k = 0; k < mat2_rows; k++) {
                total += (int)PyLong_AsLong(PyList_GetItem(mat1_current_row, k)) * (int)PyLong_AsLong(PyList_GetItem(PyList_GetItem(mat2, k), j));
            }
        PyList_SetItem(item, j, PyLong_FromLong(total));
        }
    PyList_SetItem(pyResult, i, item);
    }
    return Py_BuildValue("O", pyResult);
};

static PyObject* cos_func_np(PyObject* self, PyObject* args)
{
    PyArrayObject *arrays[2];  /* holds input and output array */
    PyObject *ret;
    NpyIter *iter;
    npy_uint32 op_flags[2];
    npy_uint32 iterator_flags;
    PyArray_Descr *op_dtypes[2];

    NpyIter_IterNextFunc *iternext;
    /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arrays[0])) {
        return NULL;
    }
    arrays[1] = NULL;  /* The result will be allocated by the iterator */
    /* Set up and create the iterator */
    iterator_flags = (NPY_ITER_ZEROSIZE_OK |
                      NPY_ITER_BUFFERED |
                      /* Manually handle innermost iteration for speed: */
                      NPY_ITER_EXTERNAL_LOOP |
                      NPY_ITER_GROWINNER);

    op_flags[0] = (NPY_ITER_READONLY |
                   /*
                    * Required that the arrays are well behaved, since the cos
                    * call below requires this.
                    */
                   NPY_ITER_NBO |
                   NPY_ITER_ALIGNED);

    /* Ask the iterator to allocate an array to write the output to */
    op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

    /*
     * Ensure the iteration has the correct type, could be checked
     * specifically here.
     */
    op_dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
    op_dtypes[1] = op_dtypes[0];

    /* Create the numpy iterator object: */
    iter = NpyIter_MultiNew(2, arrays, iterator_flags,
                            /* Use input order for output and iteration */
                            NPY_KEEPORDER,
                            /* Allow only byte-swapping of input */
                            NPY_EQUIV_CASTING, op_flags, op_dtypes);
    Py_DECREF(op_dtypes[0]);  /* The second one is identical. */

    if (iter == NULL)
        return NULL;

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    /* Fetch the output array which was allocated by the iterator: */
    ret = (PyObject *)NpyIter_GetOperandArray(iter)[1];
    Py_INCREF(ret);
    if (NpyIter_GetIterSize(iter) == 0) {
        /*
         * If there are no elements, the loop cannot be iterated.
         * This check is necessary with NPY_ITER_ZEROSIZE_OK.
         */
        NpyIter_Deallocate(iter);
        return ret;
    }

    /* The location of the data pointer which the iterator may update */
    char **dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    /* iterate over the arrays */
    do {
        npy_intp stride = strideptr[0];
        npy_intp count = *innersizeptr;
        /* out is always contiguous, so use double */
        double *out = (double *)dataptr[1];
        char *in = dataptr[0];

        /* The output is allocated and guaranteed contiguous (out++ works): */
        assert(strideptr[1] == sizeof(double));

        /*
         * For optimization it can make sense to add a check for
         * stride == sizeof(double) to allow the compiler to optimize for that.
         */
        while (count--) {
            *out = cos(*(double *)in);
            out++;
            in += stride;
        }
    } while (iternext(iter));

    /* Clean up and return the result */
    NpyIter_Deallocate(iter);
    return ret;
};

static PyMethodDef module_methods[] = {
        {"dot_product", (PyCFunction) dot_product, METH_VARARGS, "Calculates dot product of two matrices"},
        {"cos_func_np", (PyCFunction) cos_func_np, METH_VARARGS, "Calculates dot product of two matrices with np"},
        {"dot_product_optimized", (PyCFunction) dot_product_optimized, METH_VARARGS, "Optimized version of dot_product"},
        {"dot_product_optimized_parallel", (PyCFunction) dot_product_optimized_parallel, METH_VARARGS, "Optimized parallol version of dot_product"},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,
        "c_extension",
        "",
        -1,
        module_methods
};


PyMODINIT_FUNC PyInit_c_extension(void) {
    PyObject *module;
    module = PyModule_Create(&cModPyDem);
    if(module==NULL) return NULL;
    /* IMPORTANT: this must be called */
    import_array();
    if (PyErr_Occurred()) return NULL;
    return module;
};