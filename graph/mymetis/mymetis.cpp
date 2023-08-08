#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "metis.h"
#include <iostream>

using namespace std;

int long_list_to_array(PyObject *list, idx_t* &arr){
    if (Py_IsNone(list)){
        arr = NULL;
        return 0;
    }
    Py_ssize_t n, i;
    PyObject *item;
    long value;
    n = PyList_Size(list);
    arr = new idx_t[n];
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(list, i);
        value = PyLong_AsLong(item);
        if (value == -1 && PyErr_Occurred())
            return -1;
        arr[i] = value;
    }
    return i;
}

int real_list_to_array(PyObject *list, real_t* &arr){
    if (Py_IsNone(list)){
        arr = NULL;
        return 0;
    }
    Py_ssize_t n, i;
    PyObject *item;
    long value;
    n = PyList_Size(list);
    arr = new real_t[n];
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(list, i);
        value = PyFloat_AsDouble(item);
        if (value == -1. && PyErr_Occurred())
            return -1;
        arr[i] = value;
    }
    return i;
}

void print_idx_array(idx_t *arr, int size){
    if (arr==NULL){
        cout<<"NULL"<<endl;
        return;
    }
    for (int i=0;i<size;++i){
        cout<<arr[i]<<" ";
    }
    cout<<endl;
}

void print_real_array(real_t *arr, int size){
    if (arr==NULL){
        cout<<"NULL"<<endl;
        return;
    }
    for (int i=0;i<size;++i){
        cout<<arr[i]<<" ";
    }
    cout<<endl;
}

// args: xadj, adjncy, vwgt, adjwgt, nparts, tpwgts, ufactor, dbg_lvl
static PyObject *
mymetis_partition(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *xadj_list, *adjncy_list, *vwgt_list, *adjwgt_list, *tpwgts_list=Py_None;
    long nparts_long=1, ufactor_long=30, dbg_lvl_long=0, niter_long=10;
    static char *kwlist[] = {(char*)"xadj", (char*)"adjncy", (char*)"vwgt", (char*)"adjwgt", (char*)"nparts", (char*)"tpwgts", (char*)"ufactor", (char*)"niter", (char*)"dbg_lvl", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOl|Olll", kwlist,
            &xadj_list, &adjncy_list, &vwgt_list, &adjwgt_list, &nparts_long, &tpwgts_list, &ufactor_long, &niter_long, &dbg_lvl_long)){
        return NULL;
    }

    idx_t nvtxs, nadj, ncon;
    idx_t *xadj=NULL, *adjncy=NULL, *vwgt=NULL, *adjwgt=NULL; 
    real_t *tpwgts=NULL;
    idx_t *part=NULL, objval=0;
    idx_t options[METIS_NOPTIONS];
    idx_t nparts=nparts_long, ufactor=ufactor_long, dbg_lvl=dbg_lvl_long, niter=niter_long;
    int status = 0;

    nvtxs = PyList_Size(xadj_list)-1;
    nadj = PyList_Size(adjncy_list);
    ncon = 1;
    part = new idx_t[nvtxs];

    if (long_list_to_array(xadj_list, xadj)<0) return NULL;
    //Py_XDECREF(xadj_list);
    if (long_list_to_array(adjncy_list, adjncy)<0) return NULL;
    //Py_XDECREF(adjncy_list);
    if (long_list_to_array(vwgt_list, vwgt)<0) return NULL;
    //Py_XDECREF(vwgt_list);
    if (long_list_to_array(adjwgt_list, adjwgt)<0) return NULL;
    //Py_XDECREF(adjwgt_list);
    if (real_list_to_array(tpwgts_list, tpwgts)<0) return NULL;
    //Py_XDECREF(tpwgts_list);

    if (dbg_lvl>0){
        cout<<"Metis partition."<<endl;
        cout<<"nvtxs:"<<nvtxs<<",nadj:"<<nadj<<endl;
        cout<<"xadj:"<<endl;
        print_idx_array(xadj, nvtxs+1);
        cout<<"adjncy:"<<endl;
        print_idx_array(adjncy, nadj);
        cout<<"vwgt:"<<endl;
        print_idx_array(vwgt, nvtxs);
        cout<<"adjwgt:"<<endl;
        print_idx_array(adjwgt, nadj);
        cout<<"tpwgts:"<<endl;
        print_real_array(tpwgts, nparts);
    }

    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_SEED] = -1;
    options[METIS_OPTION_NCUTS] = nparts;
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_NITER] = niter;
    options[METIS_OPTION_UFACTOR] = ufactor;
    options[METIS_OPTION_DBGLVL] = dbg_lvl;

    status = METIS_PartGraphKway(&nvtxs, &ncon, xadj, 
                adjncy, vwgt, NULL, adjwgt, 
                &nparts, tpwgts, NULL, options, 
                &objval, part);
    
    if (dbg_lvl>0){
        cout<<"Status:"<<status<<endl;
        cout<<"Object:"<<objval<<endl;
        cout<<"Partitions:"<<endl;
        print_idx_array(part, nvtxs);
    }

    int objval_int = objval;
    PyObject *part_list;
    long value;
    PyObject *item;
    part_list = PyList_New((Py_ssize_t)nvtxs);
    for (idx_t i=0;i<nvtxs;++i){
        value = part[i];
        item = PyLong_FromLong(value);
        if (item==NULL&&PyErr_Occurred()){
            return NULL;
        }
        PyList_SetItem(part_list, (Py_ssize_t)i, item);
    }
    PyObject *result;
    result = Py_BuildValue("(iO)", objval_int, part_list);
    //Py_DECREF(part_list);
    if (xadj!=NULL) delete[] xadj;
    if (adjncy!=NULL) delete[] adjncy;
    if (vwgt!=NULL) delete[] vwgt;
    if (adjwgt!=NULL) delete[] adjwgt;
    if (tpwgts!=NULL) delete[] tpwgts;
    if (part!=NULL) delete[] part;

    return result;
}

static PyMethodDef MymetisMethods[] = {
    {"partition", (PyCFunction)(void(*)(void))mymetis_partition, METH_VARARGS | METH_KEYWORDS,
     "Partition a graph."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static const char *mymetis_doc="metis partition";

static struct PyModuleDef mymetismodule = {
    PyModuleDef_HEAD_INIT,
    "mymetis",   /* name of module */
    mymetis_doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    MymetisMethods
};

PyMODINIT_FUNC
PyInit_mymetis(void)
{
    PyObject *m;

    m = PyModule_Create(&mymetismodule);
    if (m == NULL)
        return NULL;

    return m;
}

/* test graph
7 11 001
5 1 3 2 2 1
1 1 3 2 4 1
5 3 4 2 2 2 1 2
2 1 3 2 6 2 7 5
1 1 3 3 6 2
5 2 4 2 7 6
6 6 4 5
*/

int main(int argc, char *argv[]){
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    if (PyImport_AppendInittab("mymetis", PyInit_mymetis) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required.
       If this step fails, it will be a fatal error. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */
    PyObject *pmodule = PyImport_ImportModule("mymetis");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'mymetis'\n");
    }
    Py_Finalize();
    PyMem_RawFree(program);
    return 0;
}

