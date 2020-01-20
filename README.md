# UnsupportedLayerTraining
The training project which includes MO and IE for OpenVINO unsupported layer - `cosh`.

### Create a tensorflow\* model with unsupported layer `cosh` and convert it to IR.

1. Create `tf_cosh.py` to generate the model and save as `cosh.pb`

    ```py
    import tensorflow as tf 
    import tensorflow.contrib.layers as layers

    import numpy as np

    weights = {
        'wc1': tf.Variable(tf.truncated_normal([1, 1, 1, 32]))
        }

    biases = {
        'bc1': tf.Variable(tf.zeros([32]))
        }

    def model (inputs) :
        conv1 = tf.nn.conv2d(inputs, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME') + biases['bc1']
        out = tf.math.cosh(conv1)
        return out

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))

    o = model(x)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    print (sess.run(o, {x : np.ones((1, 32, 32, 1))}))

    from tensorflow.python.framework import graph_io
    frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Cosh'])
    graph_io.write_graph(frozen, '.', 'cosh.pb', as_text=False)
    ```

2. Run `tf_cosh.py`

    ```sh
    $ python tf_cosh.py
    ```

3. Convert `cosh.pb` to IR

    * Run:

        ```sh
        $ python $MO_ROOT/mo.py  \
            --input_model=$AS_NEW/cosh.pb \
            --input_shape=[1,32,32,1] \
            --disable_nhwc_to_nchw \
            -o $OUTPUT_DIR
        ```

    * Output:

        ```sh
        [ ERROR ]  List of operations that cannot be converted to Inference Engine IR:
        [ ERROR ]      Cosh (1)
        [ ERROR ]          Cosh
        [ ERROR ]  Part of the nodes was not converted to IR. Stopped. 
        For more information please refer to Model Optimizer FAQ (https://docs.openvinotoolkit.org/
        latest/_docs_MO_DG_prepare_model_Model_Optimizer_FAQ.html), question #24.
        ```

        > Got the errors since the `Cosh` layer was not supported by model optimizer.

### Add unsupported TF layer into Model Optimizer

* **Use existing operation to construct a new operation** - Add `Cosh` operator into `activation_ops.py`.

    1. Add `Cosh` operator into list `activation_ops`

        * From:

            ```py
            activation_ops = ['Sigmoid', 'Tanh', 'ReLU6', 'Exp', 'Elu', 'Not', 'Floor']
            ```

        * To:

            ```py
            activation_ops = ['Sigmoid', 'Tanh', 'ReLU6', 'Exp', 'Elu', 'Not', 'Floor', 'Cosh']
            ```

    2. Create a class for `Cosh` operator which inherit from class `Activation`

        ```py
        class Cosh(Activation):
            op = 'Cosh'
            operation = staticmethod(lambda x: np.cosh(x))
        ```

    3. Create the file `cosh_ext.py` in the directory `$OPENVINO_ROOT/deployment_tools/model_optimizer/extensions/front/tf to add the layer extractor for unsupported layer` - `Cosh`

        ```py
        from extensions.ops.activation_ops import Cosh
        from mo.front.extractor import FrontExtractorOp


        class LeakyReLUFrontExtractor(FrontExtractorOp):
            op = 'Cosh'
            enabled = True

            @staticmethod
            def extract(node):
                Cosh.update_node_stat(node)
                return __class__.enabled
        ```

    4. Convert the model again

        * Run:

            ```sh
            $ python $MO_ROOT/mo.py \
                --input_model=$USING_EXISTING_LAYER/cosh.pb \
                --input_shape=[1,32,32,1] \
                --disable_nhwc_to_nchw \
                -o $OUTPUT_DIR
            ```

        * Output:

            ```sh
            [ SUCCESS ] Generated IR model.
            [ SUCCESS ] XML file: $OUTPUT_DIR/cosh.xml
            [ SUCCESS ] BIN file: $OUTPUT_DIR/cosh.bin
            [ SUCCESS ] Total execution time: 1.32 seconds.
            ```
    

* Construct a new operation 

    - Create a templates for `Cosh` custom layer

        1. Run `extgen.py` with parameter `--mo-tf-ext`, `--mo-op`, `--ie-cpu-ext`, and `--ie-gpu-ext`.

            ```sh
            python $OPENVINO_ROOT/deployment_tools/tools/extension_generator/extgen.py new \
                --mo-tf-ext \
                --mo-op \
                --ie-cpu-ext \
                --ie-gpu-ext \
                --output_dir=${OUTPUT_DIR}
            ```
        2. Information based on execution commands

            ```sh
            Generating:
              Model Optimizer:
                Extractor for Caffe Custom Layer: No
                  Extractor for MxNet Custom Layer: No
                Extractor for TensorFlow Custom Layer: Yes
                  Framework-agnostic operation extension: Yes
                Inference Engine:
                  CPU extension: Yes
                  GPU extension: Yes
            ```

        3. Answer the questions

            ```sh
            Enter layer name:   Cosh

            Do you want to automatically parse all parameters from the model file? (y/n)
              Yes means layer parameters will be automatically parsed during Model Optimizer work as is.
              No means you will be prompted for layer parameters in the following section    n

            Enter all parameters in the following format:
              <param1> <new name1> <type1>
              <param2> <new name2> <type2>
              ...
            Where type is one of the following types:

              b - Bool,                               padding - Padding type,                 list.b - List of bools,
              f - Float,                              batch - Get batch from dataFormat,      list.f - List of floats,
              i - Int,                                channel - Get channel from dataFormat,  list.i - List of ints,
              s - String,                             spatial - Get spatial from dataFormat,  list.s - List of strings,
              shape - TensorShapeProto,               list.shape - List of TensorShapeProto,  type - DataType, list.type - List of DataType,
            Example:
              length attr_length i

            If your attribute type is not shown in the list above, or you want to implement your own 
            attribute parsing, omit the <type> parameter.
            Enter 'q' when finished:    q

            **********************************************************************************************
            Check your answers for TensorFlow* extractor generation:

            1.  Layer name:                                                            Cosh
            2.  Automatically parse all parameters from model file:                    No
            3.  Parameters entered:  <param1> <new name1> <type1>                      []

            **********************************************************************************************

            Do you want to change any answer (y/n) ? Default 'no'
            n

            Do you want to use the layer name as the operation name? (y/n)    y

            Does your operation change shape? (y/n)    n


            **********************************************************************************************
            Check your answers for the Model Optimizer operation generation:

            4.  Use layer name as operation name? (y/n)                                Yes
            5.  Operation changes shape? (y/n)                                         No

            **********************************************************************************************

            Do you want to change any answer (y/n) ? Default 'no'
            n
            ```

        4. Generate files

            ```sh
            Stub file for TensorFlow Model Optimizer extractor is in ${OUTPUT_DIR}/user_mo_extensions/front/tf folder
            Stub file for the Model Optimizer operation is in ${OUTPUT_DIR}/user_mo_extensions/ops folder
            Stub files for the Inference Engine CPU extension are in ${OUTPUT_DIR}/user_ie_extensions/cpu folder
            Stub files for the Inference Engine GPU extension are in ${OUTPUT_DIR}/user_ie_extensions/gpu folder
            ```

        5. Tree structure of generate files

            ```sh
            ├── user_ie_extensions
            │   ├── cpu
            │   │   ├── CMakeLists.txt
            │   │   ├── ext_base.cpp
            │   │   ├── ext_base.hpp
            │   │   ├── ext_cosh.cpp
            │   │   ├── ext_list.cpp
            │   │   └── ext_list.hpp
            │   └── gpu
            │       ├── cosh_kernel.cl
            │       └── cosh_kernel.xml
            └── user_mo_extensions
                ├── front
                │   ├── caffe
                │   │   └── __init__.py
                │   ├── __init__.py
                │   ├── mxnet
                │   │   └── __init__.py
                │   └── tf
                │       ├── cosh_ext.py
                │       └── __init__.py
                ├── __init__.py
                └── ops
                    ├── cosh.py
                    └── __init__.py
            ```

    - **Construct a new operation** - `Cosh`

        1. Patch the file `cosh.py` and move to directory `$OPENVINO_ROOT/deployment_tools/model_optimizer/extensions/ops`

            ```py
            from mo.ops.op import Op
            from mo.front.common.partial_infer.elemental import copy_shape_infer
            from mo.graph.graph import Node


            class CoshOp(Op):
                op = 'Cosh'

                def __init__(self, graph, attrs):
                    mandatory_props = dict(
                        type=__class__.op,
                        op=__class__.op,
                        infer=CoshOp.infer            
                    )
                    super().__init__(graph, mandatory_props, attrs)

                @staticmethod
                def infer(node: Node):
                    return copy_shape_infer(node)
            ```

        2. Cosh Extractor (Choose one method)
            1. Move the file `cosh_ext.py` to directory `$OPENVINO_ROOT/deployment_tools/model_optimizer/extensions/front/tf`
                ```py
                import numpy as np

                from mo.front.extractor import FrontExtractorOp
                from mo.ops.op import Op
                from mo.front.tf.extractors.utils import *
                from mo.front.common.partial_infer.utils import convert_tf_padding_to_str

                class CoshFrontExtractor(FrontExtractorOp):
                    op = 'Cosh' 
                    enabled = True

                    @staticmethod
                    def extract(node):
                        proto_layer = node.pb
                        param = proto_layer.attr
                        # extracting parameters from TensorFlow layer and prepare them for IR
                        attrs = {
                            'op': __class__.op
                        }

                        # update the attributes of the node
                        Op.get_op_class_by_name(__class__.op).update_node_stat(node, attrs)

                        return __class__.enabled
                ```
            2. Patch the file `cosh_ext.py` and move to directory `$OPENVINO_ROOT/deployment_tools/model_optimizer/extensions/front/tf`

                ```py
                from mo.front.extractor import FrontExtractorOp
                from extensions.ops.cosh_tf import CoshOp

                class CoshFrontExtractor(FrontExtractorOp):
                    op = 'Cosh'
                    enabled = True

                    @staticmethod
                    def extract(node):
                        # update the attributes of the node
                        CoshOp.update_node_stat(node)
                        return __class__.enabled
                ```

        3. Convert the model again

           * Run:

                ```sh
                $ python $MO_ROOT/mo.py \
                    --input_model=$USING_EXISTING_LAYER/cosh.pb \
                    --input_shape=[1,32,32,1] \
                    --disable_nhwc_to_nchw \
                    -o $OUTPUT_DIR
                ```

           * Output:

                ```sh
                [ SUCCESS ] Generated IR model.
                [ SUCCESS ] XML file: $OUTPUT_DIR/cosh.xml
                [ SUCCESS ] BIN file: $OUTPUT_DIR/cosh.bin
                [ SUCCESS ] Total execution time: 1.32 seconds.
                ```
### Run inference

* Create `infer.py` for inference

    ```py
    import sys, os, time
    import numpy as np
    import argparse
    import time

    from openvino.inference_engine import IENetwork, IECore

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def parsing():
        parser = argparse.ArgumentParser(add_help = False)
        parser.add_argument('-m', '--model', default='', type=str)
        parser.add_argument('-d', '--device', default='CPU', type=str)
        parser.add_argument('-c', '--gpu_extension', default='', type=str)
        parser.add_argument('-l', '--cpu_extension', default='', type=str)

        return parser

    def main() :
        args = parsing().parse_args()

        model_graph = args.model
        model_weight = args.model[:-3] + 'bin'

        net = IENetwork(model = model_graph, 
                        weights = model_weight)

        iter_inputs = iter(net.inputs)
        iter_outputs = iter(net.outputs)
        
        inputs_num = len(net.inputs)
        print (inputs_num)

        input_blob = []
        for _inputs in iter_inputs:
            input_blob.append(_inputs)

        output_blob = []
        for _outputs in iter_outputs:
            output_blob.append(_outputs)

        input_l = []
        for i in input_blob:
            input_l.append(np.ones(shape=net.inputs[i].shape, dtype=np.float32))

        inputs = dict()
        for i in range (inputs_num):
            inputs[input_blob[i]] = input_l[i]

        plugin = IECore()
        if args.cpu_extension and 'CPU' in args.device:
            plugin.add_extension(args.cpu_extension, "CPU")
        if args.gpu_extension and 'GPU' in args.device:
            plugin.set_config({"CONFIG_FILE": args.gpu_extension}, "GPU")

        exec_net = plugin.load_network(network = net, device_name = args.device)

        s_time = time.time()
        out = exec_net.infer(inputs)
        e_time = time.time()    

        print (out)
        print ('execution time: ', e_time - s_time)


    if "__main__" :
        main()
    ```

* Run inference on **CPU**

    1. Execute `infer.py` with parameter `–m`, `–d` and `–l`.

        * Run:

            * Use existing layer to construct an operation

                ```sh
                $ python infer.py \
                    -m $USING_EXISTING_LAYER/cosh.xml \
                    -l $SAMPLE_BUILD_DIR/intel64/Release/lib/libcpu_extension.so \
                    -d CPU
                ```

            * Construct a new operation

                ```sh
                $ python infer.py \
                    -m $AS_NEW/cosh.xml \
                    -l $SAMPLE_BUILD_DIR/intel64/Release/lib/libcpu_extension.so \
                    -d CPU
                ```
        
        * Output:

            * Use existing layer to construct an operation

                ```sh
                RuntimeError: Unsupported primitive of type: cosh name: Cosh
                ```
                > Since the math extension is supported **`Cosh`** layer not **`cosh`** layer.
                > Solution: **Add CPU extension for IE**

            * Construct a new operation

                **Run the model well**
                > Since the math extension is supported ‘Cosh’ layer

    2. Add CPU extension for IE
   
        * Create the file `ext_cosh.cpp` in the directory `$OPENVINO_ROOT/deployment_tools/inference_engine/src/extensio or patch the file `ext_cosh.cpp` which under `$GEN_OUTPUT/user_ie_extensions/cpu/`

            ```cpp
            #include "ext_list.hpp"
            #include "ext_base.hpp"

            #include <algorithm>
            #include <string>
            #include <vector>
            #include <cmath>
            #include <utility>
            #include <functional>

            namespace InferenceEngine {
            namespace Extensions {
            namespace Cpu {

            class coshImpl: public ExtLayerBase {
            public:
                explicit coshImpl(const CNNLayer* layer) {
                    try {
                        if (layer->insData.size() != 1 || layer->outData.empty())
                            THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

                        addConfig(layer, {DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
                    } catch (InferenceEngine::details::InferenceEngineException &ex) {
                        errorMsg = ex.what();
                    }
                }

                StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                                ResponseDesc *resp) noexcept override {
                    SizeVector in_dims = inputs[0]->getTensorDesc().getDims();
                    SizeVector out_dims = outputs[0]->getTensorDesc().getDims();    
                
                    float* src_data = inputs[0]->buffer();
                    float* dst_data = outputs[0]->buffer();

                    int _dsize = 1;
                    for (size_t i = 0; i < in_dims.size(); i++)
                        _dsize *= in_dims[i];

                    for (int i = 0; i < _dsize; i++)
                    {
                        double __exp = exp(src_data[i]);
                        dst_data[i] = (__exp + (1 / __exp)) / 2;
                    }
                    return OK;
                }

            private:
            };

            REG_FACTORY_FOR(ImplFactory<coshImpl>, cosh);

            }  // namespace Cpu
            }  // namespace Extensions
            }  // namespace InferenceEngine
            ```

    3. Build extension file

        * Create file by yourself

            ```sh
            $ ./$OPENVINO_ROOT/deployment_tools/inference_engine/demos/build_demos.sh
            ```

        * Generate by `extgen.py`

            ```sh
            $ cd ${OUTPUT_DIR}/user_ie_extensions/cpu
            $ mkdir build && cd build 
            $ cmake .. && make -j8
            ```

    4. Execute `infer.py` with parameter `–m`, `–d` and `–l` again.

        * Run:

            * Construct a new operation

                ```sh
                $ python infer.py \
                    -m $AS_NEW/cosh.xml \
                    -l $SAMPLE_BUILD_DIR/intel64/Release/lib/libcpu_extension.so \
                    -d CPU
                ```
        
            * Generate by `extgen.py`

                ```sh
                $ python infer.py \
                    -m $AS_NEW/cosh.xml \
                    -l ${GEN_OUTPUT}/user_ie_extensions/cpu/build/intel64/Release/lib/libcpu_extension.so \
                    -d CPU
                ```


* Run inference on **GPU**

    1. Execute `infer.py` with parameter `–m` and `–d`.

        * Run:

            * Use existing layer to construct an operation

                ```sh
                $ python infer.py \
                    -m $USING_EXISTING_LAYER/cosh.xml \
                    -d GPU
                ```

            * Construct a new operation

                ```sh
                $ python infer.py \
                    -m $AS_NEW/cosh.xml \
                    -d GPU
                ```
        
        * Output:

            * Use existing layer to construct an operation

                ```sh
                  File "ie_api.pyx", line 85, in openvino.inference_engine.ie_api.IECore.load_network
                  File "ie_api.pyx", line 92, in openvino.inference_engine.ie_api.IECore.load_network
                RuntimeError: Unknown Layer Type: cosh
                ```

            * Construct a new operation

                ```sh
                  File "ie_api.pyx", line 85, in openvino.inference_engine.ie_api.IECore.load_network
                  File "ie_api.pyx", line 92, in openvino.inference_engine.ie_api.IECore.load_network
                RuntimeError: Unknown Layer Type: Cosh
                ```

                > Since the GPU didn't support **`Cosh`** and **`cosh`** layer.
                > Solution: **Add GPU extension for IE**

   2. Add GPU extension for IE

        Here provide 2 method to enable GPU extension for IE.
        * Use existing layer to construct an operation

            * Patch `cosh_kernel.cl` which under `$GEN_OUTPUT/user_ie_extensions/gpu/`

                ```cpp
                #pragma OPENCL EXTENSION cl_khr_fp16 : enable

                __kernel void cosh_kernel(
                    // Insert pointers to inputs, outputs as arguments here
                    // If your layer has one input and one output, arguments will be:
                        const __global INPUT0_TYPE*  input0, __global OUTPUT0_TYPE* output
                    )
                {
                    // Add the kernel implementation here:
                    
                    // Get data dimensions
                    
                    const uint T_ = INPUT0_DIMS[0];
                    const uint N_ = INPUT0_DIMS[1];
                    const uint X_ = INPUT0_DIMS[2];
                    const uint Y_ = INPUT0_DIMS[3];

                    // Perform the hyperbolic cosine given by: 
                    //    cosh(x) = (e^x + e^-x)/2
                    for (int ii = 0; ii < T_*N_*X_*Y_; ii++) 
                    {
                        output[ii] = (exp(input0[ii]) + exp(-input0[ii]))/2;
                    }
                }
                ```

            * Patch `cosh_kernel.xml` which under `$GEN_OUTPUT/user_ie_extensions/gpu/`

                ```xml
                <CustomLayer name="cosh" type="SimpleGPU" version="1">
                    <Kernel entry="cosh_kernel">
                        <Source filename="cosh_kernel.cl"/>
                        <!-- Parameters description /-->
                    </Kernel>
                    <!-- Buffer descriptions /-->
                    <Buffers>
                        <Tensor arg-index="0" type="input" port-index="0" format="BFYX"/>
                        <Tensor arg-index="1" type="output" port-index="1" format="BFYX"/>
                    </Buffers>
                    <CompilerOptions options="-cl-mad-enable"/>
                    <!-- define the global worksize. The formulas can use the values of the B,F,Y,X dimensions and contain the operators: +,-,/,*,% 
                        (all evaluated in integer arithmetics). Default value: global="B*F*Y*X,1,1"/-->
                    <WorkSizes global="B,F"/>
                </CustomLayer>
                ```
    

        * Construct a new operation
            * Create `Cosh_kernel.cl` which under `$GEN_OUTPUT/user_ie_extensions/gpu/`

                ```cpp
                #pragma OPENCL EXTENSION cl_khr_fp16 : enable

                __kernel void Cosh(const __global INPUT0_TYPE* input0, __global OUTPUT0_TYPE* output)
                {
                    // global index definition set in the XML configuration file
                    const uint idx = get_global_id(0);
                    const uint idy = get_global_id(1);
                    const uint idbf = get_global_id(2);
                    const uint feature = idbf%OUTPUT0_DIMS[1];
                    const uint batch = idbf/OUTPUT0_DIMS[1];

                    const uint in_id = batch*INPUT0_PITCHES[0] + feature*INPUT0_PITCHES[1] +
                            idy*INPUT0_PITCHES[2] + idx*INPUT0_PITCHES[3] + INPUT0_OFFSET;
                    const uint out_id = batch*OUTPUT0_PITCHES[0] + feature*OUTPUT0_PITCHES[1] +
                            idy*OUTPUT0_PITCHES[2] + idx*OUTPUT0_PITCHES[3] + OUTPUT0_OFFSET;

                    INPUT0_TYPE value = input0[in_id];
                        output[out_id] = (exp(value) + exp(-value))/2;
                }
                ```

            * Create `Cosh_kernel.xml` which under `$GEN_OUTPUT/user_ie_extensions/gpu/`

                ```xml
                <CustomLayer name="Cosh" type="SimpleGPU" version="1">
                    <Kernel entry="Cosh">
                        <Source filename="Cosh_kernel.cl"/>
                    </Kernel>
                    <Buffers>
                        <Tensor arg-index="0" type="input" port-index="0" format="BFYX"/>
                        <Tensor arg-index="1" type="output" port-index="0" format="BFYX"/>
                    </Buffers>
                    <CompilerOptions options="-cl-mad-enable"/>
                    <WorkSizes global="X,Y,B*F"/>
                </CustomLayer>
                ```

    3. Execute `infer.py` with parameter `–m`, `–d` and `–c`.

        * Run:

            * Construct a new operation

                ```sh
                python infer.py \
                    -m $USING_EXISTING_LAYER/cosh.xml \
                    -c $GEN_OUTPUT/user_ie_extensions/gpu//cosh_kernel.xml \
                    -d GPU 
                ```
        
            * Generate by `extgen.py`

                ```sh
                python infer.py \
                    -m $AS_NEW/cosh.xml \
                    -c $GEN_OUTPUT/user_ie_extensions/gpu//Cosh_kernel.xml \
                    -d GPU
                ```

### Reference
* [smart-video-workshop](https://github.com/intel-iot-devkit/smart-video-workshop)
* [OpenVINO-Custom-Layers](https://github.com/david-drew/OpenVINO-Custom-Layers)
