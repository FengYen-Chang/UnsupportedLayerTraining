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
