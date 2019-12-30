/*
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

/* ===============================================================================
 * Generated file for Inference Engine extension for GPU plugin
 *
 * IMPLEMENT YOUR KERNEL HERE.
 *
 * Refer to the section "Inference Engine Kernels Extensibility" in 
 * the OpenVINO Inference Engine Developer Guide 
 * ===============================================================================*/



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