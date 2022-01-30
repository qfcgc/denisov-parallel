package dev.vlde;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

public class ArrayAddition {
    public static final String PROGRAM_SOURCE = """
            __kernel void addArrays(
                    __global const float *a,
                    __global const float *b,
                               __global float *c) {
                int gid = get_global_id(0);
                c[gid] = a[gid] + b[gid];
            }""";

    public static void main(String[] args) {
        int n = 10_000_000;
        float[] sourceArrayA = new float[n];
        float[] sourceArrayB = new float[n];
        float[] destinationArray = new float[n];
        for (int i = 0; i < n; i++) {
            sourceArrayA[i] = i;
            sourceArrayB[i] = i;
        }
        Pointer sourcePointerA = Pointer.to(sourceArrayA);
        Pointer sourcePointerB = Pointer.to(sourceArrayB);
        Pointer destinationPointer = Pointer.to(destinationArray);

        // Enable exceptions and subsequently omit error checks in this program
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int[] numPlatformsArray = new int[1];
        CL.clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        CL.clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[0];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int[] numDevicesArray = new int[1];
        CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id[] devices = new cl_device_id[numDevices];
        CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_ALL, numDevices, devices, null);
        cl_device_id device = devices[0];

        // Create a context for the selected device
        cl_context context = CL.clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null);


        // Create a command-queue for the selected device
        cl_command_queue commandQueue =
                CL.clCreateCommandQueue(context, device, 0, null);

        // Allocate the memory objects for the input and output data
        cl_mem[] memObjects = new cl_mem[3];
        memObjects[0] = CL.clCreateBuffer(context,
                CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n, sourcePointerA, null);
        memObjects[1] = CL.clCreateBuffer(context,
                CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n, sourcePointerB, null);
        memObjects[2] = CL.clCreateBuffer(context,
                CL.CL_MEM_READ_WRITE,
                Sizeof.cl_float * n, null, null);

        // Create the program from the source code
        cl_program program = CL.clCreateProgramWithSource(context,
                1, new String[]{PROGRAM_SOURCE}, null, null);

        // Build the program
        CL.clBuildProgram(program, 0, null, null, null, null);

        // Create the kernel
        cl_kernel kernel = CL.clCreateKernel(program, "addArrays", null);

        // Set the arguments for the kernel
        CL.clSetKernelArg(kernel, 0,
                Sizeof.cl_mem, Pointer.to(memObjects[0]));
        CL.clSetKernelArg(kernel, 1,
                Sizeof.cl_mem, Pointer.to(memObjects[1]));
        CL.clSetKernelArg(kernel, 2,
                Sizeof.cl_mem, Pointer.to(memObjects[2]));

        // Set the work-item dimensions
        long[] global_work_size = new long[]{n};
        long[] local_work_size = new long[]{1};

        // Execute the kernel
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_size, local_work_size, 0, null, null);

        // Read the output data
        CL.clEnqueueReadBuffer(commandQueue, memObjects[2], CL.CL_TRUE, 0,
                n * Sizeof.cl_float, destinationPointer, 0, null, null);

        // Release kernel, program, and memory objects
        CL.clReleaseMemObject(memObjects[0]);
        CL.clReleaseMemObject(memObjects[1]);
        CL.clReleaseMemObject(memObjects[2]);
        CL.clReleaseKernel(kernel);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);

        for (int i = 0; i < 100; i++) {
            System.out.println(destinationArray[i]);
        }
    }
}
