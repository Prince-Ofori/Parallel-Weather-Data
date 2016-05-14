#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include "Utils.h"
#include <iomanip>
using std::vector;
using std::endl; 
using std::cout;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "kernels.cl");
		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		//Part 4 - memory allocation
		//host - input

		typedef int mytype;
		vector<mytype> A; // Input vector
		string loc;
		int year;
		int month;
		int day;
		int time;
		float temp;
		int nrBins; 

		ifstream dataFile;
		//Read Datafiles 
		dataFile.open("temp_lincolnshire.txt");
		cout << "\nAnalyising data...\n\n\t\tData Analytics" << endl;
		// While not at end of file parse data 
		while (!dataFile.eof())
		{
			dataFile >> loc >> year >> month >> day >> time >> temp;
			A.push_back((int)(temp * 10.0f));
		}
		dataFile.close();
		// Text wrapper
		cout << "___________________________________________" << endl;
		cout << "Enter the desired number of bins within the output histogram:\t";
		cin >> nrBins;
		int sizeOfA = (int)A.size();
		vector<mytype> AMin = A;
		vector<mytype> AMax = A;
		vector<mytype> AAvg = A;

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient

		size_t local_size = 1024; // See tutorial 1
		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			// Create extra vectors with neutral elements
			// Average
			vector<int> A_ext(local_size - padding_size, 0);
			AAvg.insert(AAvg.end(), A_ext.begin(), A_ext.end());
			// Min 
			vector<int> A_ext1(local_size - padding_size, INT_MIN);
			AMax.insert(AMax.end(), A_ext1.begin(), A_ext1.end());
			// Max 
			vector<int> A_ext2(local_size - padding_size, INT_MAX);
			AMin.insert(AMin.end(), A_ext2.begin(), A_ext2.end());
		}

		size_t vector_elements = AAvg.size();// Number of input elements 
		size_t vector_size = AAvg.size()*sizeof(mytype);
		size_t nr_groups = vector_elements / local_size;

		//host - output
		vector<mytype> B(vector_elements); // Min/Max/Avg output vector
		size_t output_size = B.size()*sizeof(mytype);//size in bytes

		vector<mytype> H(nrBins); // Define historgram output vector
		size_t output_Hist = H.size()*sizeof(mytype);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_H(context, CL_MEM_READ_WRITE, output_Hist);
		//Part 5 - device operations
		//5.1 copy array A to and initialise other arrays on device memory

		// MINIMUM
		// Write commands to buffer object from host memory and fill output buffer B with output pattern size
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &AMin[0]);
		queue.enqueueFillBuffer(buffer_B, INT_MAX, 0, output_size);


		cl::Kernel kernel_1 = cl::Kernel(program, "Min");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size), NULL);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]); // Read buffer and clear output buffer with 0's 
		cout << "Minimum temperature within the dataset:\t" << (float)B[0] / 10.0f << endl;
		int minVal = B[0];  // Store minimum value for later manipulation (HISTOGRAM) 


		// MAXIMUM
		// Write commands to buffer object from host memory and fill output buffer B with output pattern size
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &AMax[0]);
		queue.enqueueFillBuffer(buffer_B, INT_MIN, 0, output_size);

		kernel_1 = cl::Kernel(program, "Max");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]); // Clear output buffer

		cout << "Highest temperature witin the dataset:\t" << (float)B[0] / 10.0f << endl;
		int maxVal = B[0]; // Store maximum value for later manipulation 

		//AVERAGE 
		// Write commands to buffer object from host memory and fill output buffer B with output pattern size
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &AAvg[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

		// Pass kernel arguments 
		kernel_1 = cl::Kernel(program, "reduce_add_4");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);// Clear output buffer

		// Output average temperature to console and display within one decimal place
		cout << "Average temperature within the dataset:\t" << setprecision(2) << (float)B[0] / (float)sizeOfA / 10.0f << std::endl;

		// HISTOGRAM 
		// Write commands to buffer object from host memory and fill Hist with output size pattern
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &AMax[0]);
		queue.enqueueFillBuffer(buffer_H, 0, 0, output_Hist); 

		// Pass kernel arguments
		kernel_1 = cl::Kernel(program, "hist_simple");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_H);
		kernel_1.setArg(2, minVal);
		kernel_1.setArg(3, maxVal);
		kernel_1.setArg(4, nrBins);
		
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size)); // Execute code on device
		queue.enqueueReadBuffer(buffer_H, CL_TRUE, 0, output_Hist, &H[0]); // Read buffer and clear output buffer with 0's 
		// Output Histogram to console
		cout << H << endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	system("pause");
	return 0;
}