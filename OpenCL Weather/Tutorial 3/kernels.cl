//AVERAGE
//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
__kernel void reduce_add_4(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id]; // 

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i]; // if A>B A=B. if A<B A=B for min max//

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}

//MIN TEMP
__kernel void Min(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id]; // 

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if(scratch[lid] > scratch[lid + i])
			{
				scratch[lid] = scratch[lid + i];
			}

		barrier(CLK_LOCAL_MEM_FENCE); //wait for all local threads to finish copying from global to local memory
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_min(&B[0],scratch[lid]);
	}
}

//MAX TEMP
__kernel void Max(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id]; // 

	barrier(CLK_LOCAL_MEM_FENCE); //wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if(scratch[lid] < scratch[lid + i])
			{
				scratch[lid] = scratch[lid + i];
			}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_max(&B[0],scratch[lid]);
	}
}

// HISTOGRAM 
//a very simple histogram implementation
__kernel void hist_simple(__global const int* A, __global int* H, int minVal, int maxVal, int nrBins) { 
	int id = get_global_id(0);

	int temp = A[id];

	if (temp < -1000) return;
	
	int range = maxVal - minVal; // Define the range of the dataset
	int offset = temp - minVal; // The current temperature index subtracted by the minimum defines the offset within the data 
	int iDDX = ((offset*nrBins)/range);

	// Serial increment highly inefficient but it works :(
	if(iDDX > 0)
		atomic_inc(&H[iDDX -1]);
	else
		atomic_inc(&H[iDDX]);
	
}
