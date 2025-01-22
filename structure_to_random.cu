#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <cmath>

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif


#define THREADS_PER_BLOCK 512
#define BLOCKS 2048
// Number of uint64_t values per chunk
#define OUTPUT_CHUNK_SIZE 1048576
__managed__ uint64_t seeds_checked = 0;
__managed__ uint64_t tmax = 10000000;
__managed__ uint64_t validCount = 0;
__device__  uint64_t outputBuffer;


// Macro to check CUDA status and exit on failure
#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }


 /*
  * Function: validSeed
  * -------------------
  * Checks if a given seed is a valid random seed for Minecraft.
  *
  * Parameters:
  *   a - The seed to be validated (uint64_t).
  *
  * Returns:
  *   bool - True if the seed is valid, false otherwise.
  *
  * C port of Geosquare's original Java implementation.
  */
__device__ bool validSeed(uint64_t worldseed) {
    // LCG constants
    const int64_t LCG_MULTIPLIER = INT64_C(25214903917);
    const int64_t LCG_ADDEND = INT64_C(11);
    const int64_t LCG_MODULO = INT64_C(1) << 48;

    // Break worldseed into upper/lower halves (signed to emulate signed 64-bit behavior)
    int64_t worldseedUpper32 = static_cast<int64_t>(worldseed >> 32);
    int64_t worldseedLower32 = static_cast<int64_t>(static_cast<int32_t>(worldseed));

    // Calculate the equivalent state in the exact middle of the (theoretical) nextLong() call
    // (This seems to use some lattice magic that no one bothers to explain)
    int64_t term1 = 24667315 * worldseedUpper32 + 18218081 * worldseedLower32 + 67552711;
    int64_t term2 = -4824621 * worldseedUpper32 +  7847617 * worldseedLower32 +  7847617;
    int64_t nextLongMiddleState = 7847617 * (term1 >> 32) - 18218081 * (term2 >> 32); // Technically should be moduloed, but there's no point since all uses are moduloed later

    // Then finish emulating nextLong() and return whether the result matches the original input.
    int64_t nextLongValue = ((nextLongMiddleState % LCG_MODULO) >> 16) << 32;
    nextLongValue += static_cast<int64_t>(static_cast<int32_t>(((LCG_MULTIPLIER * nextLongMiddleState + LCG_ADDEND) % LCG_MODULO) >> 16));
    return static_cast<uint64_t>(nextLongValue) == worldseed;
}

/*
 * Function: processSeeds
 * ----------------------
 * Processes a chunk of seeds to find valid seeds.
 *
 * Parameters:
 *   input - The input buffer containing the seeds to be processed (uint64_t*).
 *   outputBuffer - The output buffer to store the valid seeds (uint64_t*).
 *   outputCount - The number of valid seeds found (size_t*).
 *   s - The starting index of the chunk (uint64_t).
 */
__global__ void processSeeds(uint64_t* input, uint64_t* outputBuffer, size_t* outputCount, uint64_t s) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x + s;
    if (idx >= tmax) return;
    uint64_t base = input[idx];

    atomicAdd(&seeds_checked, 1);

    for (uint64_t upper16 = 0; upper16 < (1ULL << 16); ++upper16) {
        uint64_t candidate = (upper16 << 48) | base;
        if (validSeed(candidate)) {
            // Atomically add the output to the global buffer
            size_t outIdx = atomicAdd(outputCount, 1);
            outputBuffer[outIdx] = candidate;
            atomicAdd(&validCount, 1);
        }
    }
}


/*
 * Function: countLines
 * ---------------------
 * Counts the number of lines in a file.
 * 
 * Parameters:
 * filename - The name of the file to be read (const char*).
 * 
 * Returns:
 * int - The number of lines in the file.
 */
int countLines(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    int lineCount = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), file)) {
        lineCount++;
    }

    fclose(file);
    return lineCount;
}


/*
 * Function: readInputFile
 * ------------------------
 * Reads a file containing a list of seeds.
 *
 * Parameters:
 * filename - The name of the file to be read (const char*).
 * data - The buffer to store the seeds (uint64_t*).
 * numValues - The number of seeds to read (int).
 */
void readInputFile(const char* filename, uint64_t* data, int numValues) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numValues; ++i) {
        if (fscanf(file, "%llu", &data[i]) != 1) {
            fprintf(stderr, "Error reading value at line %d\n", i);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}

/*
 * Function: writeChunkToFile
 * --------------------------
 * Writes a chunk of data to a file.
 *
 * Parameters:
 * file - The file to write to (FILE*).
 * buffer - The buffer containing the data to write (uint64_t*).
 * chunkSize - The number of elements in the buffer (size_t).
 */
void writeChunkToFile(FILE* file, uint64_t* buffer, size_t chunkSize) {
    for (size_t i = 0; i < chunkSize; ++i) {
        fprintf(file, "%lld\n", buffer[i]); // Use %lld for int64_t 
    } 
}

/*
 * Function: processFile
 * ---------------------
 * Processes a file containing a list of seeds.
 *
 * Parameters:
 * inputFilePath - The path to the input file (const char*).
 * outputFilePath - The path to the output file (const char*).
 * deviceBuffer - The device buffer to store the output (uint64_t*).
 * hostBuffer - The host buffer to store the output (uint64_t*).
 * 
 * This function reads seeds from the input file, processes them using CUDA to find valid seeds,
 * and writes the valid seeds to the output file. It handles memory allocation and data transfer
 * between the host and device, and ensures synchronization between CUDA kernel executions.
 */
void processFile(const char* inputFilePath, const char* outputFilePath, uint64_t* deviceBuffer, uint64_t* hostBuffer) {
    // Open the output file for writing
    FILE* file = fopen(outputFilePath, "w");
    if (file == NULL) {
        perror("fopen");
        return;
    }
    // Allocate memory on the device for the output count
    size_t* deviceOutputCount;
    cudaMalloc((void**)&deviceOutputCount, sizeof(size_t));

    // Count the number of seeds in the input file
    const char* filename = inputFilePath;
    int numValues = countLines(filename);
    size_t inputSize = numValues * sizeof(uint64_t);
    tmax = numValues;

    // Allocate memory on the host and read the input file
    uint64_t* h_input = (uint64_t*)malloc(inputSize);
    readInputFile(filename, h_input, numValues);

    // Allocate memory on the device and copy the input data to the device
    uint64_t* d_input;
    cudaMalloc(&d_input, inputSize);
    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);

    // Calculate the number of chunks to process
    const int max = (int)(ceil((double)(numValues) / (THREADS_PER_BLOCK * BLOCKS)));
    for (uint64_t s = 0; s < max; s++) {
        // Reset the output count for this chunk
        cudaMemset(deviceOutputCount, 0, sizeof(size_t));
        // Launch the CUDA kernel to process the seeds
        processSeeds<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input, deviceBuffer, deviceOutputCount, s * THREADS_PER_BLOCK * BLOCKS);
        cudaDeviceSynchronize();
        // Copy the output count and valid seeds from the device to the host
        size_t hostOutputCount;
        cudaMemcpy(&hostOutputCount, deviceOutputCount, sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostBuffer, deviceBuffer, hostOutputCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        // Write the valid seeds to the output file
        writeChunkToFile(file, hostBuffer, hostOutputCount);
    }
    // Close the output file and free allocated memory
    fclose(file);
    free(h_input);
    cudaFree(d_input);
    cudaFree(deviceOutputCount);
}

int main(void) {
    // Create CUDA events to measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    uint64_t* deviceBuffer;
    cudaMalloc((void**)&deviceBuffer, 2 * OUTPUT_CHUNK_SIZE * sizeof(uint64_t));

    uint64_t* hostBuffer = (uint64_t*)malloc(2 * OUTPUT_CHUNK_SIZE * sizeof(uint64_t));

    const char* inputDir = "path/to/structure/seeds";
    const char* outputDir = "path/to/output";

    char inputFilePath[256], outputFilePath[256];
    int fileCount = 1;

	// Check if on Windows or linux for different directory reading
#ifdef _WIN32
    WIN32_FIND_DATA findFileData;
    HANDLE hFind;
    #ifdef UNICODE
        WCHAR searchPattern[256];
        swprintf(searchPattern, sizeof(searchPattern)/sizeof(*searchPattern), reinterpret_cast<wchar_t *>("%s/*.txt"), inputDir);
        hFind = FindFirstFileW(searchPattern, &findFileData);
    #else
        char searchPattern[256];
        snprintf(searchPattern, sizeof(searchPattern)/sizeof(*searchPattern), "%s/*.txt", inputDir);
        hFind = FindFirstFileA(searchPattern, &findFileData);
    #endif

    if (hFind == INVALID_HANDLE_VALUE) {
        printf("FindFirstFile failed (%d)\n", GetLastError());
        return -1;
    }
    // Loop through each file in the directory and process
    do {
        snprintf(inputFilePath, sizeof(inputFilePath), "%s/%s", inputDir, findFileData.cFileName);
        snprintf(outputFilePath, sizeof(outputFilePath), "%s/output_%d.txt", outputDir, fileCount++);
        printf("%s\n", findFileData.cFileName);

        processFile(inputFilePath, outputFilePath, deviceBuffer, hostBuffer);


    } while (FindNextFile(hFind, &findFileData) != 0);
    FindClose(hFind);

#else
    DIR* dir;
    struct dirent* ent;

    if ((dir = opendir(inputDir)) != NULL) {
        // Loop through each file in the directory and process
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_type == DT_REG) { // Check if it's a regular file
                snprintf(inputFilePath, sizeof(inputFilePath), "%s/%s", inputDir, ent->d_name);
                snprintf(outputFilePath, sizeof(outputFilePath), "%s/output_%d.txt", outputDir, fileCount++);
                printf("%s\n", ent->d_name);
                processFile(inputFilePath, outputFilePath, deviceBuffer, hostBuffer);
            }
        }
        closedir(dir);
	} else {
		perror("opendir");
		return EXIT_FAILURE;
	}
#endif
  printf("checked: %llu\n", seeds_checked);
  printf("valid: %llu\n", validCount);

  // Record the stop event
  cudaEventRecord(stop, 0);

  // Wait for the stop event to complete
  cudaEventSynchronize(stop);
  // Calculate the elapsed time between the two events
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // Print the runtime
  printf("Kernel execution time: %f s\n", elapsedTime / 1000.0f);

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(hostBuffer);
  cudaFree(deviceBuffer);

  return 0;
}