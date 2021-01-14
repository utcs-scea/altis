////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\raytracing.cu
//
// summary:	Raytracing class
// 
// origin: Raytracing(https://github.com/rogerallen/raytracinginoneweekendincuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cuda_profiler_api.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Matching the C++ code would recurse enough into color() calls that
/// it was blowing up the stack, so we have to turn this into a
/// limited-depth loop instead.  Later code in the book limits to a max
/// depth of 50, so we adapt this a few chapters early on the GPU.. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="r">			   	A ray to process. </param>
/// <param name="world">		   	[in,out] If non-null, the world. </param>
/// <param name="local_rand_state">	[in,out] If non-null, state of the local random. </param>
///
/// <returns>	A vec3. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Random initialize. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="rand_state">	[in,out] If non-null, state of the random. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	initialize rendering. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="max_x">	 	The maximum x coordinate. </param>
/// <param name="max_y">	 	The maximum y coordinate. </param>
/// <param name="rand_state">	[in,out] If non-null, state of the random. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Renders this.  </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="fb">		 	[in,out] If non-null, the fb. </param>
/// <param name="max_x">	 	The maximum x coordinate. </param>
/// <param name="max_y">	 	The maximum y coordinate. </param>
/// <param name="ns">		 	The ns. </param>
/// <param name="cam">		 	[in,out] If non-null, the camera. </param>
/// <param name="world">	 	[in,out] If non-null, the world. </param>
/// <param name="rand_state">	[in,out] If non-null, state of the random. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Random. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define RND (curand_uniform(&local_rand_state))

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Creates a world. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_list">	 	[in,out] If non-null, the list. </param>
/// <param name="d_world">   	[in,out] If non-null, the world. </param>
/// <param name="d_camera">  	[in,out] If non-null, the camera. </param>
/// <param name="nx">		 	The nx. </param>
/// <param name="ny">		 	The ny. </param>
/// <param name="rand_state">	[in,out] If non-null, state of the random. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Free world resources. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_list">  	[in,out] If non-null, the list. </param>
/// <param name="d_world"> 	[in,out] If non-null, the world. </param>
/// <param name="d_camera">	[in,out] If non-null, the camera. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("X", OPT_INT, "1200", "specify image x dimension", '\0');
    op.addOption("Y", OPT_INT, "800", "specify image y dimension", '\0');
    op.addOption("samples", OPT_INT, "10", "specify number of iamge samples", '\0');
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="DB">	[in,out] The database. </param>
/// <param name="op">	[in,out] The options specified. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &DB, OptionParser &op) {
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));

    cudaEvent_t total_start, total_stop;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
    checkCudaErrors(cudaEventCreate(&total_start));
    checkCudaErrors(cudaEventCreate(&total_stop)); 
    checkCudaErrors(cudaEventRecord(total_start, 0));

    // Predefined image resolutions
    int xDim[5] = {400, 1200, 4096, 15360, 20480};
    int yDim[5] = {300, 800, 2160, 8640, 17280};
    int size = op.getOptionInt("size") - 1;
    int nx = xDim[size];
    int ny = yDim[size];
    if (op.getOptionInt("X") != 1200 || op.getOptionInt("Y") != 800) {
        nx = op.getOptionInt("X");
        ny = op.getOptionInt("Y");
    }
    int ns = op.getOptionInt("samples");
    assert(ns > 0);
    int tx = 8;
    int ty = 8;
    int num_passes = op.getOptionInt("passes");

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaMallocManaged((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    } else {
        checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    }

    curandState *d_rand_state2 = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaMallocManaged((void **)&d_rand_state2, num_pixels*sizeof(curandState)));
    } else {
        checkCudaErrors(cudaMalloc((void **)&d_rand_state2, num_pixels*sizeof(curandState)));
    }

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable **d_list;
    int num_hitables = 22*22+1+3;

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaMallocManaged(&d_list, num_hitables*sizeof(hitable *)));
    } else {
        checkCudaErrors(cudaMalloc(&d_list, num_hitables*sizeof(hitable *)));
    }
    
    hitable **d_world;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaMallocManaged(&d_world, num_hitables*sizeof(hitable *)));
    } else {
        checkCudaErrors(cudaMalloc(&d_world, num_hitables*sizeof(hitable *)));
    }

    camera **d_camera;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaMallocManaged(&d_camera, num_hitables*sizeof(hitable *)));
    } else {
        checkCudaErrors(cudaMalloc(&d_camera, num_hitables*sizeof(hitable *)));
    }
    
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // use cudaevent
        char atts[1024];
    sprintf(atts, "img: %d by %d, samples: %d, iter:%d", nx, ny, ns, num_passes);
    int i = 0;
    for (; i < num_passes; i++) {
        checkCudaErrors(cudaEventRecord(start, 0));
        // Render our buffer
        dim3 blocks(nx/tx+1,ny/ty+1);
        dim3 threads(tx,ty);
        render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));
        float t = 0;
        checkCudaErrors(cudaEventElapsedTime(&t, start, stop));
        DB.AddResult("raytracing rendering time", atts, "sec", t * 1.0e-3);
    }
    // std::cerr << "took " << timer_seconds << " seconds.\n";

#if 0
    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
#endif

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    checkCudaErrors(cudaEventRecord(total_stop, 0));
    checkCudaErrors(cudaEventSynchronize(total_stop));
    float total_time = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&total_time, total_start, total_stop));
    DB.AddResult("raytracing total execution time", atts, "sec", total_time * 1.0e-3);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    checkCudaErrors(cudaDeviceReset());
}
