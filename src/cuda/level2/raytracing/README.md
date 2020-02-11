Ray Tracing in One Weekend in CUDA
==================================

This is yet another _Ray Tracing in One Weekend_ clone, but this time using CUDA instead of C++.

By Roger Allen
May, 2018

See the [Master Branch](https://github.com/rogerallen/raytracinginoneweekend) for more information.

Chapter 1
---------

This introduces the basic kernel launch mechanism & host/device memory management.  We are just creating an image on the GPU device and cudaMallocmanaged allows for sharing the framebuffer and automatically copying that buffer to & from the device.

I also added a timer to see how long it takes the GPU to do rendering.

Chapter 2
---------

Because CUDA is compatible with C++ and the vec3.h class will be used on both GPU & CPU, we add `__host__` `__device__` as a prefix to all methods.

Chapter 3
---------

Since the ray class is only used on the GPU, we will just add `__device__` as a prefix to all methods.

The color function just needs a `__device__` added since this is called from the render kernel.

Note, doing a straight translation from the original C++ will mean that any floating-point constants will be doubles and math on the GPU will be forced to be double-precision.  This will hurt our performance unnecessarily.  Special attention to floating point constants must be taken (e.g. 0.5 -> 0.5f).

Use the "profile_metrics" makefile target to count inst_fp_64 and be sure that is 0.

Chapter 4
---------

We only need to add a `__device__` to the hit_sphere() call and use profile_metrics to watch for those floating-point constants.

Chapter 5
---------

Here we have to create our world of spheres on the device and get familiar with how we do memory management for CUDA C++ classes.  Note the cudaMalloc of `d_list` and `d_world` and the `create_world` kernel.

Again, attend to `__device__` and floating-point constants in hitable.h, hitable_list.h and sphere.h.

Chapter 6
---------

In this chapter we need to understand using cuRAND for per-thread random numbers.  See `d_rand_state` and `render_init`.

Note that now using debug flags in compilation makes a big difference in runtime.  Remove those flags for a signficant speedup.

Chapter 7
---------

Matching the C++ code in the color function in main.cu would recurse enough into the color() calls that it was crashing the program by overrunning the stack, so we turn this function into a limited-depth loop instead.  Later code in the book limits to a max depth of 50, so we adapt this a few chapters early on the GPU.

Chapter 8
---------

Just more plumbing for per-thread local random state, mostly.

Chapter 9
---------

Similar to previous modifications.

Chapter 10
----------

Similar to previous modifications.

Chapter 11
----------

Similar to previous modifications.

Chapter 12
----------

And we're done!
