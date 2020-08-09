////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\camera.h
//
// summary:	Declares the camera class
// 
// origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Random in unit disk. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="local_rand_state">	[in,out] If non-null, state of the local random. </param>
///
/// <returns>	A vec3. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0) - vec3(1,1,0);
    } while (dot(p,p) >= 1.0f);
    return p;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A camera. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

class camera {
public:

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	Constructor. </summary>
    ///
    /// <remarks>	Ed, 5/20/2020. </remarks>
    ///
    /// <param name="lookfrom">  	The lookfrom. </param>
    /// <param name="lookat">	 	The lookat. </param>
    /// <param name="vup">		 	The vup. </param>
    /// <param name="vfov">		 	The vfov. </param>
    /// <param name="aspect">	 	The aspect. </param>
    /// <param name="aperture">  	The aperture. </param>
    /// <param name="focus_dist">	The focus distance. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov*((float)M_PI)/180.0f;
        float half_height = tan(theta/2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
        horizontal = 2.0f*half_width*focus_dist*u;
        vertical = 2.0f*half_height*focus_dist*v;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	Gets a ray. </summary>
    ///
    /// <remarks>	Ed, 5/20/2020. </remarks>
    ///
    /// <param name="s">			   	A float to process. </param>
    /// <param name="t">			   	A float to process. </param>
    /// <param name="local_rand_state">	[in,out] If non-null, state of the local random. </param>
    ///
    /// <returns>	The ray. </returns>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    __device__ ray get_ray(float s, float t, curandState *local_rand_state) {
        vec3 rd = lens_radius*random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
    }

    /// <summary>	The origin. </summary>
    vec3 origin;
    /// <summary>	The lower left corner. </summary>
    vec3 lower_left_corner;
    /// <summary>	The horizontal. </summary>
    vec3 horizontal;
    /// <summary>	The vertical. </summary>
    vec3 vertical;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	Gets the w. </summary>
    ///
    /// <value>	The w. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    vec3 u, v, w;
    /// <summary>	The lens radius. </summary>
    float lens_radius;
};

#endif
