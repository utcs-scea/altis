////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\sphere.h
//
// summary:	Declares the sphere class
// 
// origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A sphere. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

class sphere: public hitable  {
    public:

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Default constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        __device__ sphere() {}

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="cen">	The cen. </param>
        /// <param name="r">  	A float to process. </param>
        /// <param name="m">  	[in,out] If non-null, a material to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m)  {};

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Hits. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="r">   	A ray to process. </param>
        /// <param name="tmin">	The tmin. </param>
        /// <param name="tmax">	The tmax. </param>
        /// <param name="rec"> 	[in,out] The record. </param>
        ///
        /// <returns>	True if it succeeds, false if it fails. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        /// <summary>	The center. </summary>
        vec3 center;
        /// <summary>	The radius. </summary>
        float radius;
        /// <summary>	The matrix pointer. </summary>
        material *mat_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Hits. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="r">		A ray to process. </param>
/// <param name="t_min">	The minimum. </param>
/// <param name="t_max">	The maximum. </param>
/// <param name="rec">  	[in,out] The record. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}


#endif
