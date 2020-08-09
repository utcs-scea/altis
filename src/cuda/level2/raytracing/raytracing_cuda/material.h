////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\material.h
//
// summary:	Declares the material class
// 
// origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MATERIALH
#define MATERIALH

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Information about the hit. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

struct hit_record;

#include "ray.h"
#include "hitable.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Schlicks. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="cosine"> 	The cosine. </param>
/// <param name="ref_idx">	Zero-based index of the reference. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Refracts. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="v">		 	A vec3 to process. </param>
/// <param name="n">		 	A vec3 to process. </param>
/// <param name="ni_over_nt">	The ni over NT. </param>
/// <param name="refracted"> 	[in,out] The refracted. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines randvec 3. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Random in unit sphere. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="local_rand_state">	[in,out] If non-null, state of the local random. </param>
///
/// <returns>	A vec3. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reflects. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="v">	A vec3 to process. </param>
/// <param name="n">	A vec3 to process. </param>
///
/// <returns>	A vec3. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2.0f*dot(v,n)*n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A material. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

class material  {
    public:

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Scatters. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="r_in">			   	The in. </param>
        /// <param name="rec">			   	The record. </param>
        /// <param name="attenuation">	   	[in,out] The attenuation. </param>
        /// <param name="scattered">	   	[in,out] The scattered. </param>
        /// <param name="local_rand_state">	[in,out] If non-null, state of the local random. </param>
        ///
        /// <returns>	True if it succeeds, false if it fails. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A lambertian. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

class lambertian : public material {
    public:

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="a">	A vec3 to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        __device__ lambertian(const vec3& a) : albedo(a) {}

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Scatters. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="r_in">			   	The in. </param>
        /// <param name="rec">			   	The record. </param>
        /// <param name="attenuation">	   	[in,out] The attenuation. </param>
        /// <param name="scattered">	   	[in,out] The scattered. </param>
        /// <param name="local_rand_state">	[in,out] If non-null, state of the local random. </param>
        ///
        /// <returns>	True if it succeeds, false if it fails. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
             vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
             scattered = ray(rec.p, target-rec.p);
             attenuation = albedo;
             return true;
        }

        /// <summary>	The albedo. </summary>
        vec3 albedo;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A metal. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

class metal : public material {
    public:

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="a">	A vec3 to process. </param>
        /// <param name="f">	A float to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        __device__ metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Scatters. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="r_in">			   	The in. </param>
        /// <param name="rec">			   	The record. </param>
        /// <param name="attenuation">	   	[in,out] The attenuation. </param>
        /// <param name="scattered">	   	[in,out] The scattered. </param>
        /// <param name="local_rand_state">	[in,out] If non-null, state of the local random. </param>
        ///
        /// <returns>	True if it succeeds, false if it fails. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }
        /// <summary>	The albedo. </summary>
        vec3 albedo;
        /// <summary>	The fuzz. </summary>
        float fuzz;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A dielectric. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

class dielectric : public material {
public:

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	Constructor. </summary>
    ///
    /// <remarks>	Ed, 5/20/2020. </remarks>
    ///
    /// <param name="ri">	The ri. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    __device__ dielectric(float ri) : ref_idx(ri) {}

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	Scatters. </summary>
    ///
    /// <remarks>	Ed, 5/20/2020. </remarks>
    ///
    /// <param name="r_in">			   	The in. </param>
    /// <param name="rec">			   	The record. </param>
    /// <param name="attenuation">	   	[in,out] The attenuation. </param>
    /// <param name="scattered">	   	[in,out] The scattered. </param>
    /// <param name="local_rand_state">	[in,out] If non-null, state of the local random. </param>
    ///
    /// <returns>	True if it succeeds, false if it fails. </returns>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    __device__ virtual bool scatter(const ray& r_in,
                         const hit_record& rec,
                         vec3& attenuation,
                         ray& scattered,
                         curandState *local_rand_state) const  {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx*ref_idx*(1-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    /// <summary>	Zero-based index of the reference. </summary>
    float ref_idx;
};
#endif
