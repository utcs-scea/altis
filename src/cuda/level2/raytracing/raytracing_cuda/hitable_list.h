////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\raytracing\hitable_list.h
//
// summary:	Declares the hitable list class
// 
// origin: Ray tracing(https://github.com/ssangx/raytracing.cuda)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	List of hitables. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

class hitable_list: public hitable  {
    public:

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Default constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        __device__ hitable_list() {}

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="l">	[in,out] If non-null, a hitable to process. </param>
        /// <param name="n">	An int to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        __device__ hitable_list(hitable **l, int n) {list = l; list_size = n; }

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
        /// <summary>	The list. </summary>
        hitable **list;
        /// <summary>	Size of the list. </summary>
        int list_size;
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

__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
}

#endif
