///-------------------------------------------------------------------------------------------------
// file:	genericvector.h
//
// summary:	Declares the genericvector class
///-------------------------------------------------------------------------------------------------

#ifndef __GENERIC_VECTOR_H__
#define __GENERIC_VECTOR_H__

#include <vector>
#include <math.h>

template<int R> 
struct pt {
    float m_v[R];
    pt<R>(float * p) { 
        for(int i=0;i<R;i++)
            m_v[i]=*p++;
    }
    pt<R>() { set(0.0f); }
    __device__ __host__ void set(float v) { for(int i=0;i<R;i++) m_v[i]=v; }
    __device__ __host__ void operator+=(pt<R> opt) { 
        for(int i=0;i<R;i++) 
            m_v[i]+=opt.m_v[i];
    }
    __device__ __host__ struct pt<R> operator+(pt<R> opt) { 
        pt<R> res; 
        for(int i=0;i<R;i++) 
            res.m_v[i]=m_v[i]+opt.m_v[i];
        return res;
    }
    __device__ __host__ void operator/=(int numerator) {
        if(numerator == 0) {
            set(0.0f); 
        } else {
            for(int i=0;i<R;i++) 
                m_v[i]/=numerator;
        }
    }
    void dump(FILE * fp, int nColLimit=0) {
        nColLimit = (nColLimit==0) ? R : ((nColLimit<R)?nColLimit:R);
        for(int i=0;i<nColLimit;i++) {
            if(i>0) fprintf(fp, ", ");
            fprintf(fp, "%.3f", m_v[i]);
        }
        if(nColLimit < R)
            fprintf(fp, ",...");
        fprintf(fp, "\n");
    }
};

#if 0
struct ptd {
    int m_R;
    float * m_v;
    struct ptd(int _R, float * p) { m_R = _R; m_v=new float[_R]; for(int i=0;i<m_R;i++) m_v[i]=*p++; }
    struct ptd(int _R) { m_R = _R; m_v=new float[_R]; set(0.0f); }
    __device__ __host__ void set(float v) { for(int i=0;i<m_R;i++) m_v[i]=v; }
    __device__ __host__ void operator+=(ptd opt) { 
        for(int i=0;i<min(m_R,opt.m_R);i++) 
            m_v[i]+=opt.m_v[i];
    }
    __device__ __host__ struct ptd operator+(ptd opt) { 
        ptd res(min(m_R, opt.m_R)); 
        for(int i=0;i<min(m_R,opt.m_R);i++) 
            res.m_v[i]=m_v[i]+opt.m_v[i];
        return res;
    }
    __device__ __host__ void operator/=(int numerator) {
        if(numerator == 0) {
            set(0.0f); 
        } else {
            for(int i=0;i<m_R;i++) 
                m_v[i]/=numerator;
        }
    }
    void dump(FILE * fp, int nColLimit=0) {
        nColLimit = (nColLimit==0) ? m_R : ((nColLimit<m_R)?nColLimit:m_R);
        for(int i=0;i<nColLimit;i++) {
            if(i>0) fprintf(fp, ", ");
            fprintf(fp, "%.3f", m_v[i]);
        }
        if(nColLimit < m_R)
            fprintf(fp, ",...");
        fprintf(fp, "\n");
    }
};
#endif


#endif
