///-------------------------------------------------------------------------------------------------
// file:	testsuitedecl.h
//
// summary:	macros for declaring test suite elements
///-------------------------------------------------------------------------------------------------

#pragma once

/*
#define buildfnnamebase(name) kmeans_##name
#define initfnnamebase(name) kmeans_init_##name
#define buildfnnamebaserank(name, rnk) buildfnnamebase(name)_r##rnk
#define initfnnamebaserank(name, rnk) initfnnamebase(name)_r##rnk
#define buildfnname(name, rnk, cent) buildfnnamebaserank(name,rnk)_c##cent
#define buildbncfnname(name, rnk, cent) initfnnamebaserank(name, rnk)_c##cent
*/

#define buildfnname(name, rank, cent) kmeans_ ## name ## _r ## rank ## _c ##cent
#define buildbncfnname(name, rank, cent) kmeans_init_ ## name ## _r ## rank ## _c ##cent

#define declare_testcase_hdr(name, rnk, cent, cman, accman, rowmaj)  \
double                                                               \
buildfnname(name, rnk, cent)(                                        \
    ResultDatabase &DB,                                              \
	const int nSteps,                                                \
	void * lpvPoints,                                                \
	void * lpvCenters,                                               \
	const int nPoints,                                               \
	const int nCenters,                                              \
	bool bVerify,                                                    \
	bool bVerbose                                                    \
	);                                                               \

#define declare_bnc_header(name, rnk, cent, cman, accman, rowmaj)    \
void                                                                 \
buildbncfnname(name, rnk, cent)(                                     \
    ResultDatabase &DB,                                              \
	char * lpszInputFile,                                            \
	LPFNKMEANS lpfn,                                                 \
    int nSteps,                                                      \
    int nSeed,                                                       \
    bool bVerify,                                                    \
    bool bVerbose                                                    \
	);                                                               \


#define declare_testcase(name, rnk, cent, cman, accman, rowmaj)  \
double                                                           \
buildfnname(name, rnk, cent)(                                    \
    ResultDatabase &DB,                                          \
	const int nSteps,                                            \
	void * lpvPoints,                                            \
	void * lpvCenters,                                           \
	const int nPoints,                                           \
	const int nCenters,                                          \
	bool bVerify,                                                \
	bool bVerbose                                                \
	)                                                            \
{                                                                \
                                                                 \
    return kmeansraw<rnk,                                        \
                     cent,                                       \
                     cman<rnk, cent>,                            \
                     accman<rnk, cent, rowmaj>,                  \
                     rowmaj>::benchmark(DB,                      \
                                       nSteps,                   \
                                       lpvPoints,                \
                                       lpvCenters,               \
                                       nPoints,                  \
                                       nCenters,                 \
                                       bVerify,                  \
                                       bVerbose);                \
}                           

#define declare_bnc_fn(name, rnk, cent, cman, accman, rowmaj) \
void                                                          \
buildbncfnname(name, rnk, cent)(                              \
    ResultDatabase &DB,                                       \
	char * lpszInputFile,                                     \
	LPFNKMEANS lpfn,                                          \
    int nSteps,                                               \
    int nSeed,                                                \
    bool bVerify,                                             \
    bool bVerbose                                             \
	)                                                         \
{                                                             \
    kmeansraw<rnk,                                            \
              cent,                                           \
              cman<rnk, cent>,                                \
              accman<rnk, cent, rowmaj>,                      \
              rowmaj>::bncmain(DB,                            \
                               lpszInputFile,                 \
                               lpfn,                          \
                               nSteps,                        \
                               nSeed,                         \
                               bVerify,                       \
                               bVerbose);                     \
}                                                             \
                                       
#define declare_testsuite(rank, cent)                                                                     \
declare_testcase(raw, rank, cent, centersmanagerGM, accumulatorGM, true)                                  \
declare_testcase(rawshr, rank, cent, centersmanagerGM, accumulatorSM, true)                               \
declare_testcase(constmem, rank, cent, centersmanagerRO, accumulatorGM, true)                             \
declare_testcase(constmem_memsetreset, rank, cent, centersmanagerRO, accumulatorGMMS, true)               \
declare_testcase(constmem_memsetreset_shr, rank, cent, centersmanagerRO, accumulatorSM, true)             \
declare_testcase(constmem_memsetreset_shr_map, rank, cent, centersmanagerRO, accumulatorSMMAP, true)      \
declare_testcase(cm, rank, cent, centersmanagerGM, accumulatorGM, false)                                  \
declare_testcase(cmshr, rank, cent, centersmanagerGM, accumulatorSM, false)                               \
declare_testcase(cmconstmem, rank, cent, centersmanagerRO, accumulatorGM, false)                          \
declare_testcase(cmconstmem_shr, rank, cent, centersmanagerRO, accumulatorSM, false)                      \
declare_testcase(cmconstmem_shr_map, rank, cent, centersmanagerRO, accumulatorSMMAP, false)               \
declare_bnc_fn(raw, rank, cent, centersmanagerGM, accumulatorGM, true)                                    \
declare_bnc_fn(rawshr, rank, cent, centersmanagerGM, accumulatorSM, true)                                 \
declare_bnc_fn(constmem, rank, cent, centersmanagerRO, accumulatorGM, true)                               \
declare_bnc_fn(constmem_memsetreset, rank, cent, centersmanagerRO, accumulatorGMMS, true)                 \
declare_bnc_fn(constmem_memsetreset_shr, rank, cent, centersmanagerRO, accumulatorSM, true)               \
declare_bnc_fn(constmem_memsetreset_shr_map, rank, cent, centersmanagerRO, accumulatorSMMAP, true)        \
declare_bnc_fn(cm, rank, cent, centersmanagerGM, accumulatorGM, false)                                    \
declare_bnc_fn(cmshr, rank, cent, centersmanagerGM, accumulatorSM, false)                                 \
declare_bnc_fn(cmconstmem, rank, cent, centersmanagerRO, accumulatorGM, false)                            \
declare_bnc_fn(cmconstmem_shr, rank, cent, centersmanagerRO, accumulatorSM, false)                        \
declare_bnc_fn(cmconstmem_shr_map, rank, cent, centersmanagerRO, accumulatorSMMAP, false)                 \

#define declare_testsuite_lg(rank, cent)                                                                  \
declare_testcase(raw, rank, cent, centersmanagerGM, accumulatorGM, true)                                  \
declare_testcase(rawshr, rank, cent, centersmanagerGM, accumulatorGM, true)                               \
declare_testcase(constmem, rank, cent, centersmanagerRO, accumulatorGM, true)                             \
declare_testcase(constmem_memsetreset, rank, cent, centersmanagerRO, accumulatorGM, true)                 \
declare_testcase(constmem_memsetreset_shr, rank, cent, centersmanagerRO, accumulatorGM, true)             \
declare_testcase(constmem_memsetreset_shr_map, rank, cent, centersmanagerRO, accumulatorGM, true)         \
declare_testcase(cm, rank, cent, centersmanagerGM, accumulatorGM, false)                                  \
declare_testcase(cmshr, rank, cent, centersmanagerGM, accumulatorGM, false)                               \
declare_testcase(cmconstmem, rank, cent, centersmanagerRO, accumulatorGM, false)                          \
declare_testcase(cmconstmem_shr, rank, cent, centersmanagerRO, accumulatorGM, false)                      \
declare_testcase(cmconstmem_shr_map, rank, cent, centersmanagerRO, accumulatorGM, false)                  \
declare_bnc_fn(raw, rank, cent, centersmanagerGM, accumulatorGM, true)                                 \
declare_bnc_fn(rawshr, rank, cent, centersmanagerGM, accumulatorGM, true)                                 \
declare_bnc_fn(constmem, rank, cent, centersmanagerRO, accumulatorGM, true)                               \
declare_bnc_fn(constmem_memsetreset, rank, cent, centersmanagerRO, accumulatorGM, true)                   \
declare_bnc_fn(constmem_memsetreset_shr, rank, cent, centersmanagerRO, accumulatorGM, true)               \
declare_bnc_fn(constmem_memsetreset_shr_map, rank, cent, centersmanagerRO, accumulatorGM, true)           \
declare_bnc_fn(cm, rank, cent, centersmanagerGM, accumulatorGM, false)                                    \
declare_bnc_fn(cmshr, rank, cent, centersmanagerGM, accumulatorGM, false)                                 \
declare_bnc_fn(cmconstmem, rank, cent, centersmanagerRO, accumulatorGM, false)                            \
declare_bnc_fn(cmconstmem_shr, rank, cent, centersmanagerRO, accumulatorGM, false)                        \
declare_bnc_fn(cmconstmem_shr_map, rank, cent, centersmanagerRO, accumulatorGM, false)                    \

#define declare_testsuite_hdrs(rank, cent)                                                                    \
declare_testcase_hdr(raw, rank, cent, centersmanagerGM, accumulatorGM, true)                                  \
declare_testcase_hdr(rawshr, rank, cent, centersmanagerGM, accumulatorSM, true)                               \
declare_testcase_hdr(constmem, rank, cent, centersmanagerRO, accumulatorGM, true)                             \
declare_testcase_hdr(constmem_memsetreset, rank, cent, centersmanagerRO, accumulatorGMMS, true)               \
declare_testcase_hdr(constmem_memsetreset_shr, rank, cent, centersmanagerRO, accumulatorSM, true)             \
declare_testcase_hdr(constmem_memsetreset_shr_map, rank, cent, centersmanagerRO, accumulatorSMMAP, true)      \
declare_testcase_hdr(cm, rank, cent, centersmanagerGM, accumulatorGM, false)                                  \
declare_testcase_hdr(cmshr, rank, cent, centersmanagerGM, accumulatorSMMAP, false)                            \
declare_testcase_hdr(cmconstmem, rank, cent, centersmanagerRO, accumulatorGM, false)                          \
declare_testcase_hdr(cmconstmem_shr, rank, cent, centersmanagerRO, accumulatorSM, false)                      \
declare_testcase_hdr(cmconstmem_shr_map, rank, cent, centersmanagerRO, accumulatorSMMAP, false)               \
declare_bnc_header(raw, rank, cent, centersmanagerGM, accumulatorGM, true)                                    \
declare_bnc_header(rawshr, rank, cent, centersmanagerGM, accumulatorSM, true)                                 \
declare_bnc_header(constmem, rank, cent, centersmanagerRO, accumulatorGM, true)                               \
declare_bnc_header(constmem_memsetreset, rank, cent, centersmanagerRO, accumulatorGMMS, true)                 \
declare_bnc_header(constmem_memsetreset_shr, rank, cent, centersmanagerRO, accumulatorSM, true)               \
declare_bnc_header(constmem_memsetreset_shr_map, rank, cent, centersmanagerRO, accumulatorSMMAP, true)        \
declare_bnc_header(cm, rank, cent, centersmanagerGM, accumulatorGM, false)                                    \
declare_bnc_header(cmshr, rank, cent, centersmanagerGM, accumulatorSMMAP, false)                              \
declare_bnc_header(cmconstmem, rank, cent, centersmanagerRO, accumulatorGM, false)                            \
declare_bnc_header(cmconstmem_shr, rank, cent, centersmanagerRO, accumulatorSM, false)                        \
declare_bnc_header(cmconstmem_shr_map, rank, cent, centersmanagerRO, accumulatorSMMAP, false)                 \

#define declare_testsuite_hdrs_lg(rank, cent)                                                                 \
declare_testcase_hdr(raw, rank, cent, centersmanagerGM, accumulatorGM, true)                                  \
declare_testcase_hdr(rawshr, rank, cent, centersmanagerGM, accumulatorGM, true)                               \
declare_testcase_hdr(constmem, rank, cent, centersmanagerRO, accumulatorGM, true)                             \
declare_testcase_hdr(constmem_memsetreset, rank, cent, centersmanagerRO, accumulatorGM, true)                 \
declare_testcase_hdr(constmem_memsetreset_shr, rank, cent, centersmanagerRO, accumulatorSM, true)             \
declare_testcase_hdr(constmem_memsetreset_shr_map, rank, cent, centersmanagerRO, accumulatorGM, true)         \
declare_testcase_hdr(cm, rank, cent, centersmanagerGM, accumulatorGM, false)                                  \
declare_testcase_hdr(cmshr, rank, cent, centersmanagerGM, accumulatorGM, false)                               \
declare_testcase_hdr(cmconstmem, rank, cent, centersmanagerRO, accumulatorGM, false)                          \
declare_testcase_hdr(cmconstmem_shr, rank, cent, centersmanagerRO, accumulatorGM, false)                      \
declare_testcase_hdr(cmconstmem_shr_map, rank, cent, centersmanagerRO, accumulatorGM, false)               \
declare_bnc_header(raw, rank, cent, centersmanagerGM, accumulatorGM, true)                                    \
declare_bnc_header(rawshr, rank, cent, centersmanagerGM, accumulatorGM, true)                                 \
declare_bnc_header(constmem, rank, cent, centersmanagerRO, accumulatorGM, true)                               \
declare_bnc_header(constmem_memsetreset, rank, cent, centersmanagerRO, accumulatorGM, true)                 \
declare_bnc_header(constmem_memsetreset_shr, rank, cent, centersmanagerRO, accumulatorGM, true)               \
declare_bnc_header(constmem_memsetreset_shr_map, rank, cent, centersmanagerRO, accumulatorGM, true)        \
declare_bnc_header(cm, rank, cent, centersmanagerGM, accumulatorGM, false)                                    \
declare_bnc_header(cmshr, rank, cent, centersmanagerGM, accumulatorGM, false)                              \
declare_bnc_header(cmconstmem, rank, cent, centersmanagerRO, accumulatorGM, false)                            \
declare_bnc_header(cmconstmem_shr, rank, cent, centersmanagerRO, accumulatorGM, false)                        \
declare_bnc_header(cmconstmem_shr_map, rank, cent, centersmanagerRO, accumulatorGM, false)                 \

#define declare_suite_hdrs(rank)        \
declare_testsuite_hdrs(rank, 16)        \
declare_testsuite_hdrs(rank, 32)        \
declare_testsuite_hdrs(rank, 64)        \
declare_testsuite_hdrs(rank, 128)       \
declare_testsuite_hdrs(rank, 256)       \
declare_testsuite_hdrs(rank, 512)       \

#define create_lpfn_entry(map, name, rnk, cent) map[#name][rnk][cent] = buildfnname(name, rnk, cent)
#define create_bnc_entry(map, name, rnk, cent) map[#name][rnk][cent] = buildbncfnname(name, rnk, cent)

#define create_testsuite_entries(map, imap, rank, cent)              \
create_lpfn_entry(map, raw, rank, cent);                             \
create_lpfn_entry(map, rawshr, rank, cent);                          \
create_lpfn_entry(map, constmem, rank, cent);                        \
create_lpfn_entry(map, constmem_memsetreset, rank, cent);            \
create_lpfn_entry(map, constmem_memsetreset_shr, rank, cent);        \
create_lpfn_entry(map, constmem_memsetreset_shr_map, rank, cent);    \
create_lpfn_entry(map, cm, rank, cent);                              \
create_lpfn_entry(map, cmshr, rank, cent);                           \
create_lpfn_entry(map, cmconstmem, rank, cent);                      \
create_lpfn_entry(map, cmconstmem_shr, rank, cent);                  \
create_lpfn_entry(map, cmconstmem_shr_map, rank, cent);              \
create_bnc_entry(imap, raw, rank, cent);                             \
create_bnc_entry(imap, rawshr, rank, cent);                          \
create_bnc_entry(imap, constmem, rank, cent);                        \
create_bnc_entry(imap, constmem_memsetreset, rank, cent);            \
create_bnc_entry(imap, constmem_memsetreset_shr, rank, cent);        \
create_bnc_entry(imap, constmem_memsetreset_shr_map, rank, cent);    \
create_bnc_entry(imap, cm, rank, cent);                              \
create_bnc_entry(imap, cmshr, rank, cent);                           \
create_bnc_entry(imap, cmconstmem, rank, cent);                      \
create_bnc_entry(imap, cmconstmem_shr, rank, cent);                  \
create_bnc_entry(imap, cmconstmem_shr_map, rank, cent);              \

#define create_suite_entries(map, imap, rank)                        \
create_testsuite_entries(map, imap, rank, 16)                        \
create_testsuite_entries(map, imap, rank, 32)                        \
create_testsuite_entries(map, imap, rank, 64)                        \
create_testsuite_entries(map, imap, rank, 128)                       \
create_testsuite_entries(map, imap, rank, 256)                       \
create_testsuite_entries(map, imap, rank, 512)                       \


#define decl_init_lpfn_table_begin(tbl,itbl,flag)                    \
void init_lpfn_table() {                                             \
    if(!flag) {                        

//          DECLARATIONS GO HERE!
//          create_suite_entries(tbl, itbl, 24);
//          -------

#define decl_init_lpfn_table_end(tbl,itbl, flag)                     \
        flag = true;                                                 \
    }                                                                \
}

#define declare_lpfn_finder(tbl, flag)                                                 \
LPFNKMEANS                                                                             \
find_lpfn(                                                                             \
    std::string& name,                                                                 \
    int nRank,                                                                         \
    int nCenters                                                                       \
    )                                                                                  \
{                                                                                      \
    init_lpfn_table();                                                                 \
    std::map<std::string, std::map<int, std::map<int, LPFNKMEANS>>>::iterator ni;      \
    std::map<int, std::map<int, LPFNKMEANS>>::iterator ri;                             \
    std::map<int, LPFNKMEANS>::iterator ci;                                            \
    ni=tbl.find(name);                                                                 \
    if(ni!=tbl.end()) {                                                                \
        ri=ni->second.find(nRank);                                                     \
        if(ri!=ni->second.end()) {                                                     \
            ci=ri->second.find(nCenters);                                              \
            if(ci!=ri->second.end()) {                                                 \
                return ci->second;                                                     \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
    return NULL;                                                                       \
}                                                                                      \

#define declare_bnc_finder(tbl, flag)                                                  \
LPFNBNC                                                                                \
find_bncfn(                                                                            \
    std::string& name,                                                                 \
    int nRank,                                                                         \
    int nCenters                                                                       \
    )                                                                                  \
{                                                                                      \
    init_lpfn_table();                                                                 \
    std::map<std::string, std::map<int, std::map<int, LPFNBNC>>>::iterator ni;         \
    std::map<int, std::map<int, LPFNBNC>>::iterator ri;                                \
    std::map<int, LPFNBNC>::iterator ci;                                               \
    ni=tbl.find(name);                                                                 \
    if(ni!=tbl.end()) {                                                                \
        ri=ni->second.find(nRank);                                                     \
        if(ri!=ni->second.end()) {                                                     \
            ci=ri->second.find(nCenters);                                              \
            if(ci!=ri->second.end()) {                                                 \
                return ci->second;                                                     \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
    return NULL;                                                                       \
}                                                                                      \


