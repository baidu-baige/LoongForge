#pragma once

#include <cute/tensor.hpp>
#include <cute/arch/simd_sm100.hpp>

#include "defines.h"

namespace sm100 {

using namespace cute;

__forceinline__ __device__ void cp_async_cacheglobal_l2_prefetch_256B(const void* src, void* dst) {
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst);
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n"
        :: "r"(dst_addr),
           "l"(src),
           "n"(16));
}

CUTE_DEVICE
int64_t createpolicy_evict_last() {
    int64_t res;
    asm volatile(
        "createpolicy.fractional.L2::evict_last.b64 %0, 1.0; \n\t"
        : "=l"(res)
        :
    );
    return res;
}

template<typename T>
CUTE_DEVICE
static void st_async_128b(void* dst_ptr, const T& data, const transac_bar_t* mbar_ptr) {
    static_assert(sizeof(T) == 16, "Data type must be 16 bytes (128 bits) for st_async_128b.");
    long2 data_long2 = *reinterpret_cast<const long2*>(&data);
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(mbar_ptr);
    asm volatile (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 [%0], {%1, %2}, [%3]; \n"
        :
        : "r"(dst_addr), "l"(data_long2.x), "l"(data_long2.y), "r"(mbar_addr)
    );
}


__device__ __forceinline__ void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

CUTE_DEVICE
void umma_arrive_multicast_noelect(transac_bar_t &smem_ptr, uint16_t cta_mask) {
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&smem_ptr);
  asm volatile(
    "{\n\t"
    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1; \n\t"
    "}" 
    :
    :"r"(bar_intptr), "h"(cta_mask));
}

CUTE_DEVICE
void umma_arrive_multicast_2x1SM_noelect(transac_bar_t &smem_ptr, uint16_t cta_mask) {
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&smem_ptr);
  asm volatile(
    "{\n\t"
    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1; \n\t"
    "}" 
    :
    :"r"(bar_intptr), "h"(cta_mask));
}

CUTE_DEVICE
void umma_arrive_noelect(transac_bar_t &smem_ptr) {
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&smem_ptr);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    :
    :"r"(bar_intptr));
}

CUTE_DEVICE
void umma_arrive_2x1SM_noelect(transac_bar_t &smem_ptr) {
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&smem_ptr);
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    :
    :"r"(bar_intptr));
}

CUTE_DEVICE
float2 float2_add(const float2 &a, const float2 &b) {
    float2 res;
    cute::add(res, a, b);
    return res;
}

CUTE_DEVICE
float2 float2_mul(const float2 &a, const float2 &b) {
    float2 res;
    cute::mul(res, a, b);
    return res;
}

CUTE_DEVICE
float2 float2_fma(const float2 &a, const float2 &b, const float2 &c) {
    // return a*b+c
    float2 res;
    cute::fma(res, a, b, c);
    return res;
}

CUTE_DEVICE
float2 float2_neg(const float2 &a) {
    float2 t = {-1.0f, -1.0f};
    return float2_mul(a, t);
}

template<bool USE_CTA0_MBAR = false>
CUTE_DEVICE void tma_gather4(const void* desc_ptr, transac_bar_t* mbar_ptr, void* smem_ptr, int col_idx, int4 row_idxs, TMA::CacheHintSm90 cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(mbar_ptr);
    if constexpr (USE_CTA0_MBAR) {
        mbar_addr &= Sm100MmaPeerBitMask;
    }
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx), 
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w), 
          "r"(mbar_addr), "l"(uint64_t(cache_hint))
        : "memory"
    );
}

// 32 data path lanes, 32-bit pattern, repeated N times
template <int N, typename T>
CUTE_DEVICE void tmem_ld_32dp32bNx(uint32_t const &src_addr, T* dst_ptr_) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128, "N must be a power of 2 and lies between 1 ~ 128");
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst_ptr_);

    if constexpr (N == 1) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    "{%0},"
                    "[%1];\n"
                    : "=r"(dst_ptr[0])
                    : "r"(src_addr));
    } else if constexpr (N == 2) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32"
                    "{%0, %1},"
                    "[%2];\n"
                    : "=r"(dst_ptr[0]), "=r"(dst_ptr[1])
                    : "r"(src_addr));
    } else if constexpr (N == 4) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32"
                    "{%0, %1, %2, %3},"
                    "[%4];\n"
                    : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                    "=r"(dst_ptr[3])
                    : "r"(src_addr));
    } else if constexpr (N == 8) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32"
                    "{%0, %1, %2, %3, %4, %5, %6, %7},"
                    "[%8];\n"
                    : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                    "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                    "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
                    : "r"(src_addr));
    } else if constexpr (N == 16) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32"
                    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
                    "%14, %15},"
                    "[%16];\n"
                    : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                    "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                    "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
                    "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
                    "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
                    "=r"(dst_ptr[15])
                    : "r"(src_addr));
    } else if constexpr (N == 32) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
                    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
                    "%26, %27, %28, %29, %30, %31},"
                    "[%32];\n"
                    : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                    "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                    "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
                    "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
                    "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
                    "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
                    "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
                    "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
                    "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
                    "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
                    "=r"(dst_ptr[30]), "=r"(dst_ptr[31])
                    : "r"(src_addr));
    } else if constexpr (N == 64) {
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x64.b32"
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
            "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, "
            "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, "
            "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
            "%57, %58, %59, %60, %61, %62, %63},"
            "[%64];\n"
            : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63])
            : "r"(src_addr));
    } else if constexpr (N == 128) {
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x128.b32"
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
            "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, "
            "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, "
            "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
            "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, "
            "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, "
            "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, "
            "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
            "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
            "%121, %122, %123, %124, %125, %126, %127},"
            "[%128];\n"
            : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63]), "=r"(dst_ptr[64]), "=r"(dst_ptr[65]),
            "=r"(dst_ptr[66]), "=r"(dst_ptr[67]), "=r"(dst_ptr[68]),
            "=r"(dst_ptr[69]), "=r"(dst_ptr[70]), "=r"(dst_ptr[71]),
            "=r"(dst_ptr[72]), "=r"(dst_ptr[73]), "=r"(dst_ptr[74]),
            "=r"(dst_ptr[75]), "=r"(dst_ptr[76]), "=r"(dst_ptr[77]),
            "=r"(dst_ptr[78]), "=r"(dst_ptr[79]), "=r"(dst_ptr[80]),
            "=r"(dst_ptr[81]), "=r"(dst_ptr[82]), "=r"(dst_ptr[83]),
            "=r"(dst_ptr[84]), "=r"(dst_ptr[85]), "=r"(dst_ptr[86]),
            "=r"(dst_ptr[87]), "=r"(dst_ptr[88]), "=r"(dst_ptr[89]),
            "=r"(dst_ptr[90]), "=r"(dst_ptr[91]), "=r"(dst_ptr[92]),
            "=r"(dst_ptr[93]), "=r"(dst_ptr[94]), "=r"(dst_ptr[95]),
            "=r"(dst_ptr[96]), "=r"(dst_ptr[97]), "=r"(dst_ptr[98]),
            "=r"(dst_ptr[99]), "=r"(dst_ptr[100]), "=r"(dst_ptr[101]),
            "=r"(dst_ptr[102]), "=r"(dst_ptr[103]), "=r"(dst_ptr[104]),
            "=r"(dst_ptr[105]), "=r"(dst_ptr[106]), "=r"(dst_ptr[107]),
            "=r"(dst_ptr[108]), "=r"(dst_ptr[109]), "=r"(dst_ptr[110]),
            "=r"(dst_ptr[111]), "=r"(dst_ptr[112]), "=r"(dst_ptr[113]),
            "=r"(dst_ptr[114]), "=r"(dst_ptr[115]), "=r"(dst_ptr[116]),
            "=r"(dst_ptr[117]), "=r"(dst_ptr[118]), "=r"(dst_ptr[119]),
            "=r"(dst_ptr[120]), "=r"(dst_ptr[121]), "=r"(dst_ptr[122]),
            "=r"(dst_ptr[123]), "=r"(dst_ptr[124]), "=r"(dst_ptr[125]),
            "=r"(dst_ptr[126]), "=r"(dst_ptr[127])
            : "r"(src_addr));
    } else {
        asm volatile ("trap");
    }
}

// 32 data path lanes, 32-bit pattern, repeated N times
template <int N, typename T>
CUTE_DEVICE void tmem_st_32dp32bNx(uint32_t const &dst_addr, T* src_ptr_) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128, "N must be a power of 2 and lies between 1 ~ 128");
    uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src_ptr_);

    if constexpr (N == 1) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
                    "[%1], {%0};\n"
                    :
                    : "r"(src_ptr[0]),
                      "r"(dst_addr));
    } else if constexpr (N == 2) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32"
                    "[%2], {%0, %1};\n"
                    :
                    : "r"(src_ptr[0]), "r"(src_ptr[1]),
                      "r"(dst_addr));
    } else if constexpr (N == 4) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32"
                    "[%4], {%0, %1, %2, %3};\n"
                    :
                    : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
                    "r"(src_ptr[3]),
                      "r"(dst_addr));
    } else if constexpr (N == 8) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32"
                    "[%8], {%0, %1, %2, %3, %4, %5, %6, %7};\n"
                    :
                    : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
                    "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
                    "r"(src_ptr[6]), "r"(src_ptr[7]),
                      "r"(dst_addr));
    } else if constexpr (N == 16) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32"
                    "[%16], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
                    "%14, %15};\n"
                    :
                    : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
                    "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
                    "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]),
                    "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]),
                    "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]),
                    "r"(src_ptr[15]),
                      "r"(dst_addr));
    } else if constexpr (N == 32) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32"
                    "[%32], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
                    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
                    "%26, %27, %28, %29, %30, %31};\n"
                    :
                    : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
                    "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
                    "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]),
                    "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]),
                    "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]),
                    "r"(src_ptr[15]), "r"(src_ptr[16]), "r"(src_ptr[17]),
                    "r"(src_ptr[18]), "r"(src_ptr[19]), "r"(src_ptr[20]),
                    "r"(src_ptr[21]), "r"(src_ptr[22]), "r"(src_ptr[23]),
                    "r"(src_ptr[24]), "r"(src_ptr[25]), "r"(src_ptr[26]),
                    "r"(src_ptr[27]), "r"(src_ptr[28]), "r"(src_ptr[29]),
                    "r"(src_ptr[30]), "r"(src_ptr[31]),
                      "r"(dst_addr));
    } else if constexpr (N == 64) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x64.b32"
            "[%64], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
            "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, "
            "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, "
            "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
            "%57, %58, %59, %60, %61, %62, %63};\n"
            :
            : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
            "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
            "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]),
            "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]),
            "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]),
            "r"(src_ptr[15]), "r"(src_ptr[16]), "r"(src_ptr[17]),
            "r"(src_ptr[18]), "r"(src_ptr[19]), "r"(src_ptr[20]),
            "r"(src_ptr[21]), "r"(src_ptr[22]), "r"(src_ptr[23]),
            "r"(src_ptr[24]), "r"(src_ptr[25]), "r"(src_ptr[26]),
            "r"(src_ptr[27]), "r"(src_ptr[28]), "r"(src_ptr[29]),
            "r"(src_ptr[30]), "r"(src_ptr[31]), "r"(src_ptr[32]),
            "r"(src_ptr[33]), "r"(src_ptr[34]), "r"(src_ptr[35]),
            "r"(src_ptr[36]), "r"(src_ptr[37]), "r"(src_ptr[38]),
            "r"(src_ptr[39]), "r"(src_ptr[40]), "r"(src_ptr[41]),
            "r"(src_ptr[42]), "r"(src_ptr[43]), "r"(src_ptr[44]),
            "r"(src_ptr[45]), "r"(src_ptr[46]), "r"(src_ptr[47]),
            "r"(src_ptr[48]), "r"(src_ptr[49]), "r"(src_ptr[50]),
            "r"(src_ptr[51]), "r"(src_ptr[52]), "r"(src_ptr[53]),
            "r"(src_ptr[54]), "r"(src_ptr[55]), "r"(src_ptr[56]),
            "r"(src_ptr[57]), "r"(src_ptr[58]), "r"(src_ptr[59]),
            "r"(src_ptr[60]), "r"(src_ptr[61]), "r"(src_ptr[62]),
            "r"(src_ptr[63]),
              "r"(dst_addr));
    } else if constexpr (N == 128) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x128.b32"
            "[%128], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
            "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, "
            "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, "
            "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
            "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, "
            "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, "
            "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, "
            "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
            "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
            "%121, %122, %123, %124, %125, %126, %127};\n"
            :
            : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
            "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
            "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]),
            "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]),
            "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]),
            "r"(src_ptr[15]), "r"(src_ptr[16]), "r"(src_ptr[17]),
            "r"(src_ptr[18]), "r"(src_ptr[19]), "r"(src_ptr[20]),
            "r"(src_ptr[21]), "r"(src_ptr[22]), "r"(src_ptr[23]),
            "r"(src_ptr[24]), "r"(src_ptr[25]), "r"(src_ptr[26]),
            "r"(src_ptr[27]), "r"(src_ptr[28]), "r"(src_ptr[29]),
            "r"(src_ptr[30]), "r"(src_ptr[31]), "r"(src_ptr[32]),
            "r"(src_ptr[33]), "r"(src_ptr[34]), "r"(src_ptr[35]),
            "r"(src_ptr[36]), "r"(src_ptr[37]), "r"(src_ptr[38]),
            "r"(src_ptr[39]), "r"(src_ptr[40]), "r"(src_ptr[41]),
            "r"(src_ptr[42]), "r"(src_ptr[43]), "r"(src_ptr[44]),
            "r"(src_ptr[45]), "r"(src_ptr[46]), "r"(src_ptr[47]),
            "r"(src_ptr[48]), "r"(src_ptr[49]), "r"(src_ptr[50]),
            "r"(src_ptr[51]), "r"(src_ptr[52]), "r"(src_ptr[53]),
            "r"(src_ptr[54]), "r"(src_ptr[55]), "r"(src_ptr[56]),
            "r"(src_ptr[57]), "r"(src_ptr[58]), "r"(src_ptr[59]),
            "r"(src_ptr[60]), "r"(src_ptr[61]), "r"(src_ptr[62]),
            "r"(src_ptr[63]), "r"(src_ptr[64]), "r"(src_ptr[65]),
            "r"(src_ptr[66]), "r"(src_ptr[67]), "r"(src_ptr[68]),
            "r"(src_ptr[69]), "r"(src_ptr[70]), "r"(src_ptr[71]),
            "r"(src_ptr[72]), "r"(src_ptr[73]), "r"(src_ptr[74]),
            "r"(src_ptr[75]), "r"(src_ptr[76]), "r"(src_ptr[77]),
            "r"(src_ptr[78]), "r"(src_ptr[79]), "r"(src_ptr[80]),
            "r"(src_ptr[81]), "r"(src_ptr[82]), "r"(src_ptr[83]),
            "r"(src_ptr[84]), "r"(src_ptr[85]), "r"(src_ptr[86]),
            "r"(src_ptr[87]), "r"(src_ptr[88]), "r"(src_ptr[89]),
            "r"(src_ptr[90]), "r"(src_ptr[91]), "r"(src_ptr[92]),
            "r"(src_ptr[93]), "r"(src_ptr[94]), "r"(src_ptr[95]),
            "r"(src_ptr[96]), "r"(src_ptr[97]), "r"(src_ptr[98]),
            "r"(src_ptr[99]), "r"(src_ptr[100]), "r"(src_ptr[101]),
            "r"(src_ptr[102]), "r"(src_ptr[103]), "r"(src_ptr[104]),
            "r"(src_ptr[105]), "r"(src_ptr[106]), "r"(src_ptr[107]),
            "r"(src_ptr[108]), "r"(src_ptr[109]), "r"(src_ptr[110]),
            "r"(src_ptr[111]), "r"(src_ptr[112]), "r"(src_ptr[113]),
            "r"(src_ptr[114]), "r"(src_ptr[115]), "r"(src_ptr[116]),
            "r"(src_ptr[117]), "r"(src_ptr[118]), "r"(src_ptr[119]),
            "r"(src_ptr[120]), "r"(src_ptr[121]), "r"(src_ptr[122]),
            "r"(src_ptr[123]), "r"(src_ptr[124]), "r"(src_ptr[125]),
            "r"(src_ptr[126]), "r"(src_ptr[127]),
              "r"(dst_addr));
    } else {
        asm volatile ("trap");
    }
}

static constexpr int PEER_ADDR_MASK = 16777216; // peer_addr = my_addr ^ PEER_ADDR_MASK. 不确定是不是在所有显卡上都是这个数字
template<typename T>
CUTE_DEVICE
T* get_peer_addr(const T* p) {
    return (T*)((int64_t)(p) ^ PEER_ADDR_MASK);
}


}
