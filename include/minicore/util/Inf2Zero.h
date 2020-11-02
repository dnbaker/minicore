//=================================================================================================
/*!
//  \file blaze/math/simd/Inf2Zero.h
//  \brief Header file for the SIMD binary (base-2) logarithm functionality
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_MATH_SIMD_INF2Z_H_
#define _BLAZE_MATH_SIMD_INF2Z_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>


namespace blaze {

//=================================================================================================
//
//  32-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================
template<typename T, typename=typename std::enable_if<std::is_floating_point<T>::value>::type>
BLAZE_ALWAYS_INLINE T neginf2zero(T x) {
    return x == -std::numeric_limits<T>::infinity() ? T(0) : x;
}

//*************************************************************************************************
/*!\brief Computes the binary logarithm for a vector of single precision floating point values.
// \ingroup simd
//
// \param a The vector of single precision floating point values.
// \return The resulting vector.
//
// This operation is only available via the SVML for SSE, AVX, MIC, and AVX-512.
*/
template< typename T >  // Type of the operand
BLAZE_ALWAYS_INLINE const SIMDfloat neginf2zero( const SIMDf32<T>& a ) noexcept
#if ( BLAZE_AVX512F_MODE  || BLAZE_MIC_MODE )
{
   const auto v = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
   const auto z = _mm512_set1_ps(float(0.));
   auto value = (*a).eval().value;
   return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(value, v, _CMP_EQ_OQ), value, z);
}
#elif BLAZE_AVX_MODE
{
   const auto v = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
   const auto z = _mm256_set1_ps(float(0.));
   auto value = (*a).eval().value;
   return _mm256_blendv_ps(value, z, _mm256_cmp_ps(value, v, _CMP_EQ_OQ));
}
#elif BLAZE_SSE_MODE
{
   const auto v = _mm_set1_ps(-std::numeric_limits<float>::infinity());
   const auto z = _mm_set1_ps(float(0.));
   auto value = (*a).eval().value;
   return _mm_blendv_ps(value, z, _mm_cmpeq_ps(value, v));
}
#else
= delete;
#endif
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Computes the binary logarithm for a vector of double precision floating point values.
// \ingroup simd
//
// \param a The vector of double precision floating point values.
// \return The resulting vector.
//
// This operation is only available via the SVML for SSE, AVX, MIC, and AVX-512.
*/
template< typename T >  // Type of the operand
BLAZE_ALWAYS_INLINE const SIMDdouble neginf2zero( const SIMDf64<T>& a ) noexcept
#if ( BLAZE_AVX512F_MODE  || BLAZE_MIC_MODE )
{
   const auto v = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
   const auto z = _mm512_set1_pd(0.);
   auto value = (*a).eval().value;
   return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(value, v, _CMP_EQ_OQ), value, z);
}
#elif BLAZE_AVX_MODE
{
   const auto v = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
   const auto z = _mm256_set1_pd(0.);
   auto value = (*a).eval().value;
   return _mm256_blendv_pd(value, z, _mm256_cmp_pd(value, v, _CMP_EQ_OQ));
}
#elif BLAZE_SSE_MODE
{
   const auto v = _mm_set1_pd(-std::numeric_limits<double>::infinity());
   const auto z = _mm_set1_pd(0.);
   auto value = (*a).eval().value;
   return _mm_blendv_pd(value, z, _mm_cmpeq_pd(value, v));
}
#else
= delete;
#endif
//*************************************************************************************************
struct NegInf2Zero
{
   //**********************************************************************************************
   /*!\brief Returns the result of the log2() function for the given object/value.
   //
   // \param a The given object/value.
   // \return The result of the log2() function for the given object/value.
   */
   template< typename T >
   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE decltype(auto) operator()( const T& a ) const
   {
      return neginf2zero( a );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether SIMD is enabled for the specified data type \a T.
   //
   // \return \a true in case SIMD is enabled for the data type \a T, \a false if not.
   */
   template< typename T >
   static constexpr bool simdEnabled() {
#if BLAZE_SSE_MODE || BLAZE_AVX_MODE || BLAZE_AVX512F_MODE
        return true;
#else
        return false;
#endif
    }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the operation supports padding, i.e. whether it can deal with zeros.
   //
   // \return \a true in case padding is supported, \a false if not.
   */
   static constexpr bool paddingEnabled() { return false; }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns the result of the log2() function for the given SIMD vector.
   //
   // \param a The given SIMD vector.
   // \return The result of the log2() function for the given SIMD vector.
   */
   template< typename T >
   BLAZE_ALWAYS_INLINE decltype(auto) load( const T& a ) const
   {
      BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK( T );
      return neginf2zero( a );
   }
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  YIELDSUNIFORM SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T >
struct YieldsUniform<NegInf2Zero,T>
   : public IsUniform<T>
{};
/*! \endcond */
//*************************************************************************************************




//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT >
struct YieldsSymmetric<NegInf2Zero,MT>
   : public IsSymmetric<MT>
{};
/*! \endcond */
//*************************************************************************************************

template< typename VT  // Type of the dense vector
        , bool TF >    // Transpose flag
inline decltype(auto) neginf2zero( const DenseVector<VT,TF>& dv )
{
   BLAZE_FUNCTION_TRACE;

   return map( *dv, NegInf2Zero() );
}

template< typename VT  // Type of the sparse vector
        , bool TF >    // Transpose flag
inline decltype(auto) neginf2zero( const SparseVector<VT,TF>& sv )
{
   BLAZE_FUNCTION_TRACE;

   return *sv;
}

template< typename MT  // Type of the sparse matrix
        , bool TF >    // Transpose flag
inline decltype(auto) neginf2zero( const SparseMatrix<MT,TF>& sm )
{
   BLAZE_FUNCTION_TRACE;

   return *sm;
}

template< typename MT  // Type of the sparse matrix
        , bool TF >    // Transpose flag
inline decltype(auto) neginf2zero( const DenseMatrix<MT,TF>& dm )
{
   BLAZE_FUNCTION_TRACE;

   return map( *dm, NegInf2Zero() );
}


} // namespace blaze

#endif
