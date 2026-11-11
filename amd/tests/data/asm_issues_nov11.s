// Three bugs found by running other people's HIP programs on the simulated MI250X and MI300X (issues #578, #579 and #580), each as a
// small kernel whose result is written from the documentation, not from the simulator (amd/tests/unit/test_gcn_issues_nov11.cpp).
//
// neg_scratch(out) (gfx942): "scratch_store_dwordx4 off, v[0:3], s0 offset:32" with s0 holding -32 reached for the address 2^32 and was
//   refused as past the work-item's scratch. The ISA gives the address as flat_scratch + swizzle(sgpr_offset + inst.offset, threadID)
//   (CDNA3 ISA 10.5): a 32-bit offset, here 0. Each lane stores a dword at -32 + 32 and four at -16 + 32, and loads them back.
// pk_zero(in, out): "v_pk_add_f32 v[2:3], v[2:3], 0 neg_lo:[1,1] neg_hi:[1,1]" is how LLVM writes a float negate-and-add-zero: the constant
//   0 is zero in both halves, whatever the second half of a 64-bit constant is. Lane L's pair is in[2 L], in[2 L + 1]; out holds the add at
//   2 L, the multiply by 0 at 128 + 2 L and the fused multiply-add of the pair with itself and 0 at 256 + 2 L.
// pk_const(in, out): a nonzero constant's second half stays refused (CDNA3 ISA 6.2.1.1 gives the 64-bit expansion of a literal, not the
//   packed forms' reading of an inline float constant): v_pk_add_f32 v[2:3], v[2:3], 1.0 with op_sel_hi left at its default.
// smem_vcc(in, out): "s_load_dwordx2 vcc, s[94:95], 0x0" is LLVM's form when it is short of scalar registers; the destination is VCC
//   (SDATA 106), not s0, and the executor wrote the loaded pair to s0 and s1 -- the private segment buffer's base on gfx90a, so the next
//   scratch access of the kernel read through a buffer whose address was a double (issue #579). out: vcc_lo and vcc_hi after the dwordx2
//   load of in[0], in[1]; whether s[0:1] (the kernarg pointer here) is as it was; vcc_hi after "s_load_dword vcc_lo" of in[2] (still in[1]);
//   then vcc_lo and vcc_hi after "s_load_dword vcc_hi" of in[3] (vcc_lo is now in[2]).
//
// Assembled for gfx90a and gfx942 by build.sh (clang's assembler with the C preprocessor: GFX90A or GFX942 is defined).
  .text
#if defined(GFX942)
  .globl neg_scratch
  .p2align 8
  .type neg_scratch,@function
neg_scratch:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 2, v0                  // 4 L
  v_add_u32_e32 v2, 100, v0                    // the dword, 100 + L
  v_add_u32_e32 v4, 1000, v0
  v_add_u32_e32 v5, 2000, v0
  v_add_u32_e32 v6, 3000, v0
  v_add_u32_e32 v7, 4000, v0                   // the four, 1000 + L .. 4000 + L
  s_movk_i32 s6, -32
  s_movk_i32 s7, -16
  scratch_store_dword off, v2, s6 offset:32    // at -32 + 32 = 0
  scratch_store_dwordx4 off, v[4:7], s7 offset:48   // at -16 + 48 = 32
  s_waitcnt vmcnt(0)
  scratch_load_dword v8, off, s6 offset:32
  scratch_load_dwordx4 v[10:13], off, s7 offset:48
  s_waitcnt vmcnt(0)
  global_store_dword v1, v8, s[4:5] offset:0
  v_lshlrev_b32_e32 v14, 4, v0
  global_store_dwordx4 v14, v[10:13], s[4:5] offset:256
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel neg_scratch
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 64
    .amdhsa_enable_private_segment 1
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel
#endif

  .globl pk_zero
  .p2align 8
  .type pk_zero,@function
pk_zero:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = in
  s_load_dwordx2 s[6:7], s[0:1], 0x8          // s[6:7] = out
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 3, v0                  // 8 L
  global_load_dwordx2 v[2:3], v1, s[4:5]
  s_waitcnt vmcnt(0)
  v_pk_add_f32 v[4:5], v[2:3], 0 neg_lo:[1,1] neg_hi:[1,1]
  v_pk_mul_f32 v[6:7], v[2:3], 0
  v_pk_fma_f32 v[8:9], v[2:3], v[2:3], 0
  global_store_dwordx2 v1, v[4:5], s[6:7] offset:0
  global_store_dwordx2 v1, v[6:7], s[6:7] offset:512
  global_store_dwordx2 v1, v[8:9], s[6:7] offset:1024
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel pk_zero
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 96
#if defined(GFX90A) || defined(GFX942)
    .amdhsa_accum_offset 16
#endif
  .end_amdhsa_kernel

  .globl pk_const
  .p2align 8
  .type pk_const,@function
pk_const:
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  s_load_dwordx2 s[6:7], s[0:1], 0x8
  s_waitcnt lgkmcnt(0)
  v_and_b32_e32 v0, 63, v0
  v_lshlrev_b32_e32 v1, 3, v0
  global_load_dwordx2 v[2:3], v1, s[4:5]
  s_waitcnt vmcnt(0)
  v_pk_add_f32 v[4:5], v[2:3], 1.0
  global_store_dwordx2 v1, v[4:5], s[6:7]
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel pk_const
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel

  .globl smem_vcc
  .p2align 8
  .type smem_vcc,@function
smem_vcc:
  s_load_dwordx2 s[4:5], s[0:1], 0x0          // s[4:5] = in
  s_load_dwordx2 s[6:7], s[0:1], 0x8          // s[6:7] = out
  s_waitcnt lgkmcnt(0)
  s_mov_b64 s[12:13], s[0:1]                   // the kernarg pointer, kept to see that it survives
  s_mov_b64 vcc, 0
  s_load_dwordx2 vcc, s[4:5], 0x0              // vcc = in[0], in[1]
  s_waitcnt lgkmcnt(0)
  s_mov_b64 s[8:9], vcc                        // out[0], out[1]
  s_cmp_eq_u64 s[0:1], s[12:13]
  s_cselect_b32 s10, 1, 0                      // out[2]
  s_load_dword vcc_lo, s[4:5], 0x8             // vcc_lo = in[2], vcc_hi still in[1]
  s_waitcnt lgkmcnt(0)
  s_mov_b32 s11, vcc_hi                        // out[3]
  s_load_dword vcc_hi, s[4:5], 0xc             // vcc_hi = in[3]
  s_waitcnt lgkmcnt(0)
  s_mov_b64 s[14:15], vcc                      // out[4], out[5]
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v2, s8
  v_mov_b32_e32 v3, s9
  v_mov_b32_e32 v4, s10
  v_mov_b32_e32 v5, s11
  v_mov_b32_e32 v6, s14
  v_mov_b32_e32 v7, s15
  global_store_dwordx4 v1, v[2:5], s[6:7] offset:0
  global_store_dwordx2 v1, v[6:7], s[6:7] offset:16
  s_waitcnt vmcnt(0)
  s_endpgm
  .p2align 6
  .amdhsa_kernel smem_vcc
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel

  .amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
#if defined(GFX942)
  - .name: neg_scratch
    .symbol: neg_scratch.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 64
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 16
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
#endif
  - .name: pk_zero
    .symbol: pk_zero.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 16
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
  - .name: pk_const
    .symbol: pk_const.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 16
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
  - .name: smem_vcc
    .symbol: smem_vcc.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 16
    .max_flat_workgroup_size: 64
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
  .end_amdgpu_metadata
