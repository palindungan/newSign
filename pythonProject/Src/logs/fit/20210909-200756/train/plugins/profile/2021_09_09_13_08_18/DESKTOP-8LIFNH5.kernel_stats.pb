
Z
maxwell_gcgemm_32x32_nt??`*@2?8???@?ÃHǛ?bsequential/conv2d_1/Reluhu  zA
}
maxwell_cgemm_32x32_tn`?H*@2?8Ҭ?@??zH???Xb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhu  ?A
~
maxwell_sgemm_64x64_nn|?B*@2?8?ׂ@ǩ}H?σXb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu  ?A
?
^_ZN5cudnn17winograd_nonfused26winogradForwardData9x9_5x5IffEEvNS0_18WinogradDataParamsIT_T0_EE ??*?2<8ٕ?@??[H??\Xb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu??6B
?
^_ZN5cudnn17winograd_nonfused25winogradWgradDelta9x9_5x5IffEEvNS0_19WinogradDeltaParamsIT_T0_EE*??*?2@8???@??MH??ZXb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu ?sB
?
i_Z15fft2d_c2r_32x32IfLb0ELb0ELj0ELb0ELb0EEvPT_PK6float2iiiiiiiiiffN5cudnn15reduced_divisorEbS1_S1_4int2ii@ ??*?2?8ȳ?@???H???Xb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhu  ?A
?
i_Z15fft2d_c2r_32x32IfLb0ELb1ELj0ELb0ELb0EEvPT_PK6float2iiiiiiiiiffN5cudnn15reduced_divisorEbS1_S1_4int2ii@ ??*?2?8欍@??;H??;bsequential/conv2d_1/Reluhu  ?A
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?ڳ@??2H??3b(gradient_tape/sequential/conv2d/ReluGradhu  ?B
?
g_Z17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEviiiPKT_iPS0_S2_18kernel_grad_paramsyifiiiiG?*2d8?س@??1H??5Xb;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterhu  /B
?
Y_Z15fft2d_r2c_32x32IfLb0ELj0ELb0EEvP6float2PKT_iiiiiiiiiN5cudnn15reduced_divisorEb4int2ii@ ??*?2?8???@??-H??3bsequential/conv2d_1/Reluhu  ?A
?
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb0ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_biiH?*2?	8???@??/H??0Xbsequential/conv2d/Conv2Dhu  /B
p
2_ZN10tensorflow14BiasNCHWKernelIfEEviPKT_S3_PS1_ii*?28???@??)H??*bsequential/conv2d/BiasAddhu  ?B
?
Y_Z15fft2d_r2c_32x32IfLb0ELj0ELb0EEvP6float2PKT_iiiiiiiiiN5cudnn15reduced_divisorEb4int2ii@ ??*?2?8???@£&H??)Xb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhu  ?A
?
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb0ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_biiH?*2?8?è@¶&H??&bsequential/conv2d_2/Reluhu  /B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*?28???@??%H??&b*gradient_tape/sequential/conv2d_1/ReluGradhu  ?B
?
Y_Z15fft2d_r2c_32x32IfLb0ELj5ELb0EEvP6float2PKT_iiiiiiiiiN5cudnn15reduced_divisorEb4int2ii@ ??*?2?8???@??H??+Xb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhu  ?A
?
?_ZN5cudnn45pooling_bw_kernel_max_nchw_fully_packed_smallIffLi2EL21cudnnNanPropagation_t0EEEv17cudnnTensorStructPKT_S2_S5_S2_S5_S2_PS3_18cudnnPoolingStructT0_S8_NS_15reduced_divisorES9_ ?$*?22<8???@??#H$b:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_max_opIKfSB_Li1EEEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28Փ?@??!H??"bsequential/conv2d/Reluhu  ?B
?
Y_Z15fft2d_r2c_32x32IfLb0ELj5ELb1EEvP6float2PKT_iiiiiiiiiN5cudnn15reduced_divisorEb4int2ii@ ??*?2?8Օ?@??H??bsequential/conv2d_1/Reluhu  ?A
~
maxwell_sgemm_128x64_ntx?`*?2$8˶?@??H??Xb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ?A
?
r_ZN5cudnn6detail17dgrad_alg1_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyifiF?*2?8???@??H??Xb<gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropInputhu  /B
?
?_ZN5cudnn3ops20pooling_fw_4d_kernelIffNS_15maxpooling_funcIfL21cudnnNanPropagation_t0EEEL18cudnnPoolingMode_t0ELb0EEEv17cudnnTensorStructPKT_S6_PS7_18cudnnPoolingStructT0_SC_iNS_15reduced_divisorESD_( ?*?228???@??H??b sequential/max_pooling2d/MaxPoolhu?}B
}
maxwell_sgemm_128x64_ntx?`*?2$8???@??H??Xb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhu  ?A
?
k_ZN10tensorflow7functor15RowReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*?2?8???@??H??b3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGradhu  ?B
?
^_ZN5cudnn17winograd_nonfused24winogradForwardOutput4x4IffEEvNS0_20WinogradOutputParamsIT_T0_EEP??*?2<8˸?@??H??Xb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhu  B
?
k_ZN10tensorflow7functor15RowReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*?2?8?ҏ@??H??b5gradient_tape/sequential/conv2d_1/BiasAdd/BiasAddGradhu  ?B
?
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb0ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_biiH?*2d8遏@??H??bsequential/conv2d_3/Reluhu  /B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28茌@??H??b$Adam/Adam/update_8/ResourceApplyAdamhu  ?B
?
e_ZN5cudnn17winograd_nonfused26winogradWgradOutput9x9_5x5IffEEvNS0_25WinogradWgradOutputParamsIT_T0_EE??*?28?Ά@??
H??Xb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu ??B
}
maxwell_sgemm_128x64_ntx?`*?2$8??u@??
H??Xb=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterhu  ?A
?
X_ZN5cudnn17winograd_nonfused20winogradWgradData4x4IffEEvNS0_18WinogradDataParamsIT_T0_EE@??*?228??r@??
H??
Xb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ?A
?
?_ZN5cudnn45pooling_bw_kernel_max_nchw_fully_packed_smallIffLi2EL21cudnnNanPropagation_t0EEEv17cudnnTensorStructPKT_S2_S5_S2_S5_S2_PS3_18cudnnPoolingStructT0_S8_NS_15reduced_divisorES9_ ?*?228??V@??H??b<gradient_tape/sequential/max_pooling2d_1/MaxPool/MaxPoolGradhu  ?B
?
Z_ZN5cudnn17winograd_nonfused22winogradForwardData4x4IffEEvNS0_18WinogradDataParamsIT_T0_EEk??*?28??F@??H??Xb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhu  ?A
W
sgemm_32x32x32_NN_vec???*?28??7@??H??Xbsequential/dense/MatMulhu  ?A
?
Z_ZN5cudnn17winograd_nonfused21winogradWgradDelta4x4IffEEvNS0_19WinogradDeltaParamsIT_T0_EE@??*?228??6@??H??Xb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28ð6@??H??b$Adam/Adam/update_2/ResourceApplyAdamhu  ?B
e
sgemm_32x32x32_NT_vec???*?28??3@??H??Xb%gradient_tape/sequential/dense/MatMulhu  ?A
f
sgemm_128x128x8_TN_vec???*?28??*@??H??b'gradient_tape/sequential/dense/MatMul_1hu  ?A
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28??&@??H??Xb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28??$@??H??bsequential/conv2d_1/Reluhu  ?B
?
X_ZN5cudnn17winograd_nonfused20winogradWgradData4x4IffEEvNS0_18WinogradDataParamsIT_T0_EE@??*?228??#@??H??Xb=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterhu  ?A
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@??H??b*gradient_tape/sequential/conv2d_2/ReluGradhu  ?B
?
Z_ZN5cudnn17winograd_nonfused21winogradWgradDelta4x4IffEEvNS0_19WinogradDeltaParamsIT_T0_EE@??*?228??@??H??Xb=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterhu  B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28??@??H??Xb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu  ?B
?
k_ZN10tensorflow7functor15RowReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*?2?8??@??H??b5gradient_tape/sequential/conv2d_2/BiasAdd/BiasAddGradhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorReductionOpINS0_10SumReducerIfEEKNS_9IndexListINS_10type2indexILx1EEEJEEEKNS_18TensorForcedEvalOpIKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSK_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEEKNS4_INS5_ISK_Li2ELi1ExEELi16ES7_EEEEKNSI_INS0_20scalar_difference_opIffEEKNSM_IKNSC_ISE_JiEEEKNSH_IKNS_18TensorCwiseUnaryOpINS0_13scalar_log_opIfEEKNS4_INS5_IfLi2ELi1ExEELi16ES7_EEEEEEEES14_EEEEEES7_EEEENS_9GpuDeviceEEExEEvT_T0_@*?28??@??H??b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
g
sgemm_32x32x32_NT_vec???*?28 @??H??Xb'gradient_tape/sequential/dense_1/MatMulhu  ?A
g
sgemm_32x32x32_TN_vec???*?28??@??H??b)gradient_tape/sequential/dense_1/MatMul_1hu  ?A
?
a_ZN5cudnn17winograd_nonfused22winogradWgradOutput4x4IffEEvNS0_25WinogradWgradOutputParamsIT_T0_EE5?H* 28??@??H??Xb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  HB
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_17TensorReshapingOpIKNS_9IndexListIiJEEENS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEEEKNS_17TensorReductionOpINS0_10MaxReducerIfLi0EEEKNS5_INS_10type2indexILx1EEEJEEEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEEKNS8_INS9_IKfLi2ELi1ExEELi16ESB_EEEESB_EEEENS_9GpuDeviceEEExEEvT_T0_@*?28??@??H??b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??@??H??b$Adam/Adam/update_4/ResourceApplyAdamhu  ?B
?
4_ZN5cudnn3ops24scalePackedTensor_kernelIffEEvxPT_T0_*?2?8??@??H??Xb<gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropInputhu  ?B
?
k_ZN10tensorflow7functor15RowReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*?2?8??@??H??b5gradient_tape/sequential/conv2d_3/BiasAdd/BiasAddGradhu  ?B
Y
sgemm_32x32x32_NN_vec???*?28??@??H??Xbsequential/dense_1/MatMulhu  ?A
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_17TensorReshapingOpIKNS_9IndexListIiJEEENS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEEEKNS_17TensorReductionOpINS0_10SumReducerIfEEKNS5_INS_10type2indexILx1EEEJEEEKNS_18TensorCwiseUnaryOpINS0_13scalar_exp_opIfEEKSC_EESB_EEEENS_9GpuDeviceEEExEEvT_T0_,*?28??@?|H??b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??
@?wH??b%Adam/Adam/update_10/ResourceApplyAdamhu  ?B
?
a_ZN5cudnn17winograd_nonfused22winogradWgradOutput4x4IffEEvNS0_25WinogradWgradOutputParamsIT_T0_EE5?H* 28??
@?tH??Xb=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterhu  HB
?
?_ZN5cudnn3ops20pooling_fw_4d_kernelIffNS_15maxpooling_funcIfL21cudnnNanPropagation_t0EEEL18cudnnPoolingMode_t0ELb0EEEv17cudnnTensorStructPKT_S6_PS7_18cudnnPoolingStructT0_SC_iNS_15reduced_divisorESD_( ? * 228??
@?vH?zb"sequential/max_pooling2d_1/MaxPoolhu  ?B
?
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ ?!*?228??	@?pH?wbbgradient_tape/sequential/max_pooling2d_1/MaxPool/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIhLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ ?*?228??	@?nH?sb?sequential/dropout/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ ?!*?228??	@?lH?sbHsequential/dropout/dropout/Mul_1-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
^_ZN5cudnn17winograd_nonfused24winogradForwardFilter4x4IffEEvNS0_20WinogradFilterParamsIT_T0_EE ?H* 28??	@?_H?Xb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_20TensorTupleReducerOpINS0_18ArgMaxTupleReducerINS_5TupleIxfEEEEKNS_5arrayIxLy1EEEKNS4_INS5_IKfLi2ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_ *?28??	@?fH??bArgMaxhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_20TensorTupleReducerOpINS0_18ArgMaxTupleReducerINS_5TupleIxfEEEEKNS_5arrayIxLy1EEEKNS4_INS5_IKfLi2ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_ *?28??@?eH?mbArgMax_1hu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??@?^H?db$Adam/Adam/update_6/ResourceApplyAdamhu  ?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?WH?^b.gradient_tape/sequential/dropout/dropout/Mul_1hu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28??@?VH?Ybsequential/conv2d_2/Reluhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28??@?OH?vXb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?LH?Pb*gradient_tape/sequential/conv2d_3/ReluGradhu  ?B
?
?_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*?28??@?GH?Wb7sequential/dropout/dropout/random_uniform/RandomUniformhu  ?B
?
?_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*?28??@?HH?Lb9sequential/dropout_1/dropout/random_uniform/RandomUniformhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28??@?GH?KXb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28??@?8H?kXb<gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropInputhu  ?B
?
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE?!*  28??@??H?Db2gradient_tape/sequential/dense/BiasAdd/BiasAddGradhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28??@?2H?kXb=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterhu  ?B
?
?_ZN10tensorflow66_GLOBAL__N__36_softmax_op_gpu_cu_compute_80_cpp1_ii_71bf5d47_1164822GenerateNormalizedProbIffLi4EEEvPKT_PKT0_S4_PS2_iib*?28??@?7H?<bsequential/dense_1/Softmaxhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIffEEKNS9_INS0_18scalar_quotient_opIffEEKNS_18TensorCwiseUnaryOpINS0_13scalar_exp_opIfEEKS8_EEKNS_20TensorBroadcastingOpIKNS_9IndexListINS_10type2indexILx1EEEJiEEESH_EEEEKNSK_IKNS_5arrayIxLy2EEEKNS4_INS5_IKfLi2ELi1ExEELi16ES7_EEEEEEEENS_9GpuDeviceEEExEEvT_T0_1*?28??@?3H?;b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28??@?1H?7bsequential/conv2d_3/Reluhu  ?B
k
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*?28??@?3H?8bsequential/dense/BiasAddhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorEvalToOpIKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfS6_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEEKNS_9TensorMapINS_6TensorIS6_Li2ELi1ExEELi16ENS_11MakePointerEEEEEKNS4_INS0_20scalar_difference_opIffEEKNS8_IKNS_9IndexListINS_10type2indexILx1EEEJiEEEKNS_18TensorForcedEvalOpIKNS_18TensorCwiseUnaryOpINS0_13scalar_log_opIfEEKNSC_INSD_IfLi2ELi1ExEELi16ESF_EEEEEEEESX_EEEESF_EENS_9GpuDeviceEEExEEvT_T0_**?28??@?.H?4b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?-H?1b0gradient_tape/sequential/dropout_1/dropout/Mul_1hu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??@?,H?/b"Adam/Adam/update/ResourceApplyAdamhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIKfSB_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEEKNS4_INS5_ISB_Li2ELi1ExEELi16ES7_EEEEKNSD_IKNS_9IndexListINS_10type2indexILx1EEEJiEEEKS8_EEEEEENS_9GpuDeviceEEExEEvT_T0_**?28??@?*H?/b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?,H?0b'gradient_tape/sequential/dense/ReluGradhu  ?B
?
:_ZN10tensorflow26BiasGradNHWC_SharedAtomicsIfEEviPKT_PS1_i ?*?28??@?+H?/b4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??@?(H?/b$Adam/Adam/update_9/ResourceApplyAdamhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKbLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?(H?+bsequential/dropout/dropout/Casthu  ?B
s
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28??@?'H?,b)sequential/dropout_1/dropout/GreaterEqualhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKbLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?'H?,b!sequential/dropout_1/dropout/Casthu  ?B
d
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?'H?*b"sequential/dropout_1/dropout/Mul_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKbLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?&H?*b[sequential/dropout/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Casthu  ?B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?&H?)b sequential/dropout/dropout/Mul_1hu  ?B
?
?_ZN10tensorflow7functor15RowReduceKernelIN3cub22TransformInputIteratorIfNS_66_GLOBAL__N__36_softmax_op_gpu_cu_compute_80_cpp1_ii_71bf5d47_1164821SubtractAndExpFunctorIffEENS2_21CountingInputIteratorIixEExEEPfNS2_3SumEEEvT_T0_iiT1_NSt15iterator_traitsISC_E10value_typeE*?28??@?$H?+bsequential/dense_1/Softmaxhu  ?B
?
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE?!*  28??@?&H?)b5gradient_tape/sequential/conv2d_1/BiasAdd/BiasAddGradhu  ?B
q
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28??@?'H?(b'sequential/dropout/dropout/GreaterEqualhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??@?%H?)b$Adam/Adam/update_3/ResourceApplyAdamhu  ?B
?
l_ZN10tensorflow7functor15CleanupSegmentsIPfS2_N3cub3SumEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*  28??@?%H?)b5gradient_tape/sequential/conv2d_1/BiasAdd/BiasAddGradhu  ?B
?
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE?!*  28??@?#H?(b5gradient_tape/sequential/conv2d_3/BiasAdd/BiasAddGradhu  ?B
?
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE?!*  28??@?$H?*b5gradient_tape/sequential/conv2d_2/BiasAdd/BiasAddGradhu  ?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?%H?'bsequential/dropout/dropout/Mulhu  ?B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?$H?'b sequential/dropout_1/dropout/Mulhu  ?B
?
l_ZN10tensorflow7functor15CleanupSegmentsIPfS2_N3cub3SumEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*  28??@?#H?&b3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGradhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy1EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*?28??@?"H?&b;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1hu  ?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?"H?'b.gradient_tape/sequential/dropout_1/dropout/Mulhu  ?B
E
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?ebMulhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??@?!H?%b$Adam/Adam/update_7/ResourceApplyAdamhu  ?B
J
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?&bAdam/Powhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_max_opIKfSB_Li1EEEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?!H?'bsequential/dense/Reluhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??@?!H?$b$Adam/Adam/update_1/ResourceApplyAdamhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorEvalToOpIKNS_18TensorCwiseUnaryOpINS0_13scalar_log_opIfEEKNS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEEESA_EENS_9GpuDeviceEEExEEvT_T0_*?28??@? H?$b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??@?!H?#b$Adam/Adam/update_5/ResourceApplyAdamhu  ?B
?
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE?!*  28??@?H?%b3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGradhu  ?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@? H?"b,gradient_tape/sequential/dropout/dropout/Mulhu  ?B
?
l_ZN10tensorflow7functor15CleanupSegmentsIPfS2_N3cub3SumEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*  28??@?H?$b5gradient_tape/sequential/conv2d_3/BiasAdd/BiasAddGradhu  ?B
?
l_ZN10tensorflow7functor15CleanupSegmentsIPfS2_N3cub3SumEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*  28??@?H?%b5gradient_tape/sequential/conv2d_2/BiasAdd/BiasAddGradhu  ?B
?
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE0*?28??@?H?$b*categorical_crossentropy/weighted_loss/Sumhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28??@?H?"Xb;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??@?H? b%Adam/Adam/update_11/ResourceApplyAdamhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H? bAdam/Adam/AssignAddVariableOphu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?!b,categorical_crossentropy/weighted_loss/valuehu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H? bAssignAddVariableOp_4hu  ?B
?
k_ZN10tensorflow7functor15RowReduceKernelIPKfPfN3cub3MaxEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*?28??@?H?!bsequential/dense_1/Softmaxhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?"bCasthu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28??@?H?&Xbsequential/conv2d/Conv2Dhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKxLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?bAdam/Cast_1hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel
*?28??@?H?bLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKbLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?bCast_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?bAssignAddVariableOp_2hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?b8categorical_crossentropy/weighted_loss/num_elements/Casthu  ?B
L
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28??@?H?)b
Adam/Pow_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?bEgradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nanhu  ?B
L
"AddV2_GPU_DT_INT64_DT_INT64_kernel
*?28??@?H?bAdam/addhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?!bCast_2hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H? b
div_no_nanhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?bdiv_no_nan_1hu  ?B
m
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*?28??@?H?bsequential/dense_1/BiasAddhu  ?B
H
!Equal_GPU_DT_INT64_DT_BOOL_kernel*?28??@?H?bEqualhu  ?B
?
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE0*?28??@?H?bSum_2hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?bAssignAddVariableOp_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?bAssignAddVariableOp_3hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28??@?H?bAssignAddVariableOphu  ?B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*?28?@?H?b
LogicalAndhu  ?B