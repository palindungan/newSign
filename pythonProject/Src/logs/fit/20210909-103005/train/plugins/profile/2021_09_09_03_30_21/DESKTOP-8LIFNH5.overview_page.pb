?,$	????q?A@??u??@?i?<@!k?v/YE@$	???Щ@?q??@?H)???!?+?~0@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB{O崧4B@w.??????1????8@A?|	>@I??M~?N@YX??G???rtrain 15"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsBk?v/YE@????%@1Rew?8@A?>?̔6@I?<e5]?@Yf?"?ϩ@rtrain 16"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?d??J?A@?鲘?|??12W??7@A?8~h @I?????@Y?R??Fk@rtrain 17"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?pX?>@j?J>v???11?0&?98@A??ʅ????I$H???8	@Y?k???P??rtrain 18"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsBf????>@?	MKJ??1?l??8@Ai?ai????I?2??v@YF;?I??rtrain 19"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB???f@@?Df.p???1?VC?+8@Au???mn @I??Coa@Y??q?@H??rtrain 20"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsBK?8???B@xe??i??1mu9% 8@A??v?$d@I?~m??#@Yi??U???rtrain 21"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?(	???C@?D????
@1E?A?8@A6?>W[q@I?<??@Y???o?=@rtrain 22"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB	?T?:DB@?.?o????1?YJ??08@A|,}??? @I?ߢ??&@Y?	??.?@rtrain 23"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB
q?a??{=@p%;6???1f?????7@A
?(z?? @I[}uU?V@Yr?30????rtrain 24"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?i?<@??N$????1??v?&8@AtB????I??~m?@YS?k%t???rtrain 25"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsBx*????@@.W?6????1???cw)8@A:A?>???I2?m??F@Y#k??"??rtrain 26"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?-]?@B@?vi?a	 @1?"??8@A?xy:W?@I??OU?q@Y??&????rtrain 27"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB???QD@{?\??@1????#8@A??n?!?@IİØ??@YX Sn@rtrain 28"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB$	?P?A@???+?l??1{h+ 8@AQ?i>b @I׉??
?@Y`?o`r@rtrain 29"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?`??l=@Z)r?#??1_?\6:+8@A??͋???I{??9yQ@YjhwH1??rtrain 30"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?????K@@)YNB????1??6T?8@AS???"???I???kѢ@Y?$\?#??rtrain 31"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?-X? C@?(]??$@1ݲC??8@Awd?6?O@I?CԷ? @Y??D2??rtrain 32"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?҈?}?C@Z?????17QKs+8@A&???~@Ik??=]?@Y???$x?@rtrain 33*	??? A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorDi? ?@!_??'?|W@)Di? ?@1_??'?|W@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch̘?5?N@@!??ˡ?@)̘?5?N@@1??ˡ?@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?E???~@@!Y?k??@)?j????1?Y??ŋ??:Preprocessing2F
Iterator::Model?V?9Ώ@@!??[N@)w?E???1bu?3}??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?J#f!?@!CB?~W@)?z?Fw??1@??h???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?14.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s4.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?+Mq?q@I??_?G?9@Q[CSd+Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?.???}??<??|?????N$????!????%@	!       "$	2???8@????ۧ??2W??7@!1?0&?98@*	!       2$	?іa?P@A??C??tB????!??n?!?@:$	v??/j@d>?H? @[}uU?V@!?~m??#@B	!       J$	4?$???h????@??q?@H??!???$x?@R	!       Z$	4?$???h????@??q?@H??!???$x?@b	!       JGPUY?+Mq?q@b q??_?G?9@y[CSd+Q@?"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???j???!???j???0"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInputr?SX?A??!?g??v??0":
sequential/conv2d_1/Relu_FusedConv2DQc????!???t???"F
(gradient_tape/sequential/conv2d/ReluGradReluGrado??Oxء?!?՜????"g
;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter,?6Mj]??!>p?????0"6
sequential/conv2d/Conv2DConv2DlR???ؠ?!A
?\	??0"6
sequential/conv2d/BiasAddBiasAdd?8jץ
??!	U??????"g
<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?9????!#??????0":
sequential/conv2d_2/Relu_FusedConv2D???????!???K???"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter@??,f??!???
?~??0Q      Y@Y?իW?$@aB?
*|V@q???0????yĽ?D?"?
both?Your program is MODERATELY input-bound because 5.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?14.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s4.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Maxwell)(: B 