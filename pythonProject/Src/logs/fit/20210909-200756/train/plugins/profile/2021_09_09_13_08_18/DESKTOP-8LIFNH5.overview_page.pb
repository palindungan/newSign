? $	?=???g>@??@s??}Yک? >@!??0"?@$	?y??2????6?"???N0;;????!?
?Jo_@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB??^I>@g?W????1?n?H68@Aod?C??I???p?1@YC?=?P??rtrain 40"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?ϸp 8>@H¾?D??1?lV}?"8@A?????Y??IԷ???x
@Y?E(?????rtrain 41"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB}Yک? >@]lZ)???1cE?a88@A.?_x%???I?Qԙ{?	@YDio?????rtrain 42"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB??0"?@pB!!??1Yl???V8@A?}:3??I??5&$@Y????G6??rtrain 43"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?vhX?.>@?Ȯ?????1ׅ?O?7@A??f?8??I˺,DG
@Y{?Fw;??rtrain 44"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB??ң?>@<?ן?g??1??6?38@A%?)? (??I?Ye??@YYk(???rtrain 45"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB`??-Z>@/6????1?T?-8@A?*8|??I??_vO?
@Y??"ڎ???rtrain 46"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?-z?>@?X????1Eg?E(>8@A~?Az?\??I ?E
e?@Y"5?b?i??rtrain 47"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB	???[?>@??????1?????Q8@A?:??????I??iߜ@Yȴ6?????rtrain 48*	m????cA2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator歺U?c@!Q??x??X@)歺U?c@1Q??x??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??F?0}??!???h????)??3ڪ$??1r=Jض?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?Z? m???!AP?ŭ???)?Z? m???1AP?ŭ???:Preprocessing2F
Iterator::Model}>ʈ??!?[??????)\<???r??1??Tl???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??>?c@!R???X@)???8???1?
Par??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9d??B/1??I??j?3@Qਙ?5?S@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	b?l?l??'+ѝ?????Ȯ?????!?X????	!       "$	?)?_?28@?{?(???ׅ?O?7@!Yl???V8@*	!       2$	?\zp????xC???E??od?C??!??f?8??:$	?7OĄm
@??U(Z??????p?1@!?Ye??@B	!       J$	S?#E?????.'?????"ڎ???!C?=?P??R	!       Z$	S?#E?????.'?????"ڎ???!C?=?P??b	!       JGPUYd??B/1??b q??j?3@yਙ?5?S@?"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterO?G?)V??!O?G?)V??0":
sequential/conv2d_1/Relu_FusedConv2D??????!??x2???"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInputY	?9[???!#???$???0"g
;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?3??ܡ?!_)??????0"F
(gradient_tape/sequential/conv2d/ReluGradReluGrad??>??ơ?!n3T\???"6
sequential/conv2d/Conv2DConv2DF?gԻ??!?㬚??0"6
sequential/conv2d/BiasAddBiasAdd"f?>???!?l?????"g
<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?????!?<?s???0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????K??!j? $ѥ??0":
sequential/conv2d_2/Relu_FusedConv2D??RN@??!zD?????Q      Y@Y?Vpw:d$@a$??xsV@q+?????y????RS?"?

device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Maxwell)(: B 