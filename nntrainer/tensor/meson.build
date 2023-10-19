tensor_sources = [
  'blas_interface.cpp',
  'cache_elem.cpp',
  'cache_loader.cpp',
  'cache_pool.cpp',
  'lazy_tensor.cpp',
  'manager.cpp',
  'tensor.cpp',
  'tensor_dim.cpp',
  'var_grad.cpp',
  'weight.cpp',
  'basic_planner.cpp',
  'memory_pool.cpp',
  'swap_device.cpp',
  'tensor_pool.cpp',
  'optimized_v1_planner.cpp',
  'optimized_v2_planner.cpp',
  'optimized_v3_planner.cpp',
  'task_executor.cpp',
  'opencl/cl_add_impl.cpp',
  'opencl/cl_dot_product_impl.cpp',
  'opencl/cl_sgemv_impl.cpp',
  'opencl/opencl_buffer.cpp',
  'opencl/opencl_command_queue_manager.cpp',
  'opencl/opencl_context_manager.cpp',
  'opencl/opencl_kernel.cpp',
  'opencl/opencl_loader.cpp',
  'opencl/opencl_program.cpp',
]

tensor_headers = [
  'memory_data.h',
  'tensor.h',
  'tensor_wrap_specs.h',
  'opencl/CL/cl_platform.h'
]

arch = target_machine.cpu_family()

if get_option('platform') == 'android' or arch == 'aarch64' or arch =='arm'
  tensor_sources += 'blas_neon.cpp'
endif

foreach s : tensor_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : tensor_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
