project('robomagellan', 'cpp')

opencv = dependency('opencv')
boost_options = dependency('boost', modules: ['program_options'])
uvc = find_library('uvc', required : false)
ceres = find_library('ceres', required : true)

executable('capture', 
  sources : [
    'capture.cc', 
    'camera.cc', 
    'utils.cc',
    'frame_buffer_queue.cc'],
  dependencies: [ opencv, boost_options, uvc ])

executable('process',
  sources : [
    'process.cc',
    'utils.cc',
    'raw_video_reader.cc',
    'rigid_estimator.cc',
    'clique.cc'],
  dependencies: [ opencv, boost_options ] )

library('reprojection_estimator_lib',
  sources : [
    'reprojection_estimator.cc'
  ],
  dependencies: [ ceres ])

rigid_estimator_test = executable('rigid_estimator_test',
  sources : [
    'rigid_estimator.cc',
    'rigid_estimator_test.cc'
  ],
  dependencies: [ opencv ])

clique_test = executable('clique_test',
  sources : [
    'clique.cc',
    'clique_test.cc'
  ])

test('rigid_estimator_test', rigid_estimator_test)
test('clique_test', clique_test)
