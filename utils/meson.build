project('robomagellan', 'cpp')

opencv = dependency('opencv')
boost_options = dependency('boost', modules: ['program_options'])
uvc = find_library('uvc', required : false)

executable('capture', 
  sources : [
    'capture.cc', 
    'camera.cc', 
    'frame_buffer_queue.cc'],
  dependencies: [opencv, boost_options, uvc])

executable('process',
  sources : [
    'process.cc',
    'utils.cc',
    'raw_video_reader.cc'],
  dependencies: [ opencv, boost_options ] )
