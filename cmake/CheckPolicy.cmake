file(GLOB_RECURSE _tilemega_sources
  "${CMAKE_CURRENT_LIST_DIR}/../include/*"
  "${CMAKE_CURRENT_LIST_DIR}/../lib/*")

foreach(_file IN LISTS _tilemega_sources)
  file(READ "${_file}" _text)
  if(_text MATCHES "__CUDA_ARCH__" AND
     NOT _file MATCHES "/Target/ArchDispatch\\.h$")
    message(FATAL_ERROR "__CUDA_ARCH__ outside ArchDispatch.h: ${_file}")
  endif()
  if(_text MATCHES "sm_major[ \t]*>=|sm_minor[ \t]*>=")
    message(FATAL_ERROR "architecture version comparison in business code: ${_file}")
  endif()
  if(_text MATCHES "num_sms[ \t]*=[ \t]*(128|132|108|80)|max_(dynamic_)?smem[^=]*=[ \t]*(49152|101376|227328)")
    message(FATAL_ERROR "hardware resource literal in business code: ${_file}")
  endif()
endforeach()

message(STATUS "TileMega policy checks passed")
