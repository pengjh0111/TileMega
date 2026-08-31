# V-I is populated by the cross-compilation verification. Keep configuration
# valid while verification is disabled or before its generated matrix exists.
add_custom_target(crosscompile-matrix
  COMMAND ${CMAKE_COMMAND} -E env
          TILEMEGA_SOURCE_DIR=${CMAKE_SOURCE_DIR}
          ${CMAKE_SOURCE_DIR}/docs/experiments/V_I/run.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  USES_TERMINAL)
