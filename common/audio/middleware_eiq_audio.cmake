#Description: eIQ audio stream; user_visible: False
include_guard(GLOBAL)
message("middleware_eiq_audio component is included.")

target_sources(${MCUX_SDK_PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/rtxxx/audio_capture.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rtxxx/audio_stream.c
)

target_include_directories(${MCUX_SDK_PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/rtxxx
)


