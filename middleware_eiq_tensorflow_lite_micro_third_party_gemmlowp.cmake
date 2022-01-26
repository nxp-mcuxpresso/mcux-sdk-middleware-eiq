#Description: Gemmlowp library; user_visible: False
include_guard(GLOBAL)
message("middleware_eiq_tensorflow_lite_micro_third_party_gemmlowp component is included.")

target_sources(${MCUX_SDK_PROJECT_NAME} PRIVATE
)

target_include_directories(${MCUX_SDK_PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/tensorflow-lite/third_party/gemmlowp
)


