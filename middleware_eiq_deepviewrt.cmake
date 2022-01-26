#Description: DeepViewRT Lib; user_visible: True
include_guard(GLOBAL)
message("middleware_eiq_deepviewrt component is included.")

target_sources(${MCUX_SDK_PROJECT_NAME} PRIVATE
)

target_include_directories(${MCUX_SDK_PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/deepviewrt/include
)


