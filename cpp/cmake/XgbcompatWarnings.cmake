function(xgbcompat_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target}
      PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
    )
  endif()
endfunction()

function(xgbcompat_apply_nix_cflags target)
  if(DEFINED ENV{NIX_CFLAGS_COMPILE})
    separate_arguments(XGBCOMPAT_NIX_CFLAGS UNIX_COMMAND "$ENV{NIX_CFLAGS_COMPILE}")
    set(XGBCOMPAT_EXPECT_SYSTEM_INCLUDE OFF)
    foreach(flag IN LISTS XGBCOMPAT_NIX_CFLAGS)
      if(XGBCOMPAT_EXPECT_SYSTEM_INCLUDE)
        target_compile_options(${target} PRIVATE "-isystem${flag}")
        set(XGBCOMPAT_EXPECT_SYSTEM_INCLUDE OFF)
      elseif(flag STREQUAL "-isystem")
        set(XGBCOMPAT_EXPECT_SYSTEM_INCLUDE ON)
      endif()
    endforeach()
  endif()
  foreach(dir IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
    if(EXISTS "${dir}")
      target_compile_options(${target} PRIVATE "-isystem${dir}")
    endif()
  endforeach()
endfunction()
