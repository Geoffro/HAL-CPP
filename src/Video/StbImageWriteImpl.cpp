// Sole translation unit that compiles stb_image_write's implementation
// (third_party/stb/stb_image_write.h, public domain, Sean Barrett) —
// every other file that needs its declarations includes the header
// without defining STB_IMAGE_WRITE_IMPLEMENTATION, per the library's own
// single-header convention, so the implementation is compiled exactly
// once.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
