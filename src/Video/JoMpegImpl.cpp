// Sole translation unit that compiles jo_mpeg's implementation
// (third_party/jo_mpeg/jo_mpeg.h, public domain, Jon Olick) — every other
// file that needs jo_write_mpeg() defines JO_MPEG_HEADER_FILE_ONLY before
// including the header (see VideoWriter.cpp), so the implementation is
// compiled exactly once here.
#include "jo_mpeg/jo_mpeg.h"
