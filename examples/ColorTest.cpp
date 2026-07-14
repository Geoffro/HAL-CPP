// Headless port of LEARN_HERE/Other/ColorTest.java: walk a packed ARGB
// color through the channel getters/setters and print it at each step.

#include "Hal/Color.h"

#include <cstdio>

int main()
{
    int32_t color = Hal::Rgb(1.0, 0.0, 0.0);
    std::printf("%s\n", Hal::ColorString(color).c_str());
    color = Hal::SetBlue(color, 1.0);
    std::printf("%s\n", Hal::ColorString(color).c_str());
    color = Hal::SetRed256(color, 0);
    std::printf("%s\n", Hal::ColorString(color).c_str());
    color = Hal::SetGreen256(color, Hal::GetBlue256(color));
    std::printf("%s\n", Hal::ColorString(color).c_str());
    color = Hal::SetAlpha256(color, 0);
    std::printf("%s\n", Hal::ColorString(color).c_str());
    return 0;
}
