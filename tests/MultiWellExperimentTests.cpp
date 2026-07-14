#include <doctest/doctest.h>

#include "Hal/Color.h"
#include "Hal/Tools/MultiWellExperiment.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

struct WellModel
{
    int32_t id;
    int32_t steps;
};

static void StepModel(WellModel &model, int32_t /*wellIndex*/)
{
    model.steps++;
}

// Encodes the well id, local x, and local y into the RGB channels so the
// composite can be read back and each pixel traced to its well and
// well-local coordinate.
static int32_t DrawModel(const WellModel &model, int32_t x, int32_t y)
{
    return Hal::Rgb256(model.id, x, y);
}

TEST_CASE("Hal::MultiWellExperiment: composite dimensions account for inter-well borders")
{
    std::vector<WellModel> models = {{10, 0}, {20, 0}, {30, 0}, {40, 0}};
    std::vector<WellModel *> wells = {&models[0], &models[1], &models[2], &models[3]};

    Hal::MultiWellExperiment<WellModel> experiment(2, 2, 3, 2, Hal::Rgb256(200, 100, 50), wells);
    // 2 wells * 3px + 1 border, 2 wells * 2px + 1 border.
    CHECK(experiment.CompositeWidth() == 7);
    CHECK(experiment.CompositeHeight() == 5);
    CHECK(experiment.Pixels().size() == static_cast<size_t>(7 * 5 * 4));
}

TEST_CASE("Hal::MultiWellExperiment: Step and StepMultiThread advance every well")
{
    std::vector<WellModel> models = {{10, 0}, {20, 0}, {30, 0}, {40, 0}};
    std::vector<WellModel *> wells = {&models[0], &models[1], &models[2], &models[3]};

    Hal::MultiWellExperiment<WellModel> experiment(2, 2, 3, 2, 0, wells);

    experiment.Step(StepModel);
    for (const WellModel &model : models)
    {
        CHECK(model.steps == 1);
    }

    experiment.StepMultiThread(StepModel);
    for (const WellModel &model : models)
    {
        CHECK(model.steps == 2);
    }
}

TEST_CASE("Hal::MultiWellExperiment: Render places each well at the right offset with borders between")
{
    std::vector<WellModel> models = {{10, 0}, {20, 0}, {30, 0}, {40, 0}};
    std::vector<WellModel *> wells = {&models[0], &models[1], &models[2], &models[3]};

    int32_t borderColor = Hal::Rgb256(200, 100, 50);
    Hal::MultiWellExperiment<WellModel> experiment(2, 2, 3, 2, borderColor, wells);
    experiment.Render(DrawModel);

    int32_t width = experiment.CompositeWidth();
    const std::vector<uint8_t> &pixels = experiment.Pixels();
    auto pixelAt = [&](int32_t cx, int32_t cy, int32_t channel)
    {
        return static_cast<int32_t>(pixels[(static_cast<size_t>(cy) * static_cast<size_t>(width) + static_cast<size_t>(cx)) * 4 + static_cast<size_t>(channel)]);
    };

    // Well 0 (id 10) occupies composite (0..2, 0..1). Local (0,0) at (0,0).
    CHECK(pixelAt(0, 0, 0) == 10);
    CHECK(pixelAt(0, 0, 1) == 0);
    CHECK(pixelAt(0, 0, 2) == 0);
    // Well 0 local (2,1) at composite (2,1).
    CHECK(pixelAt(2, 1, 0) == 10);
    CHECK(pixelAt(2, 1, 1) == 2);
    CHECK(pixelAt(2, 1, 2) == 1);

    // Well 2 (id 30) is the next column: xWell=1 -> xStart=4. Local (0,0)
    // at composite (4,0).
    CHECK(pixelAt(4, 0, 0) == 30);
    CHECK(pixelAt(4, 0, 1) == 0);

    // Well 1 (id 20) is below well 0: yWell=1 -> yStart=3. Local (0,0) at
    // composite (0,3).
    CHECK(pixelAt(0, 3, 0) == 20);

    // Well 3 (id 40) local (2,1) at composite (6,4).
    CHECK(pixelAt(6, 4, 0) == 40);
    CHECK(pixelAt(6, 4, 1) == 2);
    CHECK(pixelAt(6, 4, 2) == 1);

    // Border column (x=3) and border row (y=2) hold the border color.
    CHECK(pixelAt(3, 0, 0) == 200);
    CHECK(pixelAt(3, 0, 1) == 100);
    CHECK(pixelAt(3, 0, 2) == 50);
    CHECK(pixelAt(0, 2, 0) == 200);
    CHECK(pixelAt(0, 2, 1) == 100);
    CHECK(pixelAt(0, 2, 2) == 50);
}

TEST_CASE("Hal::MultiWellExperiment: unused layout cells stay border-colored")
{
    // 3 wells in a 2x2 layout leaves one cell empty.
    std::vector<WellModel> models = {{10, 0}, {20, 0}, {30, 0}};
    std::vector<WellModel *> wells = {&models[0], &models[1], &models[2]};

    int32_t borderColor = Hal::Rgb256(7, 8, 9);
    Hal::MultiWellExperiment<WellModel> experiment(2, 2, 3, 2, borderColor, wells);
    experiment.Render(DrawModel);

    int32_t width = experiment.CompositeWidth();
    const std::vector<uint8_t> &pixels = experiment.Pixels();
    // The 4th cell (well index 3: xWell=1,yWell=1 -> composite (4,3)) is
    // never drawn, so it holds the border color.
    size_t idx = (static_cast<size_t>(3) * static_cast<size_t>(width) + static_cast<size_t>(4)) * 4;
    CHECK(static_cast<int32_t>(pixels[idx]) == 7);
    CHECK(static_cast<int32_t>(pixels[idx + 1]) == 8);
    CHECK(static_cast<int32_t>(pixels[idx + 2]) == 9);
}

TEST_CASE("Hal::MultiWellExperiment: too many wells for the layout throws")
{
    std::vector<WellModel> models(5, WellModel{0, 0});
    std::vector<WellModel *> wells;
    for (WellModel &model : models)
    {
        wells.push_back(&model);
    }
    CHECK_THROWS_AS(
        (Hal::MultiWellExperiment<WellModel>(2, 2, 3, 2, 0, wells)),
        std::invalid_argument);
}
