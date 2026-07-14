// Ported from LEARN_HERE/Agents/SaveLoadModel: shows saving and restoring an
// entire model's state (all agents + RNG stream) mid-run. Java uses native
// object serialization (`SerializableModel` + `SaveState`/`LoadState`); this
// port uses the reflection-free equivalent — `AgentGrid2D::SaveAgents`/
// `LoadAgents` with a per-agent field callback, plus `Rand::Save`/`Load` for
// the RNG stream — written to a binary file (see docs/DESIGN.md).
//
// Live window: press S to save the current state, L to reload it. Pass
// "headless" to instead run a scripted save -> keep-stepping -> reload cycle
// and confirm the reload restores the saved population; dumps saveloadmodel.png.

#include "Hal/Color.h"
#include "Hal/IO/BinaryReader.h"
#include "Hal/IO/BinaryWriter.h"
#include "Hal/Log/Logger.h"

#include "common/GridWindow.h"
#include "models/BirthDeathModel.h"

#include <GLFW/glfw3.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

using Hal::Examples::BirthDeath;
using Hal::Examples::Cell;

static const std::string g_statePath = "saveloadmodel_state.bin";

static void SaveCellColor(Cell              &cell,
                          Hal::BinaryWriter &writer)
{
    writer.WriteInt32(cell.Color());
}

static void LoadCellColor(Cell              &cell,
                          Hal::BinaryReader &reader)
{
    cell.SetColor(reader.ReadInt32());
}

static void SaveState(BirthDeath &model)
{
    Hal::BinaryWriter writer(g_statePath);
    model.Rng().Save(writer);
    model.SaveAgents(writer, SaveCellColor);
}

// Rebuilds a fresh model from the saved file (matching dimensions).
static std::unique_ptr<BirthDeath> LoadState(int32_t x,
                                             int32_t y,
                                             int32_t seedColor)
{
    std::unique_ptr<BirthDeath> pModel = std::make_unique<BirthDeath>(x, y, seedColor);
    Hal::BinaryReader           reader(g_statePath);
    pModel->Rng().Load(reader);
    pModel->LoadAgents(reader, LoadCellColor);
    return pModel;
}

static void DrawModel(BirthDeath                &model,
                      Hal::Examples::GridWindow &win)
{
    for (int32_t i = 0; i < model.Length(); i++)
    {
        Cell *pCell = model.GetAgent(i);
        win.SetPix(i, pCell == nullptr ? Hal::BLACK : pCell->Color());
    }
}

int main(int argc, char **argv)
{
    try
    {
        bool          headless = (argc > 1) && (std::strcmp(argv[1], "headless") == 0);
        const int32_t x = 100;
        const int32_t y = 100;

        auto model = std::make_unique<BirthDeath>(x, y, Hal::RED);
        model->Setup(10);
        Hal::Examples::GridWindow win("Save / Load (S=save, L=load)", x, y, 6, !headless);

        if (headless)
        {
            // Scripted demo: step, save, keep stepping, then reload and confirm
            // the population matches the saved snapshot rather than the drifted
            // one.
            for (int32_t i = 0; i < 100; i++)
            {
                model->Step();
            }
            SaveState(*model);
            int32_t savedPop = model->Pop();
            for (int32_t i = 0; i < 100; i++)
            {
                model->Step();
            }
            int32_t driftedPop = model->Pop();
            model = LoadState(x, y, Hal::RED);
            std::printf("saved pop=%d, drifted pop=%d, reloaded pop=%d\n", savedPop, driftedPop, model->Pop());
            DrawModel(*model, win);
            win.ToPng("saveloadmodel.png");
            std::remove(g_statePath.c_str());
            return 0;
        }

        GLFWwindow *pWindow = static_cast<GLFWwindow *>(win.NativeHandle());
        bool        hasSaved = false;
        for (int32_t i = 0; (i < 1000000) && !win.IsClosed(); i++)
        {
            if (glfwGetKey(pWindow, GLFW_KEY_S) == GLFW_PRESS)
            {
                SaveState(*model);
                hasSaved = true;
            }
            if ((glfwGetKey(pWindow, GLFW_KEY_L) == GLFW_PRESS) && hasSaved)
            {
                model = LoadState(x, y, Hal::RED);
            }
            win.TickPause(10);
            model->Step();
            DrawModel(*model, win);
        }
        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("SaveLoadModel failed: ") + e.what());
        return 1;
    }
}
