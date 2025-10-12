#pragma once

#include <chrono>

namespace fmr {

    class frames_per_second {
        using clock = std::chrono::steady_clock;
        using time_point = clock::time_point;

    public:
        explicit frames_per_second(double smoothing = 0.9)
            : m_smoothing(smoothing), m_last(clock::now()), m_fps(0.0) {}

        void update() {
            auto now = clock::now();
            std::chrono::duration<double> delta = now - m_last;
            m_last = now;

            double instant_fps = 1.0 / std::max(delta.count(), 1e-6);
            // Apply exponential moving average (smooth FPS)
            m_fps = (m_fps == 0.0)
                        ? instant_fps
                        : (m_fps * m_smoothing + instant_fps * (1.0 - m_smoothing));
        }

        double fps() const noexcept { return m_fps; }

    private:
        double m_smoothing;
        double m_fps;
        time_point m_last;
    };

}
